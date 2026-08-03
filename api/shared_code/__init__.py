import os
import re
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from google import genai

# Initialize the Gemini client with the API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Model config
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"
MODEL_NAME = (os.environ.get("GEMINI_MODEL_NAME") or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_PROVIDER_MAX_CONCURRENCY = 4
MAX_MESSAGE_CHARS = 10000
MAX_HISTORY_MESSAGES = 20
MAX_HISTORY_MESSAGE_CHARS = 10000
MAX_HISTORY_TOTAL_CHARS = 20000
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 20

_rate_limit_lock = threading.Lock()
_rate_limit_hits = defaultdict(deque)
_timeout_executor = ThreadPoolExecutor(max_workers=8)


def _positive_int_env(name, default):
    value = (os.environ.get(name) or "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


MAX_OUTPUT_TOKENS = _positive_int_env("GEMINI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS)
REQUEST_TIMEOUT_SECONDS = _positive_int_env("GEMINI_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS)
PROVIDER_MAX_CONCURRENCY = _positive_int_env(
    "GEMINI_MAX_CONCURRENT_REQUESTS",
    DEFAULT_PROVIDER_MAX_CONCURRENCY,
)
_provider_semaphore = threading.BoundedSemaphore(PROVIDER_MAX_CONCURRENCY)

# System instruction — passed via config.system_instruction.
# The verdict phrase appears only ONCE so clean_reply can reliably distinguish
# instruction echoes (one occurrence) from real model output (last occurrence).
SYSTEM_INSTRUCTION = (
    "You are Judge Chuckles, a pompous, lovable AI courtroom judge who delivers short, absurd verdicts.\n\n"
    "Your reply must have three parts: a verdict declaration, then two funny explanation paragraphs.\n\n"
    "The verdict declaration is always one of these exact two lines and nothing else before it:\n"
    "The Court Declares: Guilty!\n"
    "The Court Declares: Not Guilty!\n\n"
    "After the verdict line write exactly two playful paragraphs in plain prose. "
    "The first paragraph gives the ruling in one or two funny sentences. "
    "The second paragraph delivers an absurd sentence or consequence in one or two funny sentences. "
    "Do not quote the verdict line. "
    "Do not add any label, header, preamble, reasoning, planning step, or self-check. "
    "Do not include word counts, checks, final plans, compliance notes, or commentary about these instructions. "
    "Do not use bullet points, dashes, numbered lists, or any markdown. "
    "Do not repeat or paraphrase the question. "
    "Keep it playful and absurd. Never offensive or biased. Entertainment only."
)

# Short priming exchange injected into contents alongside system_instruction.
# Deliberately avoids 'The Court Declares:' so clean_reply can use the LAST
# verdict match as the real answer, even if the model echoes the instruction first.
_PRIMING_INSTRUCTION = (
    "You play Judge Chuckles, a silly AI judge. "
    "Open each reply with a one-line guilty-or-not verdict declaration, "
    "then add two playful explanation paragraphs with distinct jokes. "
    "Output only the final answer — no preamble, no labels, no planning, no drafts."
)
_PRIMING_ACK = (
    "Got it! I\'m Judge Chuckles. I\'ll open with a verdict declaration and keep it short, absurd, and fun."
)

COCONUT_FALLBACK = "I... I got nothing. My brain is empty. Like a coconut."


class RequestValidationError(ValueError):
    """Raised for client-correctable chat request validation failures."""


class ProviderTimeoutError(TimeoutError):
    """Raised when the model provider call exceeds the configured timeout."""


class ProviderBusyError(RuntimeError):
    """Raised when local provider concurrency is already saturated."""


def _headers_get(headers, name):
    if not headers:
        return None
    try:
        return headers.get(name) or headers.get(name.lower())
    except AttributeError:
        return None


def client_rate_limit_key(req):
    """Return a best-effort client key from proxy headers."""
    headers = getattr(req, "headers", {}) or {}
    forwarded = _headers_get(headers, "x-forwarded-for")
    if forwarded:
        return forwarded.split(",", 1)[0].strip() or "unknown"

    for header in ("x-client-ip", "x-real-ip"):
        value = _headers_get(headers, header)
        if value:
            return value.strip()

    return "unknown"


def check_rate_limit(req):
    """In-process request throttle for anonymous chat endpoints."""
    now = time.time()
    key = client_rate_limit_key(req)

    with _rate_limit_lock:
        hits = _rate_limit_hits[key]
        while hits and now - hits[0] >= RATE_LIMIT_WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_MAX_REQUESTS:
            retry_after = max(1, int(RATE_LIMIT_WINDOW_SECONDS - (now - hits[0])))
            return False, retry_after
        hits.append(now)
        return True, None


def reset_rate_limits():
    """Test helper for clearing in-memory throttle state."""
    with _rate_limit_lock:
        _rate_limit_hits.clear()


def validate_chat_payload(data):
    """Validate and normalize a chat request payload."""
    if not isinstance(data, dict):
        raise RequestValidationError("Invalid JSON body.")

    raw_message = data.get("message")
    if not isinstance(raw_message, str):
        raise RequestValidationError("Message must be text.")

    user_message = raw_message.strip()
    if not user_message:
        raise RequestValidationError("Message is required.")
    if len(user_message) > MAX_MESSAGE_CHARS:
        raise RequestValidationError(f"Message too long. Keep it under {MAX_MESSAGE_CHARS:,} characters.")

    raw_history = data.get("history", [])
    if raw_history is None:
        raw_history = []
    if not isinstance(raw_history, list):
        raise RequestValidationError("History must be a list.")
    if len(raw_history) > MAX_HISTORY_MESSAGES:
        raise RequestValidationError(f"History too long. Keep it to {MAX_HISTORY_MESSAGES} messages.")

    history = []
    total_history_chars = 0
    for item in raw_history:
        if not isinstance(item, dict):
            raise RequestValidationError("History entries must be objects.")

        role = item.get("role")
        if role not in ("user", "assistant", "model"):
            raise RequestValidationError("History entry role is invalid.")

        content = item.get("content")
        if not isinstance(content, str):
            raise RequestValidationError("History entry content must be text.")

        content = content.strip()
        if not content:
            continue
        if len(content) > MAX_HISTORY_MESSAGE_CHARS:
            raise RequestValidationError(
                f"History messages must be under {MAX_HISTORY_MESSAGE_CHARS:,} characters."
            )

        total_history_chars += len(content)
        if total_history_chars > MAX_HISTORY_TOTAL_CHARS:
            raise RequestValidationError(
                f"Total history too long. Keep it under {MAX_HISTORY_TOTAL_CHARS:,} characters."
            )

        history.append({"role": "assistant" if role in ("assistant", "model") else "user", "content": content})

    return user_message, history


def run_with_timeout(fn):
    """Run a blocking provider operation with a bounded wait."""
    if not _provider_semaphore.acquire(blocking=False):
        raise ProviderBusyError("Provider concurrency limit reached.")

    try:
        future = _timeout_executor.submit(fn)
        try:
            return future.result(timeout=REQUEST_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            future.cancel()
            raise ProviderTimeoutError("Provider request timed out.") from exc
    finally:
        _provider_semaphore.release()

# Matches a complete verdict declaration in either guilty or not-guilty form.
# More specific than 'The Court Declares:' alone, which can appear twice inside
# an echoed instruction and cause the old first-match strategy to clip mid-phrase.
_VERDICT_RE = re.compile(r'The Court Declares:\s*(?:Not\s+)?Guilty!', re.IGNORECASE)

# Signals the model is showing a second draft or internal planning that leaked out.
_REDRAFT_RE = re.compile(
    r'\n+(?:Verdict:\s|Content:\s|User question:|Role:\s|'
    r'Constraint\s*\d*[: ]|Plain prose|Hard rules|Output format|'
    r'\(?Word count\b|Checking\s+[\'"`]|(?:Final\s+)?Plan\s*:|'
    r'Self-check\s*:|Self-correction\s*:|'
    r'Wait,?\s+(?:the\s+)?instructions?\s+(?:say|said)\b|'
    r'The\s+instructions?\s+(?:say|said)\b|'
    r'Compliance\s+(?:check|note)\s*:)',
    re.IGNORECASE,
)


def _select_verdict_match(text, matches):
    """Return the last verdict declaration that starts its own line."""
    for match in reversed(matches):
        line_start = text.rfind("\n", 0, match.start()) + 1
        if not text[line_start:match.start()].strip():
            return match
    return matches[-1]


def _select_reply_start(text, matches):
    leak = _REDRAFT_RE.search(text)
    eligible = matches
    if leak:
        before_leak = [match for match in matches if match.start() < leak.start()]
        if before_leak:
            eligible = before_leak
    return _select_verdict_match(text, eligible)


def build_contents(history, user_message):
    """Build contents with a short priming exchange that avoids the verdict phrase.

    Full persona rules go in config.system_instruction. The priming exchange gives
    an extra behavioral cue without containing 'The Court Declares:', so
    clean_reply can always locate the real verdict via the last regex match.
    """
    contents = [
        {"role": "user", "parts": [{"text": _PRIMING_INSTRUCTION}]},
        {"role": "model", "parts": [{"text": _PRIMING_ACK}]},
    ]
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


def clean_reply(text: str) -> str:
    """Trim everything outside the actual verdict.

    Uses the last complete verdict match before any planning-leak marker as the
    real answer start. If no marker appears first, this still handles instruction
    echoes by taking the last verdict match. Then cuts at re-draft / planning-leak
    markers and a second verdict declaration (model looping).
    """
    matches = list(_VERDICT_RE.finditer(text))
    if not matches:
        def _clean_line(l):
            l = l.lstrip()
            l = re.sub(r'^[-*\u2022]\s+', '', l)
            l = re.sub(r'^\d+[.)\s]\s*', '', l)
            return l
        lines = [_clean_line(l) for l in text.strip().splitlines()]
        return "\n".join(lines)

    last = _select_reply_start(text, matches)
    clipped = text[last.start():]

    # Cut at re-draft / planning-leak markers
    redraft = _REDRAFT_RE.search(clipped)
    if redraft:
        clipped = clipped[:redraft.start()]

    # Cut before a second verdict in the clipped portion (model looping)
    first_end = last.end() - last.start()
    second = _VERDICT_RE.search(clipped, first_end)
    if second:
        clipped = clipped[:second.start()]

    # Strip leading whitespace from every line so Markdown never renders
    # indented text as a <pre> code block.
    # Strip leading whitespace and markdown list markers from every line.
    def _clean_line(l):
        l = l.lstrip()
        l = re.sub(r'^[-*\u2022]\s+', '', l)  # remove - / * / • list markers
        l = re.sub(r'^\d+[.)\s]\s*', '', l)  # remove 1. / 1) numbered markers
        return l
    lines = [_clean_line(l) for l in clipped.strip().splitlines()]
    return "\n".join(lines)


def extract_reply_text(response):
    """Return (reply_text, empty_kind) where empty_kind is None when text is usable."""
    text = getattr(response, "text", None)

    if isinstance(text, str):
        cleaned = text.strip()
        if cleaned:
            return cleaned, None
        return "", "blank_text"

    if text is None:
        return "", "missing_text"

    rendered = str(text).strip()
    if rendered:
        return rendered, None
    return "", "non_string_empty"


def response_diagnostics(response):
    """Collect safe response shape metadata for logging empty-output cases."""
    text = getattr(response, "text", None)
    candidates = getattr(response, "candidates", None)
    finish_reasons = []
    if candidates:
        for candidate in candidates:
            finish_reason = getattr(candidate, "finish_reason", None)
            if finish_reason is not None:
                finish_reasons.append(str(finish_reason))

    diag = {
        "has_text_attr": hasattr(response, "text"),
        "text_type": type(text).__name__,
        "text_length": len(text) if isinstance(text, str) else None,
        "has_candidates_attr": hasattr(response, "candidates"),
        "candidate_count": len(candidates) if isinstance(candidates, list) else None,
        "finish_reasons": finish_reasons,
    }
    return diag


def classify_genai_error(exc: Exception):
    """Best-effort classification of Google GenAI failures.

    We avoid importing provider-specific exception types because the underlying
    libraries can vary in Azure builds; string-matching is more robust.
    """
    text = (str(exc) or repr(exc)).strip()
    upper = text.upper()

    # Authentication / authorization issues
    if (
        "401" in upper
        or "403" in upper
        or "UNAUTHENTICATED" in upper
        or "UNAUTHORIZED" in upper
        or "PERMISSION_DENIED" in upper
        or "INVALID API KEY" in upper
        or "API_KEY_INVALID" in upper
    ):
        return "auth_or_permission", text

    # Model/deployment lookup failures
    if (
        "404" in upper
        or "NOT_FOUND" in upper
        or "MODEL_NOT_FOUND" in upper
        or "NO SUCH MODEL" in upper
        or "UNKNOWN MODEL" in upper
    ):
        return "model_not_found", text

    # Provider-side overload / temporary outage
    if (
        "503" in upper
        or "UNAVAILABLE" in upper
        or "HIGH DEMAND" in upper
        or "CURRENTLY EXPERIENCING HIGH DEMAND" in upper
    ):
        return "provider_high_demand", text

    # Usage limits / quota / rate limits
    if (
        "429" in upper
        or "RESOURCE_EXHAUSTED" in upper
        or "QUOTA" in upper
        or "INSUFFICIENT" in upper
        or "RATE LIMIT" in upper
        or "TOO MANY REQUEST" in upper
        or "LIMIT" in upper and "TOKEN" in upper
    ):
        return "usage_limit", text

    return "unknown", text


def user_facing_error_message(exc: Exception) -> str:
    kind, _raw = classify_genai_error(exc)

    if kind == "usage_limit":
        return (
            "Sorry — we just hit today's AI usage limit. "
            "The a**hole who built this is too cheap to pay for more token usage. "
            "Try again later (or tomorrow)."
        )

    if kind == "provider_high_demand":
        return (
            "Sorry — our AI provider is experiencing high demand right now. "
            "Their servers are slammed, and my builder is too cheap to pay for higher availability. "
            "A++holes both of them. Try again in a minute."
        )

    if kind == "auth_or_permission":
        return (
            "Sorry — the AI service credentials look invalid or missing in this environment. "
            "Please ask the maintainer to check production secrets."
        )

    if kind == "model_not_found":
        return (
            "Sorry — the configured AI model could not be found. "
            "Please ask the maintainer to verify the deployed model name."
        )

    return "Sorry — the Court hit a technical snag. Please try again in a minute."
