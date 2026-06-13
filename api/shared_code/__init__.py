import os
import re
from google import genai

# Initialize the Gemini client with the API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Model config
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"
MODEL_NAME = (os.environ.get("GEMINI_MODEL_NAME") or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME

# System instruction — passed via config.system_instruction (Gemma 4 native support).
# Explicitly forbids echoing prompt structure or showing internal planning/drafts.
SYSTEM_INSTRUCTION = (
    "You are Judge Chuckles, a ridiculous courtroom judge who renders absurd moral verdicts. "
    "Follow these rules exactly:\n"
    "1. Begin EVERY reply with exactly 'The Court Declares: Guilty!' or 'The Court Declares: Not Guilty!' — no other opening, no preamble.\n"
    "2. Follow immediately with a funny 1–2 paragraph explanation, under 150 words total.\n"
    "3. Output only the final verdict and explanation. Never show planning, analysis, multiple drafts, or reasoning steps.\n"
    "4. Never repeat or paraphrase the user's question back to them.\n"
    "5. Never include labels like 'Role:', 'Verdict:', 'User question:', 'Constraint:', 'Wait...', or any internal structure.\n"
    "6. Be playful and absurd — courtroom comedy. Never offensive, mean-spirited, or biased toward any group.\n"
    "7. Entertainment only. No real legal or life advice."
)

COCONUT_FALLBACK = "I... I got nothing. My brain is empty. Like a coconut."

# Patterns that signal the model has started a second draft or is showing internal reasoning.
_REDRAFT_RE = re.compile(
    r'\n+(?:Wait[,. ]|Actually[,. ]|Hmm[,. ]|Let me |On second thought|'
    r'Let\'s reconsider|Verdict:|User question:|Role:|Constraint\s*\d*[: ])',
    re.IGNORECASE,
)


def build_contents(history, user_message):
    """Build the contents array from conversation history + current user message.

    System persona is passed separately via config.system_instruction — do NOT
    inject it here as a fake user/model exchange, as that causes Gemma 4 to echo
    the instruction labels back in its output.
    """
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


def clean_reply(text: str) -> str:
    """Trim noise before/after the actual verdict.

    1. Drop everything before the first 'The Court Declares:'.
    2. Cut at any re-draft marker (second draft, planning leakage).
    3. Cut at a second 'The Court Declares:' occurrence (model looping).
    """
    marker = "The Court Declares:"
    idx = text.find(marker)
    if idx == -1:
        # Model didn't follow the format — return as-is and let the caller decide.
        return text.strip()

    text = text[idx:]

    # Stop at re-draft / planning leakage
    m = _REDRAFT_RE.search(text)
    if m:
        text = text[:m.start()]

    # Stop at a second verdict declaration (model started looping)
    second = text.find(marker, len(marker))
    if second != -1:
        text = text[:second]

    return text.strip()


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

    diag = {
        "has_text_attr": hasattr(response, "text"),
        "text_type": type(text).__name__,
        "text_length": len(text) if isinstance(text, str) else None,
        "has_candidates_attr": hasattr(response, "candidates"),
        "candidate_count": len(candidates) if isinstance(candidates, list) else None,
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
