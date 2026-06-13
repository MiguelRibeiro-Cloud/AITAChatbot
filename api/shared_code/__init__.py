import os
import re
from google import genai

# Initialize the Gemini client with the API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Model config
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"
MODEL_NAME = (os.environ.get("GEMINI_MODEL_NAME") or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME

# System instruction — passed via config.system_instruction.
# The verdict phrase appears only ONCE so clean_reply can reliably distinguish
# instruction echoes (one occurrence) from real model output (last occurrence).
SYSTEM_INSTRUCTION = (
    "You are Judge Chuckles, a hilariously pompous AI courtroom judge.\n\n"
    "STRICT OUTPUT FORMAT — follow exactly:\n"
    "  Open with: The Court Declares: Guilty!\n"
    "  (Substitute \'Not Guilty\' when the person is clearly blameless.)\n"
    "  Then write 1-2 funny paragraphs, under 150 words total.\n\n"
    "HARD RULES:\n"
    "- Nothing before the verdict line. No preamble, no thinking, nothing.\n"
    "- One verdict only. No redrafts, no multiple attempts, no visible reasoning.\n"
    "- No labels in output: no Role, no Verdict header, no User question, no Constraint, no Wait.\n"
    "- Never repeat or paraphrase what the user said.\n"
    "- Playful and absurd only. Never offensive, mean-spirited, or biased.\n"
    "- Entertainment only — zero real legal advice."
)

# Short priming exchange injected into contents alongside system_instruction.
# Deliberately avoids 'The Court Declares:' so clean_reply can use the LAST
# verdict match as the real answer, even if the model echoes the instruction first.
_PRIMING_INSTRUCTION = (
    "You play Judge Chuckles, a silly AI judge. "
    "Open each reply with a one-line guilty-or-not verdict declaration, "
    "then add a short funny reason (under 150 words total). "
    "Output only the final answer — no preamble, no labels, no planning, no drafts."
)
_PRIMING_ACK = (
    "Got it! I\'m Judge Chuckles. I\'ll open with a verdict declaration and keep it short, absurd, and fun."
)

COCONUT_FALLBACK = "I... I got nothing. My brain is empty. Like a coconut."

# Matches a complete verdict declaration in either guilty or not-guilty form.
# More specific than 'The Court Declares:' alone, which can appear twice inside
# an echoed instruction and cause the old first-match strategy to clip mid-phrase.
_VERDICT_RE = re.compile(r'The Court Declares:\s*(?:Not\s+)?Guilty!', re.IGNORECASE)

# Signals the model is showing a second draft or internal planning that leaked out.
_REDRAFT_RE = re.compile(
    r'\n+(?:Wait[,. ]|Actually[,. ]|Hmm[,. ]|Let me |On second thought|'
    r'Let\'s reconsider|Verdict:\s|User question:|Role:\s|Constraint\s*\d*[: ])',
    re.IGNORECASE,
)


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

    Uses the LAST complete verdict match as the start of the real answer.
    If the model echoes the system instruction (one verdict phrase) before giving
    the real verdict, the last match is the real one. Then cuts at re-draft /
    planning-leak markers and a second verdict declaration (model looping).
    """
    matches = list(_VERDICT_RE.finditer(text))
    if not matches:
        return text.strip()

    last = matches[-1]
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

    return clipped.strip()


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
