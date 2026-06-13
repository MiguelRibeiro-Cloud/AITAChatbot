import os
from google import genai

# Initialize the Gemini client with the API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Model config
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"
MODEL_NAME = (os.environ.get("GEMINI_MODEL_NAME") or DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME

# System instruction
SYSTEM_INSTRUCTION = (
    "You are a silly courtroom judge. "
    "Every reply MUST start with exactly 'The Court Declares: Guilty!' or 'The Court Declares: Not Guilty!' based on the situation. "
    "Then give a brief, humorous 1-2 paragraph explanation (under 150 words). "
    "Be playful and teasing but never offensive, mean-spirited, or biased toward any group. "
    "Keep it lighthearted and absurd — like a courtroom comedy. "
    "Never give real advice. This is entertainment only."
)

SYSTEM_ACK = "Understood! I'm Judge Chuckles, ready to deliver silly verdicts. Every reply starts with 'The Court Declares: Guilty!' or 'Not Guilty!' followed by a short, funny explanation. Let's go!"

COCONUT_FALLBACK = "I... I got nothing. My brain is empty. Like a coconut."


def build_contents(history, user_message):
    """Build the contents array with system prompt injected as first exchange."""
    contents = [
        {"role": "user", "parts": [{"text": SYSTEM_INSTRUCTION}]},
        {"role": "model", "parts": [{"text": SYSTEM_ACK}]},
    ]
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


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
