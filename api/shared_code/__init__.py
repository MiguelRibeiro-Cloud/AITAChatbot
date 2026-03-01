import os
from google import genai

# Initialize the Gemini client with the API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Model config
MODEL_NAME = "gemma-3-12b-it"

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


def classify_genai_error(exc: Exception):
    """Best-effort classification of Google GenAI failures.

    We avoid importing provider-specific exception types because the underlying
    libraries can vary in Azure builds; string-matching is more robust.
    """
    text = (str(exc) or repr(exc)).strip()
    upper = text.upper()

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

    return "Sorry — the Court hit a technical snag. Please try again in a minute."
