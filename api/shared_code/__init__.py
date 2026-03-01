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
