import os
import re
import json
import time
import traceback
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize the Gemini client with the API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Model config
MODEL_NAME = "gemma-4-26b-a4b-it"

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

# Patterns that signal the model is showing a second draft or internal reasoning.
_REDRAFT_RE = re.compile(
    r'\n+(?:Wait[,. ]|Actually[,. ]|Hmm[,. ]|Let me |On second thought|'
    r"Let's reconsider|Verdict:|User question:|Role:|Constraint\s*\d*[: ])",
    re.IGNORECASE,
)


def build_contents(history, user_message):
    """Build contents from history + current message only.

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
        return text.strip()

    text = text[idx:]

    m = _REDRAFT_RE.search(text)
    if m:
        text = text[:m.start()]

    second = text.find(marker, len(marker))
    if second != -1:
        text = text[:second]

    return text.strip()

# Simple in-memory rate limiting
_rate_limit = {}
RATE_LIMIT_WINDOW = 60  # seconds
RATE_LIMIT_MAX = 20  # requests per window


def check_rate_limit(ip):
    """Basic rate limiter. Returns True if allowed, False if blocked."""
    now = time.time()
    if ip not in _rate_limit:
        _rate_limit[ip] = []
    # Clean old entries
    _rate_limit[ip] = [t for t in _rate_limit[ip] if now - t < RATE_LIMIT_WINDOW]
    if len(_rate_limit[ip]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit[ip].append(now)
    return True


@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "alive", "model": MODEL_NAME, "vibe": "chaotic"})


@app.route("/api/chat", methods=["POST"])
def chat():
    """Send a message and get a response from Gemma 3 12B."""
    if not check_rate_limit(request.remote_addr):
        return jsonify({"error": "Whoa there, speedster! Too many requests. Take a breath and try again in a minute."}), 429

    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided. Say something, coward!"}), 400

        user_message = data["message"].strip()
        if not user_message:
            return jsonify({"error": "Empty message? Really? Try harder."}), 400

        if len(user_message) > 10000:
            return jsonify({"error": "That's a novel, not a message. Keep it under 10,000 characters."}), 400

        history = data.get("history", [])
        contents = build_contents(history, user_message)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={"max_output_tokens": 300, "system_instruction": SYSTEM_INSTRUCTION},
        )

        raw = response.text if response.text else "I... I got nothing. My brain is empty. Like a coconut."
        reply = clean_reply(raw) if response.text else raw

        return jsonify({
            "reply": reply,
            "model": MODEL_NAME,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": f"Something went catastrophically wrong: {str(e)}",
            "suggestion": "Maybe try again? Or don't. I'm a chatbot, not your therapist."
        }), 500


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Stream a response from Gemma 3 12B token by token."""
    if not check_rate_limit(request.remote_addr):
        return jsonify({"error": "Rate limit exceeded. Chill for a minute."}), 429

    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided."}), 400

        user_message = data["message"].strip()
        if not user_message:
            return jsonify({"error": "Empty message."}), 400

        if len(user_message) > 10000:
            return jsonify({"error": "Message too long. Keep it under 10,000 characters."}), 400

        history = data.get("history", [])
        contents = build_contents(history, user_message)

        def generate():
            try:
                response_stream = client.models.generate_content_stream(
                    model=MODEL_NAME,
                    contents=contents,
                    config={"max_output_tokens": 300, "system_instruction": SYSTEM_INSTRUCTION},
                )
                collected = []
                for chunk in response_stream:
                    if chunk.text:
                        collected.append(chunk.text)
                reply = clean_reply("".join(collected).strip()) or "I... I got nothing. My brain is empty. Like a coconut."
                yield f"data: {json.dumps({'token': reply})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
