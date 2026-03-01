import os
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
MODEL_NAME = "gemma-3-12b-it"

# System instruction — kept short for token efficiency (~80 tokens)
# Injected as a user/model exchange since Gemma 3 doesn't support system_instruction
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
            config={"max_output_tokens": 300},
        )

        reply = response.text if response.text else "I... I got nothing. My brain is empty. Like a coconut."

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
                    config={"max_output_tokens": 300},
                )
                for chunk in response_stream:
                    if chunk.text:
                        yield f"data: {json.dumps({'token': chunk.text})}\n\n"
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
