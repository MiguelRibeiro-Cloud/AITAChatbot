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

# System instruction — passed via config.system_instruction.
# The verdict phrase appears only ONCE so clean_reply can reliably distinguish
# instruction echoes (one occurrence) from real model output (last occurrence).
SYSTEM_INSTRUCTION = (
    "You are Judge Chuckles, a pompous, lovable AI courtroom judge who delivers short, absurd verdicts.\n\n"
    "Your entire reply consists of two things only: a verdict declaration, then a brief funny explanation.\n\n"
    "The verdict declaration is always one of these exact two lines and nothing else before it:\n"
    "The Court Declares: Guilty!\n"
    "The Court Declares: Not Guilty!\n\n"
    "After the verdict line write one or two short funny paragraphs in plain prose, under 150 words total. "
    "Do not quote the verdict line. "
    "Do not add any label, header, preamble, reasoning, planning step, or self-check. "
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
    r'Let\'s reconsider|Verdict:\s|Content:\s|User question:|Role:\s|'
    r'Constraint\s*\d*[: ]|Plain prose|Hard rules|Output format)',
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
        def _clean_line(l):
            l = l.lstrip()
            l = re.sub(r'^[-*\u2022]\s+', '', l)
            l = re.sub(r'^\d+[.)\s]\s*', '', l)
            return l
        lines = [_clean_line(l) for l in text.strip().splitlines()]
        return "\n".join(lines)

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
            config={"max_output_tokens": 500, "system_instruction": SYSTEM_INSTRUCTION},
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
                    config={"max_output_tokens": 500, "system_instruction": SYSTEM_INSTRUCTION},
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
