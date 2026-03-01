import os
import json
import logging
import azure.functions as func
from google import genai

app = func.FunctionApp()

# Initialize the Gemini client with the API key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Model config
MODEL_NAME = "gemma-3-12b-it"

# System instruction — injected as a user/model exchange since Gemma 3 doesn't support system_instruction
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


@app.route(route="health", methods=[func.HttpMethod.GET])
def health(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps({"status": "alive", "model": MODEL_NAME, "vibe": "chaotic"}),
        mimetype="application/json",
    )


@app.route(route="chat", methods=[func.HttpMethod.POST])
def chat(req: func.HttpRequest) -> func.HttpResponse:
    """Send a message and get a response from Gemma 3 12B."""
    try:
        data = req.get_json()
        if not data or "message" not in data:
            return func.HttpResponse(
                json.dumps({"error": "No message provided. Say something, coward!"}),
                status_code=400,
                mimetype="application/json",
            )

        user_message = data["message"].strip()
        if not user_message:
            return func.HttpResponse(
                json.dumps({"error": "Empty message? Really? Try harder."}),
                status_code=400,
                mimetype="application/json",
            )

        if len(user_message) > 10000:
            return func.HttpResponse(
                json.dumps({"error": "That's a novel, not a message. Keep it under 10,000 characters."}),
                status_code=400,
                mimetype="application/json",
            )

        history = data.get("history", [])
        contents = build_contents(history, user_message)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={"max_output_tokens": 300},
        )

        reply = response.text if response.text else "I... I got nothing. My brain is empty. Like a coconut."

        return func.HttpResponse(
            json.dumps({"reply": reply, "model": MODEL_NAME}),
            mimetype="application/json",
        )

    except Exception as e:
        logging.error(f"Chat error: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": f"Something went catastrophically wrong: {str(e)}",
                "suggestion": "Maybe try again? Or don't. I'm a chatbot, not your therapist.",
            }),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="chat/stream", methods=[func.HttpMethod.POST])
def chat_stream(req: func.HttpRequest) -> func.HttpResponse:
    """Return response in SSE format (collected, not true streaming) for frontend compatibility."""
    try:
        data = req.get_json()
        if not data or "message" not in data:
            return func.HttpResponse(
                json.dumps({"error": "No message provided."}),
                status_code=400,
                mimetype="application/json",
            )

        user_message = data["message"].strip()
        if not user_message:
            return func.HttpResponse(
                json.dumps({"error": "Empty message."}),
                status_code=400,
                mimetype="application/json",
            )

        if len(user_message) > 10000:
            return func.HttpResponse(
                json.dumps({"error": "Message too long. Keep it under 10,000 characters."}),
                status_code=400,
                mimetype="application/json",
            )

        history = data.get("history", [])
        contents = build_contents(history, user_message)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config={"max_output_tokens": 300},
        )

        reply = response.text if response.text else "I... I got nothing. My brain is empty. Like a coconut."

        # Return full response in SSE format — frontend SSE parser handles it correctly
        sse_body = f"data: {json.dumps({'token': reply})}\n\ndata: {json.dumps({'done': True})}\n\n"

        return func.HttpResponse(
            sse_body,
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        logging.error(f"Stream error: {str(e)}")
        return func.HttpResponse(
            f"data: {json.dumps({'error': str(e)})}\n\n",
            mimetype="text/event-stream",
        )
