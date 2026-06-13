import json
import logging
import azure.functions as func
from shared_code import (
    COCONUT_FALLBACK,
    MODEL_NAME,
    build_contents,
    classify_genai_error,
    client,
    extract_reply_text,
    response_diagnostics,
    user_facing_error_message,
)


def main(req: func.HttpRequest) -> func.HttpResponse:
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

        reply, empty_kind = extract_reply_text(response)
        if empty_kind:
            logging.warning(
                "Empty GenAI response fallback used: model=%s empty_kind=%s diagnostics=%s",
                MODEL_NAME,
                empty_kind,
                response_diagnostics(response),
            )
            reply = COCONUT_FALLBACK

        return func.HttpResponse(
            json.dumps({"reply": reply, "model": MODEL_NAME}),
            mimetype="application/json",
        )

    except Exception as e:
        kind, raw = classify_genai_error(e)
        logging.error(f"Chat error ({kind}): {raw}")

        status_code = 500
        if kind == "usage_limit":
            status_code = 429
        elif kind == "provider_high_demand":
            status_code = 503
        elif kind == "auth_or_permission":
            status_code = 401
        elif kind == "model_not_found":
            status_code = 502

        return func.HttpResponse(
            json.dumps({
                "error": user_facing_error_message(e),
            }),
            status_code=status_code,
            mimetype="application/json",
        )
