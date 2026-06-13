import json
import logging
import traceback
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


def _debug_payload(stage, exc=None, empty_kind=None, response_empty=None, chunks_empty=None):
    return {
        "stage": stage,
        "type": type(exc).__name__ if exc else None,
        "message": str(exc) if exc else None,
        "model": MODEL_NAME,
        "response_text_empty": bool(response_empty) if response_empty is not None else None,
        "chunks_empty": bool(chunks_empty) if chunks_empty is not None else None,
        "empty_kind": empty_kind,
    }


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Return response in SSE format for frontend compatibility."""
    try:
        data = req.get_json()
        if not data or "message" not in data:
            return func.HttpResponse(
                json.dumps(
                    {
                        "error": "No message provided.",
                        "debug": _debug_payload(stage="validation"),
                    }
                ),
                status_code=400,
                mimetype="application/json",
            )

        user_message = data["message"].strip()
        if not user_message:
            return func.HttpResponse(
                json.dumps(
                    {
                        "error": "Empty message.",
                        "debug": _debug_payload(stage="validation"),
                    }
                ),
                status_code=400,
                mimetype="application/json",
            )

        if len(user_message) > 10000:
            return func.HttpResponse(
                json.dumps(
                    {
                        "error": "Message too long. Keep it under 10,000 characters.",
                        "debug": _debug_payload(stage="validation"),
                    }
                ),
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
                "Empty GenAI stream fallback used: model=%s empty_kind=%s diagnostics=%s",
                MODEL_NAME,
                empty_kind,
                response_diagnostics(response),
            )
            reply = COCONUT_FALLBACK

        chunk_present = bool(reply and reply.strip())

        # Return full response in SSE format — frontend SSE parser handles it correctly
        sse_body = (
            f"data: {json.dumps({'debug': _debug_payload(stage='empty_response' if empty_kind else 'genai_call', empty_kind=empty_kind, response_empty=bool(empty_kind), chunks_empty=not chunk_present)})}\n\n"
            f"data: {json.dumps({'token': reply})}\n\n"
            f"data: {json.dumps({'done': True})}\n\n"
        )

        return func.HttpResponse(
            sse_body,
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as e:
        traceback.print_exc()
        kind, raw = classify_genai_error(e)
        logging.error(f"Stream error ({kind}): {raw}")

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
            (
                f"data: {json.dumps({'error': 'DEBUG_ERROR', 'stage': 'genai_call', 'type': type(e).__name__, 'message': str(e), 'model': MODEL_NAME, 'response_text_empty': None, 'chunks_empty': None, 'classified_kind': kind, 'user_error': user_facing_error_message(e)})}\n\n"
                f"data: {json.dumps({'done': True})}\n\n"
            ),
            status_code=status_code,
            mimetype="text/event-stream",
        )
