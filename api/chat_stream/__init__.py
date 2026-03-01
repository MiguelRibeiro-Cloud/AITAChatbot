import json
import logging
import azure.functions as func
from shared_code import client, MODEL_NAME, build_contents


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Return response in SSE format for frontend compatibility."""
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
