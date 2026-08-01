import json
import logging
import traceback
import azure.functions as func
from shared_code import (
    COCONUT_FALLBACK,
    MODEL_NAME,
    SYSTEM_INSTRUCTION,
    build_contents,
    classify_genai_error,
    clean_reply,
    client,
    extract_reply_text,
    generate_with_provider_router,
    llm_provider_router_enabled,
    response_diagnostics,
    user_facing_error_message,
)


def _debug_payload(stage, exc=None, empty_kind=None, response_empty=None, model=None, provider=None):
    payload = {
        "stage": stage,
        "type": type(exc).__name__ if exc else None,
        "message": str(exc) if exc else None,
        "model": model or MODEL_NAME,
        "response_text_empty": bool(response_empty) if response_empty is not None else None,
        "chunks_empty": None,
        "empty_kind": empty_kind,
    }
    if provider is not None:
        payload["provider"] = provider
    return payload


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Send a message and get a response from Gemma 3 12B."""
    try:
        data = req.get_json()
        if not data or "message" not in data:
            return func.HttpResponse(
                json.dumps(
                    {
                        "error": "No message provided. Say something, coward!",
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
                        "error": "Empty message? Really? Try harder.",
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
                        "error": "That's a novel, not a message. Keep it under 10,000 characters.",
                        "debug": _debug_payload(stage="validation"),
                    }
                ),
                status_code=400,
                mimetype="application/json",
            )

        history = data.get("history", [])
        contents = build_contents(history, user_message)

        using_provider_router = llm_provider_router_enabled()
        if using_provider_router:
            provider_response = generate_with_provider_router(
                history,
                user_message,
                max_tokens=500,
            )
            raw_text = provider_response.text
            reply, empty_kind = (
                (raw_text.strip(), None)
                if isinstance(raw_text, str) and raw_text.strip()
                else ("", "blank_text")
            )
            response_model = provider_response.model
            response_provider = provider_response.provider
            diagnostics = {
                "provider": response_provider,
                "model": response_model,
                "key_slot": provider_response.key_slot,
            }
        else:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config={"max_output_tokens": 500, "system_instruction": SYSTEM_INSTRUCTION},
            )
            reply, empty_kind = extract_reply_text(response)
            response_model = MODEL_NAME
            response_provider = "google-ai-studio"
            diagnostics = response_diagnostics(response)

        if empty_kind:
            logging.warning(
                "Empty LLM response fallback used: provider=%s model=%s empty_kind=%s diagnostics=%s",
                response_provider,
                response_model,
                empty_kind,
                diagnostics,
            )
            reply = COCONUT_FALLBACK
            return func.HttpResponse(
                json.dumps({
                    **(
                        {"reply": reply, "model": response_model, "provider": response_provider}
                        if using_provider_router
                        else {"reply": reply, "model": response_model}
                    ),
                    "debug": _debug_payload(
                        stage="empty_response",
                        empty_kind=empty_kind,
                        response_empty=True,
                        model=response_model,
                        provider=response_provider if using_provider_router else None,
                    ),
                }
                ),
                mimetype="application/json",
            )

        reply = clean_reply(reply)

        payload = {"reply": reply, "model": response_model}
        if using_provider_router:
            payload["provider"] = response_provider

        return func.HttpResponse(json.dumps(payload), mimetype="application/json")

    except Exception as e:
        traceback.print_exc()
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
                "debug": _debug_payload(stage="genai_call", exc=e),
            }),
            status_code=status_code,
            mimetype="application/json",
        )
