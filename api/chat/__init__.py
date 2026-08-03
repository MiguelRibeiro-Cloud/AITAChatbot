import json
import logging
import azure.functions as func
from shared_code import (
    COCONUT_FALLBACK,
    MAX_OUTPUT_TOKENS,
    MODEL_NAME,
    ProviderBusyError,
    ProviderTimeoutError,
    RequestValidationError,
    SYSTEM_INSTRUCTION,
    build_contents,
    check_rate_limit,
    classify_genai_error,
    clean_reply,
    client,
    extract_reply_text,
    response_diagnostics,
    run_with_timeout,
    validate_chat_payload,
)
from db import CounterConfigError, CounterDatabaseError, increment_cases_heard

CLIENT_ERROR_MESSAGE = "The request could not be completed."


def _debug_payload(stage, empty_kind=None, response_empty=None):
    return {
        "stage": stage,
        "model": MODEL_NAME,
        "response_text_empty": bool(response_empty) if response_empty is not None else None,
        "chunks_empty": None,
        "empty_kind": empty_kind,
    }


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Send a message and get a response from Gemma 3 12B."""
    allowed, retry_after = check_rate_limit(req)
    if not allowed:
        return func.HttpResponse(
            json.dumps({"error": "Too many requests. Please try again shortly."}),
            status_code=429,
            mimetype="application/json",
            headers={"Retry-After": str(retry_after)},
        )

    try:
        try:
            data = req.get_json()
        except ValueError as exc:
            raise RequestValidationError("Invalid JSON body.") from exc
        user_message, history = validate_chat_payload(data)
        contents = build_contents(history, user_message)

        response = run_with_timeout(
            lambda: client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config={"max_output_tokens": MAX_OUTPUT_TOKENS, "system_instruction": SYSTEM_INSTRUCTION},
            )
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
                json.dumps(
                    {
                        "reply": reply,
                        "model": MODEL_NAME,
                        "debug": _debug_payload(
                            stage="empty_response",
                            empty_kind=empty_kind,
                            response_empty=True,
                        ),
                    }
                ),
                mimetype="application/json",
            )

        reply = clean_reply(reply)

        cases_heard = None
        try:
            cases_heard = increment_cases_heard()
        except (CounterConfigError, CounterDatabaseError) as counter_exc:
            logging.error("Cases-heard counter increment failed after successful chat: %s", counter_exc)

        return func.HttpResponse(
            json.dumps({"reply": reply, "model": MODEL_NAME, "casesHeard": cases_heard}),
            mimetype="application/json",
        )

    except RequestValidationError as e:
        return func.HttpResponse(
            json.dumps({"error": str(e), "debug": _debug_payload(stage="validation")}),
            status_code=400,
            mimetype="application/json",
        )

    except Exception as e:
        kind, _raw = classify_genai_error(e)
        logging.exception("Chat error (%s)", kind)

        status_code = 500
        if isinstance(e, ProviderTimeoutError):
            status_code = 504
        elif isinstance(e, ProviderBusyError):
            status_code = 503
        elif kind == "usage_limit":
            status_code = 429
        elif kind == "provider_high_demand":
            status_code = 503
        elif kind == "auth_or_permission":
            status_code = 401
        elif kind == "model_not_found":
            status_code = 502

        return func.HttpResponse(
            json.dumps({
                "error": CLIENT_ERROR_MESSAGE,
                "debug": _debug_payload(stage="genai_call"),
            }),
            status_code=status_code,
            mimetype="application/json",
        )
