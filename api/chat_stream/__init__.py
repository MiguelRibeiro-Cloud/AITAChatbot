import json
import logging
import azure.functions as func
from shared_code import (
    ProviderBusyError,
    ProviderTimeoutError,
    RequestValidationError,
    StructuredResponseError,
    build_contents,
    check_rate_limit,
    classify_genai_error,
    generate_judgment,
    run_with_timeout,
    validate_chat_payload,
)
from db import CounterConfigError, CounterDatabaseError, increment_cases_heard

CLIENT_ERROR_MESSAGE = "The request could not be completed."


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Return a validated Judge ruling in the existing SSE format."""
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

        judgment = run_with_timeout(lambda: generate_judgment(contents))

        cases_heard = None
        try:
            cases_heard = increment_cases_heard()
        except (CounterConfigError, CounterDatabaseError) as counter_exc:
            logging.error("Cases-heard counter increment failed after successful stream: %s", counter_exc)

        sse_body = (
            f"data: {json.dumps({'token': judgment['reply']})}\n\n"
            f"data: {json.dumps({'done': True, 'verdict': judgment['verdict'], 'casesHeard': cases_heard})}\n\n"
        )

        return func.HttpResponse(
            sse_body,
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except RequestValidationError as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=400,
            mimetype="application/json",
        )

    except Exception as e:
        kind, _raw = classify_genai_error(e)
        logging.exception("Stream error (%s)", kind)

        status_code = 500
        if isinstance(e, ProviderTimeoutError):
            status_code = 504
        elif isinstance(e, ProviderBusyError):
            status_code = 503
        elif isinstance(e, StructuredResponseError):
            status_code = 502
        elif kind == "usage_limit":
            status_code = 429
        elif kind == "provider_high_demand":
            status_code = 503
        elif kind == "auth_or_permission":
            status_code = 401
        elif kind == "model_not_found":
            status_code = 502

        return func.HttpResponse(
            (
                f"data: {json.dumps({'error': CLIENT_ERROR_MESSAGE})}\n\n"
                f"data: {json.dumps({'done': True})}\n\n"
            ),
            status_code=status_code,
            mimetype="text/event-stream",
        )
