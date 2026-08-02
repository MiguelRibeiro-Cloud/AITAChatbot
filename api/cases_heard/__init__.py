import json
import logging

import azure.functions as func

from db import CounterConfigError, CounterDatabaseError, get_cases_heard


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Return the deployment-wide cases-heard count without incrementing it."""
    try:
        cases_heard = get_cases_heard()
    except (CounterConfigError, CounterDatabaseError) as exc:
        logging.error("Cases-heard counter read failed: %s", exc)
        return func.HttpResponse(
            json.dumps({"error": "Cases-heard counter is temporarily unavailable."}),
            status_code=503,
            mimetype="application/json",
        )

    return func.HttpResponse(
        json.dumps({"casesHeard": cases_heard}),
        mimetype="application/json",
    )
