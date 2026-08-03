import json
import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps(
            {
                "status": "alive",
            }
        ),
        mimetype="application/json",
    )
