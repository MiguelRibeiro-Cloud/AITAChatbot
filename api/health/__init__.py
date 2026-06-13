import os
import json
import azure.functions as func
from shared_code import MODEL_NAME


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint."""
    return func.HttpResponse(
        json.dumps(
            {
                "status": "alive",
                "model": MODEL_NAME,
                "has_api_key": bool(os.getenv("GEMINI_API_KEY")),
                "model_env_present": bool(os.getenv("GEMINI_MODEL_NAME")),
                "vibe": "chaotic",
            }
        ),
        mimetype="application/json",
    )
