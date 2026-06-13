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
    user_facing_error_message,
)


def _extract_chunk_text(chunk):
    """Try multiple paths to extract text from a streaming chunk.

    Path 1 — chunk.text attribute (standard SDK shortcut).
    Path 2 — chunk.candidates[*].content.parts[*].text (full object tree).
    Path 3 — dict form via model_dump() / to_json_dict() / __dict__ for
              non-standard or future SDK shapes.
    """
    # Path 1: chunk.text
    try:
        t = getattr(chunk, "text", None)
        if isinstance(t, str) and t:
            return t
    except Exception:
        pass

    # Path 2: candidates → content → parts → text (attribute traversal)
    try:
        candidates = getattr(chunk, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                t = getattr(part, "text", None)
                if isinstance(t, str) and t:
                    return t
    except Exception:
        pass

    # Path 3: serialised dict (model_dump / to_json_dict / __dict__ / raw dict)
    try:
        raw = None
        if hasattr(chunk, "model_dump"):
            raw = chunk.model_dump()
        elif hasattr(chunk, "to_json_dict"):
            raw = chunk.to_json_dict()
        elif isinstance(chunk, dict):
            raw = chunk
        elif hasattr(chunk, "__dict__"):
            raw = vars(chunk)

        if isinstance(raw, dict):
            t = raw.get("text")
            if isinstance(t, str) and t:
                return t
            for cand in raw.get("candidates", []):
                content = cand.get("content", {}) if isinstance(cand, dict) else {}
                for part in (content.get("parts", []) if isinstance(content, dict) else []):
                    t = part.get("text") if isinstance(part, dict) else None
                    if isinstance(t, str) and t:
                        return t
    except Exception:
        pass

    return None


def _sanitize_chunk(chunk, index):
    """Build a safe debug snapshot of a chunk — no API keys or secrets."""
    try:
        raw = None
        if hasattr(chunk, "model_dump"):
            raw = chunk.model_dump()
        elif hasattr(chunk, "to_json_dict"):
            raw = chunk.to_json_dict()
        elif isinstance(chunk, dict):
            raw = chunk
        elif hasattr(chunk, "__dict__"):
            raw = vars(chunk)

        if not isinstance(raw, dict):
            return {"chunk_index": index, "raw_repr": str(raw)[:300]}

        safe_candidates = []
        for cand in raw.get("candidates", []):
            if not isinstance(cand, dict):
                continue
            content = cand.get("content", {}) if isinstance(cand, dict) else {}
            parts = content.get("parts", []) if isinstance(content, dict) else []
            safe_candidates.append({
                "finish_reason": cand.get("finish_reason"),
                "safety_ratings": cand.get("safety_ratings"),
                "content_role": content.get("role") if isinstance(content, dict) else None,
                "parts": [
                    {
                        "has_text": bool(p.get("text")) if isinstance(p, dict) else False,
                        "text_length": len(p["text"]) if isinstance(p, dict) and isinstance(p.get("text"), str) else 0,
                        "keys": list(p.keys()) if isinstance(p, dict) else [],
                    }
                    for p in parts
                ],
            })

        return {
            "chunk_index": index,
            "top_level_keys": list(raw.keys()),
            "has_text_attr": hasattr(chunk, "text"),
            "text_type": type(raw.get("text")).__name__,
            "text_length": len(raw["text"]) if isinstance(raw.get("text"), str) else None,
            "candidate_count": len(raw.get("candidates", [])),
            "candidates": safe_candidates,
        }
    except Exception as ex:
        return {"chunk_index": index, "sanitize_error": str(ex)}


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Iterate generate_content_stream chunks, extract text robustly, return SSE."""
    try:
        data = req.get_json()
        if not data or "message" not in data:
            return func.HttpResponse(
                json.dumps({"error": "No message provided.", "debug": {"stage": "validation", "model": MODEL_NAME}}),
                status_code=400,
                mimetype="application/json",
            )

        user_message = data["message"].strip()
        if not user_message:
            return func.HttpResponse(
                json.dumps({"error": "Empty message.", "debug": {"stage": "validation", "model": MODEL_NAME}}),
                status_code=400,
                mimetype="application/json",
            )

        if len(user_message) > 10000:
            return func.HttpResponse(
                json.dumps({"error": "Message too long. Keep it under 10,000 characters.", "debug": {"stage": "validation", "model": MODEL_NAME}}),
                status_code=400,
                mimetype="application/json",
            )

        history = data.get("history", [])
        contents = build_contents(history, user_message)

        # Use the streaming API so each chunk is inspectable individually.
        response_stream = client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config={"max_output_tokens": 500, "system_instruction": SYSTEM_INSTRUCTION},
        )

        collected_text = []
        chunk_shapes = []   # first 2 chunks for debug surfacing
        chunks_seen = 0

        for chunk in response_stream:
            chunks_seen += 1

            # Record and log sanitized shape of first 2 chunks
            if chunks_seen <= 2:
                shape = _sanitize_chunk(chunk, chunks_seen)
                chunk_shapes.append(shape)
                logging.info("Stream chunk %d shape: %s", chunks_seen, json.dumps(shape))

            text = _extract_chunk_text(chunk)
            if text:
                collected_text.append(text)

        reply = clean_reply("".join(collected_text).strip())

        if not reply:
            # No text extracted from any chunk — surface full debug info in SSE
            logging.warning(
                "No text from %d stream chunks: model=%s first_chunk_shapes=%s",
                chunks_seen, MODEL_NAME, json.dumps(chunk_shapes),
            )
            sse_body = (
                f"data: {json.dumps({'debug': {'stage': 'stream_parse', 'model': MODEL_NAME, 'chunks_seen': chunks_seen, 'chunks_empty': True, 'first_chunk_shapes': chunk_shapes}})}\n\n"
                f"data: {json.dumps({'token': COCONUT_FALLBACK})}\n\n"
                f"data: {json.dumps({'done': True})}\n\n"
            )
        else:
            sse_body = (
                f"data: {json.dumps({'debug': {'stage': 'genai_call', 'model': MODEL_NAME, 'chunks_seen': chunks_seen, 'chunks_empty': False, 'first_chunk_shapes': chunk_shapes}})}\n\n"
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
        logging.error("Stream error (%s): %s", kind, raw)

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
                f"data: {json.dumps({'error': 'DEBUG_ERROR', 'stage': 'genai_call', 'type': type(e).__name__, 'message': str(e), 'model': MODEL_NAME, 'chunks_seen': None, 'chunks_empty': None, 'classified_kind': kind, 'user_error': user_facing_error_message(e)})}\n\n"
                f"data: {json.dumps({'done': True})}\n\n"
            ),
            status_code=status_code,
            mimetype="text/event-stream",
        )
