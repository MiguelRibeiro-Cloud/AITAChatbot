"""
example_consumer.py

This module can itself be imported by another Python module:

    from example_consumer import ask_llm

    answer = ask_llm("Summarize dependency injection in three sentences.")
"""

from __future__ import annotations

import atexit
from threading import Lock

from llm_key_router import AllProvidersFailed, LLMRouter


_router: LLMRouter | None = None
_router_lock = Lock()


def get_router() -> LLMRouter:
    """Create one process-wide router lazily and reuse its HTTP connection pool."""
    global _router

    if _router is None:
        with _router_lock:
            if _router is None:
                _router = LLMRouter.from_env(".env")

    return _router


def ask_llm(
    prompt: str,
    *,
    system: str = "You are a concise, accurate assistant.",
    max_tokens: int = 500,
) -> str:
    """
    Callable from any other module.

    The caller does not need to know which provider/key succeeded.
    """
    return get_router().ask(
        prompt,
        system=system,
        temperature=0.2,
        max_tokens=max_tokens,
    )


def close_router() -> None:
    global _router
    if _router is not None:
        _router.close()
        _router = None


atexit.register(close_router)


def main() -> None:
    try:
        answer = ask_llm("Give one practical use for a Python context manager.")
        print(answer)

        # Optional diagnostic data: contains provider/model/key-slot numbers,
        # but never contains API-key values.
        print("\nRouter health:")
        for row in get_router().health():
            print(row)

    except AllProvidersFailed as exc:
        print("No configured provider completed the request.")
        for failure in exc.failures:
            print(
                f"- {failure.provider} key-{failure.key_slot}: "
                f"{failure.error}"
            )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
