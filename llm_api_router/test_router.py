"""Offline failover test using httpx.MockTransport."""

from __future__ import annotations

import json

import httpx

from llm_key_router import LLMRouter, ProviderConfig


def test_429_fails_over_to_next_provider() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.host)

        if request.url.host == "first.invalid":
            return httpx.Response(
                429,
                headers={"Retry-After": "1"},
                json={"error": {"message": "Free quota exhausted"}},
            )

        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Fallback worked"}}
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    providers = [
        ProviderConfig(
            name="first",
            base_url="https://first.invalid/v1",
            model="first-model",
            api_keys=("first-secret",),
        ),
        ProviderConfig(
            name="second",
            base_url="https://second.invalid/v1",
            model="second-model",
            api_keys=("second-secret",),
        ),
    ]

    router = LLMRouter(
        providers,
        cooldown_seconds=30,
        http_client=client,
    )

    response = router.chat([{"role": "user", "content": "Hello"}])

    assert response.text == "Fallback worked"
    assert response.provider == "second"
    assert calls == ["first.invalid", "second.invalid"]

    health_json = json.dumps(router.health())
    assert "first-secret" not in health_json
    assert "second-secret" not in health_json

    client.close()


if __name__ == "__main__":
    test_429_fails_over_to_next_provider()
    print("Offline failover test passed.")
