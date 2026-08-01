# Multi-provider LLM API router

This example routes non-streaming chat-completion requests through configured
API keys and providers. Every provider section is independent and can be
disabled without editing the Python code.

## Files

- `llm_key_router.py`: reusable router module.
- `.env.example`: all provider URLs, model variables, and API-key slots.
- `example_consumer.py`: importable application wrapper.
- `test_router.py`: offline failover test; no real API keys are used.
- `requirements.txt`: Python dependencies.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Fill at least one `*_API_KEYS` value in `.env`, then run:

```bash
python example_consumer.py
```

Run the offline test:

```bash
python test_router.py
```

## Import from another module

```python
from example_consumer import ask_llm

result = ask_llm("Explain decorators with one small example.")
print(result)
```

For provider metadata as well as text:

```python
from llm_key_router import LLMRouter

router = LLMRouter.from_env()
response = router.chat(
    [
        {"role": "system", "content": "Answer in plain English."},
        {"role": "user", "content": "What is eventual consistency?"},
    ],
    max_tokens=400,
)

print(response.text)
print(response.provider, response.model, response.key_slot)
```

## Rotation modes

`ROUTER_MODE=priority` uses the configured order as a fallback chain. It keeps
using the highest-priority healthy candidate.

`ROUTER_MODE=round_robin` moves the starting position after every successful
call. This distributes successful calls across all configured provider/key
candidates while retaining automatic failover.

Within each provider, comma-separated keys become separate candidates:

```dotenv
GROQ_API_KEYS=key_one,key_two,key_three
```

## What triggers failover

The router tries the next candidate after:

- Any non-2xx HTTP response, including 401, 403, 404, 429, and 5xx.
- An HTTP 200 response containing a top-level `error` object.
- A timeout or connection failure.
- Invalid JSON.
- A malformed chat-completion response.

The failed candidate is skipped for `ROUTER_COOLDOWN_SECONDS`. A numeric
`Retry-After` response header can extend that cooldown.

A request-level mistake, such as an unsupported parameter or invalid message
format, may fail on every provider. The final `AllProvidersFailed` exception
contains bounded error descriptions but never API-key values.

## Deliberate limitations

- Non-streaming chat completions only. Transparent fallback is not possible
  after a streamed response has already emitted partial output.
- Text and tool-call responses are accepted, but provider-specific features
  may require `request_overrides` and may not be portable.
- Model availability and free quotas change. Update each `*_MODEL` value from
  the provider's current model catalog.
- In-memory cooldown state is process-local. For a multi-process deployment,
  store health/quota state in a shared system such as Redis.

Use only API keys and accounts you are authorized to use. Failover should
improve reliability, not evade a provider's account-level limits or terms.
