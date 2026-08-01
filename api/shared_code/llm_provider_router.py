from __future__ import annotations

import copy
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

try:
    import httpx
except ImportError:
    httpx = None


JsonDict = dict[str, Any]
Message = Mapping[str, Any]


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_keys: tuple[str, ...]
    headers: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    provider: ProviderConfig
    key: str
    key_slot: int

    @property
    def candidate_id(self) -> str:
        return f"{self.provider.name}:key-{self.key_slot}"


@dataclass
class FailureState:
    unavailable_until: float
    last_error: str
    failure_count: int = 1


@dataclass(frozen=True)
class AttemptFailure:
    provider: str
    key_slot: int
    error: str


@dataclass(frozen=True)
class LLMProviderResponse:
    text: str
    provider: str
    model: str
    key_slot: int
    raw: JsonDict


class RouterError(RuntimeError):
    pass


class ConfigurationError(RouterError):
    pass


class ProviderCallError(RouterError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


class AllProvidersFailed(RouterError):
    def __init__(self, failures: Sequence[AttemptFailure]) -> None:
        self.failures = tuple(failures)
        summary = "; ".join(
            f"{item.provider}[key-{item.key_slot}]: {item.error}"
            for item in self.failures
        )
        super().__init__(f"All configured LLM candidates failed. {summary}")


_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "google": {
        "prefix": "GOOGLE",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash-lite",
    },
    "groq": {
        "prefix": "GROQ",
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.1-8b-instant",
    },
    "mistral": {
        "prefix": "MISTRAL",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
    },
    "openrouter": {
        "prefix": "OPENROUTER",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openrouter/free",
    },
    "cloudflare": {
        "prefix": "CLOUDFLARE",
        "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1",
        "model": "@cf/meta/llama-3.1-8b-instruct",
    },
    "cohere": {
        "prefix": "COHERE",
        "base_url": "https://api.cohere.ai/compatibility/v1",
        "model": "command-r",
    },
    "nvidia": {
        "prefix": "NVIDIA",
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "meta/llama-3.1-8b-instruct",
    },
    "huggingface": {
        "prefix": "HUGGINGFACE",
        "base_url": "https://router.huggingface.co/v1",
        "model": "meta-llama/Llama-3.1-8B-Instruct:novita",
    },
    "sambanova": {
        "prefix": "SAMBANOVA",
        "base_url": "https://api.sambanova.ai/v1",
        "model": "Meta-Llama-3.1-8B-Instruct",
    },
    "fireworks": {
        "prefix": "FIREWORKS",
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/llama-v3p1-8b-instruct",
    },
}


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be true/false, yes/no, on/off, or 1/0.")


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        result = float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be numeric.") from exc
    if result < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}.")
    return result


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        result = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
    if result < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}.")
    return result


def _split_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _safe_error_text(value: Any, limit: int = 600) -> str:
    text = str(value).replace("\n", " ").replace("\r", " ").strip()
    return text[:limit] + ("..." if len(text) > limit else "")


def _response_error_message(response, limit: int) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        body = response.text

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            detail = (
                error.get("message")
                or error.get("detail")
                or error.get("type")
                or error
            )
        else:
            detail = error or body.get("message") or body.get("detail") or body
    else:
        detail = body

    return _safe_error_text(detail, limit)


def _parse_retry_after(response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _extract_text(data: JsonDict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderCallError("Response did not contain a non-empty choices list.")

    first = choices[0]
    if not isinstance(first, dict):
        raise ProviderCallError("First choice was not a JSON object.")

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                    elif isinstance(item.get("content"), str):
                        parts.append(item["content"])
            return "".join(parts)
        if message.get("tool_calls") or message.get("function_call"):
            return ""

    legacy_text = first.get("text")
    if isinstance(legacy_text, str):
        return legacy_text

    raise ProviderCallError("Response choice did not contain assistant text.")


def _adapt_messages(provider_name: str, messages: Sequence[Message]) -> list[JsonDict]:
    adapted = [copy.deepcopy(dict(message)) for message in messages]
    if provider_name == "cohere":
        for message in adapted:
            if message.get("role") == "system":
                message["role"] = "developer"
    return adapted


def _provider_env_value(prefix: str, suffix: str, default: str) -> str:
    name = f"{prefix}_{suffix}"
    if name in os.environ:
        return os.environ[name].strip()
    return default


def load_provider_configs_from_env() -> list[ProviderConfig]:
    default_order = ",".join(_PROVIDER_DEFAULTS)
    provider_order = tuple(
        name.lower() for name in _split_csv(os.getenv("LLM_PROVIDER_ORDER", default_order))
    )

    unknown = [name for name in provider_order if name not in _PROVIDER_DEFAULTS]
    if unknown:
        raise ConfigurationError(
            "Unknown provider(s) in LLM_PROVIDER_ORDER: " + ", ".join(unknown)
        )

    configs: list[ProviderConfig] = []
    for name in provider_order:
        defaults = _PROVIDER_DEFAULTS[name]
        prefix = defaults["prefix"]
        if not _env_bool(f"{prefix}_ENABLED", True):
            continue

        api_key_value = os.getenv(f"{prefix}_API_KEYS") or os.getenv(f"{prefix}_API_KEY")
        if name == "google" and not api_key_value:
            api_key_value = os.getenv("GEMINI_API_KEY")
        api_keys = _split_csv(api_key_value)
        model = _provider_env_value(prefix, "MODEL", defaults["model"])
        base_url = _provider_env_value(prefix, "BASE_URL", defaults["base_url"])

        if name == "cloudflare":
            account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
            if not account_id:
                continue
            base_url = base_url.format(account_id=account_id)

        if not api_keys or not model or not base_url:
            continue

        headers: dict[str, str] = {}
        if name == "openrouter":
            referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
            title = os.getenv("OPENROUTER_X_TITLE", "").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            if title:
                headers["X-Title"] = title

        configs.append(
            ProviderConfig(
                name=name,
                base_url=base_url.rstrip("/"),
                model=model,
                api_keys=api_keys,
                headers=headers,
            )
        )

    return configs


class LLMProviderRouter:
    def __init__(
        self,
        providers: Sequence[ProviderConfig],
        *,
        mode: str = "priority",
        cooldown_seconds: float = 300.0,
        timeout_seconds: float = 60.0,
        max_attempts: int = 0,
        max_error_body_chars: int = 600,
        http_client: httpx.Client | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if httpx is None and http_client is None:
            raise ConfigurationError(
                "httpx is required for LLM provider routing. "
                "Install dependencies from requirements.txt before enabling it."
            )

        mode = mode.strip().lower()
        if mode not in {"priority", "round_robin"}:
            raise ConfigurationError("ROUTER_MODE must be priority or round_robin.")

        candidates: list[Candidate] = []
        for provider in providers:
            for slot, key in enumerate(provider.api_keys, start=1):
                if key.strip():
                    candidates.append(
                        Candidate(provider=provider, key=key.strip(), key_slot=slot)
                    )

        if not candidates:
            raise ConfigurationError(
                "No usable provider/API-key candidates were configured. "
                "Fill at least one provider API key and model in .envexample."
            )

        self._candidates = tuple(candidates)
        self._mode = mode
        self._cooldown_seconds = cooldown_seconds
        self._max_attempts = max_attempts
        self._max_error_body_chars = max_error_body_chars
        self._cursor = 0
        self._failures: dict[str, FailureState] = {}
        self._lock = threading.RLock()
        self._logger = logger or logging.getLogger("llm_provider_router")
        self._owns_http_client = http_client is None
        if http_client is not None:
            self._http = http_client
        else:
            self._http = httpx.Client(
                timeout=httpx.Timeout(timeout_seconds),
                follow_redirects=True,
                headers={"User-Agent": "aita-llm-provider-router/1.0"},
            )

    @classmethod
    def from_env(cls, *, http_client: httpx.Client | None = None) -> "LLMProviderRouter":
        log_level_name = os.getenv("ROUTER_LOG_LEVEL", "WARNING").upper()
        logging.getLogger("llm_provider_router").setLevel(
            getattr(logging, log_level_name, logging.WARNING)
        )
        return cls(
            load_provider_configs_from_env(),
            mode=os.getenv("ROUTER_MODE", "priority"),
            cooldown_seconds=_env_float("ROUTER_COOLDOWN_SECONDS", 300.0, minimum=0.0),
            timeout_seconds=_env_float("ROUTER_TIMEOUT_SECONDS", 60.0, minimum=0.1),
            max_attempts=_env_int("ROUTER_MAX_ATTEMPTS", 0, minimum=0),
            max_error_body_chars=_env_int("ROUTER_MAX_ERROR_BODY_CHARS", 600, minimum=50),
            http_client=http_client,
        )

    def _ordered_candidate_indexes(self, allowed_providers: set[str] | None) -> list[int]:
        now = time.monotonic()
        with self._lock:
            count = len(self._candidates)
            start = self._cursor % count if self._mode == "round_robin" else 0
            rotated = [(start + offset) % count for offset in range(count)]
            if allowed_providers is not None:
                rotated = [
                    index
                    for index in rotated
                    if self._candidates[index].provider.name in allowed_providers
                ]
            healthy = [
                index
                for index in rotated
                if (
                    self._failures.get(self._candidates[index].candidate_id) is None
                    or self._failures[self._candidates[index].candidate_id].unavailable_until
                    <= now
                )
            ]
            selected = healthy if healthy else rotated
            if self._max_attempts > 0:
                selected = selected[: self._max_attempts]
            return selected

    def _record_failure(
        self,
        candidate: Candidate,
        error: str,
        retry_after_seconds: float | None,
    ) -> None:
        cooldown = self._cooldown_seconds
        if retry_after_seconds is not None:
            cooldown = max(cooldown, retry_after_seconds)
        with self._lock:
            previous = self._failures.get(candidate.candidate_id)
            self._failures[candidate.candidate_id] = FailureState(
                unavailable_until=time.monotonic() + cooldown,
                last_error=error,
                failure_count=(previous.failure_count + 1 if previous else 1),
            )

    def _record_success(self, candidate_index: int, candidate: Candidate) -> None:
        with self._lock:
            self._failures.pop(candidate.candidate_id, None)
            if self._mode == "round_robin":
                self._cursor = (candidate_index + 1) % len(self._candidates)

    def _call_candidate(
        self,
        candidate: Candidate,
        messages: Sequence[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMProviderResponse:
        payload: JsonDict = {
            "model": candidate.provider.model,
            "messages": _adapt_messages(candidate.provider.name, messages),
            "stream": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        headers = {
            "Authorization": f"Bearer {candidate.key}",
            "Content-Type": "application/json",
            **dict(candidate.provider.headers),
        }
        url = f"{candidate.provider.base_url}/chat/completions"

        try:
            response = self._http.post(url, headers=headers, json=payload)
        except Exception as exc:
            if httpx is not None and isinstance(exc, httpx.TimeoutException):
                raise ProviderCallError(f"Request timed out: {exc}") from exc
            raise ProviderCallError(f"HTTP connection error: {exc}") from exc

        if not response.is_success:
            message = _response_error_message(response, self._max_error_body_chars)
            raise ProviderCallError(
                f"HTTP {response.status_code}: {message}",
                status_code=response.status_code,
                retry_after_seconds=_parse_retry_after(response),
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ProviderCallError("Provider returned non-JSON success response.") from exc
        if not isinstance(data, dict):
            raise ProviderCallError("Provider returned a non-object JSON response.")
        if data.get("error"):
            raise ProviderCallError(
                "Provider returned an error payload: "
                + _safe_error_text(data["error"], self._max_error_body_chars)
            )

        return LLMProviderResponse(
            text=_extract_text(data),
            provider=candidate.provider.name,
            model=candidate.provider.model,
            key_slot=candidate.key_slot,
            raw=data,
        )

    def chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allowed_providers: Iterable[str] | None = None,
    ) -> LLMProviderResponse:
        if not messages:
            raise ValueError("messages must contain at least one chat message.")

        allowed = (
            {name.strip().lower() for name in allowed_providers}
            if allowed_providers is not None
            else None
        )
        candidate_indexes = self._ordered_candidate_indexes(allowed)
        if not candidate_indexes:
            raise ConfigurationError("No candidates matched allowed_providers.")

        failures: list[AttemptFailure] = []
        for candidate_index in candidate_indexes:
            candidate = self._candidates[candidate_index]
            try:
                result = self._call_candidate(
                    candidate,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except ConfigurationError:
                raise
            except Exception as exc:
                retry_after = (
                    exc.retry_after_seconds
                    if isinstance(exc, ProviderCallError)
                    else None
                )
                error = _safe_error_text(exc, self._max_error_body_chars)
                self._record_failure(candidate, error, retry_after)
                failures.append(
                    AttemptFailure(
                        provider=candidate.provider.name,
                        key_slot=candidate.key_slot,
                        error=error,
                    )
                )
                self._logger.warning(
                    "LLM candidate failed; trying next candidate: provider=%s key_slot=%s error=%s",
                    candidate.provider.name,
                    candidate.key_slot,
                    error,
                )
                continue

            self._record_success(candidate_index, candidate)
            return result

        raise AllProvidersFailed(failures)

    def health(self) -> list[JsonDict]:
        now = time.monotonic()
        rows: list[JsonDict] = []
        with self._lock:
            for candidate in self._candidates:
                state = self._failures.get(candidate.candidate_id)
                remaining = max(0.0, state.unavailable_until - now) if state else 0.0
                rows.append(
                    {
                        "provider": candidate.provider.name,
                        "model": candidate.provider.model,
                        "key_slot": candidate.key_slot,
                        "available": remaining <= 0,
                        "cooldown_remaining_seconds": round(remaining, 1),
                        "failure_count": state.failure_count if state else 0,
                        "last_error": state.last_error if state else None,
                    }
                )
        return rows

    def reset_failures(self) -> None:
        with self._lock:
            self._failures.clear()

    def close(self) -> None:
        if self._owns_http_client:
            self._http.close()
