"""LLM service with Groq primary and OpenRouter fallback on transient errors."""
import asyncio
import re

import httpx

from app.core.config import settings

# Reasoning-capable models can spend their entire token budget on internal
# reasoning and return empty content. Retry once with a larger budget.
EMPTY_CONTENT_RETRY_FACTOR = 4
MIN_RETRY_TOKENS = 512

# Transient responses worth retrying with exponential backoff.
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

# Responses that mean the provider will not serve the request at all: the
# key is invalid or the account has no credit left. Retrying cannot help.
FATAL_STATUS_CODES = {401, 402, 403}

_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


class LLMQuotaExhaustedError(RuntimeError):
    """The provider has no request quota left for the current window."""


class LLMProviderUnavailableError(RuntimeError):
    """The provider refused the request for a reason that retrying cannot fix.

    Raised for authentication, credit and permission failures, where every
    further request would also fail.
    """


def _parse_duration(value: str | None) -> float | None:
    """Parse a Groq duration header such as ``18h51m50.4s`` into seconds."""
    if not value:
        return None
    units = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
    total = 0.0
    matched = False
    for amount, unit in _DURATION_PATTERN.findall(value):
        total += float(amount) * units[unit]
        matched = True
    return total if matched else None


class QuotaTracker:
    """Shared rate-limit accounting for every LLM service instance.

    A single evaluation query triggers several provider calls, so the tracker
    lets the harness stop before it spends the remaining daily quota on
    requests that are bound to be rejected. Quota is tracked per provider,
    because an exhausted Groq window must not stop a run that can continue
    on the fallback provider.
    """

    def __init__(self) -> None:
        self.requests_made = 0
        self.providers: dict[str, dict] = {}

    def _provider_state(self, provider: str) -> dict:
        return self.providers.setdefault(
            provider,
            {
                "remaining_requests": None,
                "remaining_tokens": None,
                "reset_seconds": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
        )

    def record_request(self, provider: str) -> None:
        self.requests_made += 1
        state = self._provider_state(provider)
        state["requests_made"] = state.get("requests_made", 0) + 1

    def record_usage(self, provider: str, usage: dict | None) -> None:
        if not usage:
            return
        state = self._provider_state(provider)
        state["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        state["completion_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    def record_response(self, provider: str, headers) -> None:
        state = self._provider_state(provider)
        remaining = headers.get("x-ratelimit-remaining-requests")
        if remaining is not None:
            try:
                state["remaining_requests"] = int(remaining)
            except ValueError:
                pass
        tokens = headers.get("x-ratelimit-remaining-tokens")
        if tokens is not None:
            try:
                state["remaining_tokens"] = int(tokens)
            except ValueError:
                pass
        state["reset_seconds"] = _parse_duration(
            headers.get("x-ratelimit-reset-requests")
        )

    def remaining_requests(self, provider: str) -> int | None:
        return self._provider_state(provider)["remaining_requests"]

    def exhausted(self, provider: str) -> bool:
        remaining = self.remaining_requests(provider)
        return remaining is not None and remaining <= 0

    def total_tokens(self, provider: str) -> int:
        state = self._provider_state(provider)
        return state["prompt_tokens"] + state["completion_tokens"]

    def snapshot(self, provider: str | None = None) -> dict:
        if provider is not None:
            state = self._provider_state(provider)
            return {
                "provider": provider,
                "requests_made": state.get("requests_made", 0),
                "prompt_tokens": state["prompt_tokens"],
                "completion_tokens": state["completion_tokens"],
                "total_tokens": state["prompt_tokens"] + state["completion_tokens"],
                "remaining_requests": state["remaining_requests"],
                "remaining_tokens": state["remaining_tokens"],
                "reset_seconds": state["reset_seconds"],
            }
        return {
            "requests_made": self.requests_made,
            "providers": {
                name: self.snapshot(name) for name in self.providers
            },
        }


QUOTA = QuotaTracker()


class LLMService:
    def __init__(self) -> None:
        self.groq_key = settings.GROQ_API_KEY
        self.groq_model = settings.GROQ_MODEL
        self.openrouter_key = getattr(settings, "OPENROUTER_API_KEY", "")
        self.openrouter_model = getattr(
            settings, "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"
        )
        self.quota = QUOTA
        self.provider = getattr(settings, "LLM_PROVIDER", "auto").strip().lower()

    def _groq_enabled(self) -> bool:
        return bool(self.groq_key) and self.provider in {"auto", "groq"}

    def _openrouter_enabled(self) -> bool:
        return bool(self.openrouter_key) and self.provider in {"auto", "openrouter"}

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """Generate text with Groq primary and OpenRouter fallback.

        Retries transient transport errors, 5xx responses and rate limits with
        exponential backoff. Non-retryable client errors are surfaced.
        """

        async def call_with_retries(provider_call, provider: str):
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await provider_call()
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code in FATAL_STATUS_CODES:
                        raise LLMProviderUnavailableError(
                            f"provider rejected the request with status "
                            f"{exc.response.status_code}"
                        ) from exc
                    if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                        raise
                    self.quota.record_response(provider, exc.response.headers)
                    if self.quota.exhausted(provider):
                        # The window quota is spent; retrying cannot succeed.
                        raise LLMQuotaExhaustedError(
                            "provider request quota exhausted"
                        ) from exc
                except httpx.TransportError as exc:
                    last_error = exc
                if attempt < max_retries:
                    await asyncio.sleep(base_delay * (2**attempt))
            if last_error is not None:
                raise last_error
            raise RuntimeError("LLM request failed without an exception")

        max_retries = max(0, settings.LLM_MAX_RETRIES)
        base_delay = settings.LLM_RETRY_BASE_DELAY

        if self._groq_enabled():
            try:
                return await call_with_retries(
                    lambda: self._call_groq(
                        prompt, model or self.groq_model, temperature, max_tokens
                    ),
                    "groq",
                )
            except (
                httpx.HTTPStatusError,
                httpx.TransportError,
                LLMQuotaExhaustedError,
                LLMProviderUnavailableError,
            ):
                # Fall back to OpenRouter when configured; otherwise surface
                # the original provider error.
                if not self._openrouter_enabled():
                    raise

        if self._openrouter_enabled():
            return await call_with_retries(
                lambda: self._call_openrouter(
                    prompt, model or self.openrouter_model, temperature, max_tokens
                ),
                "openrouter",
            )

        raise RuntimeError("No LLM provider available")

    async def _call_groq(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        return await self._call_with_retry(
            url="https://api.groq.com/openai/v1/chat/completions",
            api_key=self.groq_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            provider="groq",
        )

    async def _call_openrouter(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        return await self._call_with_retry(
            url="https://openrouter.ai/api/v1/chat/completions",
            api_key=self.openrouter_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            provider="openrouter",
        )

    async def _call_with_retry(
        self,
        url: str,
        api_key: str,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        provider: str,
    ) -> str:
        content = await self._post(
            url, api_key, model, prompt, temperature, max_tokens, provider
        )
        if content.strip():
            return content

        retry_tokens = max(max_tokens * EMPTY_CONTENT_RETRY_FACTOR, MIN_RETRY_TOKENS)
        return await self._post(
            url, api_key, model, prompt, temperature, retry_tokens, provider
        )

    async def _post(
        self,
        url: str,
        api_key: str,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
        provider: str,
    ) -> str:
        self.quota.record_request(provider)
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=60.0,
            )
            self.quota.record_response(provider, resp.headers)
            resp.raise_for_status()
            data = resp.json()
            self.quota.record_usage(provider, data.get("usage"))
            content = data["choices"][0]["message"].get("content") or ""
            return content
