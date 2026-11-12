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

_DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


class LLMQuotaExhaustedError(RuntimeError):
    """The provider has no request quota left for the current window."""


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
    requests that are bound to be rejected.
    """

    def __init__(self) -> None:
        self.requests_made = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.remaining_requests: int | None = None
        self.remaining_tokens: int | None = None
        self.reset_seconds: float | None = None

    def record_request(self) -> None:
        self.requests_made += 1

    def record_usage(self, usage: dict | None) -> None:
        if not usage:
            return
        self.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        self.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def record_response(self, headers) -> None:
        remaining = headers.get("x-ratelimit-remaining-requests")
        if remaining is not None:
            try:
                self.remaining_requests = int(remaining)
            except ValueError:
                pass
        tokens = headers.get("x-ratelimit-remaining-tokens")
        if tokens is not None:
            try:
                self.remaining_tokens = int(tokens)
            except ValueError:
                pass
        self.reset_seconds = _parse_duration(
            headers.get("x-ratelimit-reset-requests")
        )

    @property
    def exhausted(self) -> bool:
        return (
            self.remaining_requests is not None and self.remaining_requests <= 0
        )

    def snapshot(self) -> dict:
        return {
            "requests_made": self.requests_made,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "remaining_requests": self.remaining_requests,
            "remaining_tokens": self.remaining_tokens,
            "reset_seconds": self.reset_seconds,
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

        async def call_with_retries(provider_call):
            last_error: Exception | None = None
            for attempt in range(max_retries + 1):
                try:
                    return await provider_call()
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    if exc.response.status_code not in RETRYABLE_STATUS_CODES:
                        raise
                    self.quota.record_response(exc.response.headers)
                    if self.quota.exhausted:
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
                    )
                )
            except (
                httpx.HTTPStatusError,
                httpx.TransportError,
                LLMQuotaExhaustedError,
            ):
                # Fall back to OpenRouter when configured; otherwise surface
                # the original provider error.
                if not self._openrouter_enabled():
                    raise

        if self._openrouter_enabled():
            return await call_with_retries(
                lambda: self._call_openrouter(
                    prompt, model or self.openrouter_model, temperature, max_tokens
                )
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
        )

    async def _call_with_retry(
        self,
        url: str,
        api_key: str,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        content = await self._post(
            url, api_key, model, prompt, temperature, max_tokens
        )
        if content.strip():
            return content

        retry_tokens = max(max_tokens * EMPTY_CONTENT_RETRY_FACTOR, MIN_RETRY_TOKENS)
        return await self._post(
            url, api_key, model, prompt, temperature, retry_tokens
        )

    async def _post(
        self,
        url: str,
        api_key: str,
        model: str,
        prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.quota.record_request()
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
            self.quota.record_response(resp.headers)
            resp.raise_for_status()
            data = resp.json()
            self.quota.record_usage(data.get("usage"))
            content = data["choices"][0]["message"].get("content") or ""
            return content
