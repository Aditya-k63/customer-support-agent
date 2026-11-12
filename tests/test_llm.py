"""Retry behaviour of the LLM client.

A single transient provider failure used to abort a whole evaluation run, so
these tests pin the retry/fallback contract.
"""
import httpx
import pytest

from app.core.config import settings
from app.services.llm import (
    QUOTA,
    LLMQuotaExhaustedError,
    LLMService,
    _parse_duration,
)


def _status_error(code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.test/chat")
    response = httpx.Response(code, request=request)
    return httpx.HTTPStatusError("provider error", request=request, response=response)


def _service(monkeypatch, *, openrouter_key: str = "") -> LLMService:
    monkeypatch.setattr(settings, "GROQ_API_KEY", "test-groq-key")
    monkeypatch.setattr(settings, "OPENROUTER_API_KEY", openrouter_key)
    monkeypatch.setattr(settings, "LLM_MAX_RETRIES", 2)
    monkeypatch.setattr(settings, "LLM_RETRY_BASE_DELAY", 0)
    return LLMService()


@pytest.mark.asyncio
async def test_generate_retries_transport_error(monkeypatch):
    service = _service(monkeypatch)
    calls = {"count": 0}

    async def flaky(prompt, model, temperature, max_tokens):
        calls["count"] += 1
        if calls["count"] == 1:
            raise httpx.ReadError("connection reset")
        return "hello"

    monkeypatch.setattr(service, "_call_groq", flaky)

    assert await service.generate("hi") == "hello"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_generate_retries_retryable_status(monkeypatch):
    service = _service(monkeypatch)
    calls = {"count": 0}

    async def flaky(prompt, model, temperature, max_tokens):
        calls["count"] += 1
        if calls["count"] == 1:
            raise _status_error(503)
        return "hello"

    monkeypatch.setattr(service, "_call_groq", flaky)

    assert await service.generate("hi") == "hello"
    assert calls["count"] == 2


@pytest.mark.asyncio
async def test_generate_does_not_retry_client_error(monkeypatch):
    service = _service(monkeypatch)
    calls = {"count": 0}

    async def bad_request(prompt, model, temperature, max_tokens):
        calls["count"] += 1
        raise _status_error(400)

    monkeypatch.setattr(service, "_call_groq", bad_request)

    with pytest.raises(httpx.HTTPStatusError):
        await service.generate("hi")
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_generate_falls_back_to_openrouter(monkeypatch):
    service = _service(monkeypatch, openrouter_key="test-openrouter-key")

    async def always_fails(prompt, model, temperature, max_tokens):
        raise httpx.ReadError("groq down")

    async def works(prompt, model, temperature, max_tokens):
        return "from openrouter"

    monkeypatch.setattr(service, "_call_groq", always_fails)
    monkeypatch.setattr(service, "_call_openrouter", works)

    assert await service.generate("hi") == "from openrouter"


@pytest.mark.asyncio
async def test_generate_raises_without_any_provider(monkeypatch):
    service = _service(monkeypatch)
    service.groq_key = ""

    with pytest.raises(RuntimeError, match="No LLM provider available"):
        await service.generate("hi")


def test_parse_duration_reads_groq_reset_header():
    assert _parse_duration("18h51m50.4s") == pytest.approx(67910.4)
    assert _parse_duration("690ms") == pytest.approx(0.69)
    assert _parse_duration(None) is None


def test_quota_tracker_reports_exhaustion(monkeypatch):
    monkeypatch.setattr(QUOTA, "providers", {})

    assert QUOTA.exhausted("groq") is False

    QUOTA.record_response("groq", {"x-ratelimit-remaining-requests": "0"})
    assert QUOTA.exhausted("groq") is True

    QUOTA.record_response("groq", {"x-ratelimit-remaining-requests": "12"})
    assert QUOTA.exhausted("groq") is False
    assert QUOTA.remaining_requests("groq") == 12
    assert QUOTA.snapshot("groq")["remaining_requests"] == 12


def test_quota_is_tracked_per_provider(monkeypatch):
    monkeypatch.setattr(QUOTA, "providers", {})

    QUOTA.record_response("groq", {"x-ratelimit-remaining-requests": "0"})

    # An exhausted Groq window must not stop a run that continues on the
    # fallback provider.
    assert QUOTA.exhausted("groq") is True
    assert QUOTA.exhausted("openrouter") is False
    assert QUOTA.remaining_requests("openrouter") is None


@pytest.mark.asyncio
async def test_generate_stops_retrying_when_window_quota_is_spent(monkeypatch):
    service = _service(monkeypatch)
    monkeypatch.setattr(QUOTA, "providers", {})
    calls = {"count": 0}

    async def rate_limited(prompt, model, temperature, max_tokens):
        calls["count"] += 1
        request = httpx.Request("POST", "https://example.test/chat")
        response = httpx.Response(429, request=request)
        response.headers["x-ratelimit-remaining-requests"] = "0"
        raise httpx.HTTPStatusError(
            "rate limited", request=request, response=response
        )

    monkeypatch.setattr(service, "_call_groq", rate_limited)

    with pytest.raises(LLMQuotaExhaustedError):
        await service.generate("hi")
    assert calls["count"] == 1
