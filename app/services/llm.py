"""LLM service with Groq primary and OpenRouter fallback for rate limiting."""
import httpx
from app.core.config import settings


class LLMService:
    def __init__(self) -> None:
        self.groq_key = settings.GROQ_API_KEY
        self.groq_model = settings.GROQ_MODEL
        self.openrouter_key = getattr(settings, "OPENROUTER_API_KEY", "")
        self.openrouter_model = getattr(
            settings, "OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct"
        )

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """Generate text with Groq primary and OpenRouter fallback on rate limit."""
        # Try Groq first
        if self.groq_key:
            try:
                return await self._call_groq(
                    prompt, model or self.groq_model, temperature, max_tokens
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    # Rate limited — fall back to OpenRouter
                    pass
                else:
                    raise

        # Fallback to OpenRouter
        if self.openrouter_key:
            try:
                return await self._call_openrouter(
                    prompt, self.openrouter_model, temperature, max_tokens
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 429:
                    raise RuntimeError(
                        "Both Groq and OpenRouter are rate limited"
                    ) from exc
                raise

        raise RuntimeError("No LLM provider available")

    async def _call_groq(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _call_openrouter(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.openrouter_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]