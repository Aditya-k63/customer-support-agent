"""LLM service with Groq provider."""
import os
from typing import Optional
from app.core.config import settings


class LLMService:
    def __init__(self) -> None:
        self.groq_key = settings.GROQ_API_KEY
        self.groq_model = settings.GROQ_MODEL

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 500,
    ) -> str:
        """Generate text using Groq API."""
        if not self.groq_key:
            raise RuntimeError("GROQ_API_KEY not configured")

        return await self._call_groq(prompt, model or self.groq_model, temperature, max_tokens)

    async def _call_groq(
        self, prompt: str, model: str, temperature: float, max_tokens: int
    ) -> str:
        import httpx

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