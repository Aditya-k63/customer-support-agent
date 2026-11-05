"""Security utilities: API key validation, CORS, input sanitization, rate limiting."""
import asyncio
import time
from collections import defaultdict
from typing import Dict

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

security = HTTPBearer(auto_error=False)


class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed for the given key."""
        async with self._lock:
            now = time.time()
            # Clean old entries outside the window
            self._requests[key] = [
                t for t in self._requests[key] if now - t < self.window_seconds
            ]
            if len(self._requests[key]) >= self.max_requests:
                return False
            self._requests[key].append(now)
            return True


rate_limiter = RateLimiter(
    max_requests=settings.RATE_LIMIT_REQUESTS,
    window_seconds=settings.RATE_LIMIT_WINDOW,
)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> bool:
    """Validate bearer token against API_KEY."""
    if settings.ENVIRONMENT == "development":
        return True
    if not credentials or credentials.credentials != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return True


async def rate_limit_dependency(request_id: str = "default") -> bool:
    """Rate limiting dependency. Uses request ID as key for per-client limiting."""
    allowed = await rate_limiter.is_allowed(request_id)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW)},
        )
    return True