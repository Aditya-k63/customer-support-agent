"""Security utilities: API key validation, role checks, and rate limiting."""
import asyncio
import hmac
import time
from collections import defaultdict
from typing import Dict

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings

security = HTTPBearer(auto_error=False)

ROLE_CUSTOMER = "customer"
ROLE_ADMIN = "admin"


class RateLimiter:
    """Simple in-memory rate limiter using a sliding window.

    This is process-local. For multi-worker or multi-replica deployments use a
    shared backend (for example Redis) instead.
    """

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: Dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, key: str) -> bool:
        """Check if a request is allowed for the given key."""
        async with self._lock:
            now = time.time()
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


def _token_matches(token: str, expected: str) -> bool:
    """Constant-time comparison that rejects empty secrets."""
    if not expected or not token:
        return False
    return hmac.compare_digest(token, expected)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> str:
    """Validate the bearer token and return the caller role.

    There is deliberately no development bypass: every request to a protected
    route must present a valid API key, regardless of ENVIRONMENT.
    """
    token = credentials.credentials if credentials else ""
    if _token_matches(token, settings.ADMIN_API_KEY):
        return ROLE_ADMIN
    if _token_matches(token, settings.API_KEY):
        return ROLE_CUSTOMER
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )


async def require_admin(role: str = Depends(verify_api_key)) -> str:
    """Require an administrator API key."""
    if role != ROLE_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required",
        )
    return role
