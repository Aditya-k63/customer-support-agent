import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import Depends, FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.api import routes
from app.core.config import settings
from app.core.security import rate_limiter, verify_api_key
from app.db.base import engine

logging.basicConfig(format="%(message)s", level=logging.INFO)
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Apply database migrations on startup and fail fast on error."""
    if not settings.RUN_MIGRATIONS_ON_STARTUP:
        logger.warning("Startup migrations disabled by configuration")
        yield
        return

    from alembic.config import Config

    from app.db.migrations import upgrade_database

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_cfg = Config(os.path.join(project_root, "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

    try:
        await asyncio.to_thread(upgrade_database, alembic_cfg)
    except Exception:
        logger.exception("database_migration_failed")
        raise
    logger.info("Database migrations applied successfully")
    yield


app = FastAPI(
    title="Autonomous Customer Support Agent",
    version="1.0.0",
    description="LangGraph + RAG multi-agent support system",
    lifespan=lifespan,
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS.split(",")
    if settings.ALLOWED_HOSTS
    else ["localhost", "127.0.0.1"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",")
    if settings.CORS_ORIGINS
    else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/health"):
        return await call_next(request)
    client_ip = request.client.host if request.client else "unknown"
    if not await rate_limiter.is_allowed(client_ip):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."},
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW)},
        )
    return await call_next(request)


@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time_ms=round(process_time, 2),
        request_id=request_id,
    )
    return response


async def check_database() -> bool:
    """Return True when PostgreSQL accepts a query."""
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.exception("database_readiness_check_failed")
        return False


app.include_router(
    routes.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)]
)


@app.get("/health")
@app.get("/health/live")
async def health():
    """Liveness probe: the process is running."""
    return {"status": "healthy", "service": "support-agent"}


@app.get("/health/ready")
async def readiness():
    """Readiness probe: dependencies required to serve traffic are available."""
    if await check_database():
        return {"status": "ready", "service": "support-agent", "database": "ok"}
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "not_ready", "service": "support-agent", "database": "down"},
    )
