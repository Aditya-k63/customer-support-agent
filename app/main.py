from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from app.api import routes
from app.core.config import settings
from app.core.security import verify_api_key, rate_limiter
import logging
import time
import structlog

# Configure structured logging
logging.basicConfig(
    format="%(message)s",
    level=logging.INFO,
)
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
    """Run Alembic migrations on startup."""
    from alembic import command
    from alembic.config import Config
    import os

    # Find the alembic.ini file relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    alembic_ini_path = os.path.join(project_root, "alembic.ini")

    alembic_cfg = Config(alembic_ini_path)
    alembic_cfg.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

    try:
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations applied successfully")
    except Exception as exc:
        logger.warning("Failed to apply migrations on startup", error=str(exc))
        logger.warning("Migrations can be run manually with: alembic upgrade head")

    yield


app = FastAPI(
    title="Autonomous Customer Support Agent",
    version="1.0.0",
    description="LangGraph + RAG multi-agent support system",
    lifespan=lifespan,
)

# Trusted host middleware - restrict to known hosts
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS.split(",") if settings.ALLOWED_HOSTS else ["localhost", "127.0.0.1"],
)

# CORS - restrict origins (no wildcard with credentials)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS else ["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
    expose_headers=["X-Request-ID"],
)

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    allowed = await rate_limiter.is_allowed(client_ip)
    if not allowed:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."},
            headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW)},
        )
    return await call_next(request)


# Security headers middleware
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    start_time = time.time()
    request_id = request.headers.get("X-Request-ID", "")

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"

    # Log request
    logger.info(
        "request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        process_time_ms=round(process_time, 2),
        request_id=request_id,
    )

    return response

app.include_router(routes.router, prefix="/api/v1", dependencies=[Depends(verify_api_key)])


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "support-agent"}