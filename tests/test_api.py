from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CUSTOMER_HEADERS = {"Authorization": "Bearer test-api-key"}
ADMIN_HEADERS = {"Authorization": "Bearer test-admin-key"}


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


def test_liveness():
    resp = client.get("/health/live")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_ok():
    with patch("app.main.check_database", new=AsyncMock(return_value=True)):
        resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json()["database"] == "ok"


@pytest.mark.asyncio
async def test_readiness_failure():
    with patch("app.main.check_database", new=AsyncMock(return_value=False)):
        resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json()["status"] == "not_ready"


def test_chat_requires_api_key():
    resp = client.post(
        "/api/v1/chat",
        json={"message": "hello", "session_id": "s1"},
    )
    assert resp.status_code == 401


def test_chat_rejects_wrong_api_key():
    resp = client.post(
        "/api/v1/chat",
        json={"message": "hello", "session_id": "s1"},
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 401


def test_admin_requires_admin_key():
    resp = client.get("/api/v1/admin/tickets", headers=CUSTOMER_HEADERS)
    assert resp.status_code == 403
