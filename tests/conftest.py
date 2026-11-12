import os

# Must be set before app.core.config is imported by any test module.
os.environ.setdefault("RUN_MIGRATIONS_ON_STARTUP", "false")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")

import pytest  # noqa: E402


@pytest.fixture
def sample_state():
    from app.agents.state import SupportState

    return SupportState(
        user_query="How do I request a refund?",
        session_id="test-session-1",
    )
