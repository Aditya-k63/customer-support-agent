import pytest


@pytest.fixture
def sample_state():
    from app.agents.state import SupportState
    return SupportState(
        user_query="How do I request a refund?",
        session_id="test-session-1",
    )