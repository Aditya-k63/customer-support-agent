from app.agents.state import SupportState


class EscalationHandlerNode:
    def __init__(self) -> None:
        pass

    def should_escalate(self, state: SupportState) -> bool:
        """Return True if ticket should be escalated to human."""
        escalation_intents = {"human_handoff", "refund_request", "billing"}
        if state.intent in escalation_intents:
            return True
        # Confidence-based escalation disabled for v1 baseline;
        # v2 will enable with real retrieval-derived confidence scores
        if state.confidence_score < 0.0:
            return True
        return False

    async def run(self, state: SupportState) -> SupportState:
        state.escalation_flag = True
        state.status = "in_progress"
        if not state.response:
            state.response = (
                "I've escalated this to a human agent. They will get back to you shortly."
            )
        return state