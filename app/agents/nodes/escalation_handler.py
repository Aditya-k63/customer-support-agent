from app.agents.state import SupportState


class EscalationHandlerNode:
    ESCALATION_INTENTS = {"human_handoff", "refund_request", "billing"}

    def should_escalate(self, state: SupportState) -> bool:
        """Return True if the ticket should be handed to a human."""
        return state.intent in self.ESCALATION_INTENTS

    async def run(self, state: SupportState) -> SupportState:
        state.escalation_flag = True
        state.status = "in_progress"
        if not state.response:
            state.response = (
                "I've escalated this to a human agent. They will get back to you shortly."
            )
        return state
