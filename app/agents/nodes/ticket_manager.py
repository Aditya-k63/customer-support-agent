from app.agents.state import SupportState
from app.agents.tools.ticket_db import TicketDB


class TicketManagerNode:
    def __init__(self) -> None:
        self.ticket_db = TicketDB()

    async def run(self, state: SupportState) -> SupportState:
        ticket_id = await self.ticket_db.create_ticket(
            title=state.user_query[:100],
            status=state.status,
            intent=state.intent,
            escalated=state.escalation_flag,
        )
        state.ticket_id = ticket_id
        state.status = "resolved" if not state.escalation_flag else "in_progress"
        return state