import structlog

from app.agents.state import SupportState
from app.agents.tools.ticket_db import TicketDB

logger = structlog.get_logger(__name__)


class TicketManagerNode:
    def __init__(self) -> None:
        self.ticket_db = TicketDB()

    async def run(self, state: SupportState) -> SupportState:
        # Determine the final status BEFORE touching the database so the row,
        # the graph state and the API response always agree.
        final_status = "in_progress" if state.escalation_flag else "resolved"

        ticket_id = await self.ticket_db.get_or_create_ticket(
            session_id=state.session_id,
            title=state.user_query[:255],
            status=final_status,
            intent=state.intent,
            escalated=state.escalation_flag,
        )
        state.ticket_id = ticket_id
        state.status = final_status

        # Persist both sides of this turn for multi-turn conversation memory.
        await self.ticket_db.add_conversation(
            ticket_id=ticket_id, role="user", content=state.user_query
        )
        if state.response:
            await self.ticket_db.add_conversation(
                ticket_id=ticket_id, role="assistant", content=state.response
            )

        logger.info(
            "ticket_persisted",
            ticket_id=ticket_id,
            session_id=state.session_id,
            status=final_status,
            escalated=state.escalation_flag,
        )
        return state
