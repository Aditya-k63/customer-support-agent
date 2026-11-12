"""Persistent ticket and conversation storage backed by PostgreSQL."""
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.db.base import AsyncSessionLocal
from app.db.models import Conversation, Ticket


class TicketDB:
    async def create_ticket(
        self,
        title: str,
        status: str = "open",
        intent: Optional[str] = None,
        escalated: bool = False,
        session_id: Optional[str] = None,
    ) -> int:
        """Create a new ticket with its final status already applied."""
        async with AsyncSessionLocal() as session:
            ticket = Ticket(
                title=title,
                status=status,
                intent=intent,
                escalated=escalated,
                session_id=session_id,
            )
            session.add(ticket)
            await session.commit()
            await session.refresh(ticket)
            return ticket.id

    async def get_or_create_ticket(
        self,
        session_id: str,
        title: str,
        status: str,
        intent: Optional[str] = None,
        escalated: bool = False,
    ) -> int:
        """Return the ticket for this session, updating it to the final state.

        The final status is written in the same operation that creates or
        updates the row, so the API response and the database can never
        disagree about the ticket state.
        """
        async with AsyncSessionLocal() as session:
            ticket: Optional[Ticket] = None
            if session_id:
                stmt = (
                    select(Ticket)
                    .where(Ticket.session_id == session_id)
                    .order_by(Ticket.id.desc())
                    .limit(1)
                )
                result = await session.execute(stmt)
                ticket = result.scalar_one_or_none()

            if ticket is None:
                ticket = Ticket(
                    session_id=session_id or None,
                    title=title,
                    status=status,
                    intent=intent,
                    escalated=escalated,
                )
                session.add(ticket)
            else:
                ticket.title = title or ticket.title
                ticket.status = status
                ticket.intent = intent
                ticket.escalated = escalated
                ticket.updated_at = datetime.utcnow()

            await session.commit()
            await session.refresh(ticket)
            return ticket.id

    async def get_ticket(self, ticket_id: int) -> Optional[dict]:
        async with AsyncSessionLocal() as session:
            stmt = select(Ticket).where(Ticket.id == ticket_id)
            result = await session.execute(stmt)
            ticket = result.scalar_one_or_none()
            if not ticket:
                return None
            return {
                "id": ticket.id,
                "session_id": ticket.session_id,
                "title": ticket.title,
                "status": ticket.status,
                "intent": ticket.intent,
                "escalated": ticket.escalated,
                "created_at": ticket.created_at.isoformat()
                if ticket.created_at
                else None,
                "updated_at": ticket.updated_at.isoformat()
                if ticket.updated_at
                else None,
            }

    async def update_ticket_status(self, ticket_id: int, status: str) -> bool:
        async with AsyncSessionLocal() as session:
            stmt = select(Ticket).where(Ticket.id == ticket_id)
            result = await session.execute(stmt)
            ticket = result.scalar_one_or_none()
            if not ticket:
                return False
            ticket.status = status
            ticket.updated_at = datetime.utcnow()
            await session.commit()
            return True

    async def add_conversation(self, ticket_id: int, role: str, content: str) -> int:
        async with AsyncSessionLocal() as session:
            conv = Conversation(ticket_id=ticket_id, role=role, content=content)
            session.add(conv)
            await session.commit()
            await session.refresh(conv)
            return conv.id

    async def get_conversation_history(
        self, session_id: str, limit: int = 20
    ) -> list[dict]:
        """Load the most recent messages for a session, oldest first."""
        if not session_id:
            return []
        async with AsyncSessionLocal() as session:
            stmt = (
                select(Conversation)
                .join(Ticket, Conversation.ticket_id == Ticket.id)
                .where(Ticket.session_id == session_id)
                .order_by(Conversation.id.desc())
                .limit(limit)
            )
            result = await session.execute(stmt)
            rows = list(result.scalars().all())
            rows.reverse()
            return [{"role": row.role, "content": row.content} for row in rows]
