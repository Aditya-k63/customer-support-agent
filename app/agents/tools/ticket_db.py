"""Persistent ticket database backed by PostgreSQL + SQLAlchemy."""
from typing import Optional
from app.db.base import AsyncSessionLocal
from app.db.models import Ticket, Conversation
from sqlalchemy import select


class TicketDB:
    async def create_ticket(
        self,
        title: str,
        status: str = "open",
        intent: Optional[str] = None,
        escalated: bool = False,
    ) -> int:
        async with AsyncSessionLocal() as session:
            ticket = Ticket(
                title=title,
                status=status,
                intent=intent,
                escalated=escalated,
            )
            session.add(ticket)
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
            await session.commit()
            return True

    async def add_conversation(
        self, ticket_id: int, role: str, content: str
    ) -> int:
        async with AsyncSessionLocal() as session:
            conv = Conversation(ticket_id=ticket_id, role=role, content=content)
            session.add(conv)
            await session.commit()
            await session.refresh(conv)
            return conv.id