from fastapi import APIRouter, Depends, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.ticket import TicketOut, TicketCreate
from app.core.security import verify_api_key
from app.agents.graph import build_graph
from app.db.base import get_db
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """Main chat endpoint: processes user message through LangGraph agent."""
    graph = build_graph()
    initial_state = {
        "messages": [],
        "user_query": request.message,
        "intent": None,
        "context": [],
        "confidence_score": 0.0,
        "ticket_id": None,
        "status": "open",
        "escalation_flag": False,
        "session_id": request.session_id,
    }

    try:
        result = await graph.ainvoke(initial_state)
        return ChatResponse(
            response=result.get("response", "I'm sorry, I couldn't process your request."),
            ticket_id=result.get("ticket_id"),
            status=result.get("status", "open"),
            intent=result.get("intent"),
            escalated=result.get("escalation_flag", False),
            confidence=result.get("confidence_score", 0.0),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
async def get_ticket(
    ticket_id: int,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """Get ticket details by ID."""
    from app.db.models import Ticket
    from sqlalchemy import select

    stmt = select(Ticket).where(Ticket.id == ticket_id)
    result = await db.execute(stmt)
    ticket = result.scalar_one_or_none()

    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.get("/admin/tickets")
async def list_tickets(
    status: str | None = None,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(verify_api_key),
):
    """Admin endpoint: list tickets, optionally filtered by status."""
    from app.db.models import Ticket
    from sqlalchemy import select

    stmt = select(Ticket)
    if status:
        stmt = stmt.where(Ticket.status == status)
    result = await db.execute(stmt)
    tickets = result.scalars().all()
    return tickets