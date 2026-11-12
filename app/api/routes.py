import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import build_graph
from app.agents.tools.ticket_db import TicketDB
from app.core.config import settings
from app.core.security import require_admin
from app.db.base import get_db
from app.db.models import Ticket
from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.ticket import TicketOut

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """Process a user message through the LangGraph agent."""
    ticket_db = TicketDB()
    history = await ticket_db.get_conversation_history(
        request.session_id, limit=settings.CONVERSATION_HISTORY_LIMIT
    )

    graph = build_graph()
    initial_state = {
        "messages": history,
        "user_query": request.message,
        "intent": None,
        "context": [],
        "retrieved_doc_ids": [],
        "retrieval_scores": [],
        "retrieval_similarity": 0.0,
        "answer_confidence": 0.0,
        "ticket_id": None,
        "status": "open",
        "escalation_flag": False,
        "session_id": request.session_id,
        "response": "",
    }

    try:
        result = await graph.ainvoke(initial_state)
    except Exception:
        logger.exception("chat_failed", session_id=request.session_id)
        raise HTTPException(
            status_code=500, detail="An internal error occurred while processing the request."
        )

    return ChatResponse(
        response=result.get("response", "I'm sorry, I couldn't process your request."),
        ticket_id=result.get("ticket_id"),
        status=result.get("status", "open"),
        intent=result.get("intent"),
        escalated=result.get("escalation_flag", False),
        retrieval_similarity=result.get("retrieval_similarity", 0.0),
        answer_confidence=result.get("answer_confidence", 0.0),
        sources=result.get("retrieved_doc_ids", []),
    )


@router.get("/tickets/{ticket_id}", response_model=TicketOut)
async def get_ticket(
    ticket_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get ticket details by ID."""
    stmt = select(Ticket).where(Ticket.id == ticket_id)
    result = await db.execute(stmt)
    ticket = result.scalar_one_or_none()

    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.get("/admin/tickets")
async def list_tickets(
    status_filter: str | None = Query(default=None, alias="status"),
    db: AsyncSession = Depends(get_db),
    _role: str = Depends(require_admin),
):
    """Admin endpoint: list tickets, optionally filtered by status."""
    stmt = select(Ticket)
    if status_filter:
        stmt = stmt.where(Ticket.status == status_filter)
    result = await db.execute(stmt)
    return result.scalars().all()
