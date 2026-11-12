from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=255)


class ChatResponse(BaseModel):
    response: str
    ticket_id: Optional[int] = None
    status: str
    intent: Optional[str] = None
    escalated: bool = False
    retrieval_similarity: float = 0.0
    answer_confidence: float = 0.0
    sources: list[str] = []
