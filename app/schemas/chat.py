from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    ticket_id: Optional[int] = None
    status: str
    intent: Optional[str] = None
    escalated: bool = False
    confidence: float = 0.0