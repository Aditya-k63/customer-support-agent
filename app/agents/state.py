from typing import List, Optional, Literal
from pydantic import BaseModel


class SupportState(BaseModel):
    messages: List[dict] = []
    user_query: str = ""
    intent: Optional[str] = None
    context: List[str] = []
    confidence_score: float = 0.0
    ticket_id: Optional[int] = None
    status: Literal["open", "in_progress", "resolved", "closed"] = "open"
    escalation_flag: bool = False
    session_id: str = ""
    response: str = ""