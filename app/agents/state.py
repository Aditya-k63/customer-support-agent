from typing import List, Literal, Optional

from pydantic import BaseModel


class SupportState(BaseModel):
    messages: List[dict] = []
    user_query: str = ""
    intent: Optional[str] = None

    # Retrieved context plus the identifiers and similarities that produced it
    context: List[str] = []
    retrieved_doc_ids: List[str] = []
    retrieval_scores: List[float] = []

    # Retrieval similarity is NOT answer confidence. Keep them separate.
    retrieval_similarity: float = 0.0
    answer_confidence: float = 0.0

    ticket_id: Optional[int] = None
    status: Literal["open", "in_progress", "resolved", "closed"] = "open"
    escalation_flag: bool = False
    session_id: str = ""
    response: str = ""
