from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class TicketBase(BaseModel):
    title: str
    status: str = "open"
    intent: Optional[str] = None
    escalated: bool = False


class TicketCreate(TicketBase):
    pass


class TicketOut(TicketBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True