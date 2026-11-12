from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class TicketBase(BaseModel):
    title: str
    status: str = "open"
    intent: Optional[str] = None
    escalated: bool = False


class TicketCreate(TicketBase):
    pass


class TicketOut(TicketBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None