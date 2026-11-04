class TicketDB:
    def __init__(self) -> None:
        # Placeholder: integrate PostgreSQL async session here
        self._counter = 1

    async def create_ticket(
        self,
        title: str,
        status: str,
        intent: str | None,
        escalated: bool,
    ) -> int:
        ticket_id = self._counter
        self._counter += 1
        return ticket_id