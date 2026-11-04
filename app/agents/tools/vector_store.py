class VectorStore:
    def __init__(self) -> None:
        # Placeholder: integrate pgvector or Chroma here
        self.docs = [
            "Refund policy: Items can be refunded within 30 days of purchase with original receipt.",
            "Technical support: Reset your password at account settings or contact support@company.com.",
            "Billing: Invoices are generated on the 1st of each month. Payment methods include credit card and PayPal.",
            "General: Our business hours are Mon-Fri, 9 AM to 6 PM EST.",
        ]

    async def search(self, query: str, top_k: int = 3) -> list[str]:
        # Simple keyword match placeholder; replace with real vector search
        query_lower = query.lower()
        matches = [d for d in self.docs if any(w in d.lower() for w in query_lower.split())]
        return matches[:top_k] or self.docs[:top_k]