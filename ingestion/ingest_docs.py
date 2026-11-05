"""Ingest FAQ/policy documents into pgvector with real embeddings."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.tools.vector_store import VectorStore  # noqa: E402


SAMPLE_DOCS = [
    {
        "content": "Refund policy: Items can be refunded within 30 days of purchase with original receipt.",
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "content": "Technical support: Reset your password at account settings or contact support@company.com.",
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "content": "Billing: Invoices are generated on the 1st of each month. Payment methods include credit card and PayPal.",
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "content": "General: Our business hours are Mon-Fri, 9 AM to 6 PM EST.",
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "content": "Shipping: Orders are processed within 1-2 business days and shipped via standard ground delivery.",
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "content": "Password reset: Click 'Forgot Password' on the login page or visit account settings to reset your password.",
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "content": "Subscription cancellation: You can cancel anytime from your account dashboard. Access continues until the end of your billing period.",
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "content": "Order tracking: Use the tracking number in your shipping confirmation email or check your order history page.",
        "metadata": {"category": "general", "source": "faq"},
    },
]


def main() -> None:
    vs = VectorStore()
    asyncio.run(vs.add_documents(SAMPLE_DOCS))
    print(f"Indexed {len(SAMPLE_DOCS)} documents into pgvector with embeddings.")


if __name__ == "__main__":
    main()