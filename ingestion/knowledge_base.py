"""Stable knowledge base used by ingestion and evaluation."""

KNOWLEDGE_BASE: list[dict] = [
    {
        "id": "refund_policy",
        "content": (
            "Refund policy: Items can be refunded within 30 days of purchase "
            "with original receipt."
        ),
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "id": "technical_support",
        "content": (
            "Technical support: Reset your password at account settings or "
            "contact support@company.com."
        ),
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "id": "billing_invoices",
        "content": (
            "Billing: Invoices are generated on the 1st of each month. Payment "
            "methods include credit card and PayPal."
        ),
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "id": "business_hours",
        "content": "General: Our business hours are Mon-Fri, 9 AM to 6 PM EST.",
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "id": "shipping",
        "content": (
            "Shipping: Orders are processed within 1-2 business days and "
            "shipped via standard ground delivery."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "id": "password_reset",
        "content": (
            "Password reset: Click 'Forgot Password' on the login page or visit "
            "account settings to reset your password."
        ),
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "id": "subscription_cancellation",
        "content": (
            "Subscription cancellation: You can cancel anytime from your "
            "account dashboard. Access continues until the end of your "
            "billing period."
        ),
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "id": "order_tracking",
        "content": (
            "Order tracking: Use the tracking number in your shipping "
            "confirmation email or check your order history page."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
]

CATEGORY_TO_DOC_IDS: dict[str, list[str]] = {
    "billing": ["billing_invoices", "refund_policy", "subscription_cancellation"],
    "technical": ["technical_support", "password_reset"],
    "general": ["business_hours", "shipping", "order_tracking"],
    "refund_request": ["refund_policy", "subscription_cancellation"],
    "human_handoff": [],
}
