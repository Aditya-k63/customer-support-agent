"""Stable knowledge base used by ingestion and evaluation.

Every document has a stable id. Chunks derived from a document keep that id
with a ``#chunk-N`` suffix, so evaluation can compare retrieved documents by
identity instead of by text similarity.

The corpus is intentionally small but each document carries the concrete
policy facts a support agent needs, so that the golden set can be answered
from it and the retrieval metrics measure something meaningful.
"""
KNOWLEDGE_BASE: list[dict] = [
    {
        "id": "refund_policy",
        "content": (
            "Refund policy: Items can be refunded within 30 days of purchase "
            "with the original receipt. Refunds are issued to the original "
            "payment method within 5 to 7 business days. Duplicate charges "
            "are refunded in full, and a partially used service is refunded "
            "proportionally. Store credit is offered instead of a cash "
            "refund on request, and a chargeback dispute is handled by the "
            "billing team once the refund request is logged."
        ),
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "id": "billing_invoices",
        "content": (
            "Billing: Invoices are generated on the 1st of each month and are "
            "available to download from the billing history page. Billing "
            "address, tax information and company details can be updated at "
            "any time under Account Settings, and an itemized invoice can be "
            "requested for every payment. A failed payment is retried after "
            "three days, a plan change is prorated on the next invoice, and "
            "a billing adjustment can be requested with the invoice number."
        ),
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "id": "payment_methods",
        "content": (
            "Payment methods: We accept credit card, PayPal and bank "
            "transfer, and the payment method on file can be replaced from "
            "the billing settings page. A declined payment is usually caused "
            "by an expired card or an insufficient balance, and the invoice "
            "stays payable for seven days. A pending charge appears during "
            "the authorisation window and settles within two business days. "
            "Payments are made in the account currency and cannot be split "
            "across two methods."
        ),
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "id": "subscription_cancellation",
        "content": (
            "Subscription cancellation: You can cancel anytime from your "
            "account dashboard. Access continues until the end of the "
            "billing period and there is no cancellation fee. A subscription "
            "can be paused for up to three months, a downgrade takes effect "
            "on the next billing cycle, and a cancelled subscription can be "
            "reactivated within 30 days without losing saved settings."
        ),
        "metadata": {"category": "billing", "source": "faq"},
    },
    {
        "id": "technical_support",
        "content": (
            "Technical support: Clear the browser cache and cookies, update "
            "the mobile app and check the status page before contacting us "
            "at support@company.com. Errors such as 404, SSL or timeout "
            "failures are usually fixed by clearing the cache, updating the "
            "app, or switching network. We answer technical requests within "
            "one business day and the status page lists live incidents."
        ),
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "id": "password_reset",
        "content": (
            "Password reset: Click 'Forgot Password' on the login page or "
            "visit account settings to reset your password. The reset link "
            "is valid for 30 minutes, a password must be at least 12 "
            "characters long, and the account locks temporarily after five "
            "failed attempts. Two-factor authentication can be disabled "
            "again from account settings once the password is reset."
        ),
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "id": "email_verification",
        "content": (
            "Email verification: The verification link expires after 24 "
            "hours and a new link can be requested from the login page. The "
            "email address on the account is updated under Account Settings "
            "and the new address has to be verified before notifications "
            "resume. Unverified accounts keep full access but do not "
            "receive product or billing emails."
        ),
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "id": "account_access",
        "content": (
            "Account access: Signing in happens with the email address and "
            "password. If the dashboard is not reachable, sign out of all "
            "devices and sign in again, then reset the password. Two-factor "
            "authentication, a connected domain, third-party integrations, "
            "dark mode and the MFA device are all managed under Account "
            "Settings, and a locked MFA device is reset by support."
        ),
        "metadata": {"category": "technical", "source": "faq"},
    },
    {
        "id": "business_hours",
        "content": (
            "General: Our business hours are Mon-Fri, 9 AM to 6 PM EST, and "
            "our office is at 100 Market Street, New York. Email requests "
            "sent outside those hours are answered on the next business day, "
            "and the holiday schedule is published in advance. Support "
            "answers within one business day, and phone support is "
            "available for premium plans."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "id": "shipping",
        "content": (
            "Shipping: Orders are processed within 1-2 business days and "
            "shipped via standard ground delivery. Standard delivery takes "
            "3 to 5 business days and international shipping 5 to 10 "
            "business days. A shipping address can be changed until the "
            "order is processed, and a signature is required for orders "
            "above a certain value."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "id": "order_tracking",
        "content": (
            "Order tracking: Use the tracking number in your shipping "
            "confirmation email or check your order history page. The "
            "tracking number appears as soon as the label is created, and "
            "large orders may arrive in separate shipments with their own "
            "tracking numbers. If a shipment is missing after the delivery "
            "estimate, contact support@company.com with the order number."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "id": "account_management",
        "content": (
            "Account management: Create an account with an email address, "
            "update your profile and notification settings under Account "
            "Settings, and delete the account at any time. Deletion removes "
            "the profile after a 30 day grace period, and personal data can "
            "be exported as a file on request. The privacy policy explains "
            "which data is stored and for how long, and the terms of "
            "service are shown at checkout and in Account Settings. The "
            "referral programme gives 10 dollars to you and 10 dollars to "
            "the friend you refer once they subscribe, and marketing emails "
            "can be unsubscribed from the link in every newsletter."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "id": "plans_pricing",
        "content": (
            "Plans and pricing: The free tier includes one project with the "
            "core features, and paid plans add team seats, storage and "
            "priority support. Upgrading takes effect immediately and is "
            "prorated, and the features included in every tier are listed "
            "on the pricing page. Student discounts are 50 percent with a "
            "valid student email, enterprise plans are quoted per "
            "organisation, and a loyalty discount is applied automatically "
            "after three years of subscription."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
    {
        "id": "live_chat",
        "content": (
            "Customer support contact: Live chat is available Mon-Fri, "
            "9 AM to 6 PM EST, and email support@company.com is answered "
            "within one business day. The help centre holds the answers to "
            "the most common questions, the status page lists current "
            "incidents, and feedback on the support experience can be sent "
            "to support@company.com with the ticket number. Onboarding "
            "training for new teams is scheduled on request, and the company "
            "mission is to make customer support fast and fair."
        ),
        "metadata": {"category": "general", "source": "faq"},
    },
]

CATEGORY_TO_DOC_IDS: dict[str, list[str]] = {
    "billing": [
        "billing_invoices",
        "payment_methods",
        "refund_policy",
        "subscription_cancellation",
    ],
    "technical": [
        "account_access",
        "email_verification",
        "password_reset",
        "technical_support",
    ],
    "general": [
        "account_management",
        "business_hours",
        "live_chat",
        "order_tracking",
        "plans_pricing",
        "shipping",
    ],
    "refund_request": ["refund_policy", "subscription_cancellation"],
    "human_handoff": [],
}
