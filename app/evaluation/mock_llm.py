"""Deterministic keyword-based LLM mock used only by deterministic tests."""
from typing import Any, Dict, List


MOCK_INTENT_KEYWORDS: Dict[str, List[str]] = {
    "billing": [
        "refund", "charge", "invoice", "payment", "bill", "subscription",
        "credit card", "promo", "tax", "currency", "deposit", "overcharged",
        "loyalty", "discount", "receipt", "payment method", "billing address",
        "payment due", "billing plan", "accidental charge", "billing history",
        "declined", "credit card details", "itemized invoice", "promo code",
        "miss a payment", "pause", "billing adjustment", "charged after",
        "tax receipt", "pending charge", "split payment", "tax information",
        "failed service", "billing support", "loyalty discount",
    ],
    "technical": [
        "password", "login", "crash", "app", "website", "404", "cache",
        "video", "upload", "search", "notification", "dashboard", "freezing",
        "export", "SSL", "domain", "checkout", "dark mode", "email", "API",
        "integration", "timeout", "cookie", "MFA", "download", "form",
        "not loading", "log into", "two-factor", "browser", "won't play",
        "update the app", "uploading files", "search function",
        "notification settings", "access my dashboard", "page keeps freezing",
        "export my data", "SSL error", "connect my domain", "checkout button",
        "enable dark mode", "verify my email", "API is returning",
        "set up integrations", "timeout error", "clear my cookies",
        "mobile app won't open", "reset my MFA", "download my files",
        "form won't submit",
    ],
    "general": [
        "hours", "shipment", "office", "contact", "return policy",
        "shipping", "address", "privacy", "account", "mobile app",
        "delete my account", "countries", "verify my identity", "student",
        "refer a friend", "terms of service", "unsubscribe", "live chat",
        "update my profile", "refund window", "enterprise", "get started",
        "free tier", "upgrade", "training", "feedback", "mission",
        "help center", "status page", "business hours", "track my shipment",
        "office located", "customer support", "ship internationally",
        "shipping address", "privacy policy", "create an account",
        "countries do you serve", "student discounts", "unsubscribe from emails",
        "enterprise plans", "features are included", "upgrade my account",
        "offer training", "provide feedback", "company mission",
        "find your help center",
    ],
    "refund_request": [
        "refund", "cancel", "money back", "defective", "dispute",
        "return", "reverse", "transaction", "purchase", "opt out",
        "chargeback", "double billed", "store credit", "undo",
        "want to refund", "want to cancel", "defective product",
        "request a refund", "didn't use", "reverse this",
        "return my purchase", "long does a refund", "dispute this charge",
        "partial refund", "not satisfied", "i want to refund",
        "i want to cancel", "can i get a refund", "how do i request a refund",
        "i was charged for", "can i reverse", "i want to return",
        "how long does", "i want to dispute", "can i get a partial",
        "i'm not satisfied", "can i reverse a completed", "i want to opt out",
        "how do i initiate", "i was double billed", "can i get store credit",
        "i want to undo", "the product never arrived", "can i escalate",
        "i want to revoke", "how do i file", "i was charged without",
        "can i get my deposit", "i want to cancel and refund",
        "the service was not", "can i speak to someone",
        "i need an emergency", "how do i check my refund",
        "i want to reverse a recurring",
    ],
    "human_handoff": [
        "human agent", "real person", "transfer", "supervisor",
        "urgent", "manager", "review", "live agent", "bot",
        "call back", "customer service", "person", "now",
        "support staff", "human", "talk to someone",
        "phone number", "chat", "contact me directly",
        "human intervention", "routed", "in charge",
        "live help", "fix this", "transferred", "customer care",
        "speak to a human", "transfer me", "review my case",
        "can i speak to a human", "i need to talk", "can you transfer",
        "i want to speak", "this is urgent", "i need a human",
        "please connect me", "i have a complex", "can i get a live",
        "i don't want to talk to a bot", "can a human call",
        "i need to speak", "can you transfer this", "i need help from",
        "can i get a live agent", "i don't want", "can a human call me",
        "i need to speak to", "can i get a phone", "i want to chat",
        "can someone contact", "i need human", "can you route",
        "can i get live", "can i be transferred", "i want to talk",
    ],
}


def classify_intent_keywords(text: str) -> str:
    """Classify text using keyword counts (deterministic mock)."""
    lowered = text.lower()
    best_intent = "general"
    best_score = 0
    for intent, keywords in MOCK_INTENT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in lowered)
        if score > best_score:
            best_score = score
            best_intent = intent
    return best_intent


def mock_llm_generate(*args: Any, **kwargs: Any) -> str:
    """Deterministic mock for intent classification and response generation."""
    prompt = args[1] if len(args) > 1 else (args[0] if args else "")
    prompt_lower = str(prompt).lower()

    if "classify the user query" in prompt_lower:
        query = str(prompt)
        if "Query:" in query:
            query = query.split("Query:", 1)[1]
        return classify_intent_keywords(query)

    if "you are a customer support agent" in prompt_lower:
        intent = classify_intent_keywords(prompt)
        return (
            f"Here is the answer to your {intent} question. "
            "Please contact support if you need further assistance."
        )

    return "general"
