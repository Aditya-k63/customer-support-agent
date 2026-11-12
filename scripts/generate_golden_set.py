"""Generate the golden set: 150 customer support questions with document labels.

Every question is paired with the knowledge-base documents that actually
answer it. Labelling per question (instead of per intent category) is what
makes Precision@K, Recall@K and MRR meaningful: the retrieval metrics can
only score a document relevant when that document is the one that answers
the question.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.knowledge_base import KNOWLEDGE_BASE  # noqa: E402

# Questions per intent category, each paired with the documents that answer it.
TEMPLATES: dict[str, list[tuple[str, list[str]]]] = {
    "billing": [
        ("I was charged twice for the same item.", ["refund_policy"]),
        ("When will my invoice be generated?", ["billing_invoices"]),
        ("I need to update my billing address.", ["billing_invoices"]),
        ("How do I update my payment method?", ["payment_methods"]),
        ("Why is my bill higher this month?", ["billing_invoices"]),
        ("Can I get a receipt for my last payment?", ["billing_invoices"]),
        ("I don't recognize this charge on my card.", ["payment_methods"]),
        ("How do I cancel my subscription?", ["subscription_cancellation"]),
        ("What payment methods do you accept?", ["payment_methods"]),
        ("When is my next payment due?", ["billing_invoices"]),
        ("I want to change my billing plan.", ["subscription_cancellation"]),
        ("Can I get a refund for an accidental charge?", ["refund_policy"]),
        ("How do I view my billing history?", ["billing_invoices"]),
        ("Why was my payment declined?", ["payment_methods"]),
        ("I need to update my credit card details.", ["payment_methods"]),
        ("Can I get an itemized invoice?", ["billing_invoices"]),
        ("How do I apply a promo code?", ["billing_invoices"]),
        ("What happens if I miss a payment?", ["billing_invoices"]),
        ("Can I pause my subscription?", ["subscription_cancellation"]),
        ("How do I request a billing adjustment?", ["billing_invoices"]),
        ("I was charged after I cancelled.", ["subscription_cancellation"]),
        ("Can I get a tax receipt?", ["billing_invoices"]),
        ("How do I change my currency?", ["payment_methods"]),
        ("Why is there a pending charge?", ["payment_methods"]),
        ("Can I split my payment?", ["payment_methods"]),
        ("How do I update my tax information?", ["billing_invoices"]),
        ("Can I get a refund for a failed service?", ["refund_policy"]),
        ("I was overcharged.", ["refund_policy"]),
        ("How do I contact billing support?", ["live_chat"]),
        ("Can I get a discount for loyalty?", ["plans_pricing"]),
    ],
    "technical": [
        ("How do I reset my password?", ["password_reset"]),
        ("The website is not loading.", ["technical_support"]),
        ("How do I reset my password? (reference #2)", ["password_reset"]),
        ("I can't log into my account.", ["account_access"]),
        ("The app keeps crashing on my phone.", ["technical_support"]),
        ("How do I enable two-factor authentication?", ["account_access"]),
        ("I'm getting a 404 error on your site.", ["technical_support"]),
        ("How do I clear my browser cache?", ["technical_support"]),
        ("The video won't play.", ["technical_support"]),
        ("How do I update the app?", ["technical_support"]),
        ("I'm having trouble uploading files.", ["technical_support"]),
        ("The search function isn't working.", ["technical_support"]),
        ("How do I change my notification settings?", ["account_management"]),
        ("I can't access my dashboard.", ["account_access"]),
        ("The page keeps freezing.", ["technical_support"]),
        ("How do I export my data?", ["account_management"]),
        ("I'm getting an SSL error.", ["technical_support"]),
        ("How do I connect my domain?", ["account_access"]),
        ("The checkout button isn't working.", ["technical_support"]),
        ("How do I enable dark mode?", ["account_access"]),
        ("I can't verify my email address.", ["email_verification"]),
        ("The API is returning errors.", ["technical_support"]),
        ("How do I set up integrations?", ["account_access"]),
        ("I'm getting a timeout error.", ["technical_support"]),
        ("How do I clear my cookies?", ["technical_support"]),
        ("The mobile app won't open.", ["technical_support"]),
        ("How do I reset my MFA?", ["account_access"]),
        ("I can't download my files.", ["technical_support"]),
        ("The form won't submit.", ["technical_support"]),
        ("The form won't submit. (reference #29)", ["technical_support"]),
    ],
    "general": [
        ("What are your business hours?", ["business_hours"]),
        ("How do I track my shipment?", ["order_tracking"]),
        ("Where is your office located?", ["business_hours"]),
        ("How do I contact customer support?", ["live_chat"]),
        ("What is your return policy?", ["refund_policy"]),
        ("Do you ship internationally?", ["shipping"]),
        ("How long does shipping take?", ["shipping"]),
        ("Can I change my shipping address?", ["shipping"]),
        ("What is your privacy policy?", ["account_management"]),
        ("How do I create an account?", ["account_management"]),
        ("Do you have a mobile app?", ["technical_support"]),
        ("How do I delete my account?", ["account_management"]),
        ("What countries do you serve?", ["shipping"]),
        ("How do I verify my identity?", ["email_verification"]),
        ("Do you offer student discounts?", ["plans_pricing"]),
        ("How do I refer a friend?", ["account_management"]),
        ("What is your terms of service?", ["account_management"]),
        ("How do I unsubscribe from emails?", ["account_management"]),
        ("Do you have a live chat?", ["live_chat"]),
        ("How do I update my profile?", ["account_management"]),
        ("What is your refund window?", ["refund_policy"]),
        ("Do you offer enterprise plans?", ["plans_pricing"]),
        ("How do I get started?", ["plans_pricing"]),
        ("What features are included in free tier?", ["plans_pricing"]),
        ("How do I upgrade my account?", ["plans_pricing"]),
        ("Do you offer training?", ["live_chat"]),
        ("How do I provide feedback?", ["live_chat"]),
        ("What is your company mission?", ["live_chat"]),
        ("How do I find your help center?", ["live_chat"]),
        ("Do you have a status page?", ["technical_support"]),
    ],
    "refund_request": [
        ("I want to refund my order.", ["refund_policy"]),
        ("I want to cancel my subscription.", ["subscription_cancellation"]),
        ("Can I get a refund for a defective product?", ["refund_policy"]),
        ("I want my money back.", ["refund_policy"]),
        ("How do I request a refund?", ["refund_policy"]),
        ("I was charged for a service I didn't use.", ["refund_policy"]),
        ("Can I reverse this transaction?", ["refund_policy"]),
        ("I want to return my purchase.", ["refund_policy"]),
        ("How long does a refund take?", ["refund_policy"]),
        ("I want to dispute this charge.", ["refund_policy"]),
        ("Can I get a partial refund?", ["refund_policy"]),
        ("I'm not satisfied with the service.", ["refund_policy"]),
        ("Can I reverse a completed payment?", ["refund_policy"]),
        ("I want to opt out and get my money back.", ["subscription_cancellation"]),
        ("How do I initiate a chargeback?", ["refund_policy"]),
        ("I was double billed.", ["refund_policy"]),
        ("Can I get store credit instead of cash?", ["refund_policy"]),
        ("I want to undo my purchase.", ["refund_policy"]),
        ("The product never arrived, I want a refund.", ["refund_policy"]),
        ("Can I escalate this to a refund specialist?", ["live_chat"]),
        ("I want to revoke my authorization.", ["payment_methods"]),
        ("How do I file a refund dispute?", ["refund_policy"]),
        ("I was charged without consent.", ["refund_policy"]),
        ("Can I get my deposit back?", ["refund_policy"]),
        ("I want to cancel and refund immediately.", ["subscription_cancellation"]),
        ("The service was not as described.", ["refund_policy"]),
        ("Can I speak to someone about my refund?", ["live_chat"]),
        ("I need an emergency refund.", ["refund_policy"]),
        ("How do I check my refund status?", ["refund_policy"]),
        ("I want to reverse a recurring charge.", ["subscription_cancellation"]),
    ],
    "human_handoff": [
        ("Can I speak to a human agent?", []),
        ("I need to talk to a real person.", []),
        ("Can you transfer me to support?", []),
        ("I want to speak with a supervisor.", []),
        ("This is urgent, I need human help.", []),
        ("Can I escalate to a manager?", []),
        ("I need a human to review my case.", []),
        ("Please connect me to an agent.", []),
        ("I have a complex issue needing human review.", []),
        ("Can I get a live agent?", []),
        ("I don't want to talk to a bot anymore.", []),
        ("Can a human call me back?", []),
        ("I need to speak to customer service.", []),
        ("Can you transfer this to a person?", []),
        ("I need help from a real human.", []),
        ("Can I talk to someone now?", []),
        ("I want to speak to support staff.", []),
        ("Can you escalate to a human?", []),
        ("I need a person, not automation.", []),
        ("Can I get a phone number for support?", []),
        ("I want to chat with a human.", []),
        ("Can someone contact me directly?", []),
        ("I need human intervention.", []),
        ("Can you route me to an agent?", []),
        ("I want to speak to someone in charge.", []),
        ("Can I get live help?", []),
        ("I need a human to fix this.", []),
        ("Can I be transferred now?", []),
        ("I want to talk to customer care.", []),
        ("Can I get a manager on the phone?", []),
    ],
}

EXPECTED = {
    "billing": {"intent": "billing", "escalate": True, "status": "in_progress"},
    "technical": {"intent": "technical", "escalate": False, "status": "resolved"},
    "general": {"intent": "general", "escalate": False, "status": "resolved"},
    "refund_request": {
        "intent": "refund_request",
        "escalate": True,
        "status": "in_progress",
    },
    "human_handoff": {
        "intent": "human_handoff",
        "escalate": True,
        "status": "in_progress",
    },
}

QUESTIONS_PER_CATEGORY = 30


def generate_golden_set() -> list[dict]:
    """Build the golden set with per-question document labels."""
    known_ids = {document["id"] for document in KNOWLEDGE_BASE}
    golden_set: list[dict] = []
    for category, templates in TEMPLATES.items():
        if len(templates) != QUESTIONS_PER_CATEGORY:
            raise ValueError(
                f"category {category} has {len(templates)} questions, "
                f"expected {QUESTIONS_PER_CATEGORY}"
            )
        for query, relevant_ids in templates:
            unknown = [doc_id for doc_id in relevant_ids if doc_id not in known_ids]
            if unknown:
                raise ValueError(
                    f"query {query!r} references unknown documents: {unknown}"
                )
            golden_set.append(
                {
                    "query": query,
                    "expected_intent": EXPECTED[category]["intent"],
                    "expected_escalate": EXPECTED[category]["escalate"],
                    "expected_status": EXPECTED[category]["status"],
                    "category": category,
                    "relevant_doc_ids": relevant_ids,
                }
            )
    return golden_set


def main() -> None:
    golden_set = generate_golden_set()
    output_path = ROOT / "app/evaluation/golden_set.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(golden_set)} questions.")
    print(f"Saved to {output_path}")

    from collections import Counter

    dist = Counter(item["category"] for item in golden_set)
    print("\nDistribution:")
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")

    labelled = sum(1 for item in golden_set if item["relevant_doc_ids"])
    print(f"\nQuestions with labelled documents: {labelled}/{len(golden_set)}")


if __name__ == "__main__":
    main()
