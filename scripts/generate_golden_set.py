"""Generate v1 baseline golden set: 150 customer support questions."""
import json
import random
from pathlib import Path

random.seed(42)

# Question templates per intent category
TEMPLATES = {
    "billing": [
        "I was charged twice for the same item.",
        "When will my invoice be generated?",
        "I need to update my billing address.",
        "How do I update my payment method?",
        "Why is my bill higher this month?",
        "Can I get a receipt for my last payment?",
        "I don't recognize this charge on my card.",
        "How do I cancel my subscription?",
        "What payment methods do you accept?",
        "When is my next payment due?",
        "I want to change my billing plan.",
        "Can I get a refund for an accidental charge?",
        "How do I view my billing history?",
        "Why was my payment declined?",
        "I need to update my credit card details.",
        "Can I get an itemized invoice?",
        "How do I apply a promo code?",
        "What happens if I miss a payment?",
        "Can I pause my subscription?",
        "How do I request a billing adjustment?",
        "I was charged after I cancelled.",
        "Can I get a tax receipt?",
        "How do I change my currency?",
        "Why is there a pending charge?",
        "Can I split my payment?",
        "How do I update my tax information?",
        "Can I get a refund for a failed service?",
        "I was overcharged.",
        "How do I contact billing support?",
        "Can I get a discount for loyalty?",
    ],
    "technical": [
        "How do I reset my password?",
        "The website is not loading.",
        "How do I reset my password?",
        "I can't log into my account.",
        "The app keeps crashing on my phone.",
        "How do I enable two-factor authentication?",
        "I'm getting a 404 error on your site.",
        "How do I clear my browser cache?",
        "The video won't play.",
        "How do I update the app?",
        "I'm having trouble uploading files.",
        "The search function isn't working.",
        "How do I change my notification settings?",
        "I can't access my dashboard.",
        "The page keeps freezing.",
        "How do I export my data?",
        "I'm getting an SSL error.",
        "How do I connect my domain?",
        "The checkout button isn't working.",
        "How do I enable dark mode?",
        "I can't verify my email address.",
        "The API is returning errors.",
        "How do I set up integrations?",
        "I'm getting a timeout error.",
        "How do I clear my cookies?",
        "The mobile app won't open.",
        "How do I reset my MFA?",
        "I can't download my files.",
        "The form won't submit.",
    ],
    "general": [
        "What are your business hours?",
        "How do I track my shipment?",
        "Where is your office located?",
        "How do I contact customer support?",
        "What is your return policy?",
        "Do you ship internationally?",
        "How long does shipping take?",
        "Can I change my shipping address?",
        "What is your privacy policy?",
        "How do I create an account?",
        "Do you have a mobile app?",
        "How do I delete my account?",
        "What countries do you serve?",
        "How do I verify my identity?",
        "Do you offer student discounts?",
        "How do I refer a friend?",
        "What is your terms of service?",
        "How do I unsubscribe from emails?",
        "Do you have a live chat?",
        "How do I update my profile?",
        "What is your refund window?",
        "Do you offer enterprise plans?",
        "How do I get started?",
        "What features are included in free tier?",
        "How do I upgrade my account?",
        "Do you offer training?",
        "How do I provide feedback?",
        "What is your company mission?",
        "How do I find your help center?",
        "Do you have a status page?",
    ],
    "refund_request": [
        "I want to refund my order.",
        "I want to cancel my subscription.",
        "Can I get a refund for a defective product?",
        "I want my money back.",
        "How do I request a refund?",
        "I was charged for a service I didn't use.",
        "Can I reverse this transaction?",
        "I want to return my purchase.",
        "How long does a refund take?",
        "I want to dispute this charge.",
        "Can I get a partial refund?",
        "I'm not satisfied with the service.",
        "Can I reverse a completed payment?",
        "I want to opt out and get my money back.",
        "How do I initiate a chargeback?",
        "I was double billed.",
        "Can I get store credit instead of cash?",
        "I want to undo my purchase.",
        "The product never arrived, I want a refund.",
        "Can I escalate this to a refund specialist?",
        "I want to revoke my authorization.",
        "How do I file a refund dispute?",
        "I was charged without consent.",
        "Can I get my deposit back?",
        "I want to cancel and refund immediately.",
        "The service was not as described.",
        "Can I speak to someone about my refund?",
        "I need an emergency refund.",
        "How do I check my refund status?",
        "I want to reverse a recurring charge.",
    ],
    "human_handoff": [
        "Can I speak to a human agent?",
        "I need to talk to a real person.",
        "Can you transfer me to support?",
        "I want to speak with a supervisor.",
        "This is urgent, I need human help.",
        "Can I escalate to a manager?",
        "I need a human to review my case.",
        "Please connect me to an agent.",
        "I have a complex issue needing human review.",
        "Can I get a live agent?",
        "I don't want to talk to a bot anymore.",
        "Can a human call me back?",
        "I need to speak to customer service.",
        "Can you transfer this to a person?",
        "I need help from a real human.",
        "Can I talk to someone now?",
        "I want to speak to support staff.",
        "Can you escalate to a human?",
        "I need a person, not automation.",
        "Can I get a phone number for support?",
        "I want to chat with a human.",
        "Can someone contact me directly?",
        "I need human intervention.",
        "Can you route me to an agent?",
        "I want to speak to someone in charge.",
        "Can I get live help?",
        "I need a human to fix this.",
        "Can I be transferred now?",
        "I want to talk to customer care.",
    ],
}

# Relevant documents per category (matching ingest_docs.py content)
RELEVANT_DOCS = {
    "billing": [
        "Refund policy: Items can be refunded within 30 days of purchase with original receipt.",
        "Billing: Invoices are generated on the 1st of each month. Payment methods include credit card and PayPal.",
        "Subscription cancellation: You can cancel anytime from your account dashboard. Access continues until the end of your billing period.",
    ],
    "technical": [
        "Technical support: Reset your password at account settings or contact support@company.com.",
        "Password reset: Click 'Forgot Password' on the login page or visit account settings to reset your password.",
    ],
    "general": [
        "General: Our business hours are Mon-Fri, 9 AM to 6 PM EST.",
        "Shipping: Orders are processed within 1-2 business days and shipped via standard ground delivery.",
        "Order tracking: Use the tracking number in your shipping confirmation email or check your order history page.",
    ],
    "refund_request": [
        "Refund policy: Items can be refunded within 30 days of purchase with original receipt.",
        "Subscription cancellation: You can cancel anytime from your account dashboard. Access continues until the end of your billing period.",
    ],
    "human_handoff": [],
}

EXPECTED = {
    "billing": {"intent": "billing", "escalate": True, "status": "in_progress"},
    "technical": {"intent": "technical", "escalate": False, "status": "resolved"},
    "general": {"intent": "general", "escalate": False, "status": "resolved"},
    "refund_request": {"intent": "refund_request", "escalate": True, "status": "in_progress"},
    "human_handoff": {"intent": "human_handoff", "escalate": True, "status": "in_progress"},
}


def generate_golden_set() -> list[dict]:
    """Generate 150 questions: 30 per category."""
    golden_set = []
    for category, templates in TEMPLATES.items():
        for i in range(30):
            # Use template directly (cycle if more needed)
            template = templates[i % len(templates)]
            # Add slight variation for uniqueness
            if i >= len(templates):
                query = f"{template} (reference #{i})"
            else:
                query = template
            golden_set.append({
                "query": query,
                "expected_intent": EXPECTED[category]["intent"],
                "expected_escalate": EXPECTED[category]["escalate"],
                "expected_status": EXPECTED[category]["status"],
                "category": category,
                "relevant_docs": RELEVANT_DOCS[category],
            })
    return golden_set


def main() -> None:
    golden_set = generate_golden_set()
    output_path = Path("app/evaluation/golden_set.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(golden_set, f, indent=2, ensure_ascii=False)
    
    print(f"Generated {len(golden_set)} questions.")
    print(f"Saved to {output_path}")
    
    # Print distribution
    from collections import Counter
    dist = Counter(item["category"] for item in golden_set)
    print("\nDistribution:")
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()