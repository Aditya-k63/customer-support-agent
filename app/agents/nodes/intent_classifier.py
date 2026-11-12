import re

from app.agents.state import SupportState
from app.services.llm import LLMService

VALID_INTENTS = ("billing", "technical", "general", "refund_request", "human_handoff")
INTENT_ALIASES = {
    "refund request": "refund_request",
    "human handoff": "human_handoff",
    "human_hand-off": "human_handoff",
    "refund": "refund_request",
}


def parse_intent(raw: str) -> str:
    """Extract a valid intent from a free-form model response.

    Reasoning-capable models may return a sentence or a short explanation even
    when asked for a single label, so match labels anywhere in the output.
    """
    text = raw.strip().lower()
    for alias, intent in INTENT_ALIASES.items():
        if alias in text:
            return intent
    for intent in VALID_INTENTS:
        if re.search(rf"\b{re.escape(intent)}\b", text):
            return intent
    return "general"


class IntentClassifierNode:
    def __init__(self) -> None:
        self.llm = LLMService()

    async def run(self, state: SupportState) -> SupportState:
        prompt = (
            "Classify the user query into exactly one of: billing, technical, "
            "general, refund_request, human_handoff. Respond with ONLY the "
            "category name, no explanation.\n\n"
            f"Query: {state.user_query}"
        )
        # Reasoning models can consume the token budget before emitting the
        # label, so leave enough headroom for the final answer.
        result = await self.llm.generate(prompt, temperature=0.1, max_tokens=400)
        state.intent = parse_intent(result)
        return state
