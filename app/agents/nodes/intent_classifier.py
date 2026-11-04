from app.agents.state import SupportState
from app.services.llm import LLMService


class IntentClassifierNode:
    def __init__(self) -> None:
        self.llm = LLMService()

    async def run(self, state: SupportState) -> SupportState:
        prompt = (
            "Classify the user query into one of: billing, technical, general, refund_request, human_handoff. "
            "Respond with ONLY the category name.\n\n"
            f"Query: {state.user_query}"
        )
        result = await self.llm.generate(prompt, temperature=0.1, max_tokens=20)
        intent = result.strip().lower()
        valid = {"billing", "technical", "general", "refund_request", "human_handoff"}
        state.intent = intent if intent in valid else "general"
        return state