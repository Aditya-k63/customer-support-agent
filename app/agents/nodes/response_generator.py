from app.agents.state import SupportState
from app.services.llm import LLMService


class ResponseGeneratorNode:
    def __init__(self) -> None:
        self.llm = LLMService()

    async def run(self, state: SupportState) -> SupportState:
        context_str = "\n".join(state.context) if state.context else "No relevant documents found."
        prompt = (
            "You are a customer support agent. Answer the user's question using ONLY the provided context. "
            "If the answer is not in the context, say you don't know and suggest escalation. "
            "Be polite and concise.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Question: {state.user_query}\n\n"
            "Answer:"
        )
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=500)
        state.response = response
        state.confidence_score = 0.85  # Placeholder; integrate real scoring later
        return state