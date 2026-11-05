"""Response generator node with retrieval-based confidence scoring."""
from app.agents.state import SupportState
from app.services.llm import LLMService
from app.agents.tools.vector_store import VectorStore


class ResponseGeneratorNode:
    def __init__(self) -> None:
        self.llm = LLMService()
        self.vector_store = VectorStore()

    async def run(self, state: SupportState) -> SupportState:
        # Retrieve with similarity scores
        docs_with_scores = await self.vector_store.search_with_scores(
            state.user_query, top_k=5, threshold=0.6
        )
        state.context = [content for content, _ in docs_with_scores]

        # Confidence derived from retrieval quality
        if docs_with_scores:
            min_distance = min(score for _, score in docs_with_scores)
            # confidence = 1 - cosine_distance (higher = more confident)
            state.confidence_score = max(0.0, min(1.0, 1.0 - min_distance))
        else:
            state.confidence_score = 0.0

        # Generate grounded response
        context_str = "\n".join(state.context) if state.context else "No relevant documents found."
        prompt = (
            "You are a customer support agent. Answer using ONLY the provided context. "
            "If the answer is not in the context, say you don't know and suggest escalation.\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Question: {state.user_query}\n\n"
            "Answer:"
        )
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=500)
        state.response = response
        return state