"""Response generation from context already retrieved by the RAG node."""
import re

from app.agents.state import SupportState
from app.services.llm import LLMService

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "has", "have", "how", "i", "if", "in", "is", "it", "me", "my", "of", "on",
    "or", "our", "please", "the", "this", "to", "was", "we", "what", "when",
    "where", "which", "who", "why", "will", "with", "you", "your",
}


def _content_terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def estimate_answer_confidence(answer: str, context: list[str]) -> float:
    """Heuristic lexical support score for the generated answer.

    This is an estimate of how much of the answer is grounded in retrieved
    context. It is deliberately kept separate from retrieval similarity and is
    not presented as a calibrated probability of correctness.
    """
    answer_terms = _content_terms(answer)
    if not answer_terms:
        return 0.0
    context_terms = _content_terms(" ".join(context))
    supported = len(answer_terms & context_terms)
    return round(supported / len(answer_terms), 4)


def _format_history(messages: list[dict], limit: int = 6) -> str:
    recent = messages[-limit:]
    if not recent:
        return "(no previous messages)"
    return "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in recent
    )


class ResponseGeneratorNode:
    def __init__(self) -> None:
        self.llm = LLMService()

    async def run(self, state: SupportState) -> SupportState:
        # Context was retrieved exactly once by RAGRetrieverNode. Do not
        # perform a second vector search here.
        context_str = (
            "\n".join(state.context) if state.context else "No relevant documents found."
        )
        prompt = (
            "You are a customer support agent. Answer using ONLY the provided "
            "context. If the answer is not in the context, say you don't know "
            "and suggest escalation.\n\n"
            f"Conversation history:\n{_format_history(state.messages)}\n\n"
            f"Context:\n{context_str}\n\n"
            f"User Question: {state.user_query}\n\n"
            "Answer:"
        )
        response = await self.llm.generate(prompt, temperature=0.3, max_tokens=500)
        state.response = response
        state.answer_confidence = estimate_answer_confidence(
            response, state.context
        )
        return state
