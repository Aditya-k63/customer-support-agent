from app.agents.state import SupportState
from app.agents.tools.vector_store import VectorStore
from app.core.config import settings


class RAGRetrieverNode:
    """Single retrieval step. Results are carried through graph state."""

    def __init__(self) -> None:
        self.vector_store = VectorStore()

    async def run(self, state: SupportState) -> SupportState:
        chunks = await self.vector_store.search_documents(
            state.user_query,
            top_k=settings.MAX_CONTEXT_CHUNKS,
            threshold=settings.RETRIEVAL_THRESHOLD,
        )
        state.context = [chunk.content for chunk in chunks]
        state.retrieved_doc_ids = [chunk.doc_id for chunk in chunks]
        state.retrieval_scores = [chunk.similarity for chunk in chunks]
        state.retrieval_similarity = (
            max(state.retrieval_scores) if state.retrieval_scores else 0.0
        )
        return state
