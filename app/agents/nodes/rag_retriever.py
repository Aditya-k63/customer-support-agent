from app.agents.state import SupportState
from app.agents.tools.vector_store import VectorStore


class RAGRetrieverNode:
    def __init__(self) -> None:
        self.vector_store = VectorStore()

    async def run(self, state: SupportState) -> SupportState:
        docs = await self.vector_store.search(state.user_query, top_k=3)
        state.context = docs
        return state