"""One-time script to load FAQ/policy docs into the vector store."""
from app.agents.tools.vector_store import VectorStore


def main() -> None:
    vs = VectorStore()
    print(f"Indexed {len(vs.docs)} documents into vector store.")


if __name__ == "__main__":
    main()