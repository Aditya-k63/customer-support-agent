"""LLM-judge for response groundedness evaluation."""
from typing import List, Tuple
from app.services.llm import LLMService


class LLMJudge:
    def __init__(self) -> None:
        self.llm = LLMService()

    async def judge_groundedness(self, answer: str, context: str) -> bool:
        """Use LLM to judge if answer is fully supported by context."""
        prompt = (
            "Is the following answer fully supported by the provided context? "
            "Answer ONLY 'YES' or 'NO'.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}"
        )
        result = await self.llm.generate(prompt, temperature=0.1, max_tokens=10)
        return result.strip().upper() == "YES"

    async def judge_groundedness_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[bool]:
        """Judge multiple (answer, context) pairs."""
        results = []
        for answer, context in pairs:
            is_grounded = await self.judge_groundedness(answer, context)
            results.append(is_grounded)
        return results