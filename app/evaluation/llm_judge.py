"""LLM-judge for response groundedness evaluation.

The judge uses a deterministic three-way classification instead of a YES/NO
answer. Separating refusals from unsupported claims matters: an honest "I
don't have that information" is not a hallucination, but it is also not a
useful answer. The evaluation reports both groundedness (accuracy among
answered questions) and answer rate (coverage).
"""
import re
from typing import List, Tuple

from app.services.llm import LLMService

SUPPORTED = "SUPPORTED"
UNSUPPORTED = "UNSUPPORTED"
REFUSAL = "REFUSAL"

VERDICTS = (SUPPORTED, UNSUPPORTED, REFUSAL)

# UNSUPPORTED must be matched before SUPPORTED because it contains it.
_VERDICT_PATTERN = re.compile(r"\b(UNSUPPORTED|SUPPORTED|REFUSAL)\b")


class LLMJudge:
    def __init__(self) -> None:
        self.llm = LLMService()

    async def classify(self, answer: str, context: str) -> str:
        """Classify an answer as SUPPORTED, UNSUPPORTED or REFUSAL."""
        prompt = (
            "You are grading a customer-support answer against retrieved context.\n"
            "Classify the answer as exactly one of these labels:\n"
            "SUPPORTED - every factual claim is supported by the context.\n"
            "UNSUPPORTED - it asserts at least one fact not supported by the context.\n"
            "REFUSAL - it declines to answer or says it lacks the information.\n"
            "Respond with exactly one word.\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}"
        )
        # temperature=0 keeps this deterministic; reasoning models need headroom
        # before emitting the final label.
        result = await self.llm.generate(prompt, temperature=0.0, max_tokens=400)
        matches = _VERDICT_PATTERN.findall(result.upper())
        if matches:
            return matches[-1]
        # Unknown output is treated as ungrounded rather than silently passing.
        return UNSUPPORTED

    async def judge_groundedness(self, answer: str, context: str) -> bool:
        """Backward-compatible boolean judgement."""
        return await self.classify(answer, context) == SUPPORTED

    async def classify_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[str]:
        """Classify multiple ``(answer, context)`` pairs."""
        return [await self.classify(answer, context) for answer, context in pairs]

    async def judge_groundedness_batch(
        self, pairs: List[Tuple[str, str]]
    ) -> List[bool]:
        """Backward-compatible boolean judgements for multiple pairs."""
        return [
            verdict == SUPPORTED
            for verdict in await self.classify_batch(pairs)
        ]
