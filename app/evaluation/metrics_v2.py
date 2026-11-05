"""Production evaluation metrics: Precision@K, Recall@K, MRR, LLM-judge groundedness, escalation precision/recall."""
from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class EvalResultV2:
    name: str
    score: float
    total: int
    passed: int
    details: List[Dict[str, Any]]


def precision_at_k(
    retrieved: List[List[str]], relevant: List[List[str]], k: int = 3
) -> EvalResultV2:
    """Precision@K: % of top-K retrieved docs that are relevant."""
    scores = []
    for ret, rel in zip(retrieved, relevant):
        top_k = ret[:k]
        relevant_set = set(rel)
        hits = sum(1 for doc in top_k if doc in relevant_set)
        scores.append(hits / len(top_k) if top_k else 0.0)
    avg = sum(scores) / len(scores) if scores else 0.0
    return EvalResultV2(
        name=f"Precision@{k}",
        score=round(avg, 4),
        total=len(scores),
        passed=sum(1 for s in scores if s >= 0.5),
        details=[],
    )


def recall_at_k(
    retrieved: List[List[str]], relevant: List[List[str]], k: int = 3
) -> EvalResultV2:
    """Recall@K: % of relevant docs found in top-K."""
    scores = []
    for ret, rel in zip(retrieved, relevant):
        if not rel:
            continue
        top_k = set(ret[:k])
        hits = sum(1 for doc in rel if doc in top_k)
        scores.append(hits / len(rel))
    avg = sum(scores) / len(scores) if scores else 0.0
    return EvalResultV2(
        name=f"Recall@{k}",
        score=round(avg, 4),
        total=len(scores),
        passed=sum(1 for s in scores if s >= 0.5),
        details=[],
    )


def mrr(retrieved: List[List[str]], relevant: List[List[str]]) -> EvalResultV2:
    """Mean Reciprocal Rank: average of 1/rank of first relevant doc."""
    ranks = []
    for ret, rel in zip(retrieved, relevant):
        relevant_set = set(rel)
        rank = 0
        for i, doc in enumerate(ret, 1):
            if doc in relevant_set:
                rank = i
                break
        ranks.append(1.0 / rank if rank else 0.0)
    avg = sum(ranks) / len(ranks) if ranks else 0.0
    return EvalResultV2(
        name="MRR",
        score=round(avg, 4),
        total=len(ranks),
        passed=sum(1 for r in ranks if r > 0),
        details=[],
    )


def escalation_precision_recall(
    predicted: List[bool], ground_truth: List[bool]
) -> Dict[str, EvalResultV2]:
    """Escalation precision and recall."""
    tp = sum(1 for p, g in zip(predicted, ground_truth) if p and g)
    fp = sum(1 for p, g in zip(predicted, ground_truth) if p and not g)
    fn = sum(1 for p, g in zip(predicted, ground_truth) if not p and g)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return {
        "Escalation Precision": EvalResultV2(
            "Escalation Precision", round(precision, 4), len(predicted),
            int(precision * len(predicted)), []
        ),
        "Escalation Recall": EvalResultV2(
            "Escalation Recall", round(recall, 4), len(predicted),
            int(recall * len(predicted)), []
        ),
    }