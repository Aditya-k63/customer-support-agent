"""Real-model evaluation: actual LLM, real embeddings, real pgvector.

This is intentionally separate from the deterministic CI gate. It requires:
  * PostgreSQL with pgvector and the schema migrated
  * ingested knowledge base (ingestion/ingest_docs.py)
  * sentence-transformers installed
  * GROQ_API_KEY (or OPENROUTER_API_KEY)

Run it manually or on a schedule:

    python scripts/create_db.py
    python ingestion/ingest_docs.py
    python scripts/run_real_eval.py

Exits non-zero when a gated metric falls below its threshold.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.graph import build_graph  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.evaluation.llm_judge import (  # noqa: E402
    REFUSAL,
    SUPPORTED,
    UNSUPPORTED,
    LLMJudge,
)
from app.evaluation.metrics import (  # noqa: E402
    EvalResult,
    check_thresholds,
    intent_accuracy,
    status_accuracy,
)
from app.evaluation.metrics_v2 import (  # noqa: E402
    escalation_precision_recall,
    mrr,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.suite import load_golden_set  # noqa: E402
from app.services.llm import (  # noqa: E402
    QUOTA,
    LLMProviderUnavailableError,
    LLMQuotaExhaustedError,
)

JUDGE_SAMPLE_SIZE = int(os.getenv("REAL_EVAL_JUDGE_SAMPLE", "20"))
# Pause between queries to stay under free-tier rate limits.
QUERY_DELAY_SECONDS = float(os.getenv("REAL_EVAL_DELAY", "1.0"))
# A single transient provider error should not discard a full run, but a
# sustained outage must fail the gate instead of silently shrinking the suite.
MAX_FAILURE_RATE = float(os.getenv("REAL_EVAL_MAX_FAILURE_RATE", "0.05"))
# Provider requests kept in reserve so the judge phase can still run.
REQUEST_RESERVE = int(os.getenv("REAL_EVAL_REQUEST_RESERVE", "30"))
# Provider tokens allowed per minute. The harness paces itself against this
# budget instead of guessing a fixed delay, because reasoning models spend a
# variable number of tokens per query. Providers without a per-minute token
# limit (OpenRouter) do not need token pacing, so the default follows the
# selected provider.
TOKEN_BUDGET_PER_MINUTE = int(
    os.getenv(
        "REAL_EVAL_TOKEN_BUDGET",
        "5000" if getattr(settings, "LLM_PROVIDER", "auto") != "openrouter" else "0",
    )
)
# Free-tier providers often run out of quota mid-suite. Each finished query is
# appended here so a later run resumes instead of starting over.
CHECKPOINT_PATH = ROOT / os.getenv(
    "REAL_EVAL_CHECKPOINT", "real_eval_checkpoint.jsonl"
)
RESET_CHECKPOINTS = os.getenv("REAL_EVAL_RESET", "0") == "1"

REAL_THRESHOLDS = {
    "Intent Accuracy": float(os.getenv("REAL_EVAL_INTENT", "0.40")),
    "Precision@3": float(os.getenv("REAL_EVAL_PRECISION_3", "0.20")),
    "Recall@3": float(os.getenv("REAL_EVAL_RECALL_3", "0.30")),
    "MRR": float(os.getenv("REAL_EVAL_MRR", "0.30")),
    "Escalation Precision": float(os.getenv("REAL_EVAL_ESC_PRECISION", "0.50")),
    "Escalation Recall": float(os.getenv("REAL_EVAL_ESC_RECALL", "0.80")),
    "Status Accuracy": float(os.getenv("REAL_EVAL_STATUS", "0.70")),
    "LLM-Judge Groundedness": float(os.getenv("REAL_EVAL_JUDGE", "0.80")),
    "Answer Rate": float(os.getenv("REAL_EVAL_ANSWER_RATE", "0.15")),
}


def source_id(doc_id: str) -> str:
    return doc_id.split("#", 1)[0]


def make_state(item: dict, index: int) -> dict:
    return {
        "messages": [],
        "user_query": item["query"],
        "intent": None,
        "context": [],
        "retrieved_doc_ids": [],
        "retrieval_scores": [],
        "retrieval_similarity": 0.0,
        "answer_confidence": 0.0,
        "ticket_id": None,
        "status": "open",
        "escalation_flag": False,
        "session_id": f"real-eval-{index}",
        "response": "",
    }


async def evaluate_query(graph, item: dict, index: int) -> dict:
    """Run one golden query and return a JSON-serialisable result record."""
    started = time.perf_counter()
    result = await graph.ainvoke(make_state(item, index))
    return {
        "index": index,
        "query": item["query"],
        "latency_ms": (time.perf_counter() - started) * 1000,
        "intent": result.get("intent", "general"),
        "expected_intent": item["expected_intent"],
        "escalation": bool(result.get("escalation_flag", False)),
        "expected_escalate": bool(item["expected_escalate"]),
        "status": result.get("status", "open"),
        "expected_status": item["expected_status"],
        "retrieved_ids": [
            source_id(doc_id) for doc_id in result.get("retrieved_doc_ids", [])
        ],
        "relevant_ids": item.get("relevant_doc_ids", []),
        "response": result.get("response", ""),
        "context": result.get("context", []),
    }


def load_checkpoints(path: Path) -> dict[int, dict]:
    """Read previously completed queries, ignoring unreadable tail lines."""
    if not path.exists():
        return {}
    completed: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        completed[int(record["index"])] = record
    return completed


def append_checkpoint(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")


async def main() -> int:
    if not settings.GROQ_API_KEY and not settings.OPENROUTER_API_KEY:
        print("ERROR: set GROQ_API_KEY or OPENROUTER_API_KEY for the real evaluation.")
        return 2

    golden = load_golden_set()
    if not golden:
        print("ERROR: Golden set is empty.")
        return 1

    limit = int(os.getenv("REAL_EVAL_LIMIT", "0"))
    if limit > 0 and limit < len(golden):
        step = max(1, len(golden) // limit)
        golden = golden[::step][:limit]
        print(f"[INFO] sampled {len(golden)} queries for a limited run")

    # A slice lets a run be split around provider quota windows. Indices stay
    # global so checkpoints written by different slices never collide.
    start = max(0, int(os.getenv("REAL_EVAL_START", "0")))
    end = int(os.getenv("REAL_EVAL_END", "0")) or len(golden)
    end = min(end, len(golden))
    selected = [(index, item) for index, item in enumerate(golden) if start <= index < end]
    if (start, end) != (0, len(golden)):
        print(f"[INFO] evaluating slice [{start}, {end}) of {len(golden)} queries")

    provider = getattr(settings, "LLM_PROVIDER", "auto").strip().lower()
    if provider == "openrouter" or (provider == "auto" and not settings.GROQ_API_KEY):
        llm_provider, llm_model = "openrouter", settings.OPENROUTER_MODEL
    else:
        llm_provider, llm_model = "groq", settings.GROQ_MODEL

    graph = build_graph()
    judge = LLMJudge()

    if RESET_CHECKPOINTS and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print(f"[INFO] cleared checkpoint file {CHECKPOINT_PATH.name}")

    checkpoints = load_checkpoints(CHECKPOINT_PATH)
    failures: list[dict] = []
    reused = 0
    stopped_reason: str | None = None

    for index, item in selected:
        remaining = QUOTA.remaining_requests(llm_provider)
        if remaining is not None and remaining <= REQUEST_RESERVE:
            stopped_reason = "provider_quota_reserve"
            print(
                f"[WARN] stopping early: only {remaining} provider requests "
                f"left (reserve {REQUEST_RESERVE})"
            )
            break
        cached = checkpoints.get(index)
        if cached and cached.get("query") == item["query"]:
            reused += 1
            continue
        try:
            tokens_before = QUOTA.total_tokens(llm_provider)
            record = await evaluate_query(graph, item, index)
        except LLMQuotaExhaustedError as exc:
            stopped_reason = "provider_quota_exhausted"
            print(f"[WARN] provider quota exhausted at query {index + 1}: {exc}")
            break
        except LLMProviderUnavailableError as exc:
            # No credit, invalid key or no permission: every further request
            # would fail the same way, so stop and let the run resume later.
            stopped_reason = "provider_unavailable"
            print(f"[WARN] provider unavailable at query {index + 1}: {exc}")
            break
        except Exception as exc:  # noqa: BLE001 - keep the suite running
            failures.append(
                {"query": item["query"], "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"[WARN] query {index + 1} failed: {type(exc).__name__}: {exc}")
            if QUERY_DELAY_SECONDS > 0:
                await asyncio.sleep(QUERY_DELAY_SECONDS)
            continue
        append_checkpoint(CHECKPOINT_PATH, record)
        checkpoints[index] = record
        if (index + 1) % 10 == 0:
            print(f"[INFO] evaluated {index + 1}/{end} queries")
        # Pace against the provider's per-minute token budget.
        query_tokens = QUOTA.total_tokens(llm_provider) - tokens_before
        paced_delay = max(
            QUERY_DELAY_SECONDS,
            query_tokens * 60 / TOKEN_BUDGET_PER_MINUTE
            if TOKEN_BUDGET_PER_MINUTE
            else 0,
        )
        if paced_delay > 0:
            await asyncio.sleep(paced_delay)

    selected_indices = [index for index, _ in selected]
    completed = [
        checkpoints[index] for index in selected_indices if index in checkpoints
    ]
    if reused:
        print(f"[INFO] reused {reused} checkpointed queries")
    if not completed:
        print("ERROR: no queries completed; nothing to report.")
        return 1

    predictions = [record["intent"] for record in completed]
    expected_intents = [record["expected_intent"] for record in completed]
    escalations = [record["escalation"] for record in completed]
    expected_escalations = [record["expected_escalate"] for record in completed]
    statuses = [record["status"] for record in completed]
    expected_statuses = [record["expected_status"] for record in completed]
    retrieved_ids = [record["retrieved_ids"] for record in completed]
    relevant_ids = [record["relevant_ids"] for record in completed]
    latencies_ms = [record["latency_ms"] for record in completed]
    judge_candidates = [
        (record["response"], "\n".join(record["context"]))
        for record in completed
        if record["relevant_ids"]
    ]

    results: list = [
        intent_accuracy(predictions, expected_intents),
        status_accuracy(statuses, expected_statuses),
        precision_at_k(retrieved_ids, relevant_ids, k=3),
        recall_at_k(retrieved_ids, relevant_ids, k=3),
        mrr(retrieved_ids, relevant_ids),
    ]
    results.extend(
        escalation_precision_recall(escalations, expected_escalations).values()
    )

    # Sample the judge set evenly across the whole golden set so one category
    # cannot dominate the groundedness estimate.
    if JUDGE_SAMPLE_SIZE > 0 and len(judge_candidates) > JUDGE_SAMPLE_SIZE:
        step = len(judge_candidates) / JUDGE_SAMPLE_SIZE
        judge_pairs = [
            judge_candidates[int(index * step)]
            for index in range(JUDGE_SAMPLE_SIZE)
        ]
    else:
        judge_pairs = judge_candidates

    try:
        verdicts = await judge.classify_batch(judge_pairs)
    except LLMQuotaExhaustedError:
        print("[WARN] provider quota exhausted during judging; skipping judge metrics")
        stopped_reason = stopped_reason or "provider_quota_exhausted"
        verdicts = []
    except LLMProviderUnavailableError as exc:
        print(f"[WARN] provider unavailable during judging: {exc}")
        stopped_reason = stopped_reason or "provider_unavailable"
        verdicts = []
    supported = sum(1 for verdict in verdicts if verdict == SUPPORTED)
    unsupported = sum(1 for verdict in verdicts if verdict == UNSUPPORTED)
    refusals = sum(1 for verdict in verdicts if verdict == REFUSAL)
    answered = supported + unsupported
    groundedness = supported / answered if answered else 0.0
    answer_rate = answered / len(verdicts) if verdicts else 0.0
    refusal_rate = refusals / len(verdicts) if verdicts else 0.0
    unsupported_rate = unsupported / answered if answered else 0.0

    results.append(
        EvalResult(
            name="LLM-Judge Groundedness",
            score=round(groundedness, 4),
            total=answered,
            passed=supported,
            details=[],
        )
    )
    # Guard against a degenerate "refuse everything" agent passing the
    # groundedness gate by never answering.
    results.append(
        EvalResult(
            name="Answer Rate",
            score=round(answer_rate, 4),
            total=len(verdicts),
            passed=answered,
            details=[],
        )
    )

    average_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p95_latency = (
        sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))] if latencies_ms else 0.0
    )

    passed = check_thresholds(results, REAL_THRESHOLDS)

    failure_rate = (
        (len(selected) - len(completed)) / len(selected) if selected else 0.0
    )
    too_many_failures = failure_rate > MAX_FAILURE_RATE
    if too_many_failures:
        print(
            f"[FAIL] {len(selected) - len(completed)}/{len(selected)} queries have "
            f"no result ({failure_rate:.1%} > {MAX_FAILURE_RATE:.1%} allowed)"
        )
    elif failures:
        print(
            f"[WARN] {len(failures)} queries failed this run but the suite is "
            f"within the {MAX_FAILURE_RATE:.1%} tolerance; metrics exclude them"
        )

    report = {
        "suite": "real",
        "retrieval_backend": "pgvector",
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "embedding_model": settings.EMBEDDING_MODEL,
        "queries": len(selected),
        "suite_size": len(golden),
        "evaluated_range": [start, end],
        "evaluated_queries": len(completed),
        "resumed_queries": reused,
        "failed_queries": failures,
        "stopped_reason": stopped_reason,
        "provider_quota": QUOTA.snapshot(llm_provider),
        # "results" keeps this artifact compatible with baseline comparison.
        "results": [
            {
                "metric": result.name,
                "score": round(result.score, 4),
                "total": result.total,
                "passed": result.passed,
                "threshold": REAL_THRESHOLDS.get(result.name, 0.0),
                "pass_threshold": result.score
                >= REAL_THRESHOLDS.get(result.name, 0.0),
            }
            for result in results
        ],
        "judge": {
            "sample_size": len(verdicts),
            "supported": supported,
            "unsupported": unsupported,
            "refusals": refusals,
            "answer_rate": round(answer_rate, 4),
            "refusal_rate": round(refusal_rate, 4),
            "unsupported_rate": round(unsupported_rate, 4),
            "note": (
                "Groundedness = supported / (supported + unsupported). "
                "Refusals are excluded from the accuracy denominator and "
                "reported separately as coverage."
            ),
        },
        "latency_ms": {
            "average": round(average_latency, 2),
            "p95": round(p95_latency, 2),
        },
        "overall_pass": passed and not too_many_failures and not stopped_reason,
        "regressions": [],
    }
    (ROOT / "regression_report_real.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if stopped_reason:
        print(
            f"[INCOMPLETE] run stopped early ({stopped_reason}). "
            "Re-run the same command to resume from the checkpoint file."
        )
        return 3
    return 0 if passed and not too_many_failures else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
