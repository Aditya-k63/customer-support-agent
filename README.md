# 🤖 AI Customer Support Agent

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0-2e86ab.svg)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![GitHub Actions CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](https://github.com/Aditya-k63/customer-support-agent/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **agentic customer support system** built with **LangGraph**, **RAG**, and **FastAPI**. It classifies user intent, retrieves answers from a company knowledge base, manages ticket lifecycles, and escalates complex issues to human agents — with a deterministic CI gate and a separate real-model evaluation.

---

## 🏗️ Architecture

```text
┌─────────────┐      ┌─────────────────────────────────────────────────────┐
│   USER      │─────▶│              FASTAPI API (/api/v1/chat)             │
└─────────────┘      └─────────────────────────────────────────────────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  LANGGRAPH       │
                                      │  StateGraph      │
                                      │                  │
                                      │  1. Intent       │────▶ Classifies query
                                      │     Classifier   │     (billing/technical/
                                      │                  │     general/refund/handoff)
                                      │  2. RAG          │────▶ Single retrieval pass
                                      │     Retriever    │     with doc IDs + scores
                                      │  3. Response     │────▶ Drafts grounded
                                      │     Generator    │     answer from context
                                      │  4. Escalation   │────▶ Flags human handoff
                                      │     Handler      │     if needed
                                      │  5. Ticket       │────▶ Persists final status
                                      │     Manager      │     + conversation history
                                      └──────────────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  POSTGRESQL      │◀── Tickets, conversations,
                                      │  + pgvector      │     chunked knowledge base
                                      └──────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent LangGraph** | 5-node `StateGraph` with intent classification, retrieval, generation, escalation, and ticket management |
| **RAG Knowledge Base** | Chunked FAQ/policy documents with real sentence-transformer embeddings and pgvector cosine search |
| **Session-Scoped Memory** | Tickets are keyed by `session_id`; user + assistant turns are persisted and reloaded on the next message |
| **Intent Classification** | Routes queries to billing, technical, general, refund_request, or human_handoff |
| **Automated Escalation** | Refund, billing, and explicit human-handoff requests are flagged for agent review |
| **Deterministic CI Gate** | Mocked LLM + deterministic retrieval validate graph routing, ticket status, and metric plumbing without external services |
| **Real-Model Evaluation** | Optional workflow runs the real LLM and pgvector retrieval, measuring retrieval metrics and LLM-judge groundedness |
| **Role-Based API Auth** | Customer and admin API keys, with no environment-based bypass |
| **Production Ops** | Fail-fast migrations, DB readiness probe, generic client errors with internal logging, HNSW index, chunk deduplication |
| **Dockerized** | Multi-stage image with a non-root user; Compose requires explicitly set secrets |

---

## 🛠️ Tech Stack

**Backend:** FastAPI, Python 3.12, Pydantic v2  
**Agent Framework:** LangGraph, LangChain  
**Database:** PostgreSQL, pgvector (HNSW cosine index)  
**LLM Provider:** Groq (`openai/gpt-oss-20b`) with OpenRouter fallback  
**Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2`, 384 dimensions)  
**Deployment:** Docker, Docker Compose  
**CI/CD:** GitHub Actions (lint + unit tests + deterministic regression gate)  
**Testing:** pytest, pytest-asyncio  

---

## 📊 API Endpoints

All `/api/v1/*` routes require an `Authorization: Bearer <API_KEY>` header.

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/health` | none | Liveness alias |
| `GET` | `/health/live` | none | Process liveness |
| `GET` | `/health/ready` | none | Readiness probe with a real `SELECT 1` database check |
| `POST` | `/api/v1/chat` | customer | Send a message; returns response, ticket, intent, status, sources |
| `GET` | `/api/v1/tickets/{ticket_id}` | customer | Ticket details |
| `GET` | `/api/v1/admin/tickets` | admin | Ticket queue (optional `?status=`) |
| `GET` | `/docs` | none | OpenAPI/Swagger UI |

`/api/v1/admin/tickets` requires `ADMIN_API_KEY`; the customer key is rejected with `403`.

---

## 🧪 Evaluation & Regression

The project deliberately separates two evaluation modes so CI never depends on an external model.

### 1. Deterministic gate (runs in CI on every push)

```bash
python scripts/run_deterministic_eval.py
```

- Uses a keyword-based mocked LLM and a deterministic retrieval stub.
- Validates graph routing, escalation decisions, final ticket status, and that document IDs flow through retrieval metrics.
- Writes `regression_report.json` and exits non-zero if a gated metric regresses.
- **This suite does not measure real LLM or semantic retrieval quality.**

Measured deterministic scores on the 150-question golden set:

| Metric | Score | Gate |
|--------|-------|------|
| Intent Accuracy | 0.9467 | ≥ 0.50 |
| Escalation Precision | 0.9783 | ≥ 0.55 |
| Escalation Recall | 1.0000 | ≥ 0.90 |
| Status Accuracy (correctly routed) | 1.0000 | ≥ 0.95 |

### 2. Real-model evaluation (manual or scheduled workflow)

```bash
python scripts/create_db.py
python ingestion/ingest_docs.py
python scripts/run_real_eval.py
```

Requires PostgreSQL + pgvector, `requirements-embeddings.txt`, and an LLM key. It uses real embeddings, real pgvector retrieval, the real LLM, and an LLM judge.

- Intent accuracy, escalation precision/recall, and status accuracy.
- **ID-based retrieval metrics:** Precision@3, Recall@3, MRR (chunks are compared by `source_doc_id`).
- **LLM-judge groundedness:** deterministic three-way classification (`SUPPORTED` / `UNSUPPORTED` / `REFUSAL`). Groundedness is `supported / (supported + unsupported)`; refusals are excluded from that denominator and reported separately as **Answer Rate**.
- Writes `regression_report_real.json`.

`Answer Rate` is gated at a deliberately low baseline so a "refuse everything" agent cannot pass groundedness by never answering. Set `REAL_EVAL_LIMIT` to sample a subset, `REAL_EVAL_JUDGE_SAMPLE` to control judge cost, `REAL_EVAL_DELAY` to stay under free-tier rate limits, and `REAL_EVAL_MAX_FAILURE_RATE` (default `0.05`) to control how many transient provider failures a run tolerates before the gate fails.

Individual query failures are caught, logged, and excluded from the metrics; the report records them under `failed_queries` along with `evaluated_queries`, and the gate fails if the failure rate exceeds the tolerance.

### Known evaluation limitation

The bundled knowledge base is an 8-document demo corpus. Many golden-set questions are therefore outside its coverage, so the real evaluation reports a modest **Answer Rate** and a high refusal rate even though groundedness among answered questions is high. Expanding the corpus is the next data task; the harness itself is complete.

---

## 🚀 Quick Start

### Option 1: Docker Compose

```bash
git clone https://github.com/Aditya-k63/customer-support-agent.git
cd customer-support-agent

# Compose requires DB_PASSWORD and API_KEY to be set explicitly.
# ADMIN_API_KEY is optional but recommended.
cp .env.example .env
# Edit .env with your keys

docker compose up --build
```

Open <http://localhost:8000/docs>.

### Option 2: Local Development

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Core API + tests (no embedding model)
pip install -r requirements.txt

# Real semantic retrieval and the real evaluation
pip install -r requirements-embeddings.txt

cp .env.example .env
# Edit .env with DB credentials, API_KEY, ADMIN_API_KEY and GROQ_API_KEY

python scripts/create_db.py       # create DB, install pgvector, run migrations
python ingestion/ingest_docs.py   # chunk + embed the knowledge base
uvicorn app.main:app --reload
```

Migrations run automatically on startup and **fail fast** if the database is unreachable. Set `RUN_MIGRATIONS_ON_STARTUP=false` only when an external process owns migrations.

### Run Tests

```bash
pytest tests/ -q

# Integration tests need PostgreSQL and the embeddings installed
RUN_INTEGRATION=1 pytest tests/test_integration.py -v
```

---

## 📁 Project Structure

```text
customer-support-agent/
├── .github/workflows/ci.yml        # lint, unit tests, deterministic gate, manual real eval, Docker
├── app/
│   ├── main.py                     # FastAPI entrypoint, lifespan migrations, health probes
│   ├── api/
│   │   ├── deps.py
│   │   └── routes.py               # /chat, /tickets, /admin endpoints
│   ├── core/
│   │   ├── config.py               # Pydantic settings
│   │   └── security.py             # API-key roles, constant-time comparison, rate limiter
│   ├── agents/
│   │   ├── graph.py
│   │   ├── state.py                # retrieval similarity vs answer confidence
│   │   ├── nodes/                  # 5 agent nodes
│   │   └── tools/
│   │       ├── vector_store.py     # pgvector search, chunk IDs, dedupe, re-ingestion
│   │       └── ticket_db.py        # session-scoped tickets + conversation history
│   ├── schemas/
│   ├── services/
│   │   ├── llm.py                  # Groq/OpenRouter with retry + empty-content retry
│   │   └── embeddings.py           # no pseudo-embedding fallback
│   ├── db/
│   │   ├── models.py               # tickets, conversations, document_chunks
│   │   └── migrations.py           # legacy-schema adoption + fail-fast upgrades
│   └── evaluation/
│       ├── metrics.py              # intent, escalation, groundedness, status
│       ├── metrics_v2.py           # Precision@K, Recall@K, MRR
│       ├── llm_judge.py            # three-way deterministic judge
│       ├── mock_llm.py             # deterministic CI mock
│       ├── suite.py                # thresholds + report writing
│       └── golden_set.json         # 150 labelled questions with relevant_doc_ids
├── alembic/versions/
│   ├── 001_initial_schema.py
│   └── 002_hardening.py            # session_id, doc IDs, content hash, embedding model, HNSW
├── ingestion/
│   ├── knowledge_base.py           # stable source document IDs
│   ├── chunker.py                  # overlapping word chunks
│   └── ingest_docs.py              # re-ingestion entrypoint
├── scripts/
│   ├── create_db.py
│   ├── run_deterministic_eval.py   # CI gate
│   ├── run_real_eval.py            # real-model evaluation
│   ├── run_evaluation.py           # compatibility shim → deterministic gate
│   ├── generate_golden_set.py
│   └── test_db.py
├── tests/
│   ├── test_graph.py
│   ├── test_api.py
│   ├── test_evaluation.py
│   ├── test_regression.py
│   └── test_integration.py         # requires RUN_INTEGRATION=1
├── docker/
│   ├── Dockerfile                  # multi-stage, non-root, healthcheck
│   └── entrypoint.sh
├── docker-compose.yml
├── pytest.ini
├── requirements.txt                # core dependencies
├── requirements-embeddings.txt     # sentence-transformers + torch
├── .env.example
└── README.md
```

---

## 🔧 Configuration

Copy `.env.example` to `.env` and update:

```env
# Database
DB_NAME=agent
DB_USER=postgres
DB_PASSWORD=change-me
DB_HOST=localhost
DB_PORT=5432

# LLM
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
OPENROUTER_API_KEY=your_openrouter_api_key

# API authentication (no development bypass)
API_KEY=your_customer_api_key
ADMIN_API_KEY=your_admin_api_key

# Retrieval
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIM=384
RETRIEVAL_TOP_K=5
RETRIEVAL_THRESHOLD=0.0
MAX_CONTEXT_CHUNKS=5
CHUNK_SIZE=200
CHUNK_OVERLAP=30
CONVERSATION_HISTORY_LIMIT=20

# App
ENVIRONMENT=production
RUN_MIGRATIONS_ON_STARTUP=true
```

The in-memory rate limiter is process-local. Multi-replica deployments should replace it with a shared backend such as Redis.

### GitHub Actions secrets for the manual real-eval workflow

| Secret | Required | Purpose |
|--------|----------|---------|
| `GROQ_API_KEY` | yes | Real LLM calls |
| `OPENROUTER_API_KEY` | optional | Rate-limit fallback |
| `CI_DB_PASSWORD` | optional | Ephemeral Postgres password (defaults to a CI-only value) |
| `DOCKER_USERNAME` / `DOCKER_TOKEN` | optional | Enables image push and vulnerability scan |

The Docker build always validates the image without credentials; pushing is conditional on the Hub secrets.

---

## 📈 Roadmap

- [ ] React admin dashboard for ticket management
- [ ] WebSocket support for real-time chat
- [ ] Email/SMS notifications on escalation
- [ ] A/B testing for prompt variants
- [ ] LangSmith integration for production tracing
- [ ] Expand the knowledge base beyond the demo corpus

---

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📬 Contact

**Aditya Kumar**  
B.Tech CS (AI & ML) – LNCT University, Bhopal  
📧 adityakumar14072003@gmail.com  
🔗 [linkedin.com/in/aditya-kumar1407](https://linkedin.com/in/aditya-kumar1407)  
🐙 [github.com/Aditya-k63](https://github.com/Aditya-k63)  
🐳 [hub.docker.com/u/adsharma14](https://hub.docker.com/u/adsharma14)

---

*Built with ❤️ using LangGraph, FastAPI, and PostgreSQL.*
