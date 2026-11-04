# 🤖 AI Customer Support Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.0-2e86ab.svg)](https://langchain-ai.github.io/langgraph/)
[![PostgreSQL + pgvector](https://img.shields.io/badge/PostgreSQL-pgvector-336791.svg)](https://github.com/pgvector/pgvector)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![GitHub Actions CI](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF.svg)](https://github.com/Aditya-k63/customer-support-agent/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An **agentic customer support system** built with **LangGraph**, **RAG**, and **FastAPI**. It classifies user intent, retrieves answers from a company knowledge base, manages ticket lifecycles, and escalates complex issues to human agents — all with production-grade evaluation and regression testing.

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
                                      │                  │     general/refund)
                                      │  2. RAG          │────▶ Retrieves context
                                      │     Retriever    │     from pgvector
                                      │  3. Response     │────▶ Drafts grounded
                                      │     Generator    │     answer
                                      │  4. Escalation   │────▶ Flags human handoff
                                      │     Handler      │     if needed
                                      │  5. Ticket       │────▶ Persists to DB
                                      │     Manager      │     (open→resolved)
                                      └──────────────────┘
                                               │
                                               ▼
                                      ┌──────────────────┐
                                      │  POSTGRESQL      │◀── Ticket lifecycle
                                      │  + pgvector      │     & knowledge base
                                      └──────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent LangGraph** | 5-node `StateGraph` with intent classification, RAG retrieval, response generation, escalation handling, and ticket management |
| **RAG Knowledge Base** | Vector search over FAQ/policy documents using pgvector for grounded, citation-backed answers |
| **Intent Classification** | Routes queries to billing, technical, general, refund_request, or human_handoff |
| **Automated Escalation** | Refund, billing disputes, and explicit human-handoff requests are flagged for agent review |
| **Ticket Lifecycle** | Full ticket state machine: `open → in_progress → resolved → closed` |
| **Evaluation Suite** | 10-question golden set measuring intent accuracy, escalation accuracy, response groundedness, and resolution rate |
| **Regression Testing** | CI gate that fails the build if evaluation metrics drop below defined thresholds |
| **FastAPI Backend** | REST API with `/chat`, `/tickets/{id}`, and `/admin/tickets` endpoints |
| **Dockerized** | Multi-container setup with FastAPI + PostgreSQL/pgvector via Docker Compose |
| **CI/CD** | GitHub Actions with linting, unit tests, and regression evaluation on every push |

---

## 🛠️ Tech Stack

**Backend:** FastAPI, Python 3.11+, Pydantic v2  
**Agent Framework:** LangGraph, LangChain  
**Database:** PostgreSQL, pgvector (vector search)  
**LLM Providers:** Groq (`openai/gpt-oss-20b`) with dynamic model support  
**Deployment:** Docker, Docker Compose  
**CI/CD:** GitHub Actions (linting + pytest + regression gate)  
**Testing:** pytest, pytest-asyncio  

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/chat` | Send user message; returns agent response + ticket status |
| `GET` | `/api/v1/tickets/{ticket_id}` | Get ticket details and lifecycle stage |
| `GET` | `/api/v1/admin/tickets` | Admin queue (filter by status) |
| `GET` | `/docs` | Interactive Swagger UI |

---

## 🧪 Evaluation & Regression

The project includes a built-in evaluation suite that mirrors the rigor of production RAG systems.

### Golden Set Metrics

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Intent Accuracy | ≥ 80% | Correct intent classification across golden set |
| Escalation Accuracy | ≥ 80% | Correct human-handoff decisions |
| Response Groundedness | ≥ 70% | Answers supported by retrieved context |
| Ticket Resolution Rate | ≥ 60% | Tickets resolved without unnecessary escalation |

### Run Evaluation Locally

```bash
python scripts/run_evaluation.py
```

Outputs `regression_report.json` and exits with code `1` if any threshold is not met (CI gate behavior).

---

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/Aditya-k63/customer-support-agent.git
cd customer-support-agent

# Configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY and other secrets

# Build and run
docker-compose up --build

# Access the API
open http://localhost:8000/docs
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run the server
uvicorn app.main:app --reload

# Access the API
open http://localhost:8000/docs
```

### Run Tests

```bash
pytest tests/ -v
```

---

## 📁 Project Structure

```
customer-support-agent/
├── .github/
│   └── workflows/
│       └── ci.yml                 # GitHub Actions CI/CD
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── api/
│   │   └── routes.py              # /chat, /tickets, /admin endpoints
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   └── security.py            # API key validation
│   ├── agents/
│   │   ├── graph.py               # LangGraph StateGraph definition
│   │   ├── state.py               # Typed state schema
│   │   ├── nodes/                 # 5 agent nodes
│   │   │   ├── intent_classifier.py
│   │   │   ├── rag_retriever.py
│   │   │   ├── response_generator.py
│   │   │   ├── escalation_handler.py
│   │   │   └── ticket_manager.py
│   │   └── tools/
│   │       ├── vector_store.py    # pgvector wrapper
│   │       └── ticket_db.py       # Ticket CRUD
│   ├── schemas/                   # Pydantic request/response models
│   ├── services/
│   │   └── llm.py                 # Groq LLM provider
│   ├── db/                        # SQLAlchemy async models
│   └── evaluation/
│       ├── metrics.py             # Intent accuracy, groundedness, etc.
│       ├── suite.py               # Evaluation runner + thresholds
│       └── golden_set.json        # 10-question test set
├── scripts/
│   └── run_evaluation.py          # CLI regression gate
├── ingestion/
│   └── ingest_docs.py             # Knowledge base ingestion script
├── tests/
│   ├── test_graph.py              # Agent node unit tests
│   ├── test_api.py                # FastAPI endpoint tests
│   ├── test_evaluation.py         # Metric unit tests
│   └── test_regression.py         # Golden set regression tests
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
├── docker-compose.yml
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

Copy `.env.example` to `.env` and update:

```env
# Database
DB_NAME=CRAG
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# LLM
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

# API
API_KEY=your_secret_key

# App
ENVIRONMENT=development
```

---

## 📈 Roadmap

- [ ] React admin dashboard for ticket management
- [ ] WebSocket support for real-time chat
- [ ] Email/SMS notifications on escalation
- [ ] A/B testing for prompt variants
- [ ] LangSmith integration for production tracing

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