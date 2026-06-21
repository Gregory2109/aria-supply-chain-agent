# ARIA — Agentic Risk & Intelligence Assistant

> Multi-agent supply chain AI system with LangGraph orchestration, RAG over enterprise data sources, Redis-backed semantic caching, and hallucination guardrails. Built independently prior to joining LuMay AI.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square)
![LangGraph](https://img.shields.io/badge/LangGraph-1.2.4-green?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-009688?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-containerized-2496ED?style=flat-square)
![Redis](https://img.shields.io/badge/Redis-persistent%20cache-DC382D?style=flat-square)
![pgvector](https://img.shields.io/badge/pgvector-semantic%20search-336791?style=flat-square)

---

## What is ARIA?

ARIA is a production-grade agentic AI system that gives procurement and supply chain teams instant intelligence over their supplier base, ERP data, and warehouse operations.

Instead of manually searching through supplier documents, purchase orders, and shipment records, users ask ARIA natural language questions and get grounded, source-cited answers in seconds.

**Key capability:** ARIA uses a Router Agent to intelligently dispatch queries to specialized agents — each with their own dedicated vector store — then synthesizes their outputs into a single coherent answer.

---

## Architecture

```
User Query
    ↓
Redis Semantic Cache (persistent, sub-50ms on hits)
    ↓ cache miss
Router Agent — reads query, dispatches to specialists
    ↓
┌─────────────────────────────────────────┐
│                                         │
Supplier Agent    ERP Agent    WMS Agent
(pgvector)       (pgvector)   (pgvector)
aria_supplier    aria_erp     aria_wms
│                                         │
└─────────────────────────────────────────┘
    ↓
Synthesis Agent — combines outputs, applies guardrails
    ↓
Final answer → FastAPI → UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Llama 3.2 via Ollama (local, zero API cost) |
| Orchestration | LangChain + LangGraph |
| Vector DB | pgvector (PostgreSQL extension) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Semantic Cache | Redis (persistent across restarts) |
| API | FastAPI + async threadpool |
| Deployment | Docker |
| Data Sources | Supplier profiles, SAP ERP (mock), WMS (mock) |

---

## Features

### Multi-agent LangGraph architecture
Five specialized agents working in coordination:
- **Router Agent** — reads the query and decides which specialists to call
- **Supplier Specialist** — searches supplier risk, lead time, and delivery data
- **ERP Specialist** — searches purchase orders, inventory levels, and invoice status
- **WMS Specialist** — searches inbound shipments, quality holds, and warehouse alerts
- **Synthesis Agent** — combines all specialist outputs into one coherent answer

### Redis-backed semantic caching
- Cache persists across server restarts
- Cosine similarity matching at threshold 0.65
- Cache hits return in under 50ms vs 3 to 10 seconds for LLM calls
- 99%+ latency reduction on repeated or semantically similar queries
- `/cache/stats` endpoint for monitoring hit rate
- `/cache/clear` endpoint for manual cache invalidation

### RAG over enterprise data
- 15 documents indexed across 3 dedicated pgvector collections
- Semantic similarity search using vector embeddings
- Each specialist agent searches only its own collection for precision
- Source citations included in every answer

### Hallucination guardrails
- Refuses to answer when no relevant context is retrieved
- Refuses to answer when context is too sparse
- Flags low-confidence responses with uncertainty indicators
- Prompt-level instructions to never fabricate supplier names, numbers, or dates

### /reindex endpoint
- POST `/reindex` triggers live re-indexing of all 3 collections
- Designed for n8n webhook integration — SAP posts update → n8n calls `/reindex` → ARIA stays current
- Production-ready for live SAP OData connector swap

### Bloomberg-style dark terminal UI
- Real-time session metrics: queries, cache hits, hit rate, latency
- Live agent status panel: ready → active → idle per query
- Collection status showing document counts per specialist
- Source tags: `⚡ cache` vs `◈ multi-agent` with latency on every response

---

## Performance Metrics

| Metric | Value |
|---|---|
| Cache hit latency | ~20 to 50ms |
| LLM response latency | 3,000 to 10,000ms |
| Latency improvement on cache hits | 99.5%+ |
| Cache similarity threshold | 0.65 cosine similarity |
| Documents indexed | 15 (5 supplier + 5 ERP + 5 WMS) |
| Semantic match example | "unreliable vendors" matched "highest delivery risk" at 0.955 similarity |

---

## Project Structure

```
aria-supply-chain-agent/
├── aria_graph.py        # Multi-agent LangGraph pipeline + Redis cache
├── main.py              # FastAPI server with all endpoints
├── ui.html              # Bloomberg-style dark terminal UI
├── supplier_docs.py     # Supplier profile data
├── erp_data.py          # Simulated SAP ERP data
├── wms_data.py          # Simulated WMS warehouse data
├── agent.py             # Legacy single-agent (reference only)
├── Dockerfile           # Container configuration
├── requirements.txt     # Python dependencies
├── .dockerignore        # Docker build exclusions
└── .gitignore           # Git exclusions
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask` | Query ARIA — checks Redis cache first, then multi-agent pipeline |
| POST | `/reindex` | Re-index all 3 pgvector collections from source data |
| GET | `/cache/stats` | Redis cache metrics: hits, misses, hit rate, entries |
| GET | `/cache/clear` | Clear Redis cache |
| GET | `/ui` | Bloomberg-style chat interface |
| GET | `/docs` | FastAPI interactive API documentation |
| GET | `/` | Health check |

---

## Quick Start

### Prerequisites
- Docker Desktop
- Ollama with Llama 3.2 (`ollama pull llama3.2`)

### Run with Docker

```bash
# Start pgvector database
docker run -d --name pgvector-db \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=supplychain \
  -p 5432:5432 ankane/pgvector

# Start Redis cache
docker run -d --name redis -p 6379:6379 redis:alpine

# Build and run ARIA
docker build -t aria-supply-chain .
docker run --name aria -p 8000:8000 \
  -e DATABASE_URL="postgresql+psycopg2://postgres:password@host.docker.internal:5432/supplychain" \
  -e OLLAMA_HOST="http://host.docker.internal:11434" \
  aria-supply-chain
```

Open **http://localhost:8000/ui**

### Run locally

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## Production Roadmap

| Feature | Status |
|---|---|
| Multi-agent LangGraph | ✅ Complete |
| Redis persistent cache | ✅ Complete |
| Hallucination guardrails | ✅ Complete |
| /reindex endpoint for live data | ✅ Complete |
| SAP OData live connector | 🔄 In progress |
| n8n webhook automation | 🔄 Planned |
| HDBSCAN supplier clustering | 🔄 Planned |
| Azure OpenAI swap for enterprise privacy | 🔄 Planned |

---

## About

Built by **Gregory Jaison** — MS Technology Management, University of Illinois Urbana-Champaign (Supply Chain concentration).

ARIA was designed to demonstrate the intersection of supply chain domain expertise and production-grade agentic AI engineering. The architecture mirrors enterprise AI platforms like LuMay AI, with connector-agnostic design allowing live SAP, Oracle, or WMS data sources to be swapped in with a single file change.

- LinkedIn: [linkedin.com/in/gregory-jaison](https://linkedin.com/in/gregory-jaison)
- Email: gregory.jaison@gmail.com
