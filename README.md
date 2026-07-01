---
title: ARIA Supply Chain Agent
emoji: 🔗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

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

**Key capability:** ARIA retrieves across a single merged knowledge store (supplier, ERP, and WMS data), synthesizes a grounded answer with hallucination guardrails, and learns from user feedback over time.

---

## Architecture

```
User Query (+ session_id)
    ↓
Redis Semantic Cache (persistent, sub-50ms on hits)
    ↓ cache miss
Retrieve Agent — similarity search over aria_knowledge (pgvector)
    ↓
Synthesis Agent — grounds answer in retrieved context + recent
                   session history, applies guardrails
    ↓
Final answer → FastAPI → UI
    ↓
👍 / 👎 feedback → Redis log → promoted into aria_knowledge
                                 (background, off the request path)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) in production; set `LLM_PROVIDER=ollama` to swap to local Ollama for zero-cost local dev |
| Orchestration | LangChain + LangGraph |
| Vector DB | pgvector (PostgreSQL extension) |
| Embeddings | all-MiniLM-L6-v2 (HuggingFace, local) |
| Semantic Cache | Redis (persistent across restarts) |
| API | FastAPI + async threadpool, API-key auth on admin endpoints |
| Deployment | Docker, Render Blueprint |
| Data Sources | SAP OData live (purchase orders, material stock via API Business Hub), supplier enrichment layer (18 suppliers, real SAP IDs), WMS (mock) |
| SAP Connector | `API_PURCHASEORDER_PROCESS_SRV`, `API_MATERIAL_STOCK_SRV`, `API_BUSINESS_PARTNER` — swappable between API Business Hub sandbox and direct S/4HANA tenant |

---

## Features

### LangGraph retrieval + synthesis pipeline
Two-node graph, kept intentionally lean:
- **Retrieve Agent** — single similarity search over the merged knowledge store (replaces the old router + 3 separate specialist collections, cutting retrieval to one DB call per query)
- **Synthesis Agent** — grounds the answer in retrieved context plus recent session history, applies guardrails

### Multi-turn session memory
- Each `/ask` call carries a `session_id`; the UI persists it in `localStorage`
- Last 6 turns per session are kept in-process and fed to the synthesis prompt so follow-ups like "what about that supplier?" resolve correctly
- History is used only to resolve references — never treated as a source of facts

### Feedback-driven self-learning
- 👍 / 👎 buttons on every answer call `POST /feedback`
- Feedback is logged to Redis inline (cheap — no embedding/LLM call on the request path)
- Helpful Q&A pairs are promoted into the `aria_knowledge` vector store as a background task, so future similar questions retrieve richer context without adding latency to `/ask` or `/feedback`

### Redis-backed semantic caching
- Cache persists across server restarts
- Cosine similarity matching at threshold 0.65
- Cache hits return in under 50ms vs 3 to 10 seconds for LLM calls
- 99%+ latency reduction on repeated or semantically similar queries
- `/cache/stats` endpoint for monitoring hit rate
- `/cache/clear` endpoint for manual cache invalidation

### Live SAP OData integration
- Connects to SAP S/4HANA Cloud via `API_PURCHASEORDER_PROCESS_SRV` (purchase orders), `API_MATERIAL_STOCK_SRV` (inventory), and `API_BUSINESS_PARTNER` (supplier master data)
- Dual auth modes: API Business Hub sandbox (API key) or direct S/4HANA tenant (Communication User / Basic Auth) — switched via env vars, no code changes
- 18-supplier enrichment layer (`data/supplier_enrichment.py`) keyed to real SAP Supplier IDs, carrying lead times, on-time delivery rates, quality rejection rates, risk levels, and YTD spend — fields the BP master data API does not expose
- `POST /reindex` rebuilds the knowledge store from live SAP data on demand; `GET /sap/test` probes all three OData services without side effects

### RAG over enterprise data
- 200+ documents indexed in a single merged pgvector collection (`aria_knowledge`) — 100 live purchase orders, 100 material stock records, 18 enriched supplier profiles, and WMS data
- Semantic similarity search using vector embeddings
- Source citations included in every answer
- Grows over time as helpful feedback is promoted back into the collection

### Hallucination guardrails
- Refuses to answer when no relevant context is retrieved
- Refuses to answer when context is too sparse
- Flags low-confidence responses with uncertainty indicators
- Prompt-level instructions to never fabricate supplier names, numbers, or dates

### /reindex endpoint
- POST `/reindex` triggers a live rebuild of the `aria_knowledge` collection from source data
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
| Documents indexed | 200+ (100 live POs, 100 stock records, 18 enriched suppliers, WMS) + learned Q&A pairs over time |
| Semantic match example | "unreliable vendors" matched "highest delivery risk" at 0.955 similarity |

---

## Project Structure

```
aria-supply-chain-agent/
├── app/
│   ├── main.py           # FastAPI server with all endpoints
│   └── aria_graph.py     # LangGraph retrieval + synthesis pipeline, Redis cache,
│                         # session memory, feedback/self-learning loop
├── data/
│   ├── sap_connector.py      # Live SAP OData connector (API Business Hub + direct tenant)
│   ├── supplier_enrichment.py# Performance metrics keyed by real SAP Supplier IDs
│   ├── supplier_docs.py      # Fallback mock supplier profiles (used when SAP creds absent)
│   ├── erp_data.py           # Fallback mock ERP data
│   └── wms_data.py           # WMS warehouse data (mock — no free WMS OData API)
├── static/
│   └── ui.html           # Bloomberg-style dark terminal UI
├── Dockerfile            # Container configuration
├── render.yaml           # Render Blueprint (web service + Postgres)
├── requirements.txt      # Python dependencies
├── .env.example          # Required/optional environment variables
├── .dockerignore         # Docker build exclusions
└── .gitignore            # Git exclusions
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/ask` | Query ARIA — checks Redis cache first, then retrieval + synthesis pipeline. Accepts an optional `session_id` for multi-turn context |
| POST | `/feedback` | Record 👍/👎 on an answer; helpful pairs are promoted into the knowledge store in the background |
| POST | `/reindex` 🔒 | Re-index the `aria_knowledge` pgvector collection from source data |
| GET | `/cache/stats` | Redis cache metrics: hits, misses, hit rate, entries |
| GET | `/cache/clear` 🔒 | Clear Redis cache |

🔒 = requires an `X-API-Key` header matching `ARIA_API_KEY` once that env var is set (see [Production Deployment](#production-deployment)). Unset locally, so these are open in dev.
| GET | `/ui` | Bloomberg-style chat interface |
| GET | `/docs` | FastAPI interactive API documentation |
| GET | `/sap/test` 🔒 | Probe all three SAP OData services with `$top=1` — returns per-service status without modifying data |
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
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## Production Deployment (Free Stack)

Runs entirely on free tiers — no credit card required.

| Service | Role | Why |
|---|---|---|
| Hugging Face Spaces | Web service (Docker) | Free CPU with 16GB RAM — handles sentence-transformers + torch comfortably |
| Supabase | Postgres + pgvector | Free tier, pgvector extension built-in |
| Upstash | Redis | Free tier (10k commands/day) |
| Groq | LLM | Free tier, hosts Llama 3.1 |

### Step 1 — Supabase (Postgres + pgvector)

1. Sign up at [supabase.com](https://supabase.com) → New Project
2. In **SQL Editor**, run:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Go to **Settings → Database → Connection string** — select the **Session pooler** tab (not Direct connection — that URL is IPv6-only and unreachable from HF Spaces)
4. Copy the pooler URL, change `postgresql://` to `postgresql+psycopg2://`, and URL-encode any special characters in the password (e.g. `!` → `%21`). It will look like:
   ```
   postgresql+psycopg2://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres
   ```
   That's your `DATABASE_URL`.

### Step 2 — Upstash (Redis)

1. Sign up at [upstash.com](https://upstash.com) → Create Database → choose Redis
2. Pick a region, create it
3. Copy the **Redis URL** (starts with `rediss://`) — that's your `REDIS_URL`

### Step 3 — Groq (LLM)

1. Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create
2. Copy the key — that's your `GROQ_API_KEY`

### Step 4 — Hugging Face Spaces

1. Sign up at [huggingface.co](https://huggingface.co) → New Space
2. Name it `aria-supply-chain-agent`, set SDK to **Docker**, visibility to **Public**
3. Under **Files**, link it to this GitHub repo (or push directly)
4. Go to **Settings → Variables and Secrets** and add:

| Key | Value |
|---|---|
| `DATABASE_URL` | your Supabase connection string |
| `REDIS_URL` | your Upstash Redis URL |
| `LLM_PROVIDER` | `groq` |
| `GROQ_API_KEY` | your Groq key |
| `GROQ_MODEL` | `llama-3.1-8b-instant` |
| `ARIA_API_KEY` | any strong random string — protects `/reindex`, `/cache/clear`, `/sap/test` |
| `SAP_API_HUB_KEY` | API key from [api.sap.com](https://api.sap.com) (profile → Show API Key) |
| `HF_HUB_DISABLE_TELEMETRY` | `1` |
| `TOKENIZERS_PARALLELISM` | `false` |

5. Click **Factory reboot** (or it'll deploy automatically). First build takes ~5-10 min (pip install + model bake). After that your app is live at:
   ```
   https://huggingface.co/spaces/<your-username>/aria-supply-chain-agent
   ```
   The UI is at that URL directly (HF Spaces proxies the root).

**Note:** HF Spaces sleeps after ~30 min of inactivity. First request after sleep takes ~20-30s to wake up. This is normal for the free tier.

Calling a protected endpoint:
```bash
curl -X POST https://<your-username>-aria-supply-chain-agent.hf.space/reindex \
  -H "X-API-Key: <your ARIA_API_KEY>"
```

### Paid alternative (Render)

If you need no sleep/cold starts, `render.yaml` in this repo provisions everything on Render with a credit card. See [render.com](https://render.com).

---

## n8n Automation

ARIA's knowledge store is kept fresh via an n8n workflow running on [n8n cloud](https://n8n.io) (free tier).

### What it does

| Trigger | Behaviour |
|---|---|
| **Daily schedule** (midnight UTC) | Automatically calls `POST /reindex` — pulls fresh SAP OData data and rebuilds the pgvector knowledge store |
| **Webhook** (on-demand) | `POST https://gregory2109.app.n8n.cloud/webhook/aria-reindex` — triggers an immediate reindex without needing to touch the server |

### How to set it up

1. Sign up at [n8n.io](https://n8n.io) → New workflow
2. Add a **Schedule Trigger** node — set to every 1 day at midnight
3. Add a **Webhook** node — path `aria-reindex`, method POST
4. Connect both triggers to a single **HTTP Request** node:
   - Method: `POST`
   - URL: `https://<your-hf-username>-aria-supply-chain-agent.hf.space/reindex`
   - Authentication: Generic Credential Type → Header Auth
   - Header name: `X-API-Key`, value: your `ARIA_API_KEY`
5. Save and toggle the workflow **Active**

The production webhook URL (`/webhook/aria-reindex`) then works anytime — the test URL (`/webhook-test/aria-reindex`) only works while the editor is open.

---

## Production Roadmap

| Feature | Status |
|---|---|
| LangGraph retrieval + synthesis pipeline | ✅ Complete |
| Redis persistent cache | ✅ Complete |
| Hallucination guardrails | ✅ Complete |
| Multi-turn session memory | ✅ Complete |
| Feedback-driven self-learning loop | ✅ Complete |
| /reindex endpoint for live data | ✅ Complete |
| Swappable hosted LLM backend (Groq) + API-key auth | ✅ Complete |
| Free production deployment (HF Spaces + Supabase + Upstash + Groq) | ✅ Complete |
| Render deployment blueprint (paid, no cold starts) | ✅ Complete |
| SAP OData live connector (API Business Hub + direct tenant) | ✅ Complete |
| Supplier enrichment layer (18 suppliers, real SAP IDs, full KPIs) | ✅ Complete |
| n8n webhook automation (daily schedule + on-demand webhook → `/reindex`) | ✅ Complete |
| HDBSCAN supplier clustering (auto-segmentation, cluster docs in RAG, /clusters endpoint) | ✅ Complete |
| Azure OpenAI swap for enterprise privacy | 🔄 Planned |

---

## About

Built by **Gregory Jaison** — MS Technology Management, University of Illinois Urbana-Champaign (Supply Chain concentration).

ARIA was designed to demonstrate the intersection of supply chain domain expertise and production-grade agentic AI engineering. The architecture mirrors enterprise AI platforms like LuMay AI, with connector-agnostic design allowing live SAP, Oracle, or WMS data sources to be swapped in with a single file change.

- LinkedIn: [linkedin.com/in/gregory-jaison](https://linkedin.com/in/gregory-jaison)
- Email: gregory.jaison@gmail.com
