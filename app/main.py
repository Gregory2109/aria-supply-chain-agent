import os
from pathlib import Path
from typing import Optional
from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from app.aria_graph import ask_aria_multi, cache, reindex_all, record_feedback, promote_learned_knowledge

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

ARIA_API_KEY = os.getenv("ARIA_API_KEY")
if not ARIA_API_KEY:
    print("[AUTH] WARNING: ARIA_API_KEY not set — /reindex and /cache/clear are UNAUTHENTICATED")

def require_api_key(x_api_key: Optional[str] = Header(None)):
    if ARIA_API_KEY and x_api_key != ARIA_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")

app = FastAPI(title="ARIA — Agentic Risk & Intelligence Assistant")

class Query(BaseModel):
    question: str
    session_id: Optional[str] = None

class Feedback(BaseModel):
    question: str
    answer: str
    helpful: bool
    session_id: Optional[str] = None

@app.post("/ask")
async def ask_agent(query: Query):
    result = await run_in_threadpool(ask_aria_multi, query.question, query.session_id)
    return {
        "question": query.question,
        "answer": result["answer"],
        "source": result["source"],
        "session_id": result["session_id"],
        "latency_ms": result["latency_ms"]
    }

@app.post("/feedback")
async def feedback(fb: Feedback, background_tasks: BackgroundTasks):
    await run_in_threadpool(record_feedback, fb.question, fb.answer, fb.helpful, fb.session_id)
    background_tasks.add_task(promote_learned_knowledge)
    return {"status": "recorded"}

@app.post("/reindex", dependencies=[Depends(require_api_key)])
async def reindex():
    result = await run_in_threadpool(reindex_all)
    return result

@app.get("/cache/stats")
async def cache_stats():
    return cache.stats()

@app.get("/cache/clear", dependencies=[Depends(require_api_key)])
async def clear_cache():
    cache.clear()
    return {"status": "cache cleared"}

@app.get("/")
async def health():
    return {"status": "ARIA is running"}

@app.get("/sap/test", dependencies=[Depends(require_api_key)])
async def sap_test():
    """Probe all three SAP OData APIs and report per-service status."""
    sap_url = os.getenv("SAP_BASE_URL")
    if not sap_url:
        return {"configured": False, "message": "SAP_BASE_URL not set"}
    from data.sap_connector import test_connection
    results = await run_in_threadpool(test_connection)
    all_ok = all(v["ok"] for v in results.values())
    return {"configured": True, "all_ok": all_ok, "services": results}

@app.get("/ui")
async def ui():
    return FileResponse(STATIC_DIR / "ui.html")