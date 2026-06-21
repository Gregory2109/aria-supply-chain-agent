from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from aria_graph import ask_aria_multi, cache, reindex_all

app = FastAPI(title="ARIA — Agentic Risk & Intelligence Assistant")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_agent(query: Query):
    result = await run_in_threadpool(ask_aria_multi, query.question)
    return {
        "question": query.question,
        "answer": result["answer"],
        "source": result["source"],
        "latency_ms": result["latency_ms"]
    }

@app.post("/reindex")
async def reindex():
    result = await run_in_threadpool(reindex_all)
    return result

@app.get("/cache/stats")
async def cache_stats():
    return cache.stats()

@app.get("/cache/clear")
async def clear_cache():
    cache.clear()
    return {"status": "cache cleared"}

@app.get("/")
async def health():
    return {"status": "ARIA is running"}

@app.get("/ui")
async def ui():
    return FileResponse("ui.html")