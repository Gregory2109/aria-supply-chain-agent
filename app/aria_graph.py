from dotenv import load_dotenv
load_dotenv()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import uuid
from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from data.supplier_docs import supplier_documents as _mock_supplier_docs
from data.erp_data import erp_data as _mock_erp_data
from data.wms_data import wms_data as _mock_wms_data

# --- SHARED SETUP ---
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)

CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:password@localhost:5432/supplychain"
)

# LLM_PROVIDER=ollama (default, local dev) or groq (hosted, used in production
# where the host has no GPU/RAM budget to run Ollama itself).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
if LLM_PROVIDER == "groq":
    from langchain_groq import ChatGroq
    groq_model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    llm = ChatGroq(model=groq_model, api_key=os.getenv("GROQ_API_KEY"))
    print(f"[LLM] Using Groq ({groq_model})")
else:
    from langchain_ollama import OllamaLLM
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm = OllamaLLM(model="llama3.2", base_url=ollama_host)
    print(f"[LLM] Using Ollama ({ollama_host})")

# ChatGroq returns a message object, OllamaLLM returns a plain string —
# normalize both to str so synthesis_agent doesn't need to know which backend is active.
llm_chain = llm | StrOutputParser()

# --- SINGLE MERGED KNOWLEDGE COLLECTION ---
# One combined vector store replaces the old router + 3 separate specialist
# collections — a question now costs one retrieval call instead of up to three.
print("Setting up knowledge store...")

KNOWLEDGE_COLLECTION = "aria_knowledge"

def _all_seed_documents():
    # Use live SAP data when credentials are present; fall back per-source to
    # mock data so local dev (no SAP creds) keeps working without any changes.
    sap_configured = bool(
        os.getenv("SAP_API_HUB_KEY")
        or (os.getenv("SAP_BASE_URL") and os.getenv("SAP_USERNAME") and os.getenv("SAP_PASSWORD"))
    )

    if sap_configured:
        from data.sap_connector import fetch_all_sap_data
        live = fetch_all_sap_data()
        supplier_texts = live["suppliers"]       or _mock_supplier_docs
        erp_texts      = live["purchase_orders"] + live["material_stock"] or _mock_erp_data
        wms_texts      = _mock_wms_data  # no live WMS API yet — keep mock
        source_label = "live SAP"
    else:
        supplier_texts = _mock_supplier_docs
        erp_texts      = _mock_erp_data
        wms_texts      = _mock_wms_data
        source_label = "mock"

    print(f"[SEED] Using {source_label} data: {len(supplier_texts)} supplier, {len(erp_texts)} ERP, {len(wms_texts)} WMS docs")

    return (
        [Document(page_content=t, metadata={"source": "supplier"}) for t in supplier_texts]
        + [Document(page_content=t, metadata={"source": "erp"})      for t in erp_texts]
        + [Document(page_content=t, metadata={"source": "wms"})      for t in wms_texts]
    )

def _load_or_build_store():
    """Reuse an already-indexed collection instead of re-embedding every
    document on every process start. Full rebuilds happen only via reindex_all()."""
    store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name=KNOWLEDGE_COLLECTION,
    )
    if store.similarity_search("ping", k=1):
        print(f"[STARTUP] '{KNOWLEDGE_COLLECTION}' already indexed — reusing existing collection")
        return store
    print(f"[STARTUP] Indexing '{KNOWLEDGE_COLLECTION}'...")
    return PGVector.from_documents(
        documents=_all_seed_documents(),
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name=KNOWLEDGE_COLLECTION,
    )

knowledge_store = _load_or_build_store()
print("Knowledge store ready!")

# --- REDIS (semantic cache + feedback log) ---
import redis
import json

class RedisSemanticCache:
    def __init__(self, embedding_model, threshold=0.65, redis_url=None, redis_host="localhost", redis_port=6379,
                 max_entries=500, ttl_seconds=7 * 24 * 3600):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        # Managed Redis (e.g. Render) is reached via a connection URL;
        # local dev falls back to a plain host/port.
        if redis_url:
            self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
            print(f"[REDIS] Connected to Redis via REDIS_URL")
        else:
            self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
            print(f"[REDIS] Connected to Redis at {redis_host}:{redis_port}")
        self.cache_key = "aria:semantic_cache"
        self.lock_key = "aria:semantic_cache:lock"

    def cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    def _load_entries(self):
        cached_data = self.redis.get(self.cache_key)
        if not cached_data:
            return []
        entries = json.loads(cached_data)
        if self.ttl_seconds is None:
            return entries
        cutoff = time.time() - self.ttl_seconds
        return [e for e in entries if e.get("timestamp", 0) >= cutoff]

    def get(self, question):
        try:
            query_embedding = self.embedding_model.embed_query(question)
            entries = self._load_entries()
        except redis.RedisError as e:
            print(f"[REDIS] get() failed, treating as cache miss: {e}")
            self.misses += 1
            return None

        if not entries:
            self.misses += 1
            print(f"[CACHE MISS] Redis cache empty | '{question}'")
            return None

        best_score = 0
        best_answer = None

        for entry in entries:
            similarity = self.cosine_similarity(query_embedding, entry["embedding"])
            if similarity > best_score:
                best_score = similarity
                best_answer = entry["answer"]

        if best_score >= self.threshold:
            self.hits += 1
            print(f"[CACHE HIT] score={best_score:.3f} | '{question}'")
            return best_answer

        self.misses += 1
        print(f"[CACHE MISS] best score={best_score:.3f} | '{question}'")
        return None

    def set(self, question, answer):
        try:
            query_embedding = self.embedding_model.embed_query(question)
            # Lock guards the read-modify-write below — without it, concurrent
            # requests can race on the get/append/set and silently drop entries.
            with self.redis.lock(self.lock_key, timeout=10):
                entries = self._load_entries()
                entries.append({
                    "embedding": query_embedding,
                    "question": question,
                    "answer": answer,
                    "timestamp": time.time()
                })
                # Cap growth — without this the blob (and every get/set scan
                # over it) grows unbounded forever.
                if len(entries) > self.max_entries:
                    entries = entries[-self.max_entries:]
                self.redis.set(self.cache_key, json.dumps(entries))
            print(f"[REDIS] Cached answer for: '{question}'")
        except redis.RedisError as e:
            print(f"[REDIS] set() failed, skipping cache write: {e}")

    def stats(self):
        try:
            entries = self._load_entries()
        except redis.RedisError as e:
            print(f"[REDIS] stats() failed: {e}")
            entries = []
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "total_queries": total,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cached_entries": len(entries),
            "max_entries": self.max_entries,
            "cache_backend": "Redis",
            "persistent": True
        }

    def clear(self):
        self.redis.delete(self.cache_key)
        print("[REDIS] Cache cleared")

print("Initializing Redis semantic cache...")
cache = RedisSemanticCache(
    embeddings,
    threshold=0.65,
    redis_url=os.getenv("REDIS_URL"),
    redis_host=os.getenv("REDIS_HOST", "localhost"),
    redis_port=int(os.getenv("REDIS_PORT", "6379")),
)
print("Redis semantic cache ready!")

redis_client = cache.redis  # shared connection, also used by the feedback/learning loop below

# --- DEFINE STATE ---
class ARIAState(TypedDict):
    question: str
    session_id: str
    history: List[dict]
    context: str
    final_answer: str

# --- SESSION MEMORY (short-term, multi-turn, in-process) ---
MAX_HISTORY_TURNS = 6
_conversation_memory: Dict[str, List[dict]] = {}

def _get_history(session_id: str) -> List[dict]:
    return _conversation_memory.get(session_id, [])

def _append_history(session_id: str, question: str, answer: str):
    history = _conversation_memory.setdefault(session_id, [])
    history.append({"question": question, "answer": answer})
    if len(history) > MAX_HISTORY_TURNS:
        del history[:-MAX_HISTORY_TURNS]

# --- AGENT 1: RETRIEVE (single merged search, replaces router + 3 specialists) ---
def retrieve_agent(state: ARIAState):
    print("[RETRIEVE AGENT] Searching knowledge store...")
    docs = knowledge_store.similarity_search(state["question"], k=6)
    context = "\n\n".join(f"[{doc.metadata.get('source', 'unknown')}] {doc.page_content}" for doc in docs)
    print(f"[RETRIEVE AGENT] Found {len(docs)} documents")
    return {"context": context}

# --- AGENT 2: SYNTHESIS WITH HALLUCINATION GUARDRAILS ---
def synthesis_agent(state: ARIAState):
    print("[SYNTHESIS AGENT] Generating answer...")
    combined_context = state.get("context", "")

    # --- GUARDRAIL 1: No context retrieved ---
    if not combined_context.strip():
        print("[GUARDRAIL] No context retrieved — refusing to answer")
        return {"final_answer": "I was unable to find relevant information in the available data to answer this question. Please rephrase your query or check if the data source contains this information."}

    # --- GUARDRAIL 2: Context too short to be meaningful ---
    if len(combined_context.strip()) < 100:
        print("[GUARDRAIL] Context too sparse — refusing to answer")
        return {"final_answer": "I found very limited information related to your question. I cannot provide a reliable answer without sufficient data. Please try a more specific query."}

    history = state.get("history") or []
    history_block = ""
    if history:
        turns = "\n".join(f"Q: {h['question']}\nA: {h['answer']}" for h in history)
        history_block = f"\nConversation so far (most recent last):\n{turns}\n"

    prompt = f"""You are ARIA, an Agentic Risk and Intelligence Assistant for supply chain management.
You have access to a knowledge base covering supplier profiles, ERP systems, and WMS warehouse data.

IMPORTANT INSTRUCTIONS:
- Use ONLY the context below to answer the question
- If the context does not contain enough information to answer confidently, say exactly: "I don't have sufficient information in the available data to answer this question reliably."
- Never make up supplier names, numbers, dates, or facts not present in the context
- Always mention which data source supports your answer
- If you are uncertain, say so explicitly
- Use the conversation history only to resolve references (e.g. "that supplier") — never as a source of facts
{history_block}
Context:
{combined_context}

Question: {state['question']}

Answer:"""

    try:
        answer = llm_chain.invoke(prompt)
    except Exception as e:
        print(f"[SYNTHESIS AGENT] LLM call failed: {e}")
        return {"final_answer": "ARIA's language model is currently unavailable. Please try again shortly."}

    # --- GUARDRAIL 3: Detect if LLM admitted it doesn't know ---
    uncertainty_phrases = [
        "i don't have sufficient",
        "i do not have sufficient",
        "not enough information",
        "cannot answer",
        "no information available",
        "i cannot determine",
        "insufficient information",
    ]

    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in uncertainty_phrases):
        print("[GUARDRAIL] LLM expressed uncertainty — flagging response")
        answer = f"⚠ Low confidence: {answer}"

    print("[SYNTHESIS AGENT] Answer generated")
    return {"final_answer": answer}

# --- BUILD THE GRAPH ---
print("Building ARIA graph...")

workflow = StateGraph(ARIAState)
workflow.add_node("retrieve", retrieve_agent)
workflow.add_node("synthesis", synthesis_agent)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "synthesis")
workflow.add_edge("synthesis", END)

aria_graph = workflow.compile()
print("ARIA graph ready!")

# --- CACHED ENTRY POINT ---
def ask_aria_multi(question: str, session_id: Optional[str] = None) -> dict:
    start = time.time()
    session_id = session_id or str(uuid.uuid4())

    # Check cache first
    cached_answer = cache.get(question)
    if cached_answer:
        elapsed = time.time() - start
        _append_history(session_id, question, cached_answer)
        return {
            "answer": cached_answer,
            "source": "cache",
            "session_id": session_id,
            "latency_ms": round(elapsed * 1000, 2)
        }

    # Cache miss — run the retrieval + synthesis pipeline
    history = _get_history(session_id)
    result = aria_graph.invoke({"question": question, "session_id": session_id, "history": history})
    answer = result["final_answer"]

    # Store in cache and conversation memory
    cache.set(question, answer)
    _append_history(session_id, question, answer)

    elapsed = time.time() - start
    return {
        "answer": answer,
        "source": "multi-agent",
        "session_id": session_id,
        "latency_ms": round(elapsed * 1000, 2)
    }

# --- SELF-LEARNING LOOP ---
# Feedback is logged inline (cheap — one Redis append, no embedding/LLM call) but
# promotion into the knowledge store always runs out-of-band so it never adds
# latency to /ask or /feedback.
FEEDBACK_KEY = "aria:feedback"
FEEDBACK_OFFSET_KEY = "aria:feedback:promoted_offset"

def record_feedback(question: str, answer: str, helpful: bool, session_id: Optional[str] = None):
    try:
        entry = {
            "question": question,
            "answer": answer,
            "helpful": helpful,
            "session_id": session_id,
            "timestamp": time.time(),
        }
        redis_client.rpush(FEEDBACK_KEY, json.dumps(entry))
        print(f"[FEEDBACK] Recorded helpful={helpful} for: '{question}'")
    except redis.RedisError as e:
        print(f"[FEEDBACK] Failed to record feedback: {e}")

def promote_learned_knowledge():
    """Fold confirmed-helpful Q&A pairs back into the knowledge store so future
    similar questions retrieve richer context. Intended to run as a background
    task, not inline on a request."""
    try:
        offset = int(redis_client.get(FEEDBACK_OFFSET_KEY) or 0)
        raw_entries = redis_client.lrange(FEEDBACK_KEY, offset, -1)
    except redis.RedisError as e:
        print(f"[LEARNING] Could not read feedback log: {e}")
        return {"promoted": 0}

    new_docs = [
        Document(
            page_content=f"Q: {entry['question']}\nA: {entry['answer']}",
            metadata={"source": "learned"}
        )
        for entry in (json.loads(raw) for raw in raw_entries)
        if entry.get("helpful")
    ]

    if new_docs:
        knowledge_store.add_documents(new_docs)
        print(f"[LEARNING] Promoted {len(new_docs)} learned Q&A pairs into '{KNOWLEDGE_COLLECTION}'")
    else:
        print("[LEARNING] No new helpful feedback to promote")

    try:
        redis_client.set(FEEDBACK_OFFSET_KEY, offset + len(raw_entries))
    except redis.RedisError as e:
        print(f"[LEARNING] Failed to advance feedback offset: {e}")

    return {"promoted": len(new_docs)}

# --- REINDEX FUNCTION ---
def reindex_all():
    global knowledge_store
    print("[REINDEX] Starting reindex of knowledge store...")

    # Release the old store's DB connection before replacing it — otherwise
    # repeated reindexing leaks one engine per call.
    if hasattr(knowledge_store._bind, "dispose"):
        knowledge_store._bind.dispose()

    knowledge_store = PGVector.from_documents(
        documents=_all_seed_documents(),
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name=KNOWLEDGE_COLLECTION,
        pre_delete_collection=True
    )
    print("[REINDEX] Knowledge store reindexed successfully!")
    return {"status": "reindexed", "collections": [KNOWLEDGE_COLLECTION]}

# --- TEST ---
if __name__ == "__main__":
    print("\n--- ARIA Multi-Agent + Semantic Cache ---\n")
    questions = [
        "Which suppliers have the highest delivery risk?",
        "Which vendors are most at risk for delays?",
        "Are there any overdue purchase orders?",
        "Are there any delayed shipments?",
    ]
    session = str(uuid.uuid4())
    for q in questions:
        print(f"\nQ: {q}")
        result = ask_aria_multi(q, session_id=session)
        print(f"A: {result['answer']}")
        print(f"[Source: {result['source']} | Latency: {result['latency_ms']}ms]")
