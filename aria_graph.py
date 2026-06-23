from dotenv import load_dotenv
load_dotenv()
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import re
import time
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from supplier_docs import supplier_documents
from erp_data import erp_data
from wms_data import wms_data

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

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
llm = OllamaLLM(model="llama3.2", base_url=ollama_host)

# --- CREATE 3 SEPARATE VECTOR COLLECTIONS ---
print("Setting up specialist vector stores...")

def _load_or_build_store(collection_name, documents, source_label):
    """Reuse an already-indexed collection instead of re-embedding every
    document on every process start. Full rebuilds happen only via reindex_all()."""
    store = PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    if store.similarity_search("ping", k=1):
        print(f"[STARTUP] '{collection_name}' already indexed — reusing existing collection")
        return store
    print(f"[STARTUP] Indexing '{collection_name}'...")
    return PGVector.from_documents(
        documents=[Document(page_content=t, metadata={"source": source_label}) for t in documents],
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name=collection_name,
    )

supplier_store = _load_or_build_store("aria_supplier", supplier_documents, "supplier")
erp_store = _load_or_build_store("aria_erp", erp_data, "erp")
wms_store = _load_or_build_store("aria_wms", wms_data, "wms")
print("All specialist stores ready!")

# --- REDIS SEMANTIC CACHE ---
import redis
import json

class RedisSemanticCache:
    def __init__(self, embedding_model, threshold=0.65, redis_host="localhost", redis_port=6379,
                 max_entries=500, ttl_seconds=7 * 24 * 3600):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self.hits = 0
        self.misses = 0
        # Connect to Redis
        self.redis = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.cache_key = "aria:semantic_cache"
        self.lock_key = "aria:semantic_cache:lock"
        print(f"[REDIS] Connected to Redis at {redis_host}:{redis_port}")

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
cache = RedisSemanticCache(embeddings, threshold=0.65)
print("Redis semantic cache ready!")

# --- DEFINE STATE ---
class ARIAState(TypedDict):
    question: str
    agents_to_call: List[str]
    supplier_context: str
    erp_context: str
    wms_context: str
    final_answer: str

# --- AGENT 1: ROUTER ---
def _contains_keyword(question: str, keyword: str) -> bool:
    # Word-boundary match — plain substring containment let short keywords
    # like "po" match inside "support"/"report"/"deployment" and misroute.
    return re.search(rf"\b{re.escape(keyword)}\b", question) is not None

def router_agent(state: ARIAState):
    question = state["question"].lower()
    agents = []

    if any(_contains_keyword(question, word) for word in [
        "risk", "delay", "lead time", "delivery", "supplier",
        "vendor", "reliable", "unreliable", "performance", "on-time"
    ]):
        agents.append("supplier")

    if any(_contains_keyword(question, word) for word in [
        "purchase order", "po", "invoice", "inventory", "reorder",
        "stock", "replenish", "erp", "payment", "overdue order"
    ]):
        agents.append("erp")

    if any(_contains_keyword(question, word) for word in [
        "shipment", "warehouse", "quality hold", "overdue", "customs",
        "inbound", "wms", "shipping", "carrier", "dispatch"
    ]):
        agents.append("wms")

    if not agents:
        agents = ["supplier", "erp", "wms"]

    print(f"[ROUTER] Dispatching to: {agents}")
    return {"agents_to_call": agents}

# --- AGENT 2: SUPPLIER SPECIALIST ---
def supplier_agent(state: ARIAState):
    if "supplier" not in state["agents_to_call"]:
        print("[SUPPLIER AGENT] Skipped")
        return {"supplier_context": ""}
    print("[SUPPLIER AGENT] Searching supplier documents...")
    retriever = supplier_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"[SUPPLIER AGENT] Found {len(docs)} documents")
    return {"supplier_context": context}

# --- AGENT 3: ERP SPECIALIST ---
def erp_agent(state: ARIAState):
    if "erp" not in state["agents_to_call"]:
        print("[ERP AGENT] Skipped")
        return {"erp_context": ""}
    print("[ERP AGENT] Searching ERP data...")
    retriever = erp_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"[ERP AGENT] Found {len(docs)} documents")
    return {"erp_context": context}

# --- AGENT 4: WMS SPECIALIST ---
def wms_agent(state: ARIAState):
    if "wms" not in state["agents_to_call"]:
        print("[WMS AGENT] Skipped")
        return {"wms_context": ""}
    print("[WMS AGENT] Searching WMS data...")
    retriever = wms_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(state["question"])
    context = "\n\n".join(doc.page_content for doc in docs)
    print(f"[WMS AGENT] Found {len(docs)} documents")
    return {"wms_context": context}

# --- AGENT 5: SYNTHESIS WITH HALLUCINATION GUARDRAILS ---
def synthesis_agent(state: ARIAState):
    print("[SYNTHESIS AGENT] Combining specialist outputs...")
    combined_context = ""
    if state.get("supplier_context"):
        combined_context += f"SUPPLIER DATA:\n{state['supplier_context']}\n\n"
    if state.get("erp_context"):
        combined_context += f"ERP DATA:\n{state['erp_context']}\n\n"
    if state.get("wms_context"):
        combined_context += f"WMS DATA:\n{state['wms_context']}\n\n"

    # --- GUARDRAIL 1: No context retrieved ---
    if not combined_context.strip():
        print("[GUARDRAIL] No context retrieved — refusing to answer")
        return {"final_answer": "I was unable to find relevant information in the supplier, ERP, or WMS data to answer this question. Please rephrase your query or check if the data source contains this information."}

    # --- GUARDRAIL 2: Context too short to be meaningful ---
    if len(combined_context.strip()) < 100:
        print("[GUARDRAIL] Context too sparse — refusing to answer")
        return {"final_answer": "I found very limited information related to your question. I cannot provide a reliable answer without sufficient data. Please try a more specific query."}

    prompt = f"""You are ARIA, an Agentic Risk and Intelligence Assistant for supply chain management.
You have received intelligence from specialist agents covering supplier profiles, ERP systems, and WMS warehouse data.

IMPORTANT INSTRUCTIONS:
- Use ONLY the context below to answer the question
- If the context does not contain enough information to answer confidently, say exactly: "I don't have sufficient information in the available data to answer this question reliably."
- Never make up supplier names, numbers, dates, or facts not present in the context
- Always mention which data source supports your answer
- If you are uncertain, say so explicitly

Context from specialist agents:
{combined_context}

Question: {state['question']}

Answer:"""

    try:
        answer = llm.invoke(prompt)
    except Exception as e:
        print(f"[SYNTHESIS AGENT] LLM call failed: {e}")
        return {"final_answer": "ARIA's language model is currently unavailable. Please try again shortly."}

    # --- GUARDRAIL 3: Detect if LLM admitted it doesn't know ---
    uncertainty_phrases = [
        "i don't have",
        "i do not have",
        "not enough information",
        "cannot answer",
        "no information",
        "not mentioned",
        "not provided",
        "i cannot determine",
        "insufficient"
    ]

    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in uncertainty_phrases):
        print("[GUARDRAIL] LLM expressed uncertainty — flagging response")
        answer = f"⚠ Low confidence: {answer}"

    print("[SYNTHESIS AGENT] Answer generated")
    return {"final_answer": answer}

# --- BUILD THE GRAPH WITH PARALLEL EXECUTION ---
print("Building ARIA multi-agent graph...")

workflow = StateGraph(ARIAState)
workflow.add_node("router", router_agent)
workflow.add_node("supplier", supplier_agent)
workflow.add_node("erp", erp_agent)
workflow.add_node("wms", wms_agent)
workflow.add_node("synthesis", synthesis_agent)

workflow.set_entry_point("router")
workflow.add_edge("router", "supplier")
workflow.add_edge("router", "erp")
workflow.add_edge("router", "wms")
workflow.add_edge("supplier", "synthesis")
workflow.add_edge("erp", "synthesis")
workflow.add_edge("wms", "synthesis")
workflow.add_edge("synthesis", END)

aria_graph = workflow.compile()
print("ARIA multi-agent graph ready!")

# --- CACHED ENTRY POINT ---
def ask_aria_multi(question: str) -> dict:
    start = time.time()

    # Check cache first
    cached_answer = cache.get(question)
    if cached_answer:
        elapsed = time.time() - start
        return {
            "answer": cached_answer,
            "source": "cache",
            "latency_ms": round(elapsed * 1000, 2)
        }

    # Cache miss — run multi-agent pipeline
    result = aria_graph.invoke({"question": question})
    answer = result["final_answer"]

    # Store in cache
    cache.set(question, answer)

    elapsed = time.time() - start
    return {
        "answer": answer,
        "source": "multi-agent",
        "latency_ms": round(elapsed * 1000, 2)
    }
# --- REINDEX FUNCTION ---
def reindex_all():
    global supplier_store, erp_store, wms_store
    print("[REINDEX] Starting reindex of all collections...")

    # Release the old stores' DB connections before replacing them —
    # otherwise repeated reindexing leaks one engine per call.
    for old_store in (supplier_store, erp_store, wms_store):
        if hasattr(old_store._bind, "dispose"):
            old_store._bind.dispose()

    supplier_store = PGVector.from_documents(
        documents=[Document(page_content=t, metadata={"source": "supplier"}) for t in supplier_documents],
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name="aria_supplier",
        pre_delete_collection=True
    )
    print("[REINDEX] Supplier collection done")

    erp_store = PGVector.from_documents(
        documents=[Document(page_content=t, metadata={"source": "erp"}) for t in erp_data],
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name="aria_erp",
        pre_delete_collection=True
    )
    print("[REINDEX] ERP collection done")

    wms_store = PGVector.from_documents(
        documents=[Document(page_content=t, metadata={"source": "wms"}) for t in wms_data],
        embedding=embeddings,
        connection_string=CONNECTION_STRING,
        collection_name="aria_wms",
        pre_delete_collection=True
    )
    print("[REINDEX] WMS collection done")
    print("[REINDEX] All collections reindexed successfully!")
    return {"status": "reindexed", "collections": ["aria_supplier", "aria_erp", "aria_wms"]}
# --- TEST ---
if __name__ == "__main__":
    print("\n--- ARIA Multi-Agent + Semantic Cache ---\n")
    questions = [
        "Which suppliers have the highest delivery risk?",
        "Which vendors are most at risk for delays?",
        "Are there any overdue purchase orders?",
        "Are there any delayed shipments?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = ask_aria_multi(q)
        print(f"A: {result['answer']}")
        print(f"[Source: {result['source']} | Latency: {result['latency_ms']}ms]")