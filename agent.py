import os
import time
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from supplier_docs import supplier_documents
from erp_data import erp_data
from wms_data import wms_data

# --- STEP 1: Embedding model ---
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    cache_folder="E:/supply-chain-agent/model_cache"
)

# --- STEP 2: Connect to pgvector ---
CONNECTION_STRING = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:password@localhost:5432/supplychain"
)

# Combine all data sources — supplier docs, ERP, and WMS
all_documents = supplier_documents + erp_data + wms_data
docs = [
    Document(
        page_content=text,
        metadata={"source": "supplier_docs" if text in supplier_documents else "erp" if text in erp_data else "wms"}
    )
    for text in all_documents
]

print("Indexing supplier documents into pgvector...")
vectorstore = PGVector.from_existing_index(
    embedding=embeddings,
    connection_string=CONNECTION_STRING,
    collection_name="supplier_docs"
)
print("Documents indexed successfully!")

# --- STEP 3: Retriever ---
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- STEP 4: LLM ---
print("Loading Ollama LLM...")
ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
llm = OllamaLLM(model="llama3.2", base_url=ollama_host)

# --- STEP 5: Semantic Cache ---
# Simple in-memory semantic cache using embeddings
# Stores previous queries and their answers
# If a new query is semantically similar (cosine similarity > threshold), return cached answer

class SemanticCache:
    def __init__(self, embedding_model, threshold=0.85):
        self.embedding_model = embedding_model
        self.threshold = threshold  # similarity score needed to hit cache
        self.cache = []  # list of {embedding, question, answer}
        self.hits = 0
        self.misses = 0

    def cosine_similarity(self, a, b):
        # measures angle between two vectors
        # 1.0 = identical meaning, 0.0 = completely different
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x ** 2 for x in a) ** 0.5
        norm_b = sum(x ** 2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0
        return dot / (norm_a * norm_b)

    def get(self, question):
        query_embedding = self.embedding_model.embed_query(question)
    
        best_score = 0
        best_answer = None
        # compare against all cached questions
        for entry in self.cache:
            similarity = self.cosine_similarity(query_embedding, entry["embedding"])
            print(f"[DEBUG] similarity={similarity:.3f} vs '{entry['question']}'")
            if similarity > best_score:
                best_score = similarity
                best_answer = entry["answer"]
    
        if best_score >= self.threshold:
            self.hits += 1
            print(f"[CACHE HIT] score={best_score:.3f}")
            return best_answer
    
        self.misses += 1
        print(f"[CACHE MISS] best score was {best_score:.3f}, threshold={self.threshold}")
        return None

    def set(self, question, answer):
        query_embedding = self.embedding_model.embed_query(question)
        self.cache.append({
            "embedding": query_embedding,
            "question": question,
            "answer": answer
        })

    def stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "total_queries": total,
            "cache_hits": self.hits,
            "cache_misses": self.misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "cached_entries": len(self.cache)
        }

# Initialize cache
print("Initializing semantic cache...")
cache = SemanticCache(embeddings, threshold=0.65)
print("Semantic cache ready!")

# --- STEP 6: Prompt ---
prompt = PromptTemplate.from_template("""You are ARIA, an Agentic Risk & Intelligence Assistant for supply chain management.
You have access to three enterprise data sources:
- Supplier profiles (risk levels, lead times, delivery performance)
- ERP system data (purchase orders, inventory planning, invoice status)
- WMS data (inbound shipments, warehouse inventory, quality holds)

Use the context below to answer the question accurately.
Only use the provided context — do not make up information.
Always mention which data source supports your answer.

Context:
{context}

Question: {question}

Answer:""")

# --- STEP 7: Build chain ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# --- STEP 8: Cached invoke function ---
def ask_aria(question: str) -> dict:
    start = time.time()
    
    # check cache first
    cached_answer = cache.get(question)
    
    if cached_answer:
        elapsed = time.time() - start
        return {
            "answer": cached_answer,
            "source": "cache",
            "latency_ms": round(elapsed * 1000, 2)
        }
    
    # cache miss — call the full RAG pipeline
    answer = chain.invoke(question)
    
    # store in cache for future similar questions
    cache.set(question, answer)
    
    elapsed = time.time() - start
    return {
        "answer": answer,
        "source": "llm",
        "latency_ms": round(elapsed * 1000, 2)
    }

# --- STEP 9: Test it ---
if __name__ == "__main__":
    print("\n--- ARIA: Agentic Risk & Intelligence Assistant ---\n")
    
    test_questions = [
        "Which suppliers have the highest delivery risk?",
        "Which vendors have the highest delivery risk?",  # almost identical — will hit cache
        "Which supplier has the fastest lead time?",
        "Which supplier delivers the fastest?",  # very similar — will hit cache
        "Are there any suppliers with quality issues?"
    ]
    
    for question in test_questions:
        print(f"\nQ: {question}")
        result = ask_aria(question)
        print(f"A: {result['answer']}")
        print(f"[Source: {result['source']} | Latency: {result['latency_ms']}ms]")
    
    print("\n--- Cache Stats ---")
    print(cache.stats())