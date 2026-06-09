"""
RCSA Intelligence Platform — Backend
═══════════════════════════════════════════════════════════════════════
Real RAG pipeline: Word docs → ChromaDB → GPT-4o → Sourced responses

Architecture:
  1. On startup → load all .docx files from /control_docs
  2. Chunk + embed via OpenAI text-embedding-3-small → store in ChromaDB
  3. Each chat message → semantic search → retrieve top 4 chunks
  4. Inject chunks as context → GPT-4o generates response
  5. Return response + source attribution to frontend
═══════════════════════════════════════════════════════════════════════
"""
import os
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from docx import Document as DocxDocument

# ── Configuration ───────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

DOCS_PATH = Path(__file__).parent.parent / "control_docs"
CHROMA_PATH = Path(__file__).parent / "chroma_data"
COLLECTION_NAME = "rcsa_controls"
CHAT_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K_RETRIEVAL = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s · %(levelname)s · %(message)s")
log = logging.getLogger("rcsa")

# ── Clients ─────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name=EMBEDDING_MODEL,
)

# ── FastAPI app ─────────────────────────────────────────────────────────
app = FastAPI(
    title="RCSA Intelligence Platform API",
    description="Real RAG over banking control documents · GPT-4o + ChromaDB",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Document loading & chunking ─────────────────────────────────────────
def load_docx(path: Path) -> str:
    """Extract all text from a .docx file."""
    doc = DocxDocument(str(path))
    parts = []
    # Paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    # Tables (control docs have many)
    for table in doc.tables:
        for row in table.rows:
            row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            if row_text:
                parts.append(row_text)
    return "\n".join(parts)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Sliding-window chunking with paragraph awareness."""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break on paragraph boundary
        if end < len(text):
            last_break = text.rfind("\n", start, end)
            if last_break > start + size // 2:
                end = last_break
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end
    return [c for c in chunks if len(c) > 50]


def ingest_documents():
    """Ingest all .docx files into ChromaDB on startup."""
    # Reset collection (idempotent on cold starts)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
        log.info("Existing collection deleted")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"},
    )

    if not DOCS_PATH.exists():
        log.warning(f"Docs path {DOCS_PATH} does not exist — skipping ingestion")
        return collection

    doc_files = sorted(DOCS_PATH.glob("*.docx"))
    log.info(f"Found {len(doc_files)} documents to ingest")

    all_chunks = []
    all_ids = []
    all_metadata = []

    for doc_path in doc_files:
        try:
            text = load_docx(doc_path)
            chunks = chunk_text(text)
            doc_name = doc_path.stem
            # Parse control ID from filename (e.g. RB-CTRL-02_Control_Narrative)
            parts = doc_name.split("_", 1)
            control_id = parts[0] if parts else doc_name
            doc_type = parts[1].replace("_", " ") if len(parts) > 1 else "Document"

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"{doc_name}_chunk_{i}")
                all_metadata.append({
                    "source": doc_name,
                    "control_id": control_id,
                    "doc_type": doc_type,
                    "chunk_index": i,
                })

            log.info(f"  ✓ {doc_name}: {len(chunks)} chunks")
        except Exception as e:
            log.error(f"  ✗ Failed to ingest {doc_path.name}: {e}")

    if all_chunks:
        # Batch insert (Chroma handles embedding via the embedding_function)
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            collection.add(
                documents=all_chunks[i:i+batch_size],
                ids=all_ids[i:i+batch_size],
                metadatas=all_metadata[i:i+batch_size],
            )
        log.info(f"✓ Ingested {len(all_chunks)} total chunks into vector store")

    return collection


# ── Initialise collection on startup ────────────────────────────────────
collection = None

@app.on_event("startup")
async def startup_event():
    global collection
    log.info("═" * 60)
    log.info("RCSA Intelligence Platform — Starting up")
    log.info("═" * 60)
    collection = ingest_documents()
    log.info("✓ Ready to serve requests")


# ── Pydantic models ─────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    business_unit: Optional[str] = "Retail Banking"


class RetrievedChunk(BaseModel):
    source: str
    control_id: str
    doc_type: str
    snippet: str
    similarity: float


class ChatResponse(BaseModel):
    response: str
    retrieved_chunks: List[RetrievedChunk]
    model: str
    timestamp: str


# ── System prompt ───────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert RCSA (Risk and Control Self-Assessment) AI Agent for a large global retail bank. You conduct structured risk interviews with control owners as part of the quarterly RCSA cycle.

You have access to the bank's full library of control documentation via semantic retrieval — control narratives, performance evaluations, design testing evidence, owner interview transcripts. Each user message triggers a search across this knowledge base and the relevant chunks are injected below.

INTERVIEW GUIDELINES:
1. Conduct a professional, structured RCSA interview — adaptive, not scripted
2. Ask one focused question at a time, with intelligent follow-up
3. When you identify risks, flag them explicitly with severity (HIGH/MEDIUM/LOW) and the specific regulatory citation
4. Reference specific incidents, control IDs, regulations, and dates from the retrieved knowledge
5. After 6-8 exchanges, produce a structured assessment summary with inherent risk, residual risk, and 3 priority actions
6. Keep responses concise — max 3-4 sentences per turn unless producing a summary
7. Be professional and respectful — you are assisting the risk owner, not interrogating

OUTPUT FORMAT:
- For risk flags, use: **Risk flagged — [HIGH/MEDIUM/LOW]** with regulatory citation
- For final summaries, structure as: Inherent Risk Score (x/25), Residual Risk Score (x/25), Top 3 Priority Actions
"""


# ── Routes ──────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "RCSA Intelligence Platform",
        "status": "operational",
        "model": CHAT_MODEL,
        "documents_indexed": collection.count() if collection else 0,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.get("/knowledge_base")
async def list_kb():
    """List all documents currently in the knowledge base."""
    if not collection:
        raise HTTPException(503, "Knowledge base not initialised")

    # Get unique sources
    all_items = collection.get(include=["metadatas"])
    sources = {}
    for meta in all_items["metadatas"]:
        src = meta["source"]
        if src not in sources:
            sources[src] = {
                "source": src,
                "control_id": meta["control_id"],
                "doc_type": meta["doc_type"],
                "chunk_count": 0,
            }
        sources[src]["chunk_count"] += 1

    return {
        "total_documents": len(sources),
        "total_chunks": len(all_items["metadatas"]),
        "documents": sorted(sources.values(), key=lambda x: x["source"]),
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main RAG endpoint — retrieve relevant chunks + generate response."""
    if not collection:
        raise HTTPException(503, "Knowledge base not initialised")

    # Get the latest user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(400, "No user message provided")
    latest_query = user_messages[-1].content

    # ── 1. Retrieve relevant chunks ──
    try:
        results = collection.query(
            query_texts=[latest_query],
            n_results=TOP_K_RETRIEVAL,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        log.error(f"Retrieval failed: {e}")
        raise HTTPException(500, f"Retrieval error: {str(e)}")

    retrieved_chunks = []
    context_block = ""
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = max(0.0, 1.0 - dist)  # cosine distance → similarity
            retrieved_chunks.append(RetrievedChunk(
                source=meta["source"],
                control_id=meta["control_id"],
                doc_type=meta["doc_type"],
                snippet=doc[:300] + ("..." if len(doc) > 300 else ""),
                similarity=round(similarity, 3),
            ))
            context_block += f"\n--- From {meta['source']} ({meta['doc_type']}) ---\n{doc}\n"

    # ── 2. Build LLM messages ──
    llm_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": f"RETRIEVED KNOWLEDGE BASE CONTEXT:\n{context_block}\n\n--- END RETRIEVED CONTEXT ---"},
    ]
    # Include conversation history
    for msg in request.messages:
        llm_messages.append({"role": msg.role, "content": msg.content})

    # ── 3. Generate response ──
    try:
        completion = openai_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=llm_messages,
            max_tokens=600,
            temperature=0.7,
        )
        response_text = completion.choices[0].message.content
    except Exception as e:
        log.error(f"LLM call failed: {e}")
        raise HTTPException(500, f"LLM error: {str(e)}")

    log.info(f"Chat · {len(retrieved_chunks)} chunks retrieved · query: {latest_query[:80]}...")

    return ChatResponse(
        response=response_text,
        retrieved_chunks=retrieved_chunks,
        model=CHAT_MODEL,
        timestamp=datetime.utcnow().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
