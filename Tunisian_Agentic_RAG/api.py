from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import threading

from rag_system import MultilingualRAGSystem

app = FastAPI(title="Multilingual RAG Service")

# Global RAG system instance and readiness flag
_rag = None
_rag_lock = threading.Lock()
_ready = False


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict]
    scraped_content: List[Dict]
    relevant_history: List[Dict]


def _initialize_rag(persist_directory: str = "chromadb_data", collection_name: str = "wie_rag_collection"):
    global _rag, _ready
    with _rag_lock:
        if _rag is None:
            _rag = MultilingualRAGSystem(persist_directory=persist_directory, collection_name=collection_name)
            # ensure collection is loaded (lazy load behavior as in main.py)
            stats = _rag.get_stats()
            total_docs = stats['vector_store'].get('total_documents', 0)
            if total_docs == 0:
                # attempt to load default PDF path if available
                try:
                    _rag.load_pdf(_rag_system_pdf_path())
                except Exception:
                    pass
            _ready = True


def _rag_system_pdf_path() -> str:
    # reuse the same default path as main.py if present
    # In container, this is mounted at /app/data
    import os
    default_paths = [
        "/app/data/wie_rag.pdf",  # In container (bind mount at /app/data)
        "/home/ahmed-bensalah/sight/wie_rag/data/wie_rag.pdf",  # Local dev
        "/wie_rag/data/wie_rag.pdf"  # Legacy container path
    ]
    for path in default_paths:
        if os.path.exists(path):
            return path
    return default_paths[0]  # Return first as fallback


@app.on_event("startup")
def startup_event():
    # Do not block startup — leave initialization to the /initialize endpoint or background
    pass


@app.get("/health")
def health():
    return {"status": "ready" if _ready else "initializing"}


@app.post("/initialize")
def initialize():
    """Synchronous initialization endpoint (useful for testing)."""
    global _ready
    if _ready:
        return {"status": "already_ready"}

    try:
        _initialize_rag()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"status": "initialized"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    global _rag, _ready
    if not _ready or _rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized. POST /initialize first.")

    # Only accept a single `query` field per request (simple payload)
    print(f"\n📡 API /query called: {req.query}")
    result = _rag.query(req.query, n_results=5, use_memory=True, scrape_urls=False)

    # Return structured response
    response = QueryResponse(
        answer=result.get('answer', ''),
        sources=result.get('sources', []),
        scraped_content=result.get('scraped_content', []),
        relevant_history=result.get('relevant_history', [])
    )
    
    print(f"📤 API /query response: answer length={len(response.answer)}")
    return response


@app.get("/stats")
def stats():
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return _rag.get_stats()
