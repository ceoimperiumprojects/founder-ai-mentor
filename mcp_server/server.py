from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.vector_store import VectorStore

app = FastAPI(
    title="Startup Knowledge MCP Server",
    description="Semanticko pretrazivanje startup literature",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize vector store
project_root = Path(__file__).parent.parent
db_path = project_root / "chroma_db"
vector_store = VectorStore(persist_directory=str(db_path))


class SearchRequest(BaseModel):
    query: str
    k: int = 5


class SearchResult(BaseModel):
    text: str
    source: str
    relevance_score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total_results: int


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semanticka pretraga kroz knowledge base."""
    try:
        results = vector_store.search(request.query, k=request.k)

        formatted_results = [
            SearchResult(
                text=r["text"],
                source=r["metadata"].get("source", "unknown"),
                relevance_score=round(1 - r["distance"], 4)
            )
            for r in results
        ]

        return SearchResponse(
            results=formatted_results,
            total_results=len(formatted_results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "documents_count": stats["total_documents"]
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Startup Knowledge RAG",
        "version": "1.0.0",
        "endpoints": ["/search", "/health"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
