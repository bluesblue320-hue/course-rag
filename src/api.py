"""Expose semantic retrieval through a minimal FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from pydantic import BaseModel

from src.chunker import split_text
from src.embedding import EmbeddingService
from src.loader import load_text
from src.retriever import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"


def initialize_search(app: FastAPI) -> None:
    """Build the semantic index and store reusable objects on the app."""
    text = load_text(str(KNOWLEDGE_PATH))
    chunks = split_text(text)
    embedding_service = EmbeddingService()
    document_embeddings = embedding_service.encode_documents(chunks)
    retriever = SemanticRetriever(chunks, document_embeddings)

    app.state.embedding_service = embedding_service
    app.state.retriever = retriever
    app.state.chunk_count = len(chunks)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the search index before accepting requests."""
    initialize_search(app)
    yield


class HealthResponse(BaseModel):
    """Report that the API and its semantic index are ready."""

    status: str
    chunk_count: int


app = FastAPI(
    title="Course RAG Retrieval API",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return readiness and the number of indexed chunks."""
    return HealthResponse(
        status="ok",
        chunk_count=request.app.state.chunk_count,
    )
