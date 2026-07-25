"""Expose semantic retrieval through a minimal FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator

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
    app.state.model_name = embedding_service.model_name


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize the search index before accepting requests."""
    initialize_search(app)
    yield


class HealthResponse(BaseModel):
    """Report that the API and its semantic index are ready."""

    status: str
    chunk_count: int


class SearchRequest(BaseModel):
    """Describe one validated semantic retrieval request."""

    query: str
    top_k: int = Field(default=3, gt=0, strict=True)

    @field_validator("query")
    @classmethod
    def clean_query(cls, value: str) -> str:
        """Trim the query and reject blank text."""
        cleaned_query = value.strip()
        if not cleaned_query:
            raise ValueError("query 不能为空")
        return cleaned_query


class SearchResultResponse(BaseModel):
    """Describe one ranked source chunk."""

    rank: int
    score: float
    text: str
    chunk_index: int


class SearchResponse(BaseModel):
    """Return the cleaned query and ranked source chunks."""

    query: str
    elapsed_ms: float
    indexed_chunks: int
    model: str
    results: list[SearchResultResponse]


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


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Encode one query and return its most relevant source chunks."""
    started_at = perf_counter()
    query_embedding = request.app.state.embedding_service.encode_query(
        payload.query
    )
    results = request.app.state.retriever.search(
        query_embedding,
        top_k=payload.top_k,
    )
    elapsed_ms = round((perf_counter() - started_at) * 1000, 2)
    return SearchResponse(
        query=payload.query,
        elapsed_ms=elapsed_ms,
        indexed_chunks=request.app.state.chunk_count,
        model=request.app.state.model_name,
        results=[SearchResultResponse(**result) for result in results],
    )
