"""Expose semantic retrieval through a minimal FastAPI application."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.chunker import split_text
from src.embedding import EmbeddingService
from src.exceptions import (
    GenerationConfigurationError,
    GenerationError,
    RagConfigurationError,
)
from src.generation import GenerationService
from src.loader import load_text
from src.prompt_builder import PromptBuilder
from src.rag_service import (
    RagService,
    resolve_min_relevance_score,
)
from src.retriever import SemanticRetriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"


def initialize_search(app: FastAPI) -> None:
    """Build retrieval dependencies and optionally enable answer generation."""
    text = load_text(str(KNOWLEDGE_PATH))
    chunks = split_text(text)
    embedding_service = EmbeddingService()
    document_embeddings = embedding_service.encode_documents(chunks)
    retriever = SemanticRetriever(chunks, document_embeddings)
    prompt_builder = PromptBuilder()

    app.state.embedding_service = embedding_service
    app.state.retriever = retriever
    app.state.chunk_count = len(chunks)
    app.state.model_name = embedding_service.model_name
    app.state.retrieval_ready = True

    app.state.generation_service = None
    app.state.rag_service = None
    app.state.llm_model_name = None
    app.state.generation_ready = False
    app.state.rag_ready = False
    app.state.generation_configuration_error = None
    app.state.rag_configuration_error = None
    app.state.min_relevance_score = None

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    try:
        generation_service = GenerationService()
    except GenerationConfigurationError as exc:
        app.state.generation_configuration_error = exc
    else:
        app.state.generation_service = generation_service
        app.state.llm_model_name = generation_service.model_name
        app.state.generation_ready = True
        try:
            min_relevance_score = resolve_min_relevance_score()
            rag_service = RagService(
                embedding_service=embedding_service,
                retriever=retriever,
                prompt_builder=prompt_builder,
                generation_service=generation_service,
                min_relevance_score=min_relevance_score,
            )
        except RagConfigurationError as exc:
            app.state.rag_configuration_error = exc
            return
        app.state.rag_service = rag_service
        app.state.rag_ready = True
        app.state.min_relevance_score = min_relevance_score


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize shared services and close owned resources on shutdown."""
    try:
        initialize_search(app)
        yield
    finally:
        generation_service = getattr(
            app.state,
            "generation_service",
            None,
        )
        if generation_service is not None:
            generation_service.close()


class HealthResponse(BaseModel):
    """Report retrieval and optional generation readiness."""

    status: str
    chunk_count: int
    retrieval_ready: bool
    generation_ready: bool
    rag_ready: bool
    min_relevance_score: float | None


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


class AskRequest(BaseModel):
    """Describe one validated retrieval-augmented question."""

    question: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=3, ge=1, le=10, strict=True)

    @field_validator("question", mode="before")
    @classmethod
    def clean_question(cls, value: object) -> object:
        """Trim the question and reject blank text."""
        if not isinstance(value, str):
            return value
        cleaned_question = value.strip()
        if not cleaned_question:
            raise ValueError("question 不能为空")
        return cleaned_question


class AskSourceResponse(BaseModel):
    """Describe one source used to ground an answer."""

    rank: int
    score: float
    text: str
    chunk_index: int


class AskResponse(BaseModel):
    """Return one generated answer, its sources, models, and timings."""

    question: str
    answer: str
    answer_status: Literal["answered", "insufficient_context"]
    max_relevance_score: float | None
    relevance_threshold: float
    retrieval_elapsed_ms: float
    generation_elapsed_ms: float
    total_elapsed_ms: float
    embedding_model: str
    llm_model: str
    sources: list[AskSourceResponse]


app = FastAPI(
    title="Course RAG Retrieval API",
    lifespan=lifespan,
)


@app.exception_handler(GenerationError)
async def handle_generation_error(
    _request: Request,
    _error: GenerationError,
) -> JSONResponse:
    """Hide provider errors behind a stable public response."""
    return JSONResponse(
        status_code=502,
        content={
            "code": "GENERATION_FAILED",
            "message": "回答生成失败，请稍后重试",
        },
    )


@app.exception_handler(GenerationConfigurationError)
async def handle_generation_configuration_error(
    _request: Request,
    _error: GenerationConfigurationError,
) -> JSONResponse:
    """Report unavailable LLM configuration without exposing secrets."""
    return JSONResponse(
        status_code=503,
        content={
            "code": "LLM_NOT_CONFIGURED",
            "message": "问答服务尚未完成配置",
        },
    )


@app.exception_handler(RagConfigurationError)
async def handle_rag_configuration_error(
    _request: Request,
    _error: RagConfigurationError,
) -> JSONResponse:
    """Report invalid RAG configuration without exposing raw values."""
    return JSONResponse(
        status_code=503,
        content={
            "code": "RAG_NOT_CONFIGURED",
            "message": "问答相关性配置无效",
        },
    )


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return readiness for retrieval and answer generation."""
    return HealthResponse(
        status="ok",
        chunk_count=request.app.state.chunk_count,
        retrieval_ready=request.app.state.retrieval_ready,
        generation_ready=request.app.state.generation_ready,
        rag_ready=request.app.state.rag_ready,
        min_relevance_score=request.app.state.min_relevance_score,
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


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Delegate one validated question to the initialized RAG pipeline."""
    rag_service = request.app.state.rag_service
    if rag_service is None:
        rag_configuration_error = request.app.state.rag_configuration_error
        if rag_configuration_error is not None:
            raise rag_configuration_error
        raise GenerationConfigurationError("问答服务尚未完成配置")

    result = rag_service.answer(
        payload.question,
        top_k=payload.top_k,
    )
    return AskResponse.model_validate(
        {
            **result,
            "embedding_model": request.app.state.model_name,
            "llm_model": request.app.state.llm_model_name,
        }
    )
