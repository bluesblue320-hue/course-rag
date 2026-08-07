"""Expose semantic retrieval, document management, and RAG answering."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.database.config import (
    resolve_database_url_for_backend,
    resolve_vector_store_backend,
)
from src.documents import DocumentRecord
from src.embedding import EmbeddingService
from src.exceptions import (
    BuiltinDocumentDeletionError,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseOperationError,
    DatabaseSchemaError,
    DocumentIngestionError,
    DocumentMetadataError,
    DocumentNotFoundError,
    DocumentParseError,
    EmptyDocumentError,
    GenerationConfigurationError,
    GenerationError,
    RagConfigurationError,
    RagError,
    UnsupportedDocumentTypeError,
    UploadConfigurationError,
    UploadTooLargeError,
)
from src.generation import GenerationService
from src.ingestion_service import (
    DEFAULT_MAX_UPLOAD_BYTES,
    resolve_max_upload_bytes,
)
from src.prompt_builder import PromptBuilder
from src.rag_service import (
    RagService,
    resolve_min_relevance_score,
)
from src.reranker import (
    RerankerConfig,
    requested_reranker_enabled,
    resolve_reranker_config,
    safe_reranker_model_display_name,
)
from src.retrieval_service import RetrievalService, build_reranker
from src.storage.runtime import (
    StorageRuntime,
    build_memory_runtime,
    build_pgvector_runtime,
)

#: Public reranker readiness states exposed by /health.
RerankerStatus = Literal["disabled", "ready", "load_failed", "config_invalid"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
METADATA_PATH = RUNTIME_DIR / "documents.json"


def create_app(
    *,
    embedding_service_factory: Callable[[], Any] | None = None,
    generation_service_factory: Callable[[], Any] | None = None,
    reranker_builder: Callable[..., Any] | None = None,
) -> FastAPI:
    """Return the application with optional test dependency injection.

    When a factory is not provided, the module-level default class is
    resolved at application startup, which keeps the existing monkeypatch
    based test suite working and keeps ``uvicorn src.api:app`` unchanged.
    """
    app.state.embedding_service_factory = embedding_service_factory
    app.state.generation_service_factory = generation_service_factory
    app.state.reranker_builder = reranker_builder
    return app


def _build_reranker_config(
    final_top_k: int,
) -> tuple[bool, RerankerConfig | None, RerankerStatus]:
    """Resolve reranker configuration without loading any model.

    Returns ``(enabled, config, status)``; the status reports
    ``config_invalid`` when the enable flag or config cannot be parsed.
    """
    requested_enable = requested_reranker_enabled()
    if requested_enable is None:
        return False, None, "config_invalid"
    try:
        reranker_config = resolve_reranker_config(final_top_k=final_top_k)
    except (ValueError, RagError):
        return requested_enable, None, "config_invalid"
    if not reranker_config.enabled:
        return False, None, "disabled"
    return True, reranker_config, "ready"


def initialize_search(app: FastAPI) -> None:
    """Build the selected storage runtime and optional RAG dependencies."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    try:
        max_upload_bytes = resolve_max_upload_bytes()
    except UploadConfigurationError as exc:
        app.state.upload_configuration_error = exc
        max_upload_bytes = DEFAULT_MAX_UPLOAD_BYTES
    else:
        app.state.upload_configuration_error = None

    embedding_factory = (
        getattr(app.state, "embedding_service_factory", None)
        or EmbeddingService
    )
    embedding_service = embedding_factory()
    app.state.embedding_service = embedding_service
    app.state.model_name = embedding_service.model_name

    # Assemble the selected storage runtime.  Any database failure keeps the
    # ASGI app alive in a degraded state: /health still answers, storage
    # endpoints return stable 503s, and memory mode never touches the
    # database at all.
    backend: Literal["memory", "pgvector"] = "memory"
    storage_runtime: StorageRuntime | None = None
    storage_error: RagError | None = None
    try:
        backend = resolve_vector_store_backend()
        if backend == "memory":
            storage_runtime = build_memory_runtime(
                embedding_service=embedding_service,
                max_upload_bytes=max_upload_bytes,
                knowledge_path=KNOWLEDGE_PATH,
                upload_dir=UPLOAD_DIR,
                metadata_path=METADATA_PATH,
            )
        else:
            database_url = resolve_database_url_for_backend("pgvector")
            assert database_url is not None
            storage_runtime = build_pgvector_runtime(
                embedding_service=embedding_service,
                max_upload_bytes=max_upload_bytes,
                database_url=database_url,
                knowledge_path=KNOWLEDGE_PATH,
                upload_dir=UPLOAD_DIR,
                script_location=PROJECT_ROOT / "migrations",
                alembic_config_path=PROJECT_ROOT / "alembic.ini",
            )
    except (
        DatabaseConfigurationError,
        DatabaseConnectionError,
        DatabaseSchemaError,
        DatabaseOperationError,
    ) as exc:
        storage_error = exc

    app.state.storage_backend = backend
    app.state.storage_error = storage_error
    app.state.storage_runtime = storage_runtime
    document_manager = (
        storage_runtime.document_manager if storage_runtime is not None else None
    )
    retriever = storage_runtime.retriever if storage_runtime is not None else None
    app.state.document_manager = document_manager
    app.state.retriever = retriever
    app.state.ingestion_service = document_manager
    app.state.retrieval_ready = storage_runtime is not None
    app.state.chunk_count = (
        document_manager.chunk_count if document_manager is not None else 0
    )
    # Compatibility: memory mode keeps exposing the index object; new API code
    # must use document_manager / retriever instead of this attribute.
    app.state.knowledge_index = retriever if backend == "memory" else None

    # Build the optional reranker (disabled by default; safe degradation on
    # failure).  Only possible when the storage runtime is ready.
    reranker_instance = None
    retrieval_service: RetrievalService | None = None
    reranker_enabled = False
    reranker_ready = False
    reranker_model_name: str | None = None
    reranker_status: RerankerStatus = "disabled"

    if storage_runtime is not None:
        requested_enable, reranker_config, reranker_status = _build_reranker_config(
            final_top_k=3,
        )
        reranker_enabled = bool(requested_enable)
        if reranker_config is not None and reranker_config.enabled:
            reranker_model_name = reranker_config.model_name or None
            reranker_builder = (
                getattr(app.state, "reranker_builder", None) or build_reranker
            )
            try:
                reranker_instance = reranker_builder(reranker_config)
                retrieval_service = RetrievalService(
                    retriever=retriever,
                    config=reranker_config,
                    reranker=reranker_instance,
                )
                reranker_ready = retrieval_service.reranker_ready
                reranker_status = "ready" if reranker_ready else "load_failed"
            except Exception:
                # Model present in config but could not be loaded (missing,
                # corrupt, offline-only miss, etc.).  Keep vector-only path.
                reranker_status = "load_failed"

    app.state.retrieval_service = retrieval_service
    app.state.reranker_enabled = reranker_enabled
    app.state.reranker_ready = reranker_ready
    app.state.reranker_model = reranker_model_name
    app.state.reranker_status = reranker_status

    app.state.generation_service = None
    app.state.rag_service = None
    app.state.llm_model_name = None
    app.state.generation_ready = False
    app.state.rag_ready = False
    app.state.generation_configuration_error = None
    app.state.rag_configuration_error = None
    app.state.min_relevance_score = None

    generation_factory = (
        getattr(app.state, "generation_service_factory", None)
        or GenerationService
    )
    try:
        generation_service = generation_factory()
    except GenerationConfigurationError as exc:
        app.state.generation_configuration_error = exc
    else:
        app.state.generation_service = generation_service
        app.state.llm_model_name = generation_service.model_name
        app.state.generation_ready = True
        if storage_runtime is None:
            return
        try:
            min_relevance_score = resolve_min_relevance_score()
            rag_service = RagService(
                embedding_service=embedding_service,
                retriever=retriever,
                prompt_builder=PromptBuilder(),
                generation_service=generation_service,
                min_relevance_score=min_relevance_score,
                retrieval_service=retrieval_service,
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
        storage_runtime = getattr(app.state, "storage_runtime", None)
        if storage_runtime is not None:
            storage_runtime.close()


class HealthResponse(BaseModel):
    """Report retrieval and optional generation readiness."""

    status: str
    chunk_count: int
    retrieval_ready: bool
    generation_ready: bool
    rag_ready: bool
    min_relevance_score: float | None
    reranker_enabled: bool = False
    reranker_ready: bool = False
    reranker_model: str | None = None
    reranker_status: RerankerStatus = "disabled"


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
    """Describe one ranked source chunk with document metadata."""

    rank: int
    score: float
    text: str
    chunk_index: int
    document_id: str
    filename: str
    page_number: int | None
    retrieval_rank: int | None = None
    rerank_score: float | None = None
    reranker_applied: bool = False


class SearchResponse(BaseModel):
    """Return the cleaned query and ranked source chunks."""

    query: str
    elapsed_ms: float
    indexed_chunks: int
    model: str
    results: list[SearchResultResponse]
    # Whether this request actually used the reranker (or fell back).
    reranker_applied: bool = False
    reranker_fallback: bool = False


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
    document_id: str
    filename: str
    page_number: int | None
    retrieval_rank: int | None = None
    rerank_score: float | None = None
    reranker_applied: bool = False


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
    # Whether this request actually used the reranker (or fell back).
    reranker_applied: bool = False
    reranker_fallback: bool = False


class DocumentResponse(BaseModel):
    """Describe one managed knowledge document."""

    document_id: str
    filename: str
    content_type: str
    size_bytes: int
    text_length: int
    chunk_count: int
    created_at: str
    is_builtin: bool
    index_status: Literal["ready"] = "ready"


class DocumentListResponse(BaseModel):
    """Return the managed document list and index totals."""

    documents: list[DocumentResponse]
    document_count: int
    chunk_count: int


class DeleteDocumentResponse(BaseModel):
    """Report one successful document deletion."""

    document_id: str
    deleted: bool
    document_count: int
    chunk_count: int


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


@app.exception_handler(UploadConfigurationError)
async def handle_upload_configuration_error(
    _request: Request,
    _error: UploadConfigurationError,
) -> JSONResponse:
    """Report an invalid upload limit without exposing raw values."""
    return JSONResponse(
        status_code=503,
        content={
            "code": "UPLOAD_NOT_CONFIGURED",
            "message": "文档上传配置无效",
        },
    )


@app.exception_handler(DatabaseConfigurationError)
async def handle_database_configuration_error(
    _request: Request,
    _error: DatabaseConfigurationError,
) -> JSONResponse:
    """Report invalid storage configuration without exposing values."""
    return JSONResponse(
        status_code=503,
        content={
            "code": "STORAGE_NOT_CONFIGURED",
            "message": "存储服务配置无效",
        },
    )


@app.exception_handler(DatabaseConnectionError)
async def handle_database_connection_error(
    _request: Request,
    _error: DatabaseConnectionError,
) -> JSONResponse:
    """Report an unreachable storage backend with a stable message."""
    return JSONResponse(
        status_code=503,
        content={
            "code": "STORAGE_UNAVAILABLE",
            "message": "存储服务暂时不可用",
        },
    )


@app.exception_handler(DatabaseSchemaError)
async def handle_database_schema_error(
    _request: Request,
    _error: DatabaseSchemaError,
) -> JSONResponse:
    """Report a not-yet-migrated database schema."""
    return JSONResponse(
        status_code=503,
        content={
            "code": "STORAGE_SCHEMA_NOT_READY",
            "message": "数据库结构尚未准备完成",
        },
    )


@app.exception_handler(DatabaseOperationError)
async def handle_database_operation_error(
    _request: Request,
    _error: DatabaseOperationError,
) -> JSONResponse:
    """Report a failed database operation with a stable message."""
    return JSONResponse(
        status_code=503,
        content={
            "code": "STORAGE_UNAVAILABLE",
            "message": "存储服务暂时不可用",
        },
    )


@app.exception_handler(UnsupportedDocumentTypeError)
async def handle_unsupported_document_type(
    _request: Request,
    _error: UnsupportedDocumentTypeError,
) -> JSONResponse:
    """Reject unsupported upload types with a stable public message."""
    return JSONResponse(
        status_code=415,
        content={
            "code": "UNSUPPORTED_DOCUMENT_TYPE",
            "message": "仅支持 .txt、.md 和文本型 .pdf 文件",
        },
    )


@app.exception_handler(EmptyDocumentError)
async def handle_empty_document(
    _request: Request,
    _error: EmptyDocumentError,
) -> JSONResponse:
    """Reject empty uploads with a stable public message."""
    return JSONResponse(
        status_code=422,
        content={
            "code": "EMPTY_DOCUMENT",
            "message": "文档内容为空",
        },
    )


@app.exception_handler(DocumentParseError)
async def handle_document_parse_error(
    _request: Request,
    _error: DocumentParseError,
) -> JSONResponse:
    """Report unparseable uploads without exposing internal details."""
    return JSONResponse(
        status_code=422,
        content={
            "code": "DOCUMENT_PARSE_FAILED",
            "message": "文档解析失败，请确认文件内容有效",
        },
    )


@app.exception_handler(UploadTooLargeError)
async def handle_upload_too_large(
    _request: Request,
    _error: UploadTooLargeError,
) -> JSONResponse:
    """Reject oversized uploads with a stable public message."""
    return JSONResponse(
        status_code=413,
        content={
            "code": "UPLOAD_TOO_LARGE",
            "message": "文件超过上传大小限制",
        },
    )


@app.exception_handler(DocumentNotFoundError)
async def handle_document_not_found(
    _request: Request,
    _error: DocumentNotFoundError,
) -> JSONResponse:
    """Report a missing document id."""
    return JSONResponse(
        status_code=404,
        content={
            "code": "DOCUMENT_NOT_FOUND",
            "message": "文档不存在",
        },
    )


@app.exception_handler(BuiltinDocumentDeletionError)
async def handle_builtin_document_deletion(
    _request: Request,
    _error: BuiltinDocumentDeletionError,
) -> JSONResponse:
    """Reject deletion of the built-in knowledge document."""
    return JSONResponse(
        status_code=409,
        content={
            "code": "BUILTIN_DOCUMENT_CANNOT_BE_DELETED",
            "message": "内置文档不能删除",
        },
    )


@app.exception_handler(DocumentMetadataError)
async def handle_document_metadata_error(
    _request: Request,
    _error: DocumentMetadataError,
) -> JSONResponse:
    """Report metadata store failures without exposing internal details."""
    return JSONResponse(
        status_code=500,
        content={
            "code": "DOCUMENT_INGESTION_FAILED",
            "message": "文档处理失败，请稍后重试",
        },
    )


@app.exception_handler(DocumentIngestionError)
async def handle_document_ingestion_error(
    _request: Request,
    _error: DocumentIngestionError,
) -> JSONResponse:
    """Report failed ingestion or deletion without exposing internal paths."""
    return JSONResponse(
        status_code=500,
        content={
            "code": "DOCUMENT_INGESTION_FAILED",
            "message": "文档处理失败，请稍后重试",
        },
    )


def _raise_storage_error(app: FastAPI) -> None:
    """Raise the startup storage failure so handlers return a stable 503."""
    storage_error = getattr(app.state, "storage_error", None)
    if storage_error is not None:
        raise storage_error


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return readiness for retrieval and answer generation."""
    # The model name is redacted through safe_reranker_model_display_name so a
    # local absolute path in the configuration never leaks into the payload.
    raw_reranker_model = getattr(request.app.state, "reranker_model", None)
    storage_error = getattr(request.app.state, "storage_error", None)
    return HealthResponse(
        status="degraded" if storage_error is not None else "ok",
        chunk_count=request.app.state.chunk_count,
        retrieval_ready=request.app.state.retrieval_ready,
        generation_ready=request.app.state.generation_ready,
        rag_ready=request.app.state.rag_ready,
        min_relevance_score=request.app.state.min_relevance_score,
        reranker_enabled=getattr(request.app.state, "reranker_enabled", False),
        reranker_ready=getattr(request.app.state, "reranker_ready", False),
        reranker_model=safe_reranker_model_display_name(raw_reranker_model),
        reranker_status=getattr(request.app.state, "reranker_status", "disabled"),
    )


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Encode one query and return its most relevant source chunks."""
    _raise_storage_error(request.app)
    started_at = perf_counter()
    query_embedding = request.app.state.embedding_service.encode_query(
        payload.query
    )
    retrieval_service = getattr(request.app.state, "retrieval_service", None)
    reranker_applied = False
    reranker_fallback = False
    if retrieval_service is not None:
        outcome = retrieval_service.retrieve(
            query_embedding,
            payload.query,
            final_top_k=payload.top_k,
        )
        reranker_applied = outcome.reranker_applied
        reranker_fallback = outcome.reranker_fallback
        results = [
            {
                "rank": chunk.final_rank,
                "score": chunk.retrieval_score,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "document_id": chunk.document_id,
                "filename": chunk.filename,
                "page_number": chunk.page_number,
                "retrieval_rank": chunk.retrieval_rank,
                "rerank_score": chunk.rerank_score,
                "reranker_applied": outcome.reranker_applied,
            }
            for chunk in outcome.sources
        ]
    else:
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
        reranker_applied=reranker_applied,
        reranker_fallback=reranker_fallback,
    )


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Delegate one validated question to the initialized RAG pipeline."""
    _raise_storage_error(request.app)
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


def _document_response(record: DocumentRecord) -> DocumentResponse:
    return DocumentResponse(
        document_id=record.document_id,
        filename=record.original_filename,
        content_type=record.content_type,
        size_bytes=record.size_bytes,
        text_length=record.text_length,
        chunk_count=record.chunk_count,
        created_at=record.created_at,
        is_builtin=record.is_builtin,
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request) -> DocumentListResponse:
    """Return the unified document list and current index totals."""
    _raise_storage_error(request.app)
    document_manager = request.app.state.document_manager
    documents = document_manager.list_documents()
    return DocumentListResponse(
        documents=[_document_response(record) for record in documents],
        document_count=len(documents),
        chunk_count=document_manager.chunk_count,
    )


def read_upload_with_limit(
    upload: UploadFile,
    max_bytes: int,
) -> bytes:
    """Read at most max_bytes + 1 bytes so oversized uploads stay bounded."""
    data = upload.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise UploadTooLargeError("文件超过上传大小限制")
    return data


@app.post(
    "/documents",
    response_model=DocumentResponse,
    status_code=201,
)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
) -> DocumentResponse:
    """Validate, store, and index one uploaded knowledge document."""
    _raise_storage_error(request.app)
    upload_configuration_error = request.app.state.upload_configuration_error
    if upload_configuration_error is not None:
        raise upload_configuration_error

    document_manager = request.app.state.document_manager
    data = read_upload_with_limit(
        file,
        document_manager.max_upload_bytes,
    )
    record = document_manager.ingest(
        file.filename or "",
        file.content_type or "",
        data,
    )
    request.app.state.chunk_count = document_manager.chunk_count
    return _document_response(record)


@app.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
)
def delete_document(
    document_id: str,
    request: Request,
) -> DeleteDocumentResponse:
    """Delete one uploaded document and its stored chunks."""
    _raise_storage_error(request.app)
    document_manager = request.app.state.document_manager
    document_manager.delete_document(document_id)
    chunk_count = document_manager.chunk_count
    request.app.state.chunk_count = chunk_count
    return DeleteDocumentResponse(
        document_id=document_id,
        deleted=True,
        document_count=len(document_manager.list_documents()),
        chunk_count=chunk_count,
    )
