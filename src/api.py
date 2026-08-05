"""Expose semantic retrieval, document management, and RAG answering."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.document_chunker import chunk_document
from src.document_loaders import (
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TextDocumentLoader,
)
from src.document_repository import DocumentRepository
from src.documents import DocumentRecord
from src.embedding import EmbeddingService
from src.exceptions import (
    BuiltinDocumentDeletionError,
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
    IngestionService,
    resolve_max_upload_bytes,
)
from src.knowledge_index import KnowledgeIndex
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

#: Public reranker readiness states exposed by /health.
RerankerStatus = Literal["disabled", "ready", "load_failed", "config_invalid"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = PROJECT_ROOT / "data" / "knowledge.txt"
RUNTIME_DIR = PROJECT_ROOT / "data" / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
METADATA_PATH = RUNTIME_DIR / "documents.json"
BUILTIN_DOCUMENT_ID = "builtin-knowledge"
BUILTIN_FILENAME = "knowledge.txt"


def _file_mtime_iso(path: Path) -> str:
    """Return a file's modification time as a UTC ISO 8601 string."""
    modified_at = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )
    return modified_at.isoformat(timespec="seconds").replace("+00:00", "Z")


def _build_ingestion_dependencies(
    embedding_service: EmbeddingService,
    max_upload_bytes: int,
) -> tuple[IngestionService, DocumentRecord]:
    """Build the document repository, upload store, and initial index."""
    loader = TextDocumentLoader()
    builtin_loaded = loader.load(KNOWLEDGE_PATH)
    builtin_document = DocumentRecord(
        document_id=BUILTIN_DOCUMENT_ID,
        original_filename=BUILTIN_FILENAME,
        stored_filename=None,
        content_type="text/plain",
        size_bytes=KNOWLEDGE_PATH.stat().st_size,
        text_length=builtin_loaded.text_length,
        chunk_count=len(
            chunk_document(
                builtin_loaded,
                BUILTIN_DOCUMENT_ID,
                BUILTIN_FILENAME,
            )
        ),
        created_at=_file_mtime_iso(KNOWLEDGE_PATH),
        is_builtin=True,
    )

    repository = DocumentRepository(METADATA_PATH)
    index = KnowledgeIndex()
    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        repository=repository,
        index=index,
        upload_dir=UPLOAD_DIR,
        builtin_document=builtin_document,
        builtin_path=KNOWLEDGE_PATH,
        loaders={
            ".txt": TextDocumentLoader(),
            ".md": MarkdownDocumentLoader(),
            ".pdf": PdfDocumentLoader(),
        },
        max_upload_bytes=max_upload_bytes,
    )

    valid_uploads = ingestion_service.reconcile_persisted_documents()
    ingestion_service.initialize_index([builtin_document, *valid_uploads])
    return ingestion_service, builtin_document


def initialize_search(app: FastAPI) -> None:
    """Build retrieval, document, and optional generation dependencies."""
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

    try:
        max_upload_bytes = resolve_max_upload_bytes()
    except UploadConfigurationError as exc:
        app.state.upload_configuration_error = exc
        max_upload_bytes = DEFAULT_MAX_UPLOAD_BYTES
    else:
        app.state.upload_configuration_error = None

    embedding_service = EmbeddingService()
    ingestion_service, _builtin_document = _build_ingestion_dependencies(
        embedding_service,
        max_upload_bytes,
    )
    index = ingestion_service.index

    app.state.embedding_service = embedding_service
    app.state.ingestion_service = ingestion_service
    app.state.knowledge_index = index
    app.state.chunk_count = index.chunk_count
    app.state.model_name = embedding_service.model_name
    app.state.retrieval_ready = True

    # Build optional reranker (disabled by default; safe degradation on failure)
    # Config resolution and model loading are kept separate so that a load
    # failure can be reported as `load_failed` instead of silently appearing
    # as "not enabled".  The health endpoint never leaks paths, stacks, or
    # internal exception text, and the model name is redacted when it looks
    # like a local path.
    reranker_instance = None
    retrieval_service: RetrievalService | None = None
    reranker_enabled = False
    reranker_ready = False
    reranker_model_name: str | None = None
    reranker_status: RerankerStatus = "disabled"

    # First resolve whether the user actually requested reranking.  When the
    # enable flag itself cannot be parsed (e.g. RAG_RERANKER_ENABLED=maybe),
    # the system cannot confirm a request to enable, so the public
    # `reranker_enabled` stays false while the status becomes config_invalid.
    requested_enable = requested_reranker_enabled()
    if requested_enable is None:
        reranker_status = "config_invalid"
    else:
        reranker_enabled = requested_enable
        try:
            reranker_config = resolve_reranker_config(final_top_k=3)
        except (ValueError, RagError):
            reranker_status = "config_invalid"
        else:
            reranker_model_name = reranker_config.model_name or None
            if not reranker_config.enabled:
                reranker_status = "disabled"
            else:
                try:
                    reranker_instance = build_reranker(reranker_config)
                    retrieval_service = RetrievalService(
                        retriever=index,
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
                retriever=index,
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


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return readiness for retrieval and answer generation."""
    # The model name is redacted through safe_reranker_model_display_name so a
    # local absolute path in the configuration never leaks into the payload.
    raw_reranker_model = getattr(request.app.state, "reranker_model", None)
    return HealthResponse(
        status="ok",
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
        results = request.app.state.knowledge_index.search(
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
    documents = request.app.state.ingestion_service.list_documents()
    return DocumentListResponse(
        documents=[_document_response(record) for record in documents],
        document_count=len(documents),
        chunk_count=request.app.state.knowledge_index.chunk_count,
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
    upload_configuration_error = request.app.state.upload_configuration_error
    if upload_configuration_error is not None:
        raise upload_configuration_error

    ingestion_service = request.app.state.ingestion_service
    data = read_upload_with_limit(
        file,
        ingestion_service.max_upload_bytes,
    )
    record = ingestion_service.ingest(
        file.filename or "",
        file.content_type or "",
        data,
    )
    request.app.state.chunk_count = request.app.state.knowledge_index.chunk_count
    return _document_response(record)


@app.delete(
    "/documents/{document_id}",
    response_model=DeleteDocumentResponse,
)
def delete_document(
    document_id: str,
    request: Request,
) -> DeleteDocumentResponse:
    """Delete one uploaded document and rebuild the active index."""
    request.app.state.ingestion_service.delete_document(document_id)
    chunk_count = request.app.state.knowledge_index.chunk_count
    request.app.state.chunk_count = chunk_count
    return DeleteDocumentResponse(
        document_id=document_id,
        deleted=True,
        document_count=len(
            request.app.state.ingestion_service.list_documents()
        ),
        chunk_count=chunk_count,
    )
