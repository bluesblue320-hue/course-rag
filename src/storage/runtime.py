"""Selectable storage runtime assembly for memory and pgvector backends.

Importing this module never connects to a database and never creates an
engine or session.  All engines are created explicitly inside
:func:`build_pgvector_runtime` and disposed either on build failure or when
the returned :class:`StorageRuntime` is closed.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal

from src.database.engine import (
    check_database_connection,
    check_database_schema,
    create_database_engine,
    create_session_factory,
)
from src.document_chunker import chunk_document
from src.document_loaders import (
    DocumentLoader,
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TextDocumentLoader,
)
from src.document_repository import DocumentRepository
from src.documents import DocumentRecord
from src.ingestion_service import (
    DEFAULT_MAX_UPLOAD_BYTES,
    IngestionService,
)
from src.knowledge_index import KnowledgeIndex
from src.pgvector_ingestion_service import PgVectorIngestionService
from src.storage.pgvector_store import PgVectorStore
from src.storage.protocols import (
    ChunkRetrieverProtocol,
    DocumentManagerProtocol,
)

logger = logging.getLogger(__name__)

#: Document id reserved for the built-in knowledge document in both backends.
BUILTIN_DOCUMENT_ID = "builtin-knowledge"
BUILTIN_FILENAME = "knowledge.txt"


def _file_mtime_iso(path: Path) -> str:
    """Return a file's modification time as a UTC ISO 8601 string."""
    modified_at = datetime.fromtimestamp(
        path.stat().st_mtime,
        tz=timezone.utc,
    )
    return modified_at.isoformat(timespec="seconds").replace("+00:00", "Z")


def make_builtin_document(knowledge_path: Path) -> DocumentRecord:
    """Build the built-in document record from the knowledge text file."""
    loader = TextDocumentLoader()
    builtin_loaded = loader.load(knowledge_path)
    return DocumentRecord(
        document_id=BUILTIN_DOCUMENT_ID,
        original_filename=BUILTIN_FILENAME,
        stored_filename=None,
        content_type="text/plain",
        size_bytes=knowledge_path.stat().st_size,
        text_length=builtin_loaded.text_length,
        chunk_count=len(
            chunk_document(
                builtin_loaded,
                BUILTIN_DOCUMENT_ID,
                BUILTIN_FILENAME,
            )
        ),
        created_at=_file_mtime_iso(knowledge_path),
        is_builtin=True,
    )


def _standard_loaders() -> dict[str, DocumentLoader]:
    """Return the production document loader map shared by both backends."""
    return {
        ".txt": TextDocumentLoader(),
        ".md": MarkdownDocumentLoader(),
        ".pdf": PdfDocumentLoader(),
    }


@dataclass
class StorageRuntime:
    """One assembled, closable storage runtime.

    ``close`` is idempotent and never raises: each owned component is closed
    individually and failures are logged instead of propagated.
    """

    backend: Literal["memory", "pgvector"]
    document_manager: DocumentManagerProtocol
    retriever: ChunkRetrieverProtocol
    close_callback: Callable[[], None]
    _closed: bool = False

    def close(self) -> None:
        """Release owned resources; safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        for closer in (
            self.document_manager.close,
            self.retriever.close,
            self.close_callback,
        ):
            try:
                closer()
            except Exception:
                logger.warning("存储运行时关闭失败", exc_info=True)


def build_memory_runtime(
    *,
    embedding_service: Any,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    knowledge_path: Path,
    upload_dir: Path,
    metadata_path: Path,
) -> StorageRuntime:
    """Assemble the existing JSON + in-memory runtime.

    The memory runtime never reads ``DATABASE_URL``, never creates an engine,
    and never connects to PostgreSQL.
    """
    builtin_document = make_builtin_document(knowledge_path)
    repository = DocumentRepository(metadata_path)
    index = KnowledgeIndex()
    ingestion_service = IngestionService(
        embedding_service=embedding_service,
        repository=repository,
        index=index,
        upload_dir=upload_dir,
        builtin_document=builtin_document,
        builtin_path=knowledge_path,
        loaders=_standard_loaders(),
        max_upload_bytes=max_upload_bytes,
    )
    valid_uploads = ingestion_service.reconcile_persisted_documents()
    ingestion_service.initialize_index([builtin_document, *valid_uploads])
    return StorageRuntime(
        backend="memory",
        document_manager=ingestion_service,
        retriever=index,
        close_callback=lambda: None,
    )


def build_pgvector_runtime(
    *,
    embedding_service: Any,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    database_url: str,
    knowledge_path: Path,
    upload_dir: Path,
    script_location: str | Path = "migrations",
    alembic_config_path: str | Path | None = None,
) -> StorageRuntime:
    """Assemble the PostgreSQL + pgvector runtime.

    Raises :class:`DatabaseConnectionError` / :class:`DatabaseSchemaError` /
    :class:`DatabaseOperationError` when the database is not ready.  The
    engine is disposed on any build failure so no connection pool leaks.
    """
    engine = create_database_engine(database_url)
    try:
        check_database_connection(engine)
        check_database_schema(
            engine,
            script_location=script_location,
            alembic_config_path=alembic_config_path,
        )
        session_factory = create_session_factory(engine)
        store = PgVectorStore(session_factory, engine=engine)
        builtin_document = make_builtin_document(knowledge_path)
        ingestion_service = PgVectorIngestionService(
            embedding_service=embedding_service,
            store=store,
            upload_dir=upload_dir,
            builtin_document=builtin_document,
            builtin_path=knowledge_path,
            loaders=_standard_loaders(),
            max_upload_bytes=max_upload_bytes,
        )
        ingestion_service.reconcile_tombstones()
        ingestion_service.ensure_builtin_document()
    except Exception:
        engine.dispose()
        raise
    return StorageRuntime(
        backend="pgvector",
        document_manager=ingestion_service,
        retriever=store,
        close_callback=engine.dispose,
    )
