"""pgvector-backed ingestion service: files, chunks, embeddings, and store.

This service owns the *file* half of the ``pgvector`` runtime:

* upload parameter validation (same rules as the memory path)
* safe storage of original uploaded files
* document parsing and chunk splitting
* embedding only the newly created chunks
* transactional writes through :class:`PgVectorStore`
* coordinated file + database deletion with tombstone compensation
* built-in document initialization

It never writes SQL and never talks to the database directly; every database
operation goes through the injected store.
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.document_chunker import chunk_document
from src.document_loaders import DocumentLoader
from src.documents import (
    DocumentRecord,
    sort_documents,
    utc_now_iso,
)
from src.exceptions import (
    BuiltinDocumentDeletionError,
    DatabaseOperationError,
    DocumentIngestionError,
    DocumentMetadataError,
    DocumentNotFoundError,
    DocumentParseError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
    UploadTooLargeError,
)
from src.ingestion_service import (
    DEFAULT_MAX_UPLOAD_BYTES,
    _ALLOWED_CONTENT_TYPES,
    _STORED_FILENAME_RE,
    _unlink_quietly,
    _validate_max_upload_bytes,
)
from src.storage.pgvector_store import PgVectorStore

logger = logging.getLogger(__name__)

#: Document id reserved for the built-in knowledge document in both backends.
BUILTIN_DOCUMENT_ID = "builtin-knowledge"

#: Uploaded document ids are always ``uuid4().hex`` (32 lowercase hex digits).
_DOCUMENT_ID_RE = re.compile(r"[0-9a-f]{32}")

#: Tombstones are hidden files that can never collide with the strict
#: ``<32 hex>.<suffix>`` stored-filename format and are never picked up by
#: the document loaders (which only accept .txt/.md/.pdf suffixes).
#: Every tombstone carries enough identity metadata to be reconciled against
#: the database at startup:
#: ``.tombstone-{document_id}-{stored_filename}-{token}.tmp``.
_TOMBSTONE_PREFIX = ".tombstone-"
_TOMBSTONE_SUFFIX = ".tmp"
_TOMBSTONE_RE = re.compile(
    r"\.tombstone-"
    r"(?P<document_id>[0-9a-f]{32})-"
    r"(?P<stored_filename>[0-9a-f]{32}\.(?:txt|md|pdf))-"
    r"(?P<token>[0-9a-f]{32})"
    r"\.tmp"
)


def _parse_tombstone_name(path: Path) -> tuple[str, str] | None:
    """Parse a strict tombstone filename into ``(document_id, stored_filename)``.

    Returns ``None`` when the name does not match the strict format.  The
    caller must never guess about such files, must never log the raw name,
    and must never log the original exception.
    """
    match = _TOMBSTONE_RE.fullmatch(path.name)
    if match is None:
        return None
    return match.group("document_id"), match.group("stored_filename")


class PgVectorIngestionService:
    """Coordinate uploads, deletions, and built-in initialization on pgvector."""

    def __init__(
        self,
        *,
        embedding_service: Any,
        store: PgVectorStore,
        upload_dir: Path,
        builtin_document: DocumentRecord,
        builtin_path: Path,
        loaders: dict[str, DocumentLoader],
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        self._embedding_service = embedding_service
        self._store = store
        self._upload_dir = upload_dir
        self._builtin_document = builtin_document
        self._builtin_path = builtin_path
        self._loaders = dict(loaders)
        self._max_upload_bytes = _validate_max_upload_bytes(max_upload_bytes)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # DocumentManagerProtocol surface
    # ------------------------------------------------------------------

    @property
    def max_upload_bytes(self) -> int:
        """Return the enforced upload size limit in bytes."""
        return self._max_upload_bytes

    @property
    def chunk_count(self) -> int:
        """Return the number of ready chunks persisted in PostgreSQL."""
        return self._store.chunk_count

    def list_documents(self) -> list[DocumentRecord]:
        """Return the built-in document followed by stored uploads."""
        return sort_documents(self._store.list_documents())

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one ready document by id or None when absent."""
        return self._store.get_document(document_id)

    def close(self) -> None:
        """No-op: the engine is owned and disposed by StorageRuntime."""

    # ------------------------------------------------------------------
    # Startup: tombstones and the built-in document
    # ------------------------------------------------------------------

    def reconcile_tombstones(self) -> None:
        """Reconcile leftover tombstones against the current database state.

        Called at startup.  For every parseable tombstone:

        * when the database record still exists and the original file is
          missing, the tombstone is restored to the original filename;
        * when the database record still exists and the original file is
          already present, the tombstone is a duplicate and is removed;
        * when the database record no longer exists, the deletion already
          committed and the leftover tombstone is removed;
        * malformed or inconsistent tombstones are kept for manual
          inspection.

        Regular uploaded files (which never start with the tombstone
        prefix) are ignored entirely: no parsing, no database lookup, no
        warnings, and no file mutations.

        Failures are logged with fixed messages (never with file paths or
        exception traces) and never block application startup.
        """
        if not self._upload_dir.is_dir():
            return
        for candidate in sorted(
            self._upload_dir.iterdir(),
            key=lambda path: path.name,
        ):
            # Regular uploaded files never begin with the tombstone prefix
            # and must be skipped entirely: no parsing, no database lookup,
            # no warnings, and no file mutations.
            if not candidate.name.startswith(_TOMBSTONE_PREFIX):
                continue

            parsed = _parse_tombstone_name(candidate)
            if parsed is None:
                logger.warning(
                    "发现无法安全识别的tombstone，已保留供人工检查"
                )
                continue
            document_id, stored_filename = parsed
            try:
                record = self._store.get_document(document_id)
            except DatabaseOperationError:
                logger.warning(
                    "暂时无法核对tombstone对应的数据库记录，将在下次启动重试"
                )
                continue
            if record is None:
                self._remove_tombstone(candidate)
                continue
            if record.is_builtin or record.stored_filename != stored_filename:
                logger.warning(
                    "tombstone与数据库记录不一致，已保留供人工检查"
                )
                continue
            try:
                target_path = self._resolve_stored_path(stored_filename)
            except DocumentMetadataError:
                logger.warning(
                    "tombstone与数据库记录不一致，已保留供人工检查"
                )
                continue
            if target_path.exists():
                # The database record and the original file are both intact;
                # this tombstone is a duplicate leftover of a completed delete.
                self._remove_tombstone(candidate)
            else:
                try:
                    os.replace(candidate, target_path)
                except OSError:
                    logger.warning(
                        "数据库删除失败后暂时无法恢复原文件，将在下次启动重试"
                    )

    def _remove_tombstone(self, tombstone: Path) -> None:
        """Best-effort removal of one tombstone; failures are logged only."""
        try:
            tombstone.unlink()
        except OSError:
            logger.warning(
                "暂时无法清理已完成删除的tombstone，将在下次启动重试"
            )

    def cleanup_tombstones(self) -> None:
        """Deprecated compatibility alias for :meth:`reconcile_tombstones`."""
        self.reconcile_tombstones()

    def ensure_builtin_document(self) -> None:
        """Write the built-in knowledge document once and reuse it on restarts.

        When the document already exists the method verifies its integrity
        (builtin flag and chunk count) without re-embedding anything and
        without loading any vectors into Python memory.
        """
        existing = self._store.get_document(BUILTIN_DOCUMENT_ID)
        if existing is not None:
            if not existing.is_builtin:
                raise DatabaseOperationError("数据库操作失败，请稍后重试")
            persisted_chunks = self._store.count_document_chunks(
                BUILTIN_DOCUMENT_ID
            )
            if existing.chunk_count != persisted_chunks:
                raise DatabaseOperationError("数据库操作失败，请稍后重试")
            return

        loader = self._loaders[".txt"]
        loaded = loader.load(self._builtin_path)
        chunks = chunk_document(
            loaded,
            BUILTIN_DOCUMENT_ID,
            self._builtin_document.original_filename,
        )
        if len(chunks) != self._builtin_document.chunk_count:
            raise DocumentIngestionError("内置文档切分结果不一致")
        embeddings = self._embedding_service.encode_documents(
            [chunk.text for chunk in chunks]
        )
        self._store.insert_document(
            self._builtin_document,
            chunks,
            embeddings,
        )

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def ingest(
        self,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> DocumentRecord:
        """Validate, store, parse, and persist one uploaded file.

        Only the newly created chunks are embedded, and only this document's
        metadata and chunks are written to the database.  On any failure the
        new file is removed and no partial database state remains.
        """
        original_filename = filename.strip()
        if not original_filename:
            raise UnsupportedDocumentTypeError("缺少文件名")

        suffix = Path(original_filename).suffix.lower()
        loader = self._loaders.get(suffix)
        if loader is None:
            raise UnsupportedDocumentTypeError("仅支持 .txt、.md、.pdf 文件")
        if content_type not in _ALLOWED_CONTENT_TYPES:
            raise UnsupportedDocumentTypeError("文件类型不受支持")
        if not data:
            raise EmptyDocumentError("文件内容为空")
        if len(data) > self._max_upload_bytes:
            raise UploadTooLargeError("文件超过上传大小限制")

        document_id = uuid4().hex
        stored_filename = f"{uuid4().hex}{suffix}"
        target_path = self._upload_dir / stored_filename
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        if target_path.exists():
            raise DocumentIngestionError("文件保存冲突，请重试")
        try:
            target_path.write_bytes(data)
        except OSError as exc:
            raise DocumentIngestionError("文件保存失败") from exc

        try:
            loaded = loader.load(target_path)
            new_chunks = chunk_document(
                loaded,
                document_id,
                original_filename,
            )
            embeddings = self._embedding_service.encode_documents(
                [chunk.text for chunk in new_chunks]
            )
            record = DocumentRecord(
                document_id=document_id,
                original_filename=original_filename,
                stored_filename=stored_filename,
                content_type=content_type or "application/octet-stream",
                size_bytes=len(data),
                text_length=loaded.text_length,
                chunk_count=len(new_chunks),
                created_at=utc_now_iso(),
                is_builtin=False,
            )
            with self._lock:
                self._store.insert_document(record, new_chunks, embeddings)
            return record
        except (
            EmptyDocumentError,
            DocumentParseError,
            DocumentMetadataError,
            UnsupportedDocumentTypeError,
        ):
            _unlink_quietly(target_path)
            raise
        except DatabaseOperationError:
            _unlink_quietly(target_path)
            raise
        except Exception as exc:
            _unlink_quietly(target_path)
            raise DocumentIngestionError("文档处理失败") from exc

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> DocumentRecord:
        """Delete one uploaded document and its stored chunks.

        The local file and the database cannot form one true transaction, so
        the original file is first renamed to a parseable tombstone, the
        database transaction runs, and only after it commits is the tombstone
        removed.  A failed database transaction restores the original file;
        when that immediate restore fails the tombstone is kept and
        :meth:`reconcile_tombstones` restores it on the next startup.
        """
        with self._lock:
            record = self._store.get_document(document_id)
            if record is None:
                raise DocumentNotFoundError("文档不存在")
            if record.is_builtin:
                raise BuiltinDocumentDeletionError("内置文档不能删除")

            target_path: Path | None = None
            tombstone_path: Path | None = None
            stored_filename = record.stored_filename
            if stored_filename is not None:
                target_path = self._resolve_stored_path(stored_filename)
                if target_path.is_file():
                    tombstone_path = self._new_tombstone_path(
                        document_id,
                        stored_filename,
                    )
                    try:
                        os.replace(target_path, tombstone_path)
                    except OSError as exc:
                        raise DocumentIngestionError("文档删除失败") from exc

            try:
                self._store.delete_document(document_id)
            except Exception:
                if tombstone_path is not None and target_path is not None:
                    try:
                        os.replace(tombstone_path, target_path)
                    except OSError:
                        # The database record still exists, so the next
                        # startup reconciles this tombstone back to the
                        # original file.  Never delete it here.
                        logger.warning(
                            "数据库删除失败后暂时无法恢复原文件，将在下次启动重试"
                        )
                raise

            if tombstone_path is not None:
                # The database deletion already committed; removing the
                # tombstone is best-effort.  A leftover tombstone is removed
                # by the next startup once the database record is gone.
                self._remove_tombstone(tombstone_path)
            return record

    # ------------------------------------------------------------------
    # Path safety
    # ------------------------------------------------------------------

    def _resolve_stored_path(self, stored_filename: str) -> Path:
        """Resolve one stored filename strictly inside the upload directory."""
        upload_root = self._upload_dir.resolve()
        if not _STORED_FILENAME_RE.fullmatch(stored_filename):
            raise DocumentMetadataError("文档元数据损坏，无法读取")
        candidate = (self._upload_dir / stored_filename).resolve()
        if candidate.parent != upload_root:
            raise DocumentMetadataError("文档元数据损坏，无法读取")
        return candidate

    def _is_tombstone(self, path: Path) -> bool:
        """Return whether one file looks like a parseable deletion tombstone."""
        return _parse_tombstone_name(path) is not None

    def _new_tombstone_path(
        self,
        document_id: str,
        stored_filename: str,
    ) -> Path:
        """Return a fresh, parseable tombstone path for one upload.

        Both identifiers are validated before use so the generated name
        always matches :data:`_TOMBSTONE_RE` and never contains user
        controlled input, path separators, or dot segments.
        """
        if not _DOCUMENT_ID_RE.fullmatch(document_id):
            raise DocumentMetadataError("文档元数据损坏，无法读取")
        if not _STORED_FILENAME_RE.fullmatch(stored_filename):
            raise DocumentMetadataError("文档元数据损坏，无法读取")
        upload_root = self._upload_dir.resolve()
        tombstone = upload_root / (
            f"{_TOMBSTONE_PREFIX}{document_id}-{stored_filename}-"
            f"{uuid4().hex}{_TOMBSTONE_SUFFIX}"
        )
        if tombstone.exists():
            raise DocumentIngestionError("文件保存冲突，请重试")
        return tombstone
