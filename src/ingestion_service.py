"""Coordinate upload validation, index rebuilding, and document deletion."""

import os
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

from src.document_chunker import chunk_document
from src.document_loaders import DocumentLoader
from src.document_repository import DocumentRepository
from src.documents import (
    ChunkRecord,
    DocumentRecord,
    sort_documents,
    utc_now_iso,
)
from src.exceptions import (
    BuiltinDocumentDeletionError,
    DocumentIngestionError,
    DocumentMetadataError,
    DocumentNotFoundError,
    DocumentParseError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
    UploadConfigurationError,
    UploadTooLargeError,
)
from src.knowledge_index import KnowledgeIndex

DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_ALLOWED_CONTENT_TYPES = {
    "text/plain",
    "text/markdown",
    "text/x-markdown",
    "application/pdf",
    "application/x-pdf",
    "application/octet-stream",
}
_MAX_UPLOAD_ERROR = "MAX_UPLOAD_BYTES 必须是正整数"


def _validate_max_upload_bytes(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UploadConfigurationError(_MAX_UPLOAD_ERROR)
    if value <= 0:
        raise UploadConfigurationError(_MAX_UPLOAD_ERROR)
    return value


def resolve_max_upload_bytes(explicit_value: object | None = None) -> int:
    """Resolve and validate the upload size limit without mutating state."""
    if explicit_value is not None:
        return _validate_max_upload_bytes(explicit_value)

    raw_value = os.getenv("MAX_UPLOAD_BYTES")
    if raw_value is None:
        return DEFAULT_MAX_UPLOAD_BYTES

    try:
        parsed_value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise UploadConfigurationError(_MAX_UPLOAD_ERROR) from exc
    return _validate_max_upload_bytes(parsed_value)


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


class IngestionService:
    """Rebuild the shared index after every upload or deletion succeeds."""

    def __init__(
        self,
        *,
        embedding_service: Any,
        repository: DocumentRepository,
        index: KnowledgeIndex,
        upload_dir: Path,
        builtin_document: DocumentRecord,
        builtin_path: Path,
        loaders: dict[str, DocumentLoader],
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    ) -> None:
        self._embedding_service = embedding_service
        self._repository = repository
        self._index = index
        self._upload_dir = upload_dir
        self._builtin_document = builtin_document
        self._builtin_path = builtin_path
        self._loaders = dict(loaders)
        self._max_upload_bytes = _validate_max_upload_bytes(max_upload_bytes)
        self._lock = threading.RLock()

    @property
    def index(self) -> KnowledgeIndex:
        return self._index

    @property
    def max_upload_bytes(self) -> int:
        return self._max_upload_bytes

    def list_documents(self) -> list[DocumentRecord]:
        """Return the built-in document followed by stored uploads."""
        return sort_documents(
            [self._builtin_document, *self._repository.list_documents()]
        )

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return one document across the unified list or None."""
        for record in self.list_documents():
            if record.document_id == document_id:
                return record
        return None

    def chunks_for_document(self, record: DocumentRecord) -> list[ChunkRecord]:
        """Load one document from disk and convert it to chunk records."""
        if record.is_builtin:
            loader = self._loaders[".txt"]
            path = self._builtin_path
        else:
            stored_filename = record.stored_filename
            if stored_filename is None:
                raise DocumentMetadataError("文档元数据损坏，无法读取")
            suffix = Path(stored_filename).suffix.lower()
            loader = self._loaders.get(suffix)
            if loader is None:
                raise DocumentParseError("文档类型不受支持")
            path = self._resolve_stored_path(stored_filename)
        loaded = loader.load(path)
        return chunk_document(loaded, record.document_id, record.original_filename)

    def _resolve_stored_path(self, stored_filename: str) -> Path:
        """Resolve one stored filename strictly inside the upload directory."""
        upload_root = self._upload_dir.resolve()
        candidate = (self._upload_dir / stored_filename).resolve()
        if candidate.parent != upload_root:
            raise DocumentMetadataError("文档元数据损坏，无法读取")
        return candidate

    def reconcile_persisted_documents(self) -> list[DocumentRecord]:
        """Validate persisted uploads and atomically clean invalid records."""
        persisted_uploads = self._repository.list_documents()
        valid_uploads: list[DocumentRecord] = []
        invalid_uploads: list[DocumentRecord] = []
        for record in persisted_uploads:
            if record.is_builtin:
                invalid_uploads.append(record)
                continue
            try:
                if not self._resolve_stored_path(
                    record.stored_filename or ""
                ).is_file():
                    invalid_uploads.append(record)
                    continue
                chunks = self.chunks_for_document(record)
                if not chunks:
                    invalid_uploads.append(record)
                    continue
            except (DocumentParseError, EmptyDocumentError):
                invalid_uploads.append(record)
                continue
            valid_uploads.append(record)

        if invalid_uploads:
            self._repository.save_documents(valid_uploads)
            for record in invalid_uploads:
                stored_filename = record.stored_filename
                if stored_filename is None:
                    continue
                _unlink_quietly(self._resolve_stored_path(stored_filename))
        return valid_uploads

    def initialize_index(self, documents: list[DocumentRecord]) -> None:
        """Build the initial unified index from a list of documents."""
        chunks = self._collect_chunks(documents)
        embeddings = self._embedding_service.encode_documents(
            [chunk.text for chunk in chunks]
        )
        self._index.replace(chunks, embeddings)

    def _all_records(
        self,
        upload_records: list[DocumentRecord],
    ) -> list[DocumentRecord]:
        """Return the built-in document followed by stored uploads."""
        return sort_documents([self._builtin_document, *upload_records])

    def _collect_chunks(
        self,
        documents: list[DocumentRecord],
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        for record in documents:
            chunks.extend(self.chunks_for_document(record))
        return chunks

    def ingest(
        self,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> DocumentRecord:
        """Validate, store, parse, and index one uploaded file atomically."""
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
                previous_records = self._repository.list_documents()
                all_chunks = self._collect_chunks(
                    self._all_records(previous_records)
                ) + new_chunks
                embeddings = self._embedding_service.encode_documents(
                    [chunk.text for chunk in all_chunks]
                )
                self._repository.save_documents(
                    [*previous_records, record]
                )
                try:
                    self._index.replace(all_chunks, embeddings)
                except Exception:
                    self._repository.save_documents(previous_records)
                    raise
            return record
        except (
            EmptyDocumentError,
            DocumentParseError,
            DocumentMetadataError,
            DocumentIngestionError,
        ):
            _unlink_quietly(target_path)
            raise
        except Exception as exc:
            _unlink_quietly(target_path)
            raise DocumentIngestionError("文档处理失败") from exc

    def delete_document(self, document_id: str) -> DocumentRecord:
        """Remove one uploaded document while keeping every failure safe."""
        try:
            return self._delete_document_locked(document_id)
        except (
            DocumentNotFoundError,
            BuiltinDocumentDeletionError,
            DocumentMetadataError,
            DocumentIngestionError,
        ):
            raise
        except Exception as exc:
            raise DocumentIngestionError("文档处理失败") from exc

    def _delete_document_locked(self, document_id: str) -> DocumentRecord:
        with self._lock:
            previous_records = self._all_records(
                self._repository.list_documents()
            )
            record = next(
                (
                    candidate
                    for candidate in previous_records
                    if candidate.document_id == document_id
                ),
                None,
            )
            if record is None:
                raise DocumentNotFoundError("文档不存在")
            if record.is_builtin:
                raise BuiltinDocumentDeletionError("内置文档不能删除")

            remaining_records = [
                candidate
                for candidate in previous_records
                if candidate.document_id != document_id
            ]
            previous_uploads = [
                candidate
                for candidate in previous_records
                if not candidate.is_builtin
            ]
            remaining_uploads = [
                candidate
                for candidate in remaining_records
                if not candidate.is_builtin
            ]
            all_chunks = self._collect_chunks(remaining_records)
            embeddings = self._embedding_service.encode_documents(
                [chunk.text for chunk in all_chunks]
            )

            self._repository.save_documents(remaining_uploads)
            try:
                self._index.replace(all_chunks, embeddings)
            except Exception:
                self._repository.save_documents(previous_uploads)
                raise

            if record.stored_filename:
                target_path = self._resolve_stored_path(record.stored_filename)
                try:
                    target_path.unlink()
                except OSError as exc:
                    try:
                        self._repository.save_documents(previous_uploads)
                    except Exception:
                        pass
                    try:
                        rollback_chunks = self._collect_chunks(previous_records)
                        rollback_embeddings = self._embedding_service.encode_documents(
                            [chunk.text for chunk in rollback_chunks]
                        )
                        self._index.replace(rollback_chunks, rollback_embeddings)
                    except Exception:
                        pass
                    raise DocumentIngestionError("文档删除失败") from exc
        return record
