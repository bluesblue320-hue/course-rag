"""Unit tests for PgVectorIngestionService without a real database."""

import math
from pathlib import Path

import numpy as np
import pytest

from src.document_chunker import chunk_document
from src.document_loaders import TextDocumentLoader
from src.documents import DocumentRecord, utc_now_iso
from src.exceptions import (
    BuiltinDocumentDeletionError,
    DatabaseOperationError,
    DocumentIngestionError,
    DocumentMetadataError,
    DocumentNotFoundError,
    DocumentParseError,
    UnsupportedDocumentTypeError,
)
from src.pgvector_ingestion_service import PgVectorIngestionService

BUILTIN_ID = "builtin-knowledge"


class FakeStore:
    """In-memory stand-in for PgVectorStore's service-facing surface."""

    def __init__(
        self,
        *,
        documents: list[DocumentRecord] | None = None,
        chunk_counts: dict[str, int] | None = None,
        fail_insert: bool = False,
        fail_delete: bool = False,
    ) -> None:
        self._documents = {record.document_id: record for record in (documents or [])}
        self._chunk_counts = dict(chunk_counts or {})
        self.fail_insert = fail_insert
        self.fail_delete = fail_delete
        self.inserted: list[tuple[DocumentRecord, list]] = []
        self.deleted_ids: list[str] = []

    @property
    def chunk_count(self) -> int:
        return sum(self._chunk_counts.values())

    def list_documents(self) -> list[DocumentRecord]:
        return list(self._documents.values())

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return self._documents.get(document_id)

    def count_document_chunks(self, document_id: str) -> int:
        return self._chunk_counts.get(document_id, 0)

    def insert_document(
        self,
        document: DocumentRecord,
        chunks: list,
        _embeddings: object,
    ) -> None:
        if self.fail_insert:
            raise DatabaseOperationError("数据库操作失败，请稍后重试")
        self.inserted.append((document, list(chunks)))
        self._documents[document.document_id] = document
        self._chunk_counts[document.document_id] = len(chunks)

    def delete_document(self, document_id: str) -> DocumentRecord:
        if self.fail_delete:
            raise DatabaseOperationError("数据库操作失败，请稍后重试")
        record = self._documents.pop(document_id)
        self._chunk_counts.pop(document_id, None)
        self.deleted_ids.append(document_id)
        return record


class FakeEmbeddingService:
    """Record every encode_documents call with a deterministic 384-dim vector."""

    def __init__(self) -> None:
        self.encode_calls: list[list[str]] = []

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.encode_calls.append(list(texts))
        if not texts:
            return np.empty((0, 384))
        return np.full((len(texts), 384), 1.0 / math.sqrt(384))


class FailingEmbeddingService:
    def encode_documents(self, _texts: list[str]) -> np.ndarray:
        raise RuntimeError("embedding exploded")


class FailingLoader:
    def load(self, _path: Path):
        raise DocumentParseError("文档无法解析")


def make_upload_record(
    document_id: str = "doc-1",
    stored_filename: str | None = "a" * 32 + ".txt",
    is_builtin: bool = False,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename="knowledge.txt" if is_builtin else "notes.txt",
        stored_filename=stored_filename,
        content_type="text/plain",
        size_bytes=10,
        text_length=10,
        chunk_count=1,
        created_at=utc_now_iso(),
        is_builtin=is_builtin,
    )


def make_service(
    tmp_path: Path,
    *,
    store: FakeStore | None = None,
    embedding: object | None = None,
    loaders: dict[str, object] | None = None,
    builtin_text: str = "内置课程知识内容。",
) -> tuple[PgVectorIngestionService, Path, Path]:
    upload_dir = tmp_path / "uploads"
    knowledge_path = tmp_path / "knowledge.txt"
    knowledge_path.write_text(builtin_text, encoding="utf-8")
    loaded = TextDocumentLoader().load(knowledge_path)
    builtin_document = DocumentRecord(
        document_id=BUILTIN_ID,
        original_filename="knowledge.txt",
        stored_filename=None,
        content_type="text/plain",
        size_bytes=knowledge_path.stat().st_size,
        text_length=loaded.text_length,
        chunk_count=len(
            chunk_document(loaded, BUILTIN_ID, "knowledge.txt")
        ),
        created_at=utc_now_iso(),
        is_builtin=True,
    )
    service = PgVectorIngestionService(
        embedding_service=embedding or FakeEmbeddingService(),
        store=store or FakeStore(),
        upload_dir=upload_dir,
        builtin_document=builtin_document,
        builtin_path=knowledge_path,
        loaders=loaders or {".txt": TextDocumentLoader()},
        max_upload_bytes=1024 * 1024,
    )
    return service, upload_dir, knowledge_path


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


class TestUpload:
    def test_upload_encodes_only_new_chunks(self, tmp_path: Path) -> None:
        embedding = FakeEmbeddingService()
        store = FakeStore()
        service, upload_dir, _ = make_service(
            tmp_path,
            store=store,
            embedding=embedding,
        )

        record = service.ingest(
            "notes.txt",
            "text/plain",
            "上传的课程内容 alpha beta gamma".encode("utf-8"),
        )

        assert len(embedding.encode_calls) == 1
        encoded_texts = embedding.encode_calls[0]
        assert len(encoded_texts) == record.chunk_count
        # Historical documents were never read: exactly one insert happened.
        assert len(store.inserted) == 1
        inserted_document, inserted_chunks = store.inserted[0]
        assert inserted_document.document_id == record.document_id
        assert [chunk.text for chunk in inserted_chunks] == encoded_texts
        # The uploaded file is kept after a successful upload.
        assert (upload_dir / record.stored_filename).is_file()

    def test_upload_rejects_unsupported_suffix(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path)
        with pytest.raises(UnsupportedDocumentTypeError):
            service.ingest("evil.exe", "text/plain", b"data")

    def test_upload_rejects_unsupported_content_type(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path)
        with pytest.raises(UnsupportedDocumentTypeError):
            service.ingest("notes.txt", "application/json", b"data")

    def test_parse_failure_cleans_file(self, tmp_path: Path) -> None:
        store = FakeStore()
        service, upload_dir, _ = make_service(
            tmp_path,
            store=store,
            loaders={".txt": FailingLoader()},
        )

        with pytest.raises(DocumentParseError):
            service.ingest("bad.txt", "text/plain", b"data")

        assert list(upload_dir.iterdir()) == []
        assert store.inserted == []

    def test_embedding_failure_cleans_file(self, tmp_path: Path) -> None:
        store = FakeStore()
        service, upload_dir, _ = make_service(
            tmp_path,
            store=store,
            embedding=FailingEmbeddingService(),
        )

        with pytest.raises(DocumentIngestionError):
            service.ingest("notes.txt", "text/plain", b"data")

        assert list(upload_dir.iterdir()) == []
        assert store.inserted == []

    def test_database_failure_cleans_file(self, tmp_path: Path) -> None:
        store = FakeStore(fail_insert=True)
        service, upload_dir, _ = make_service(tmp_path, store=store)

        with pytest.raises(DatabaseOperationError):
            service.ingest("notes.txt", "text/plain", b"data")

        assert list(upload_dir.iterdir()) == []
        assert store.inserted == []

    def test_stored_filename_is_safe_uuid(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path)
        record = service.ingest("notes.txt", "text/plain", b"data")

        import re

        assert re.fullmatch(r"[0-9a-f]{32}\.txt", record.stored_filename or "")


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_builtin_rejected(self, tmp_path: Path) -> None:
        builtin = make_upload_record(document_id=BUILTIN_ID, is_builtin=True)
        store = FakeStore(documents=[builtin])
        service, _, _ = make_service(tmp_path, store=store)

        with pytest.raises(BuiltinDocumentDeletionError):
            service.delete_document(BUILTIN_ID)

    def test_delete_missing_document_rejected(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path, store=FakeStore())

        with pytest.raises(DocumentNotFoundError):
            service.delete_document("missing")

    def test_delete_renames_to_tombstone_then_removes(
        self, tmp_path: Path
    ) -> None:
        stored = "b" * 32 + ".txt"
        document = make_upload_record(document_id="doc-1", stored_filename=stored)
        store = FakeStore(documents=[document], chunk_counts={"doc-1": 1})
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"original")

        record = service.delete_document("doc-1")

        assert record.document_id == "doc-1"
        assert "doc-1" in store.deleted_ids
        # The original file is gone and no tombstone is left behind.
        assert list(upload_dir.iterdir()) == []

    def test_delete_database_failure_restores_file(self, tmp_path: Path) -> None:
        stored = "c" * 32 + ".txt"
        document = make_upload_record(document_id="doc-1", stored_filename=stored)
        store = FakeStore(documents=[document], fail_delete=True)
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"original")

        with pytest.raises(DatabaseOperationError):
            service.delete_document("doc-1")

        # The original file must be restored and the tombstone removed.
        assert (upload_dir / stored).read_bytes() == b"original"
        assert list(upload_dir.iterdir()) == [upload_dir / stored]

    def test_delete_missing_file_still_deletes_database(
        self, tmp_path: Path
    ) -> None:
        stored = "d" * 32 + ".txt"
        document = make_upload_record(document_id="doc-1", stored_filename=stored)
        store = FakeStore(documents=[document])
        service, _, _ = make_service(tmp_path, store=store)

        record = service.delete_document("doc-1")

        assert record.document_id == "doc-1"
        assert "doc-1" in store.deleted_ids


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


class TestPathSafety:
    def test_backslash_traversal_rejected(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path)
        with pytest.raises(DocumentMetadataError):
            service._resolve_stored_path("..\\..\\evil.txt")

    def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path)
        with pytest.raises(DocumentMetadataError):
            service._resolve_stored_path("/etc/passwd")

    def test_illegal_suffix_rejected(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path)
        with pytest.raises(DocumentMetadataError):
            service._resolve_stored_path("e" * 32 + ".exe")

    def test_dot_segment_rejected(self, tmp_path: Path) -> None:
        service, _, _ = make_service(tmp_path)
        with pytest.raises(DocumentMetadataError):
            service._resolve_stored_path("../" + "g" * 32 + ".txt")

    def test_valid_stored_filename_resolves_inside_upload_dir(
        self, tmp_path: Path
    ) -> None:
        service, upload_dir, _ = make_service(tmp_path)
        stored = "h" * 32 + ".txt"
        with pytest.raises(DocumentMetadataError):
            service._resolve_stored_path(stored)
        valid_stored = "a" * 32 + ".txt"
        resolved = service._resolve_stored_path(valid_stored)
        assert resolved == (upload_dir / valid_stored).resolve()


# ---------------------------------------------------------------------------
# Built-in document initialization
# ---------------------------------------------------------------------------


class TestBuiltinInitialization:
    def test_first_startup_writes_builtin_once(self, tmp_path: Path) -> None:
        embedding = FakeEmbeddingService()
        store = FakeStore()
        service, _, _ = make_service(tmp_path, store=store, embedding=embedding)

        service.ensure_builtin_document()

        assert len(store.inserted) == 1
        assert store.inserted[0][0].document_id == BUILTIN_ID
        assert len(embedding.encode_calls) == 1

    def test_restart_does_not_reembed_builtin(self, tmp_path: Path) -> None:
        embedding = FakeEmbeddingService()
        store = FakeStore()
        service, _, _ = make_service(tmp_path, store=store, embedding=embedding)
        service.ensure_builtin_document()
        first_embed_count = len(embedding.encode_calls)

        # Simulate a restart: a second service sees the builtin already there.
        second_embedding = FakeEmbeddingService()
        second_service, _, _ = make_service(
            tmp_path,
            store=store,
            embedding=second_embedding,
        )
        second_service.ensure_builtin_document()

        assert first_embed_count == 1
        assert len(second_embedding.encode_calls) == 0
        assert len(store.inserted) == 1

    def test_builtin_with_wrong_flag_is_data_anomaly(
        self, tmp_path: Path
    ) -> None:
        broken = make_upload_record(document_id=BUILTIN_ID, is_builtin=False)
        store = FakeStore(documents=[broken], chunk_counts={BUILTIN_ID: 1})
        service, _, _ = make_service(tmp_path, store=store)

        with pytest.raises(DatabaseOperationError):
            service.ensure_builtin_document()

    def test_builtin_chunk_count_mismatch_is_data_anomaly(
        self, tmp_path: Path
    ) -> None:
        builtin = make_upload_record(document_id=BUILTIN_ID, is_builtin=True)
        store = FakeStore(documents=[builtin], chunk_counts={BUILTIN_ID: 99})
        service, _, _ = make_service(tmp_path, store=store)

        with pytest.raises(DatabaseOperationError):
            service.ensure_builtin_document()


# ---------------------------------------------------------------------------
# Tombstone cleanup
# ---------------------------------------------------------------------------


class TestTombstoneCleanup:
    def test_cleanup_removes_leftovers_only(self, tmp_path: Path) -> None:
        service, upload_dir, _ = make_service(tmp_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / ".tombstone-aaaa.tmp").write_bytes(b"x")
        (upload_dir / ".tombstone-bbbb.tmp").write_bytes(b"y")
        (upload_dir / ("i" * 32 + ".txt")).write_bytes(b"keep")

        service.cleanup_tombstones()

        assert not (upload_dir / ".tombstone-aaaa.tmp").exists()
        assert not (upload_dir / ".tombstone-bbbb.tmp").exists()
        assert len(list(upload_dir.iterdir())) == 1

    def test_cleanup_failure_does_not_raise(self, tmp_path: Path) -> None:
        service, upload_dir, _ = make_service(tmp_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        # A directory with a tombstone name cannot be unlinked -> OSError.
        (upload_dir / ".tombstone-dir.tmp").mkdir()

        service.cleanup_tombstones()  # must not raise
