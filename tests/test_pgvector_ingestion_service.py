"""Unit tests for PgVectorIngestionService without a real database."""

import logging
import math
import os
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
from src.pgvector_ingestion_service import (
    PgVectorIngestionService,
    _parse_tombstone_name,
)

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
    document_id: str = "a" * 32,
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


def make_tombstone_name(
    document_id: str,
    stored_filename: str,
    token: str | None = None,
) -> str:
    """Return a strict, parseable tombstone filename."""
    return (
        f".tombstone-{document_id}-{stored_filename}-"
        f"{token or 'c' * 32}.tmp"
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
        document_id = "b" * 32
        stored = "b" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(
            documents=[document], chunk_counts={document_id: 1}
        )
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"original")

        record = service.delete_document(document_id)

        assert record.document_id == document_id
        assert document_id in store.deleted_ids
        # The original file is gone and no tombstone is left behind.
        assert list(upload_dir.iterdir()) == []

    def test_delete_database_failure_restores_file(self, tmp_path: Path) -> None:
        document_id = "c" * 32
        stored = "c" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(documents=[document], fail_delete=True)
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"original")

        with pytest.raises(DatabaseOperationError):
            service.delete_document(document_id)

        # The original file must be restored and the tombstone removed.
        assert (upload_dir / stored).read_bytes() == b"original"
        assert list(upload_dir.iterdir()) == [upload_dir / stored]

    def test_delete_missing_file_still_deletes_database(
        self, tmp_path: Path
    ) -> None:
        document_id = "d" * 32
        stored = "d" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(documents=[document])
        service, _, _ = make_service(tmp_path, store=store)

        record = service.delete_document(document_id)

        assert record.document_id == document_id
        assert document_id in store.deleted_ids


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
# Tombstone parsing
# ---------------------------------------------------------------------------


class TestTombstoneParsing:
    def test_tombstone_name_is_parseable(self, tmp_path: Path) -> None:
        document_id = "a" * 32
        stored = "b" * 32 + ".txt"
        name = make_tombstone_name(document_id, stored)
        path = tmp_path / name

        parsed = _parse_tombstone_name(path)

        assert parsed is not None
        parsed_document_id, parsed_stored = parsed
        assert parsed_document_id == document_id
        assert parsed_stored == stored
        assert len(name.split("-")[-1].split(".")[0]) == 32  # token exists

    def test_non_matching_name_returns_none(self, tmp_path: Path) -> None:
        cases = [
            "tombstone-aaaa.txt.tmp",  # no leading dot
            ".tombstone-aaaa.tmp",  # not enough segments
            ".tombstone-AAAAAAAA-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.txt-cccccccccccccccccccccccccccccccc.tmp",  # uppercase hex
            ".tombstone-aaaa-../../evil.txt-cccccccccccccccccccccccccccccccc.tmp",  # traversal
            ".tombstone-aaaa-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.exe-cccccccccccccccccccccccccccccccc.tmp",  # bad suffix
        ]
        for name in cases:
            assert _parse_tombstone_name(tmp_path / name) is None, name


# ---------------------------------------------------------------------------
# Tombstone reconciliation at startup
# ---------------------------------------------------------------------------


class TestTombstoneReconcile:
    def test_malformed_tombstone_is_preserved(
        self, tmp_path: Path, caplog
    ) -> None:
        service, upload_dir, _ = make_service(tmp_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        malformed = upload_dir / ".tombstone-aaaa.tmp"
        malformed.write_bytes(b"precious")

        service.reconcile_tombstones()

        assert malformed.is_file()
        assert malformed.read_bytes() == b"precious"
        assert "人工检查" in caplog.text
        assert str(tmp_path) not in caplog.text
        assert "Traceback" not in caplog.text

    def test_normal_upload_files_are_ignored_without_warning(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class FailingOnGetStore(FakeStore):
            def get_document(
                self,
                document_id: str,
            ) -> DocumentRecord | None:
                raise AssertionError("普通上传文件不应触发数据库查询")

        store = FailingOnGetStore()
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)

        files = {
            "a" * 32 + ".txt": b"normal txt",
            "b" * 32 + ".md": b"normal markdown",
            "c" * 32 + ".pdf": b"normal pdf",
        }
        for filename, content in files.items():
            (upload_dir / filename).write_bytes(content)

        with caplog.at_level(logging.WARNING):
            service.reconcile_tombstones()

        for filename, content in files.items():
            path = upload_dir / filename
            assert path.is_file()
            assert path.read_bytes() == content
        assert caplog.text == ""

    def test_mixed_directory_reconcile(
        self,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        class RecordingStore(FakeStore):
            def __init__(self) -> None:
                super().__init__()
                self.get_document_calls: list[str] = []

            def get_document(
                self,
                document_id: str,
            ) -> DocumentRecord | None:
                self.get_document_calls.append(document_id)
                return super().get_document(document_id)

        store = RecordingStore()
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # A regular upload, a malformed tombstone, and a valid tombstone
        # whose database record no longer exists.
        normal = upload_dir / ("a" * 32 + ".txt")
        normal.write_bytes(b"normal")
        malformed = upload_dir / ".tombstone-aaaa.tmp"
        malformed.write_bytes(b"bad")
        stale_document_id = "d" * 32
        stale_stored = "d" * 32 + ".txt"
        stale = upload_dir / make_tombstone_name(stale_document_id, stale_stored)
        stale.write_bytes(b"stale")

        with caplog.at_level(logging.WARNING):
            service.reconcile_tombstones()

        # The regular upload is untouched and never queried.
        assert normal.read_bytes() == b"normal"
        # The malformed tombstone is preserved with one fixed warning.
        assert malformed.is_file()
        assert malformed.read_bytes() == b"bad"
        assert caplog.text.count("人工检查") == 1
        # The stale valid tombstone is removed after its database lookup.
        assert not stale.exists()
        # Only the valid tombstone's document id was queried.
        assert store.get_document_calls == [stale_document_id]

    def test_record_exists_and_file_missing_restores(
        self, tmp_path: Path
    ) -> None:
        document_id = "b" * 32
        stored = "b" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(documents=[document])
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        tombstone = upload_dir / make_tombstone_name(document_id, stored)
        tombstone.write_bytes(b"original content")

        service.reconcile_tombstones()

        restored = upload_dir / stored
        assert restored.is_file()
        assert restored.read_bytes() == b"original content"
        assert not tombstone.exists()

    def test_record_exists_and_file_present_removes_duplicate(
        self, tmp_path: Path
    ) -> None:
        document_id = "c" * 32
        stored = "c" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(documents=[document])
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"existing")
        tombstone = upload_dir / make_tombstone_name(document_id, stored)
        tombstone.write_bytes(b"old")

        service.reconcile_tombstones()

        assert (upload_dir / stored).read_bytes() == b"existing"
        assert not tombstone.exists()

    def test_record_missing_removes_tombstone(self, tmp_path: Path) -> None:
        service, upload_dir, _ = make_service(tmp_path, store=FakeStore())
        upload_dir.mkdir(parents=True, exist_ok=True)
        document_id = "d" * 32
        stored = "d" * 32 + ".txt"
        tombstone = upload_dir / make_tombstone_name(document_id, stored)
        tombstone.write_bytes(b"leftover")

        service.reconcile_tombstones()

        assert not tombstone.exists()
        assert not (upload_dir / stored).exists()

    def test_stored_filename_mismatch_is_preserved(
        self, tmp_path: Path, caplog
    ) -> None:
        document_id = "e" * 32
        stored_in_record = "e" * 32 + ".txt"
        stored_in_tombstone = "f" * 32 + ".md"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored_in_record
        )
        store = FakeStore(documents=[document])
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        tombstone = upload_dir / make_tombstone_name(
            document_id, stored_in_tombstone
        )
        tombstone.write_bytes(b"mismatch")

        service.reconcile_tombstones()

        assert tombstone.is_file()
        assert not (upload_dir / stored_in_tombstone).exists()
        assert not (upload_dir / stored_in_record).exists()
        assert "不一致" in caplog.text

    def test_builtin_record_tombstone_is_preserved(
        self, tmp_path: Path, caplog
    ) -> None:
        document_id = "a" * 32
        stored = "b" * 32 + ".txt"
        # A database record flagged as builtin (with a hex document id)
        # must never trigger a tombstone restore or removal.
        builtin = make_upload_record(
            document_id=document_id, stored_filename=stored, is_builtin=True
        )
        store = FakeStore(documents=[builtin])
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        tombstone = upload_dir / make_tombstone_name(document_id, stored)
        tombstone.write_bytes(b"unexpected")

        service.reconcile_tombstones()

        assert tombstone.is_file()
        assert not (upload_dir / stored).exists()
        assert "不一致" in caplog.text


# ---------------------------------------------------------------------------
# Delete compensation semantics
# ---------------------------------------------------------------------------


class TestDeleteCompensation:
    def test_db_failure_with_immediate_restore_failure_keeps_tombstone(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        document_id = "a" * 32
        stored = "a" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(documents=[document], fail_delete=True)
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"original")

        real_replace = os.replace
        replace_calls = []

        def flaky_replace(src, dst):
            replace_calls.append((Path(src).name, Path(dst).name))
            if len(replace_calls) == 2:
                raise OSError("simulated restore failure")
            return real_replace(src, dst)

        monkeypatch.setattr("os.replace", flaky_replace)

        with pytest.raises(DatabaseOperationError):
            service.delete_document(document_id)

        # The database record still exists, the original file is temporarily
        # missing, and the tombstone is preserved for the next startup.
        assert store.get_document(document_id) is not None
        assert not (upload_dir / stored).exists()
        tombstones = [
            path for path in upload_dir.iterdir()
            if _parse_tombstone_name(path) is not None
        ]
        assert len(tombstones) == 1
        assert tombstones[0].read_bytes() == b"original"

    def test_restart_reconcile_restores_after_failed_restore(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        document_id = "a" * 32
        stored = "a" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(documents=[document], fail_delete=True)
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"original")

        real_replace = os.replace
        replace_calls = []

        def flaky_replace(src, dst):
            replace_calls.append(Path(src).name)
            if len(replace_calls) == 2:
                raise OSError("simulated restore failure")
            return real_replace(src, dst)

        monkeypatch.setattr("os.replace", flaky_replace)

        with pytest.raises(DatabaseOperationError):
            service.delete_document(document_id)

        # Simulate an application restart: the same database record exists.
        store.fail_delete = False
        service.reconcile_tombstones()

        assert (upload_dir / stored).read_bytes() == b"original"
        assert store.get_document(document_id) is not None
        tombstones = [
            path for path in upload_dir.iterdir()
            if _parse_tombstone_name(path) is not None
        ]
        assert tombstones == []

    def test_db_success_but_unlink_failure_keeps_tombstone(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        document_id = "a" * 32
        stored = "a" * 32 + ".txt"
        document = make_upload_record(
            document_id=document_id, stored_filename=stored
        )
        store = FakeStore(documents=[document])
        service, upload_dir, _ = make_service(tmp_path, store=store)
        upload_dir.mkdir(parents=True, exist_ok=True)
        (upload_dir / stored).write_bytes(b"original")

        original_unlink = Path.unlink
        unlink_calls = []

        def flaky_unlink(self, *args, **kwargs):
            if ".tombstone-" in self.name and not unlink_calls:
                unlink_calls.append(self.name)
                raise OSError("simulated unlink failure")
            return original_unlink(self, *args, **kwargs)

        monkeypatch.setattr(Path, "unlink", flaky_unlink)

        # The API deletion still succeeds even though the tombstone removal
        # failed; the leftover is cleaned up by the next startup.
        record = service.delete_document(document_id)

        assert record.document_id == document_id
        assert document_id in store.deleted_ids
        tombstones = [
            path for path in upload_dir.iterdir()
            if _parse_tombstone_name(path) is not None
        ]
        assert len(tombstones) == 1

        # Next startup: the database record is gone, so the tombstone is
        # removed and no original file is created.
        service.reconcile_tombstones()
        assert list(upload_dir.iterdir()) == []


# ---------------------------------------------------------------------------
# Recovery log privacy
# ---------------------------------------------------------------------------


class TestRecoveryLogPrivacy:
    def test_logs_do_not_leak_paths_or_identifiers(
        self, tmp_path: Path, caplog
    ) -> None:
        service, upload_dir, _ = make_service(tmp_path)
        upload_dir.mkdir(parents=True, exist_ok=True)
        document_id = "a" * 32
        stored = "b" * 32 + ".txt"
        # A malformed tombstone logs a fixed warning.
        (upload_dir / ".tombstone-aaaa.tmp").write_bytes(b"x")
        # A database-mismatch tombstone logs a fixed warning.
        store = FakeStore()
        service._store = store
        mismatch = upload_dir / make_tombstone_name(document_id, stored)
        mismatch.write_bytes(b"y")

        with caplog.at_level(logging.WARNING):
            service.reconcile_tombstones()

        assert str(tmp_path) not in caplog.text
        assert stored not in caplog.text
        assert document_id not in caplog.text
        assert "Traceback" not in caplog.text
