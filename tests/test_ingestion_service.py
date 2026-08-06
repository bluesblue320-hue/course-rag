"""Offline tests for the document ingestion and deletion service."""

from pathlib import Path

import numpy as np
import pytest

from src.document_loaders import (
    MarkdownDocumentLoader,
    PdfDocumentLoader,
    TextDocumentLoader,
)
from src.document_repository import DocumentRepository
from src.documents import DocumentRecord, utc_now_iso
from src.exceptions import (
    BuiltinDocumentDeletionError,
    DocumentIngestionError,
    DocumentMetadataError,
    DocumentNotFoundError,
    EmptyDocumentError,
    UnsupportedDocumentTypeError,
    UploadTooLargeError,
)
from src.ingestion_service import IngestionService
from src.knowledge_index import KnowledgeIndex

from pdf_helpers import make_text_pdf

BUILTIN_ID = "builtin-knowledge"


class FakeEmbeddingService:
    def __init__(self) -> None:
        self.document_encode_count = 0
        self.fail_documents = False
        self.queries: list[str] = []

    def _vector_for(self, text: str) -> list[float]:
        if "alpha" in text:
            return [1.0, 0.0]
        if "beta" in text:
            return [0.0, 1.0]
        return [0.5, 0.5]

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        self.document_encode_count += 1
        if self.fail_documents:
            raise RuntimeError("embedding failed")
        return np.asarray([self._vector_for(text) for text in texts])

    def encode_query(self, query: str) -> np.ndarray:
        self.queries.append(query)
        return np.asarray(self._vector_for(query))


def _builtin_document() -> DocumentRecord:
    return DocumentRecord(
        document_id=BUILTIN_ID,
        original_filename="knowledge.txt",
        stored_filename=None,
        content_type="text/plain",
        size_bytes=12,
        text_length=12,
        chunk_count=1,
        created_at="2026-01-01T00:00:00Z",
        is_builtin=True,
    )


def _make_service(
    tmp_path: Path,
    embedding: FakeEmbeddingService | None = None,
    max_upload_bytes: int = 1024 * 1024,
) -> tuple[IngestionService, FakeEmbeddingService, Path]:
    upload_dir = tmp_path / "runtime" / "uploads"
    builtin_path = tmp_path / "knowledge.txt"
    builtin_path.write_text("内置知识库内容", encoding="utf-8")
    fake_embedding = embedding or FakeEmbeddingService()
    repository = DocumentRepository(tmp_path / "runtime" / "documents.json")
    index = KnowledgeIndex()
    service = IngestionService(
        embedding_service=fake_embedding,
        repository=repository,
        index=index,
        upload_dir=upload_dir,
        builtin_document=_builtin_document(),
        builtin_path=builtin_path,
        loaders={
            ".txt": TextDocumentLoader(),
            ".md": MarkdownDocumentLoader(),
            ".pdf": PdfDocumentLoader(),
        },
        max_upload_bytes=max_upload_bytes,
    )
    service.initialize_index([_builtin_document()])
    return service, fake_embedding, upload_dir


def _upload_dir_files(upload_dir: Path) -> list[Path]:
    if not upload_dir.exists():
        return []
    return list(upload_dir.iterdir())


def test_upload_txt_success(tmp_path: Path) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)

    record = service.ingest("notes.txt", "text/plain", "alpha 核心内容".encode())

    assert record.is_builtin is False
    assert record.original_filename == "notes.txt"
    assert record.stored_filename is not None
    assert record.stored_filename.endswith(".txt")
    assert record.content_type == "text/plain"
    assert record.size_bytes == len("alpha 核心内容".encode())
    assert record.chunk_count == 1
    assert record.document_id != BUILTIN_ID
    assert (upload_dir / record.stored_filename).is_file()
    assert service.get_document(record.document_id) is not None


def test_upload_markdown_success(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    record = service.ingest("notes.md", "text/markdown", "# alpha 标题".encode())

    assert record.stored_filename is not None
    assert record.stored_filename.endswith(".md")


def test_upload_pdf_success(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)
    pdf_bytes = make_text_pdf(["alpha page one", "alpha page two"])

    record = service.ingest("course.pdf", "application/pdf", pdf_bytes)

    assert record.stored_filename is not None
    assert record.stored_filename.endswith(".pdf")
    assert record.chunk_count == 2


def test_uploaded_content_becomes_searchable_with_metadata(tmp_path: Path) -> None:
    service, embedding, _upload_dir = _make_service(tmp_path)
    record = service.ingest("notes.txt", "text/plain", "alpha 核心内容".encode())

    results = service.index.search(embedding.encode_query("alpha"), top_k=3)

    assert results[0]["document_id"] == record.document_id
    assert results[0]["filename"] == "notes.txt"
    assert results[0]["page_number"] is None
    assert "alpha" in results[0]["text"]


def test_pdf_upload_preserves_page_numbers_in_sources(tmp_path: Path) -> None:
    service, embedding, _upload_dir = _make_service(tmp_path)
    pdf_bytes = make_text_pdf(["alpha page one", "alpha page two"])
    service.ingest("course.pdf", "application/pdf", pdf_bytes)

    results = service.index.search(embedding.encode_query("alpha"), top_k=2)

    assert [result["page_number"] for result in results] == [1, 2]


def test_embedding_service_is_called_and_index_replaced(tmp_path: Path) -> None:
    service, embedding, _upload_dir = _make_service(tmp_path)
    before_count = embedding.document_encode_count
    before_chunks = service.index.chunk_count

    service.ingest("notes.txt", "text/plain", "alpha 内容".encode())

    assert embedding.document_encode_count == before_count + 1
    assert service.index.chunk_count == before_chunks + 1


def test_parse_failure_keeps_old_index_and_cleans_file(tmp_path: Path) -> None:
    service, embedding, upload_dir = _make_service(tmp_path)
    before_chunks = service.index.chunk_count

    with pytest.raises(Exception):
        service.ingest("broken.pdf", "application/pdf", b"not a real pdf")

    assert service.index.chunk_count == before_chunks
    assert _upload_dir_files(upload_dir) == []
    assert service.list_documents() == [_builtin_document()]


def test_embedding_failure_keeps_old_index_and_cleans_file(
    tmp_path: Path,
) -> None:
    embedding = FakeEmbeddingService()
    service, _embedding, upload_dir = _make_service(tmp_path, embedding)
    embedding.fail_documents = True
    before_chunks = service.index.chunk_count

    with pytest.raises(DocumentIngestionError, match="文档处理失败"):
        service.ingest("notes.txt", "text/plain", "alpha 内容".encode())

    assert service.index.chunk_count == before_chunks
    assert _upload_dir_files(upload_dir) == []
    assert service.list_documents() == [_builtin_document()]


def test_metadata_failure_keeps_old_index_and_cleans_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    before_chunks = service.index.chunk_count

    def fail_save(_documents: list[DocumentRecord]) -> None:
        raise DocumentMetadataError("磁盘写入失败")

    monkeypatch.setattr(service._repository, "save_documents", fail_save)

    with pytest.raises(DocumentMetadataError):
        service.ingest("notes.txt", "text/plain", "alpha 内容".encode())

    assert service.index.chunk_count == before_chunks
    assert _upload_dir_files(upload_dir) == []
    assert service.list_documents() == [_builtin_document()]


def test_traversal_filename_cannot_escape_upload_dir(tmp_path: Path) -> None:
    service, embedding, upload_dir = _make_service(tmp_path)

    record = service.ingest(
        "../../escape.txt",
        "text/plain",
        "alpha 内容".encode(),
    )

    assert record.stored_filename is not None
    assert (upload_dir / record.stored_filename).is_file()
    assert not (tmp_path / "escape.txt").exists()
    assert not (tmp_path / ".." / "escape.txt").exists()
    assert record.original_filename == "../../escape.txt"
    results = service.index.search(embedding.encode_query("alpha"), top_k=1)
    assert results[0]["document_id"] == record.document_id


def test_oversized_file_is_rejected(tmp_path: Path) -> None:
    service, _embedding, upload_dir = _make_service(
        tmp_path,
        max_upload_bytes=5,
    )

    with pytest.raises(UploadTooLargeError):
        service.ingest("big.txt", "text/plain", b"too large content")

    assert _upload_dir_files(upload_dir) == []


def test_empty_file_is_rejected(tmp_path: Path) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)

    with pytest.raises(EmptyDocumentError):
        service.ingest("empty.txt", "text/plain", b"")

    assert _upload_dir_files(upload_dir) == []


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    with pytest.raises(UnsupportedDocumentTypeError):
        service.ingest("notes.docx", "application/vnd.openxmlformats", b"x")


def test_unsupported_content_type_is_rejected(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    with pytest.raises(UnsupportedDocumentTypeError):
        service.ingest("notes.txt", "image/png", b"alpha")


def test_blank_filename_is_rejected(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    with pytest.raises(UnsupportedDocumentTypeError):
        service.ingest("   ", "text/plain", b"alpha")


def test_delete_uploaded_document_success(tmp_path: Path) -> None:
    service, embedding, upload_dir = _make_service(tmp_path)
    record = service.ingest("notes.txt", "text/plain", "alpha 内容".encode())
    stored_path = upload_dir / (record.stored_filename or "")
    assert stored_path.is_file()

    deleted = service.delete_document(record.document_id)

    assert deleted.document_id == record.document_id
    assert not stored_path.exists()
    assert service.get_document(record.document_id) is None
    assert service.list_documents() == [_builtin_document()]
    assert all(
        result["document_id"] != record.document_id
        for result in service.index.search(
            embedding.encode_query("alpha"),
            top_k=3,
        )
    )


def test_delete_removes_content_from_search(tmp_path: Path) -> None:
    service, embedding, _upload_dir = _make_service(tmp_path)
    record = service.ingest("notes.txt", "text/plain", "alpha 内容".encode())
    service.ingest("other.txt", "text/plain", "beta 内容".encode())

    service.delete_document(record.document_id)

    results = service.index.search(embedding.encode_query("alpha"), top_k=3)
    assert all(
        result["document_id"] != record.document_id
        for result in results
    )
    beta_results = service.index.search(embedding.encode_query("beta"), top_k=3)
    assert beta_results[0]["filename"] == "other.txt"


def test_delete_unknown_id_raises(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    with pytest.raises(DocumentNotFoundError):
        service.delete_document("does-not-exist")


def test_delete_builtin_document_raises(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    with pytest.raises(BuiltinDocumentDeletionError):
        service.delete_document(BUILTIN_ID)


def test_delete_rebuild_failure_keeps_document_and_index(
    tmp_path: Path,
) -> None:
    embedding = FakeEmbeddingService()
    service, _embedding, upload_dir = _make_service(tmp_path, embedding)
    record = service.ingest("notes.txt", "text/plain", "alpha 内容".encode())
    embedding.fail_documents = True

    with pytest.raises(DocumentIngestionError, match="文档处理失败"):
        service.delete_document(record.document_id)

    embedding.fail_documents = False
    assert service.get_document(record.document_id) is not None
    stored_path = upload_dir / (record.stored_filename or "")
    assert stored_path.is_file()
    results = service.index.search(
        service._embedding_service.encode_query("alpha"),
        top_k=3,
    )
    assert results[0]["document_id"] == record.document_id


def test_delete_disk_failure_rolls_back_and_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, embedding, upload_dir = _make_service(tmp_path)
    record = service.ingest("notes.txt", "text/plain", "alpha 内容".encode())

    def fail_unlink(_path: Path) -> None:
        raise OSError("disk busy")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    with pytest.raises(DocumentIngestionError):
        service.delete_document(record.document_id)

    monkeypatch.setattr(Path, "unlink", Path.unlink)
    assert service.get_document(record.document_id) is not None
    assert (upload_dir / (record.stored_filename or "")).exists()
    results = service.index.search(embedding.encode_query("alpha"), top_k=3)
    assert results[0]["document_id"] == record.document_id


def test_created_at_is_utc_iso_8601(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    record = service.ingest("notes.txt", "text/plain", "alpha 内容".encode())

    from datetime import datetime, timezone

    parsed = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
    assert parsed.tzinfo == timezone.utc


def test_document_ids_are_unique_across_uploads(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)

    first = service.ingest("a.txt", "text/plain", "alpha 一".encode())
    second = service.ingest("b.txt", "text/plain", "alpha 二".encode())

    assert first.document_id != second.document_id


def _stored_name(seed: str, suffix: str = "txt") -> str:
    hex_slug = "".join(
        character for character in seed if character in "0123456789abcdef"
    )
    return f"{hex_slug}{'a' * (32 - len(hex_slug))}.{suffix}"


def _upload_record(
    document_id: str,
    stored_filename: str,
    *,
    is_builtin: bool = False,
    content_type: str = "text/plain",
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename=f"{document_id}.txt",
        stored_filename=None if is_builtin else stored_filename,
        content_type=content_type,
        size_bytes=10,
        text_length=10,
        chunk_count=1,
        created_at=utc_now_iso(),
        is_builtin=is_builtin,
    )


def test_reconcile_removes_missing_file_record(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)
    ghost = _upload_record("doc-ghost", _stored_name("ghost"))
    service._repository.save_documents([ghost])

    valid = service.reconcile_persisted_documents()

    assert valid == []
    assert service._repository.list_documents() == []


def test_reconcile_removes_corrupt_pdf_and_deletes_file(tmp_path: Path) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    corrupt_name = _stored_name("corrupt", "pdf")
    (upload_dir / corrupt_name).write_bytes(b"not a real pdf")
    service._repository.save_documents(
        [
            _upload_record(
                "doc-corrupt",
                corrupt_name,
                content_type="application/pdf",
            )
        ]
    )

    valid = service.reconcile_persisted_documents()

    assert valid == []
    assert not (upload_dir / corrupt_name).exists()
    assert service._repository.list_documents() == []


def test_reconcile_removes_empty_file_record(tmp_path: Path) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    empty_name = _stored_name("empty")
    (upload_dir / empty_name).write_bytes(b"")
    service._repository.save_documents(
        [_upload_record("doc-empty", empty_name)]
    )

    valid = service.reconcile_persisted_documents()

    assert valid == []
    assert service._repository.list_documents() == []


def test_reconcile_keeps_valid_uploads_and_repository(tmp_path: Path) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    valid_name = _stored_name("valid")
    (upload_dir / valid_name).write_text("alpha 有效内容", encoding="utf-8")
    record = _upload_record("doc-valid", valid_name)
    service._repository.save_documents([record])

    valid = service.reconcile_persisted_documents()

    assert valid == [record]
    assert service._repository.list_documents() == [record]


def test_reconcile_cleans_builtin_records_from_repository(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)
    builtin_record = _upload_record(BUILTIN_ID, "", is_builtin=True)
    service._repository.save_documents([builtin_record])

    valid = service.reconcile_persisted_documents()

    assert valid == []
    assert service._repository.list_documents() == []
    assert service.list_documents() == [_builtin_document()]


def test_reconcile_propagates_corrupted_json(tmp_path: Path) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)
    service._repository._json_path.parent.mkdir(parents=True, exist_ok=True)
    service._repository._json_path.write_text("{ broken json", encoding="utf-8")

    with pytest.raises(DocumentMetadataError):
        service.reconcile_persisted_documents()


def test_reconcile_delete_failure_does_not_break_valid_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    corrupt_name = _stored_name("corrupt", "pdf")
    (upload_dir / corrupt_name).write_bytes(b"not a real pdf")
    valid_name = _stored_name("valid")
    (upload_dir / valid_name).write_text("alpha 有效内容", encoding="utf-8")
    valid_record = _upload_record("doc-valid", valid_name)
    service._repository.save_documents(
        [
            _upload_record(
                "doc-corrupt",
                corrupt_name,
                content_type="application/pdf",
            ),
            valid_record,
        ]
    )

    def fail_unlink(_path: Path) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "unlink", fail_unlink)

    valid = service.reconcile_persisted_documents()

    monkeypatch.setattr(Path, "unlink", Path.unlink)
    assert valid == [valid_record]
    assert service._repository.list_documents() == [valid_record]
    assert (upload_dir / corrupt_name).exists()


@pytest.mark.parametrize(
    "dangerous",
    [
        "../../escape.txt",
        "../escape.txt",
        "nested/file.txt",
        "..\\escape.txt",
        "..\\..\\escape.txt",
        "C:/absolute/escape.txt",
        "/absolute/escape.txt",
        "0123456789abcdef0123456789abcdef",
        "0123456789abcdef0123456789abcdef.exe",
        "0123456789abcdef0123456789abcdef.txt/pwn",
        "0x23456789abcdef0123456789abcdef.txt",
    ],
)
def test_resolve_stored_path_rejects_dangerous_names(
    tmp_path: Path,
    dangerous: str,
) -> None:
    service, _embedding, _upload_dir = _make_service(tmp_path)
    sentinel = tmp_path / "escape.txt"
    sentinel.write_text("sentinel", encoding="utf-8")

    with pytest.raises(DocumentMetadataError):
        service._resolve_stored_path(dangerous)

    assert sentinel.read_text(encoding="utf-8") == "sentinel"


def test_resolve_stored_path_accepts_valid_uuids(tmp_path: Path) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    for suffix in ("txt", "md", "pdf"):
        name = _stored_name(f"ok{suffix}", suffix)
        resolved = service._resolve_stored_path(name)
        assert resolved.parent == upload_dir.resolve()
        assert resolved.name == name


def test_chunks_for_document_never_reads_outside_upload_dir(
    tmp_path: Path,
) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    upload_dir.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / "outside.txt"
    sentinel.write_text("outside-secret", encoding="utf-8")
    record = _upload_record("doc-evil", "../../outside.txt")

    with pytest.raises(DocumentMetadataError):
        service.chunks_for_document(record)

    assert sentinel.read_text(encoding="utf-8") == "outside-secret"
    assert list(upload_dir.iterdir()) == []


def test_delete_resolves_stored_path_through_safe_helper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, _embedding, upload_dir = _make_service(tmp_path)
    record = service.ingest("notes.txt", "text/plain", "alpha 内容".encode())
    original_resolve = service._resolve_stored_path
    resolved_names: list[str] = []

    def recording_resolve(stored_filename: str) -> Path:
        resolved_names.append(stored_filename)
        return original_resolve(stored_filename)

    monkeypatch.setattr(service, "_resolve_stored_path", recording_resolve)

    service.delete_document(record.document_id)

    assert resolved_names == [record.stored_filename]
    assert not (upload_dir / (record.stored_filename or "")).exists()
