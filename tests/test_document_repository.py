"""Offline tests for the JSON document metadata repository."""

from pathlib import Path

import pytest

from src.document_repository import DocumentRepository
from src.documents import DocumentRecord
from src.exceptions import DocumentMetadataError


def _record(
    document_id: str,
    *,
    created_at: str = "2026-08-03T08:00:00Z",
    is_builtin: bool = False,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename=f"{document_id}.txt",
        stored_filename=None if is_builtin else f"stored-{document_id}.txt",
        content_type="text/plain",
        size_bytes=10,
        text_length=5,
        chunk_count=1,
        created_at=created_at,
        is_builtin=is_builtin,
    )


def test_empty_repository_returns_no_documents(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "documents.json")

    assert repository.list_documents() == []
    assert repository.get_document("anything") is None


def test_save_and_read_round_trip(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "documents.json")
    records = [
        _record("doc-1", created_at="2026-08-03T08:00:00Z"),
        _record("doc-2", created_at="2026-08-03T09:00:00Z"),
    ]

    repository.save_documents(records)

    loaded = repository.list_documents()
    assert {record.document_id for record in loaded} == {"doc-1", "doc-2"}
    assert repository.get_document("doc-1") is not None
    assert repository.get_document("doc-1").document_id == "doc-1"
    assert repository.get_document("missing") is None
    assert (tmp_path / "documents.json").is_file()


def test_save_writes_atomically_without_previous_file(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "documents.json")

    repository.save_documents([_record("doc-1")])

    leftovers = [
        path.name for path in tmp_path.glob("documents-*.tmp")
    ]
    assert leftovers == []


def test_corrupted_json_raises_instead_of_clearing(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    path.write_text("{ not valid json !!", encoding="utf-8")
    repository = DocumentRepository(path)

    with pytest.raises(DocumentMetadataError):
        repository.list_documents()


def test_non_document_payload_raises(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    path.write_text('{"documents": "wrong"}', encoding="utf-8")
    repository = DocumentRepository(path)

    with pytest.raises(DocumentMetadataError):
        repository.list_documents()


def test_missing_required_fields_raise(tmp_path: Path) -> None:
    path = tmp_path / "documents.json"
    path.write_text(
        '{"documents": [{"document_id": "only-id"}]}',
        encoding="utf-8",
    )
    repository = DocumentRepository(path)

    with pytest.raises(DocumentMetadataError):
        repository.list_documents()


def test_list_sorts_builtin_first_then_newest(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "documents.json")
    records = [
        _record("old", created_at="2026-08-01T08:00:00Z"),
        _record("new", created_at="2026-08-03T08:00:00Z"),
        _record("builtin", created_at="2026-01-01T00:00:00Z", is_builtin=True),
    ]

    repository.save_documents(records)

    assert [
        record.document_id for record in repository.list_documents()
    ] == ["builtin", "new", "old"]


def test_writing_order_is_stable_across_saves(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "documents.json")
    records = [_record("doc-1"), _record("doc-2")]

    repository.save_documents(records)
    first = repository.list_documents()
    repository.save_documents(records)
    second = repository.list_documents()

    assert [record.document_id for record in first] == [
        record.document_id for record in second
    ]


def test_stored_filename_round_trips_as_string(tmp_path: Path) -> None:
    repository = DocumentRepository(tmp_path / "documents.json")
    repository.save_documents([_record("doc-1")])

    loaded = repository.list_documents()

    assert isinstance(loaded[0].stored_filename, str)
    assert loaded[0].stored_filename == "stored-doc-1.txt"
