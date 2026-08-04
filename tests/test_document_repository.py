"""Offline tests for the JSON document metadata repository."""

from pathlib import Path

import pytest

from src.document_repository import DocumentRepository
from src.documents import DocumentRecord
from src.exceptions import DocumentMetadataError


def _stored_name(document_id: str) -> str:
    hex_slug = "".join(
        character
        for character in document_id
        if character in "0123456789abcdef"
    )
    return f"{hex_slug}{'a' * (32 - len(hex_slug))}.txt"


def _record(
    document_id: str,
    *,
    created_at: str = "2026-08-03T08:00:00Z",
    is_builtin: bool = False,
) -> DocumentRecord:
    return DocumentRecord(
        document_id=document_id,
        original_filename=f"{document_id}.txt",
        stored_filename=None if is_builtin else _stored_name(document_id),
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
    repository.save_documents([_record("doc1")])

    loaded = repository.list_documents()

    assert isinstance(loaded[0].stored_filename, str)
    assert loaded[0].stored_filename == _stored_name("doc1")


def _write_payload(tmp_path: Path, records: list[dict[str, object]]) -> Path:
    path = tmp_path / "documents.json"
    path.write_text(
        __import__("json").dumps({"documents": records}),
        encoding="utf-8",
    )
    return path


def _record_payload(
    *,
    stored_filename: object,
    is_builtin: bool = False,
) -> dict[str, object]:
    return {
        "document_id": "doc-1",
        "original_filename": "notes.txt",
        "stored_filename": stored_filename,
        "content_type": "text/plain",
        "size_bytes": 10,
        "text_length": 10,
        "chunk_count": 1,
        "created_at": "2026-08-03T08:00:00Z",
        "is_builtin": is_builtin,
    }


@pytest.mark.parametrize(
    "stored_filename",
    [
        "../../escape.txt",
        "../escape.txt",
        "nested/file.txt",
        "..\\escape.txt",
        "..\\..\\escape.txt",
        "C:/absolute/escape.txt",
        "C:\\absolute\\escape.txt",
        "/absolute/escape.txt",
        "escape.txt",
        "a" * 31 + ".txt",
        "A" * 32 + ".txt",
        "a" * 32 + ".TXT",
        "a" * 32 + ".docx",
        "a" * 32 + ".pdf.exe",
        "aaaa" + ".txt",
    ],
)
def test_repository_rejects_dangerous_stored_filenames(
    tmp_path: Path,
    stored_filename: str,
) -> None:
    path = _write_payload(
        tmp_path,
        [_record_payload(stored_filename=stored_filename)],
    )

    with pytest.raises(DocumentMetadataError):
        DocumentRepository(path).list_documents()


@pytest.mark.parametrize("suffix", ["txt", "md", "pdf"])
def test_repository_accepts_valid_uuid_stored_filenames(
    tmp_path: Path,
    suffix: str,
) -> None:
    stored_filename = f"{'c' * 32}.{suffix}"
    path = _write_payload(
        tmp_path,
        [_record_payload(stored_filename=stored_filename)],
    )

    loaded = DocumentRepository(path).list_documents()

    assert loaded[0].stored_filename == stored_filename


def test_repository_rejects_upload_without_stored_filename(
    tmp_path: Path,
) -> None:
    path = _write_payload(tmp_path, [_record_payload(stored_filename=None)])

    with pytest.raises(DocumentMetadataError):
        DocumentRepository(path).list_documents()


def test_repository_rejects_builtin_with_stored_filename(
    tmp_path: Path,
) -> None:
    path = _write_payload(
        tmp_path,
        [_record_payload(stored_filename=f"{'d' * 32}.txt", is_builtin=True)],
    )

    with pytest.raises(DocumentMetadataError):
        DocumentRepository(path).list_documents()


def test_repository_accepts_builtin_without_stored_filename(
    tmp_path: Path,
) -> None:
    path = _write_payload(
        tmp_path,
        [_record_payload(stored_filename=None, is_builtin=True)],
    )

    loaded = DocumentRepository(path).list_documents()

    assert loaded[0].is_builtin is True
    assert loaded[0].stored_filename is None
