"""Domain models for uploaded and built-in knowledge documents."""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable


@dataclass(frozen=True)
class DocumentRecord:
    """Describe one knowledge document tracked by the repository."""

    document_id: str
    original_filename: str
    stored_filename: str | None
    content_type: str
    size_bytes: int
    text_length: int
    chunk_count: int
    created_at: str
    is_builtin: bool


@dataclass(frozen=True)
class LoadedPage:
    """Describe one extracted page with an optional one-based number."""

    page_number: int | None
    text: str


@dataclass(frozen=True)
class LoadedDocument:
    """Describe one parsed document as an ordered tuple of pages."""

    pages: tuple[LoadedPage, ...]
    text_length: int


@dataclass(frozen=True)
class ChunkRecord:
    """Describe one searchable chunk with its source metadata."""

    chunk_id: str
    document_id: str
    filename: str
    text: str
    chunk_index: int
    page_number: int | None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a Z suffix."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _created_at_timestamp(record: DocumentRecord) -> float:
    """Parse a created_at timestamp, treating invalid values as oldest."""
    try:
        parsed = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
        return parsed.timestamp()
    except ValueError:
        return float("-inf")


def sort_documents(documents: Iterable[DocumentRecord]) -> list[DocumentRecord]:
    """Return built-in documents first, then uploads by newest created_at."""
    return sorted(
        documents,
        key=lambda record: (
            0 if record.is_builtin else 1,
            -_created_at_timestamp(record),
            record.document_id,
        ),
    )
