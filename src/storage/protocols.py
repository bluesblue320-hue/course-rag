"""Application-level storage protocols shared by both backends.

These are the *only* contracts the API layer depends on.  They deliberately
hide SQLAlchemy sessions, ORM models, database URLs, and transaction details
so swapping between the ``memory`` and ``pgvector`` runtimes never touches
route logic.
"""

from typing import Protocol

from src.documents import DocumentRecord


class ChunkRetrieverProtocol(Protocol):
    """Describe the search surface used by retrieval and RAG services."""

    @property
    def chunk_count(self) -> int:
        """Return the number of ready chunks in the active store."""
        ...

    def search(
        self,
        query_embedding: object,
        top_k: int,
    ) -> list[dict[str, object]]:
        """Return the top-k chunks ranked by cosine similarity."""
        ...

    def close(self) -> None:
        """Release any owned resources; may be called more than once."""
        ...


class DocumentManagerProtocol(Protocol):
    """Describe the document management surface used by the API."""

    @property
    def max_upload_bytes(self) -> int:
        """Return the enforced upload size limit in bytes."""
        ...

    @property
    def chunk_count(self) -> int:
        """Return the number of ready chunks in the active store."""
        ...

    def list_documents(self) -> list[DocumentRecord]:
        """Return all ready documents in stable display order."""
        ...

    def get_document(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        """Return one ready document by id or None when absent."""
        ...

    def ingest(
        self,
        filename: str,
        content_type: str,
        data: bytes,
    ) -> DocumentRecord:
        """Validate, store, and index one uploaded document."""
        ...

    def delete_document(
        self,
        document_id: str,
    ) -> DocumentRecord:
        """Delete one uploaded document and its chunks."""
        ...

    def close(self) -> None:
        """Release any owned resources; may be called more than once."""
        ...
