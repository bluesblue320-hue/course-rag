"""SQLAlchemy 2.x ORM models for the PostgreSQL persistence layer.

These models are independent of the domain dataclasses in ``src.documents``.
They exist so the schema is prepared for a later vector store integration;
the running application still uses the in-memory index and the JSON metadata
repository by default.
"""

from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.database.base import DEFAULT_EMBEDDING_DIMENSION, Base

#: Allowed values for the documents.status column.
DOCUMENT_STATUS_VALUES = ("pending", "processing", "ready", "failed", "deleting")


class DocumentModel(Base):
    """Persist one managed knowledge document."""

    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint("size_bytes >= 0", name="ck_documents_size_bytes_non_negative"),
        CheckConstraint(
            "text_length >= 0",
            name="ck_documents_text_length_non_negative",
        ),
        CheckConstraint(
            "chunk_count >= 0",
            name="ck_documents_chunk_count_non_negative",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed', 'deleting')",
            name="ck_documents_status_values",
        ),
    )

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    stored_filename: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
    )
    content_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    text_length: Mapped[int] = mapped_column(BigInteger, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False)
    is_builtin: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        server_default="ready",
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    chunks: Mapped[list["ChunkModel"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        """Return a compact, non-secret representation."""
        return (
            f"DocumentModel(document_id={self.document_id!r}, "
            f"status={self.status!r})"
        )


class ChunkModel(Base):
    """Persist one searchable chunk with its embedding vector."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunks_document_id_chunk_index",
        ),
        CheckConstraint(
            "chunk_index >= 0",
            name="ck_chunks_chunk_index_non_negative",
        ),
        CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="ck_chunks_page_number_positive",
        ),
        CheckConstraint(
            "content <> ''",
            name="ck_chunks_content_not_empty",
        ),
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # NOTE: "metadata" is reserved by the SQLAlchemy Declarative API, so the
    # Python attribute is named chunk_metadata and mapped to the "metadata"
    # database column explicitly.
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=func.jsonb_build_object(),
    )
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(DEFAULT_EMBEDDING_DIMENSION),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    document: Mapped[DocumentModel] = relationship(
        back_populates="chunks",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        """Return a compact representation that never prints the embedding."""
        return (
            f"ChunkModel(chunk_id={self.chunk_id!r}, "
            f"document_id={self.document_id!r}, "
            f"chunk_index={self.chunk_index!r})"
        )
