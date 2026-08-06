"""create pgvector schema (documents and chunks)

Revision ID: 9ecde52cf1bd
Revises:
Create Date: 2026-08-06

Enable the PostgreSQL vector extension, then create the ``documents`` and
``chunks`` tables with their constraints and plain B-tree indexes.  No ANN
(HNSW/IVFFlat) index is created in this foundation PR.

Downgrade drops the project tables but deliberately keeps the ``vector``
extension: the extension may be shared by other schemas or managed by the
database administrator.  For a full cleanup an administrator must explicitly
run ``DROP EXTENSION vector``.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from pgvector.sqlalchemy import VECTOR

from src.database.base import DEFAULT_EMBEDDING_DIMENSION

# revision identifiers, used by Alembic.
revision: str = "9ecde52cf1bd"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable the vector extension and create both tables."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("original_filename", sa.String(length=512), nullable=False),
        sa.Column("stored_filename", sa.String(length=128), nullable=True),
        sa.Column("content_type", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("text_length", sa.BigInteger(), nullable=False),
        sa.Column("chunk_count", sa.Integer(), nullable=False),
        sa.Column(
            "is_builtin",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            nullable=False,
            server_default=sa.text("'ready'"),
        ),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "size_bytes >= 0",
            name="ck_documents_size_bytes_non_negative",
        ),
        sa.CheckConstraint(
            "text_length >= 0",
            name="ck_documents_text_length_non_negative",
        ),
        sa.CheckConstraint(
            "chunk_count >= 0",
            name="ck_documents_chunk_count_non_negative",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'ready', 'failed', 'deleting')",
            name="ck_documents_status_values",
        ),
        sa.PrimaryKeyConstraint("document_id"),
    )

    op.create_table(
        "chunks",
        sa.Column("chunk_id", sa.String(length=64), nullable=False),
        sa.Column("document_id", sa.String(length=64), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("filename", sa.String(length=512), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "embedding",
            VECTOR(DEFAULT_EMBEDDING_DIMENSION),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.CheckConstraint(
            "chunk_index >= 0",
            name="ck_chunks_chunk_index_non_negative",
        ),
        sa.CheckConstraint(
            "page_number IS NULL OR page_number >= 1",
            name="ck_chunks_page_number_positive",
        ),
        sa.CheckConstraint("content <> ''", name="ck_chunks_content_not_empty"),
        sa.ForeignKeyConstraint(
            ["document_id"],
            ["documents.document_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("chunk_id"),
        sa.UniqueConstraint(
            "document_id",
            "chunk_index",
            name="uq_chunks_document_id_chunk_index",
        ),
    )
    op.create_index(
        "ix_chunks_document_id",
        "chunks",
        ["document_id"],
        unique=False,
    )


def downgrade() -> None:
    """Drop the project tables; the vector extension is intentionally kept."""
    op.drop_index("ix_chunks_document_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("documents")
