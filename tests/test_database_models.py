"""Tests for the SQLAlchemy ORM metadata.

These tests inspect the in-memory schema only; they never connect to a
database.
"""

from sqlalchemy import CheckConstraint, MetaData, UniqueConstraint, inspect
from sqlalchemy.orm import InstrumentedAttribute

from src.database.base import DEFAULT_EMBEDDING_DIMENSION, Base
from src.database.models import ChunkModel, DocumentModel

_TABLE_NAMES = {"documents", "chunks"}


def test_only_project_tables_are_registered() -> None:
    assert set(Base.metadata.tables) == _TABLE_NAMES


def test_documents_primary_key() -> None:
    table = DocumentModel.__table__
    assert list(table.primary_key.columns) == [table.c.document_id]


def test_chunks_primary_key() -> None:
    table = ChunkModel.__table__
    assert list(table.primary_key.columns) == [table.c.chunk_id]


def test_chunks_document_fk_cascades() -> None:
    foreign_keys = list(ChunkModel.__table__.c.document_id.foreign_keys)
    assert len(foreign_keys) == 1
    assert foreign_keys[0].target_fullname == "documents.document_id"
    assert foreign_keys[0].ondelete == "CASCADE"


def test_unique_document_chunk_index() -> None:
    constraints = ChunkModel.__table__.constraints
    matching = [
        constraint
        for constraint in constraints
        if isinstance(constraint, UniqueConstraint)
        and set(constraint.columns) == {
            ChunkModel.__table__.c.document_id,
            ChunkModel.__table__.c.chunk_index,
        }
    ]
    assert len(matching) == 1


def test_embedding_column_uses_384_dimensions() -> None:
    column = ChunkModel.__table__.c.embedding
    assert str(column.type) == f"VECTOR({DEFAULT_EMBEDDING_DIMENSION})"
    assert DEFAULT_EMBEDDING_DIMENSION == 384


def test_metadata_database_column_exists() -> None:
    assert "metadata" in ChunkModel.__table__.c


def test_python_attribute_uses_chunk_metadata() -> None:
    # The Declarative attribute is chunk_metadata and it maps to the
    # "metadata" database column.  The reserved "metadata" class attribute
    # must remain the shared MetaData object, never a mapped column.
    assert hasattr(ChunkModel, "chunk_metadata")
    assert isinstance(ChunkModel.metadata, MetaData)
    assert not isinstance(ChunkModel.metadata, InstrumentedAttribute)


def test_status_check_constraint_exists() -> None:
    constraints = DocumentModel.__table__.constraints
    status_constraints = [
        constraint
        for constraint in constraints
        if isinstance(constraint, CheckConstraint)
        and "status" in constraint.sqltext.text
    ]
    assert len(status_constraints) == 1
    constraint_text = str(status_constraints[0].sqltext)
    for allowed in ("pending", "processing", "ready", "failed", "deleting"):
        assert allowed in constraint_text


def test_non_negative_check_constraints_exist() -> None:
    documents = DocumentModel.__table__
    for column_name in ("size_bytes", "text_length", "chunk_count"):
        matching = [
            constraint
            for constraint in documents.constraints
            if isinstance(constraint, CheckConstraint)
            and column_name in constraint.sqltext.text
            and ">=" in str(constraint.sqltext)
        ]
        assert matching, f"missing non-negative constraint for {column_name}"

    chunks = ChunkModel.__table__
    chunk_index_constraints = [
        constraint
        for constraint in chunks.constraints
        if isinstance(constraint, CheckConstraint)
        and "chunk_index" in constraint.sqltext.text
    ]
    assert chunk_index_constraints


def test_no_ann_indexes_are_created() -> None:
    for table in (DocumentModel.__table__, ChunkModel.__table__):
        for index in table.indexes:
            assert index.name is not None
            assert not index.name.startswith("hnsw"), f"unexpected HNSW index {index.name}"
            assert not index.name.startswith(
                "ivfflat"
            ), f"unexpected IVFFlat index {index.name}"


def test_repr_never_prints_embedding() -> None:
    chunk = ChunkModel(
        chunk_id="chunk-1",
        document_id="doc-1",
        chunk_index=0,
        filename="f.txt",
        content="text",
        page_number=None,
        chunk_metadata={},
        embedding=[0.0] * 384,
    )
    representation = repr(chunk)
    assert "embedding" not in representation


def test_relationships_configured() -> None:
    document_chunks = inspect(DocumentModel).relationships["chunks"]
    assert "delete" in document_chunks.cascade
    assert "delete-orphan" in document_chunks.cascade
    assert document_chunks.passive_deletes is True
    chunk_document = inspect(ChunkModel).relationships["document"]
    assert chunk_document.passive_deletes is True
