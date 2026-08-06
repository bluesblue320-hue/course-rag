"""Declarative Base and shared schema constants for the database layer."""

from sqlalchemy.orm import DeclarativeBase

#: Embedding dimension used by the default sentence-transformers model
#: (paraphrase-multilingual-MiniLM-L12-v2).  The pgvector column is created
#: with this dimension and must match the vectors written in a later PR.
DEFAULT_EMBEDDING_DIMENSION = 384


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""

    pass


if DEFAULT_EMBEDDING_DIMENSION <= 0:
    raise RuntimeError("DEFAULT_EMBEDDING_DIMENSION 必须为正整数")
