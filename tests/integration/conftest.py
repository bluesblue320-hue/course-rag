"""Shared fixtures and helpers for real-PostgreSQL integration tests.

These tests require a running PostgreSQL with the pgvector extension and an
applied Alembic migration.  They never use SQLite and never download models.
"""

import hashlib
import os

import numpy as np
import pytest
from sqlalchemy import create_engine, text

from src.database.base import DEFAULT_EMBEDDING_DIMENSION
from src.database.engine import create_session_factory
from src.storage.pgvector_store import PgVectorStore

DIMENSION = DEFAULT_EMBEDDING_DIMENSION


def integration_database_url() -> str:
    """Return the required real PostgreSQL connection string."""
    database_url = os.environ.get("DATABASE_URL")
    assert database_url and database_url.startswith(
        "postgresql+psycopg://"
    ), "集成测试需要真实的 DATABASE_URL (postgresql+psycopg://)"
    return database_url


def deterministic_vector(text: str) -> np.ndarray:
    """Return a deterministic, finite, normalized 384-dim unit vector.

    Python's builtin ``hash`` is intentionally avoided because it is
    randomized across processes; hashlib.sha256 is stable everywhere.
    """
    values: list[int] = []
    counter = 0
    while len(values) < DIMENSION:
        digest = hashlib.sha256(f"{text}:{counter}".encode("utf-8")).digest()
        values.extend(digest)
        counter += 1
    array = np.asarray(values[:DIMENSION], dtype=float) / 255.0
    array = array + 1e-9  # never a zero vector
    return array / np.linalg.norm(array)


def document_embeddings(chunks: list) -> np.ndarray:
    """Return one deterministic unit vector per chunk."""
    return np.stack([deterministic_vector(chunk.text) for chunk in chunks])


def query_embedding(query: str) -> np.ndarray:
    """Return the deterministic query vector for one text."""
    return deterministic_vector(query)


@pytest.fixture
def engine():
    """One SQLAlchemy engine bound to the real test database."""
    url = integration_database_url()
    engine = create_engine(url, pool_pre_ping=True)
    yield engine
    engine.dispose()


@pytest.fixture
def session_factory(engine):
    """A session factory bound to the integration engine."""
    return create_session_factory(engine)


@pytest.fixture
def store(session_factory, engine):
    """A PgVectorStore wired to the real database."""
    return PgVectorStore(session_factory, engine=engine)


@pytest.fixture
def cleanup_uploads(session_factory):
    """Remove all non-builtin documents after each test."""

    def _cleanup() -> None:
        with session_factory.begin() as session:
            session.execute(
                text(
                    "DELETE FROM documents "
                    "WHERE document_id <> 'builtin-knowledge'"
                )
            )

    yield _cleanup
    _cleanup()
