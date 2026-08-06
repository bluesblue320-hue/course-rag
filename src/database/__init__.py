"""PostgreSQL + pgvector infrastructure for the RAG project.

This package only provisions the database layer.  The running application
keeps using the in-memory KnowledgeIndex and the JSON metadata repository
unless the vector store backend is explicitly switched to ``pgvector``.
"""

from src.database.base import DEFAULT_EMBEDDING_DIMENSION, Base
from src.database.config import (
    resolve_database_url,
    resolve_database_url_for_backend,
    resolve_vector_store_backend,
)
from src.database.engine import (
    check_database_connection,
    create_database_engine,
    create_session_factory,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIMENSION",
    "Base",
    "check_database_connection",
    "create_database_engine",
    "create_session_factory",
    "resolve_database_url",
    "resolve_database_url_for_backend",
    "resolve_vector_store_backend",
]
