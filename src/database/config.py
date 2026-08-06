"""Resolve vector store backend and PostgreSQL connection configuration.

Configuration is resolved lazily by explicit function calls.  Importing this
module never connects to a database and never prints secrets.  ``memory``
mode completely ignores ``DATABASE_URL``: it is neither read, validated, nor
connected to.
"""

import os
from typing import Literal

from src.exceptions import DatabaseConfigurationError

VectorStoreBackend = Literal["memory", "pgvector"]

#: Environment variable that selects the active vector store backend.
VECTOR_STORE_BACKEND_ENV = "VECTOR_STORE_BACKEND"

#: Environment variable holding the PostgreSQL connection string.
DATABASE_URL_ENV = "DATABASE_URL"

#: Accepted DATABASE_URL scheme.  Only Psycopg 3 is supported; sqlite,
#: psycopg2, and bare http(s) URLs are rejected by the prefix check.
_DATABASE_URL_PREFIX = "postgresql+psycopg://"

_INVALID_BACKEND_MESSAGE = "数据库配置无效"
_INVALID_URL_MESSAGE = "数据库配置无效"


def _redact(raw_value: str) -> str:
    """Return a stable redacted value suitable for error messages."""
    del raw_value
    return "postgresql+psycopg://***@***:***/***"


def resolve_vector_store_backend(
    raw_value: str | None = None,
) -> VectorStoreBackend:
    """Return the normalized vector store backend.

    When ``raw_value`` is not given the ``VECTOR_STORE_BACKEND`` environment
    variable is read instead.  Missing or blank values default to
    ``memory``.  The backend value is case-insensitive and leading/trailing
    whitespace is ignored.  Any value other than ``memory`` or ``pgvector``
    raises :class:`DatabaseConfigurationError`.
    """
    value = raw_value if raw_value is not None else os.getenv(
        VECTOR_STORE_BACKEND_ENV
    )
    value = (value or "").strip().lower()
    if value in ("", "memory"):
        return "memory"
    if value == "pgvector":
        return "pgvector"
    raise DatabaseConfigurationError(_INVALID_BACKEND_MESSAGE)


def _validate_database_url(database_url: str) -> str:
    """Validate one non-empty connection string and return it unchanged."""
    candidate = database_url.strip()
    if not candidate:
        raise DatabaseConfigurationError(_INVALID_URL_MESSAGE)
    if not candidate.startswith(_DATABASE_URL_PREFIX):
        raise DatabaseConfigurationError(_INVALID_URL_MESSAGE)
    return candidate


def resolve_database_url(
    raw_value: str | None = None,
    *,
    required: bool = False,
) -> str | None:
    """Resolve the DATABASE_URL environment value.

    Returns ``None`` when the value is missing or blank and not required.
    When ``required`` is true a missing or invalid value raises
    :class:`DatabaseConfigurationError` with no secrets in the message.
    """
    value = raw_value if raw_value is not None else os.getenv(DATABASE_URL_ENV)
    if value is None or not value.strip():
        if required:
            raise DatabaseConfigurationError(_INVALID_URL_MESSAGE)
        return None
    return _validate_database_url(value)


def resolve_database_url_for_backend(
    backend: VectorStoreBackend,
    raw_value: str | None = None,
) -> str | None:
    """Resolve the database URL that a backend requires.

    The ``memory`` backend completely ignores ``DATABASE_URL``: it never
    reads the environment, never validates any value, and always returns
    ``None``.  The ``pgvector`` backend requires a valid
    ``postgresql+psycopg://`` URL.
    """
    if backend == "memory":
        return None
    return resolve_database_url(raw_value, required=True)


def redacted_database_url() -> str:
    """Return a stable placeholder for logs that never contains secrets."""
    return _redact("")
