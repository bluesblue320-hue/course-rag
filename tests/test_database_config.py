"""Tests for database configuration resolution.

These tests never connect to a database and never read the process
environment unless the test explicitly passes a raw value.
"""

import pytest

from src.database.config import (
    resolve_database_url,
    resolve_database_url_for_backend,
    resolve_vector_store_backend,
)
from src.exceptions import DatabaseConfigurationError

_SECRET_PASSWORD = "super-secret-test-password"


def test_backend_defaults_to_memory() -> None:
    assert resolve_vector_store_backend(None) == "memory"


def test_backend_empty_string_defaults_to_memory() -> None:
    assert resolve_vector_store_backend("") == "memory"


def test_backend_whitespace_defaults_to_memory() -> None:
    assert resolve_vector_store_backend("   ") == "memory"


def test_backend_memory_normalizes_case_and_whitespace() -> None:
    assert resolve_vector_store_backend(" MEMORY ") == "memory"
    assert resolve_vector_store_backend("memory") == "memory"


def test_backend_pgvector_normalizes_case_and_whitespace() -> None:
    assert resolve_vector_store_backend(" PGVECTOR ") == "pgvector"
    assert resolve_vector_store_backend("PgVector") == "pgvector"


@pytest.mark.parametrize("invalid", ["redis", "postgres", "vector", "MYSQL"])
def test_backend_rejects_invalid_values(invalid: str) -> None:
    with pytest.raises(DatabaseConfigurationError):
        resolve_vector_store_backend(invalid)


def test_memory_backend_accepts_empty_database_url() -> None:
    assert resolve_database_url_for_backend("memory", None) is None
    assert resolve_database_url_for_backend("memory", "") is None


def test_pgvector_backend_requires_database_url() -> None:
    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url_for_backend("pgvector", None)
    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url_for_backend("pgvector", "   ")


def test_accepts_postgresql_psycopg_url() -> None:
    url = f"postgresql+psycopg://user:{_SECRET_PASSWORD}@localhost:5432/course_rag"
    assert resolve_database_url(url) == url


def test_rejects_sqlite_url() -> None:
    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url("sqlite:///course_rag.db")


def test_rejects_psycopg2_url() -> None:
    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url(
            f"postgresql+psycopg2://user:{_SECRET_PASSWORD}@localhost/course_rag"
        )


def test_rejects_bare_http_url() -> None:
    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url("http://localhost:5432/course_rag")


def test_rejects_bare_https_url() -> None:
    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url("https://localhost:5432/course_rag")


def test_rejects_garbage_url() -> None:
    with pytest.raises(DatabaseConfigurationError):
        resolve_database_url("not a url at all")


@pytest.mark.parametrize(
    "invalid_url",
    [
        f"sqlite:///course_rag.db?password={_SECRET_PASSWORD}",
        f"postgresql+psycopg2://user:{_SECRET_PASSWORD}@localhost/course_rag",
        f"postgresql://user:{_SECRET_PASSWORD}@localhost/course_rag",
    ],
)
def test_error_text_never_contains_password(invalid_url: str) -> None:
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        resolve_database_url(invalid_url)
    assert _SECRET_PASSWORD not in str(exc_info.value)


def test_error_message_is_stable() -> None:
    with pytest.raises(DatabaseConfigurationError) as exc_info:
        resolve_database_url_for_backend("pgvector", None)
    assert str(exc_info.value) == "数据库配置无效"

    with pytest.raises(DatabaseConfigurationError) as exc_info:
        resolve_vector_store_backend("bogus")
    assert str(exc_info.value) == "数据库配置无效"
