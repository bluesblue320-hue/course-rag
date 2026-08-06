"""Tests for engine/session factory creation and connection failure handling.

These tests never talk to a real PostgreSQL instance.  Connection failures
are simulated by patching the engine's ``connect`` method; the driver error
is converted to :class:`DatabaseConnectionError` and the public message must
not leak the connection string or its password.
"""

import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

from src.database.engine import (
    check_database_connection,
    create_database_engine,
    create_session_factory,
)
from src.exceptions import DatabaseConnectionError

_URL = "postgresql+psycopg://user:super-secret-test-password@127.0.0.1:1/course_rag"

REPO_ROOT = Path(__file__).resolve().parents[1]


def _engine_with_failing_connect(monkeypatch: pytest.MonkeyPatch) -> Engine:
    engine = create_database_engine(_URL)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("connection refused")

    monkeypatch.setattr(engine, "connect", _fail)
    return engine


def test_import_does_not_connect() -> None:
    """Importing the engine module must not attempt any network access."""
    result = subprocess.run(
        [sys.executable, "-c", "import src.database.engine; print('ok')"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0
    assert result.stdout.strip() == "ok"


def test_create_database_engine_returns_engine() -> None:
    engine = create_database_engine(_URL)
    assert isinstance(engine, Engine)
    engine.dispose()


def test_engine_uses_pool_pre_ping() -> None:
    engine = create_database_engine(_URL)
    try:
        assert engine.pool._pre_ping is True
    finally:
        engine.dispose()


def test_session_factory_is_constructible() -> None:
    engine = create_database_engine(_URL)
    try:
        session_factory = create_session_factory(engine)
        assert isinstance(session_factory, sessionmaker)
        session = session_factory()
        assert session is not None
        session.close()
    finally:
        engine.dispose()


def test_connection_error_is_converted(monkeypatch: pytest.MonkeyPatch) -> None:
    engine = _engine_with_failing_connect(monkeypatch)
    try:
        with pytest.raises(DatabaseConnectionError) as exc_info:
            check_database_connection(engine)
        assert str(exc_info.value) == "数据库连接不可用"
    finally:
        engine.dispose()


def test_connection_error_chains_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_with_failing_connect(monkeypatch)
    try:
        with pytest.raises(DatabaseConnectionError) as exc_info:
            check_database_connection(engine)
        assert exc_info.value.__cause__ is not None
    finally:
        engine.dispose()


def test_connection_error_never_leaks_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = _engine_with_failing_connect(monkeypatch)
    try:
        with pytest.raises(DatabaseConnectionError) as exc_info:
            check_database_connection(engine)
        message = str(exc_info.value)
        assert "super-secret-test-password" not in message
        assert "127.0.0.1" not in message
    finally:
        engine.dispose()
