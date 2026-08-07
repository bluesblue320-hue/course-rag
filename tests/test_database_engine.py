"""Tests for engine/session factory creation and connection failure handling.

These tests never talk to a real PostgreSQL instance.  Connection failures
are simulated by patching the engine's ``connect`` method; the driver error
is converted to :class:`DatabaseConnectionError` and the public message must
not leak the connection string or its password.

Alembic readiness failures (missing migration directory, unreadable config,
multiple migration heads) are converted into the stable
:class:`DatabaseSchemaError` before any database connection is attempted.
"""

import configparser
import subprocess
import sys
from pathlib import Path

import pytest
from alembic.util.exc import CommandError as AlembicCommandError
from sqlalchemy import Engine
from sqlalchemy.orm import sessionmaker

import src.database.engine as engine_module
from src.database.engine import (
    check_database_connection,
    check_database_schema,
    create_database_engine,
    create_session_factory,
)
from src.exceptions import DatabaseConnectionError, DatabaseSchemaError

_URL = "postgresql+psycopg://user:super-secret-test-password@127.0.0.1:1/course_rag"

REPO_ROOT = Path(__file__).resolve().parents[1]

_SCHEMA_NOT_READY = "数据库结构尚未准备完成"


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


# ---------------------------------------------------------------------------
# Alembic readiness conversion
# ---------------------------------------------------------------------------


def _schema_error_from_local_head(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> DatabaseSchemaError:
    """Run check_database_schema with a failing local-head resolution."""
    engine = create_database_engine(_URL)

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(engine_module, "_resolve_local_head", _fail)
    try:
        with pytest.raises(DatabaseSchemaError) as exc_info:
            check_database_schema(engine, script_location="migrations")
        return exc_info.value
    finally:
        engine.dispose()


def test_missing_migration_directory_is_schema_error(tmp_path: Path) -> None:
    engine = create_database_engine(_URL)
    try:
        with pytest.raises(DatabaseSchemaError) as exc_info:
            check_database_schema(
                engine,
                script_location=tmp_path / "no-such-migrations",
            )
        message = str(exc_info.value)
        assert message == _SCHEMA_NOT_READY
        assert str(tmp_path) not in message
    finally:
        engine.dispose()


def test_alembic_command_error_is_schema_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _schema_error_from_local_head(
        monkeypatch,
        AlembicCommandError("Path doesn't exist: /secret/migrations"),
    )
    assert str(error) == _SCHEMA_NOT_READY
    assert "secret" not in str(error)
    assert "migrations" not in str(error)


def test_multiple_migration_heads_is_schema_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _schema_error_from_local_head(
        monkeypatch,
        AlembicCommandError("The script directory has multiple heads"),
    )
    assert str(error) == _SCHEMA_NOT_READY


def test_invalid_alembic_config_is_schema_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _schema_error_from_local_head(
        monkeypatch,
        configparser.Error("source contains parsing errors"),
    )
    assert str(error) == _SCHEMA_NOT_READY


def test_unreadable_alembic_ini_is_schema_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = _schema_error_from_local_head(
        monkeypatch,
        OSError("permission denied"),
    )
    assert str(error) == _SCHEMA_NOT_READY
    assert "permission" not in str(error)


def test_schema_error_chains_original_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = AlembicCommandError("multiple heads")
    error = _schema_error_from_local_head(monkeypatch, original)
    assert error.__cause__ is original
