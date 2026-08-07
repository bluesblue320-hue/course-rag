"""Create SQLAlchemy engines and session factories without global state.

Importing this module never connects to a database.  Engines are created
explicitly via :func:`create_database_engine` and must be disposed by the
caller once they are no longer needed.
"""

from pathlib import Path

from alembic.config import Config as AlembicConfig
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine, create_engine, inspect, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from src.exceptions import (
    DatabaseConnectionError,
    DatabaseSchemaError,
)

_CONNECTION_CHECK_SQL = text("SELECT 1")
_SCHEMA_NOT_READY = "数据库结构尚未准备完成"


def create_database_engine(database_url: str) -> Engine:
    """Create a Psycopg 3 SQLAlchemy engine.

    ``pool_pre_ping`` revalidates pooled connections so a restarted database
    does not serve stale connections to later requests.  A 5-second
    ``connect_timeout`` makes a down database fail fast during startup
    instead of hanging the degraded-mode initialization.
    """
    return create_engine(
        database_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a sessionmaker bound to one engine."""
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def check_database_connection(engine: Engine) -> None:
    """Verify the database is reachable with a minimal ``SELECT 1`` probe.

    Raises :class:`DatabaseConnectionError` (chained from the driver
    exception) when the database cannot be reached.  The public message never
    contains the connection URL, credentials, or driver error text.
    """
    try:
        with engine.connect() as connection:
            connection.execute(_CONNECTION_CHECK_SQL)
    except Exception as exc:  # conversion boundary for driver failures
        raise DatabaseConnectionError("数据库连接不可用") from exc


def _resolve_local_head(
    script_location: str | Path,
    alembic_config_path: str | Path | None,
) -> str | None:
    """Return the local Alembic migration head revision id."""
    config = (
        AlembicConfig(str(alembic_config_path))
        if alembic_config_path is not None
        else AlembicConfig()
    )
    config.set_main_option("script_location", str(script_location))
    script = ScriptDirectory.from_config(config)
    return script.get_current_head()


def _table_exists(connection: object, name: str) -> bool:
    """Return whether one table exists in the connected database."""
    return inspect(connection).has_table(name)


def _vector_extension_available(connection: object) -> bool:
    """Return whether the pgvector ``vector`` extension is installed."""
    row = connection.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    ).first()
    return row is not None


def check_database_schema(
    engine: Engine,
    *,
    script_location: str | Path = "migrations",
    alembic_config_path: str | Path | None = None,
) -> None:
    """Verify the database is migrated to the current Alembic head.

    This is a read-only readiness probe.  It never applies migrations,
    never creates tables, and never modifies the schema.  The database
    revision must equal the local migration head, the ``vector`` extension
    must be installed, and both ``documents`` and ``chunks`` tables must
    exist; otherwise :class:`DatabaseSchemaError` is raised.
    """
    local_head = _resolve_local_head(script_location, alembic_config_path)
    try:
        with engine.connect() as connection:
            if not _table_exists(connection, "alembic_version"):
                raise DatabaseSchemaError(_SCHEMA_NOT_READY)
            context = MigrationContext.configure(connection)
            current_revision = context.get_current_revision()
            if current_revision != local_head:
                raise DatabaseSchemaError(_SCHEMA_NOT_READY)
            if not _vector_extension_available(connection):
                raise DatabaseSchemaError(_SCHEMA_NOT_READY)
            if not _table_exists(connection, "documents"):
                raise DatabaseSchemaError(_SCHEMA_NOT_READY)
            if not _table_exists(connection, "chunks"):
                raise DatabaseSchemaError(_SCHEMA_NOT_READY)
    except DatabaseSchemaError:
        raise
    except OperationalError as exc:
        raise DatabaseConnectionError("数据库连接不可用") from exc
    except SQLAlchemyError as exc:
        raise DatabaseSchemaError(_SCHEMA_NOT_READY) from exc
