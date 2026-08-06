"""Alembic migration environment for the course-rag PostgreSQL schema.

Migrations are explicit operations: importing this module never connects to
the database and the application never runs ``alembic upgrade`` at startup.
The database URL is read from the DATABASE_URL environment variable and
overrides the placeholder in ``alembic.ini``.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from src.database.base import Base
from src.database.config import resolve_database_url
from src.database import models  # noqa: F401  (register ORM models)

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Importing src.database.models registers both tables on Base.metadata, so
# autogenerate and offline compilation see the full schema.
target_metadata = Base.metadata


def _configured_database_url() -> str:
    """Return the DATABASE_URL value required for any migration run."""
    database_url = resolve_database_url(required=True)
    assert database_url is not None  # required=True raises when missing
    return database_url


def run_migrations_offline() -> None:
    """Render migration SQL to a script without connecting to a database."""
    # Offline mode passes the raw URL straight to SQLAlchemy; no ConfigParser
    # interpolation happens on this path, so percent signs stay untouched.
    context.configure(
        url=_configured_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    # Alembic's ConfigParser treats ``%`` as an interpolation marker, so the
    # URL is stored with doubled percent signs; engine_from_config reads it
    # back through the parser, which restores the original ``%`` characters.
    config.set_main_option(
        "sqlalchemy.url",
        _configured_database_url().replace("%", "%%"),
    )
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
