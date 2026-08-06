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
    """Return the DATABASE_URL value required for any migration run.

    Alembic's ConfigParser interprets ``%`` as an interpolation marker, so
    escaped percent signs are unescaped for SQLAlchemy.
    """
    database_url = resolve_database_url(required=True)
    assert database_url is not None  # required=True raises when missing
    return database_url.replace("%", "%%")


def run_migrations_offline() -> None:
    """Render migration SQL to a script without connecting to a database."""
    url = _configured_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live database connection."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _configured_database_url()
    connectable = engine_from_config(
        section,
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
