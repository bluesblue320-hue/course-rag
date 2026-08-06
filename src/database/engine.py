"""Create SQLAlchemy engines and session factories without global state.

Importing this module never connects to a database.  Engines are created
explicitly via :func:`create_database_engine` and must be disposed by the
caller once they are no longer needed.
"""

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.exceptions import DatabaseConnectionError

_CONNECTION_CHECK_SQL = text("SELECT 1")


def create_database_engine(database_url: str) -> Engine:
    """Create a Psycopg 3 SQLAlchemy engine.

    ``pool_pre_ping`` revalidates pooled connections so a restarted database
    does not serve stale connections to later requests.
    """
    return create_engine(
        database_url,
        pool_pre_ping=True,
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
