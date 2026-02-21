"""Database connection management with scoped sessions."""

import functools
import logging

from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from rohan.config import DatabaseSettings

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Manages database connection pool and thread-local Sessions.

    Instantiate directly for custom configuration (e.g. in tests with an
    in-memory SQLite URL).  For normal application use, prefer the module-level
    ``get_database_connector()`` factory which returns a single long-lived
    instance per process via ``@functools.lru_cache``.

    Uses ``scoped_session`` to provide thread-local sessions, ensuring
    proper cleanup and avoiding session leaks across threads.
    """

    def __init__(self) -> None:
        """Initialize the database engine and session factory."""
        settings = DatabaseSettings()

        # SQLite does not support pool_size / max_overflow arguments.
        engine_kwargs: dict = {"future": True}
        if not settings.connection_string.startswith("sqlite"):
            engine_kwargs["pool_size"] = settings.pool_size
            engine_kwargs["max_overflow"] = settings.max_overflow

        self._engine: Engine = create_engine(settings.connection_string, **engine_kwargs)
        self._session_factory: sessionmaker[Session] = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
        self._scoped_session: scoped_session[Session] = scoped_session(self._session_factory)
        logger.info(
            "Database connector initialised: %s",
            settings.connection_string.split("@")[-1] if "@" in settings.connection_string else settings.connection_string,
        )

    @property
    def engine(self) -> Engine:
        """Return the SQLAlchemy Engine."""
        return self._engine

    def get_session(self) -> Session:
        """Return a scoped (thread-local) Session instance.

        The scoped session ensures that within a single thread, the same
        session is returned, and it is automatically removed at the end
        of the request/scope.
        """
        return self._scoped_session()

    def remove_session(self) -> None:
        """Remove the current scoped session, releasing the connection back to the pool."""
        self._scoped_session.remove()

    def create_tables(self) -> None:
        """Create all tables defined in models.py."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def dispose(self) -> None:
        """Dispose of the engine and all connections.  Used for clean shutdown."""
        self._scoped_session.remove()
        self._engine.dispose()
        logger.info("Database engine disposed")


@functools.lru_cache(maxsize=1)
def get_database_connector() -> DatabaseConnector:
    """Return the process-wide ``DatabaseConnector`` singleton.

    ``@lru_cache(maxsize=1)`` is safe here because ``DatabaseConnector`` is a
    lightweight, intentionally long-lived object (it holds a connection pool,
    not large data payloads).  This avoids the surprising ``__new__``-based
    singleton anti-pattern while still ensuring a single connector per process.

    In tests, call ``get_database_connector.cache_clear()`` before each test
    that needs a fresh connector, or inject a ``DatabaseConnector`` directly:

        def test_something():
            db = DatabaseConnector()  # fresh instance, custom URL via env
            repo = ArtifactStore(db)
            ...
    """
    return DatabaseConnector()


def get_db() -> DatabaseConnector:
    """Convenience alias kept for backward compatibility.  Prefer ``get_database_connector()``."""
    return get_database_connector()
