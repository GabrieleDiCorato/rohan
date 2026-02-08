"""Database connection management with scoped sessions."""

import logging

from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, scoped_session, sessionmaker

from rohan.config import DatabaseSettings

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """
    Singleton class to manage database connection and sessions.

    Uses ``scoped_session`` to provide thread-local sessions, ensuring
    proper cleanup and avoiding session leaks across threads.
    """

    _instance = None
    _engine: Engine | None = None
    _session_factory: sessionmaker[Session] | None = None
    _scoped_session: scoped_session[Session] | None = None

    def __new__(cls) -> "DatabaseConnector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the database engine and session factory."""
        settings = DatabaseSettings()

        # SQLite does not support pool_size / max_overflow arguments.
        engine_kwargs: dict = {"future": True}
        if not settings.connection_string.startswith("sqlite"):
            engine_kwargs["pool_size"] = settings.pool_size
            engine_kwargs["max_overflow"] = settings.max_overflow

        self._engine = create_engine(settings.connection_string, **engine_kwargs)
        self._session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)
        self._scoped_session = scoped_session(self._session_factory)
        logger.info(
            "Database connector initialised: %s",
            settings.connection_string.split("@")[-1] if "@" in settings.connection_string else settings.connection_string,
        )

    @property
    def engine(self) -> Engine:
        """Return the SQLAlchemy Engine."""
        if self._engine is None:
            self._initialize()
        assert self._engine is not None
        return self._engine

    def get_session(self) -> Session:
        """Return a scoped (thread-local) Session instance.

        The scoped session ensures that within a single thread, the same
        session is returned, and it is automatically removed at the end
        of the request/scope.
        """
        if self._scoped_session is None:
            self._initialize()
        assert self._scoped_session is not None
        return self._scoped_session()

    def remove_session(self) -> None:
        """Remove the current scoped session, releasing the connection back to the pool."""
        if self._scoped_session is not None:
            self._scoped_session.remove()

    def create_tables(self) -> None:
        """Create all tables defined in models.py."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created successfully")

    def dispose(self) -> None:
        """Dispose of the engine and all connections.  Used for clean shutdown."""
        if self._scoped_session is not None:
            self._scoped_session.remove()
        if self._engine is not None:
            self._engine.dispose()
            logger.info("Database engine disposed")

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton instance. Used primarily in testing."""
        if cls._instance is not None:
            cls._instance.dispose()
        cls._instance = None
        cls._engine = None
        cls._session_factory = None
        cls._scoped_session = None


def get_db() -> DatabaseConnector:
    """Functions as a dependency to get the Database instance."""
    return DatabaseConnector()
