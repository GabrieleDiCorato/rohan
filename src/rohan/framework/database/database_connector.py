from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from rohan.config import DatabaseSettings

from .models import Base


class DatabaseConnector:
    """
    Singleton class to manage database connection and sessions.
    """

    _instance = None
    _engine: Engine | None = None
    _session_maker: sessionmaker[Session] | None = None

    def __new__(cls) -> "DatabaseConnector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the database engine and session factory."""
        settings = DatabaseSettings()
        self._engine = create_engine(
            settings.connection_string,
            pool_size=settings.pool_size,
            max_overflow=settings.max_overflow,
            future=True,
        )
        self._session_maker = sessionmaker(autocommit=False, autoflush=False, bind=self._engine)

    @property
    def engine(self) -> Engine:
        """Return the SQLAlchemy Engine."""
        if self._engine is None:
            self._initialize()
        assert self._engine is not None
        return self._engine

    def get_session(self) -> Session:
        """Return a new Session instance."""
        if self._session_maker is None:
            self._initialize()
        assert self._session_maker is not None
        return self._session_maker()

    def create_tables(self) -> None:
        """Create all tables defined in models.py."""
        Base.metadata.create_all(bind=self.engine)


def get_db() -> DatabaseConnector:
    """Functions as a dependency to get the Database instance."""
    return DatabaseConnector()
