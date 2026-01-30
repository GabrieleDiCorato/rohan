from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Configuration for the database connection."""

    connection_string: str = Field(
        default="sqlite:///./rohan.db",
        description="SQLAlchemy connection string for the database.",
    )
    pool_size: int = Field(default=10, description="Database connection pool size.")
    max_overflow: int = Field(
        default=20,
        description="Maximum number of connections to create beyond pool_size.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="db_",
        case_sensitive=False,
        extra="ignore",
    )
