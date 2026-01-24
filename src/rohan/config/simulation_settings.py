"""Simulation settings configuration. Defines the structure and default values for simulation parameters.
This is agnostic to the simulation engine used (e.g., ABIDES) and focuses on high-level settings."""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_TICKER: str = "FTSEMIB"


class LatencyType(str, Enum):
    """Specifies the type of latency model to be used in the simulation.
    See LatencyModel class in abides_core.latency_model for details."""

    NO_LATENCY = ("no_latency",)
    DETERMINISTIC = ("deterministic",)
    CUBIC = "cubic"


class LatencyModelSettings(BaseModel):
    """Configuration for latency model."""

    type: LatencyType = Field(default=LatencyType.DETERMINISTIC, description="Type of latency model to use in the simulation.")
    jitter: float = Field(default=0.5, description="Jitter parameter for cubic latency model.")
    jitter_clip: float = Field(default=0.1, description="Jitter clip parameter for cubic latency model.")
    jitter_unit: float = Field(default=10.0, description="Jitter unit parameter for cubic latency model.")


class SimulationSettings(BaseSettings):
    """Configuration for the simulation environment."""

    seed: int = Field(default=int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1), description="Random seed for the simulation.")
    date: str = Field(default="20210205", description="Date of the simulation.")
    start_time: str = Field(default="09:00:00", description="Start time of the simulation.")
    end_time: str = Field(default="10:00:00", description="End time of the simulation.")
    stdout_log_level: str = Field(default="INFO", description="Log level for stdout logging.")
    ticker: str = Field(default=DEFAULT_TICKER, description="Ticker symbol for the simulation.")
    starting_cash: int = Field(default=10_000_000, description="Starting cash for agents in CENTS.")
    log_orders: bool = Field(default=True, description="Whether to log all orders.")

    computation_delay_ns: int = Field(default=50, description="Computation delay in nanoseconds.")
    latency: LatencyModelSettings = Field(default_factory=LatencyModelSettings, description="Settings for the latency model.")

    model_config = SettingsConfigDict(
        env_file=".env",  # path to the .env file. Overwrite with _env_file when instantiating
        # env_file_encoding="utf-8", # encoding of the .env file
        env_prefix="sim_",  # prefix for environment variables
        env_nested_delimiter="__",  # delimiter for nested fields
        validate_default=True,  # validate default values
        case_sensitive=False,  # environment variable names are case insensitive
        env_ignore_empty=False,  # do not ignore empty environment variables
        enable_decoding=False,  # disable JSON parsing of all fields
        # alias_generator=AliasChoices.SNAKE_CASE,
        extra="forbid",  # forbid extra fields not defined in the model,
        use_enum_values=True,  # use enum values instead of names
        env_parse_enums=True,  # parse enums from environment variables
    )
