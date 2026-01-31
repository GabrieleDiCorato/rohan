"""Simulation settings configuration. Defines the structure and default values for simulation parameters.
This is agnostic to the simulation engine used (e.g., ABIDES) and focuses on high-level settings.
"""

from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .agent_settings import AgentSettings
from .latency_settings import LatencyModelSettings

# TODO this is hardcoded in ABIDES, will be fixed in a new version
DEFAULT_TICKER: str = "ABM"


class SimulationEngine(str, Enum):
    """Enumeration of supported simulation engines."""

    ABIDES = "ABIDES"


class SimulationSettings(BaseSettings):
    """Configuration for the simulation environment."""

    engine: SimulationEngine = Field(default=SimulationEngine.ABIDES, description="The simulation engine to use.")
    seed: int = Field(
        default_factory=lambda: int(datetime.now().timestamp() * 1_000_000) % (2**32 - 1),
        description="Random seed for the simulation.",
    )
    date: str = Field(default="20210205", description="Date of the simulation.")
    start_time: str = Field(default="09:30:00", description="Start time of the simulation.")
    end_time: str = Field(default="10:00:00", description="End time of the simulation.")
    stdout_log_level: str = Field(default="INFO", description="Log level for stdout logging.")
    ticker: str = Field(default=DEFAULT_TICKER, description="Ticker symbol for the simulation.")
    starting_cash: int = Field(default=10_000_000, description="Starting cash for agents in CENTS.")
    log_orders: bool = Field(default=True, description="Whether to log all orders.")

    computation_delay_ns: int = Field(default=50, description="Computation delay in nanoseconds.")
    latency: LatencyModelSettings = Field(
        default_factory=LatencyModelSettings,
        description="Settings for the latency model.",
    )

    agents: AgentSettings = Field(
        default_factory=AgentSettings,
        description="Settings for the agents in the simulation.",
    )

    @field_validator("end_time")
    @classmethod
    def validate_minimum_duration(cls, end_time_str: str, info) -> str:
        """Ensure simulation runs for at least 5 minutes to avoid edge cases like division by zero."""
        if "start_time" in info.data:
            start = datetime.strptime(info.data["start_time"], "%H:%M:%S").time()
            end = datetime.strptime(end_time_str, "%H:%M:%S").time()

            # Convert to seconds since midnight for comparison
            start_seconds = start.hour * 3600 + start.minute * 60 + start.second
            end_seconds = end.hour * 3600 + end.minute * 60 + end.second
            duration_seconds = end_seconds - start_seconds

            if duration_seconds < 300:  # Less than 5 minutes (300 seconds)
                raise ValueError(
                    f"Simulation duration must be at least 5 minutes (300 seconds). " f"Current duration: {duration_seconds} seconds " f"(from {info.data['start_time']} to {end_time_str})"
                )
        return end_time_str

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
