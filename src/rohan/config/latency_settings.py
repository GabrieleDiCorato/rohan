from enum import StrEnum

from pydantic import BaseModel, Field


class LatencyType(StrEnum):
    """Specifies the type of latency model to be used in the simulation.
    See LatencyModel class in abides_core.latency_model for details."""

    NO_LATENCY = "no_latency"
    DETERMINISTIC = "deterministic"
    CUBIC = "cubic"


class LatencyModelSettings(BaseModel):
    """Configuration for latency model."""

    type: LatencyType = Field(default=LatencyType.DETERMINISTIC, description="Type of latency model to use in the simulation.")
    jitter: float = Field(default=0.5, description="Jitter parameter for cubic latency model.")
    jitter_clip: float = Field(default=0.1, description="Jitter clip parameter for cubic latency model.")
    jitter_unit: float = Field(default=10.0, description="Jitter unit parameter for cubic latency model.")
