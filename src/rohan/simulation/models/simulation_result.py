"""Context and result models for simulation execution.

This module provides data structures for tracking simulation execution lifecycle:

- SimulationContext: Created at simulation start, tracks metadata during execution
- SimulationResult: Created at simulation end, contains output and execution status

These models enable uniform handling of single and batch simulations, with
comprehensive tracking of run IDs, timing, success/failure, and error information.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from rohan.config import SimulationSettings
from rohan.simulation.models.simulation_output import SimulationOutput


class SimulationContext(BaseModel):
    """Context information for a simulation run.

    Created at the start of simulation execution to track metadata
    and provide a consistent run identifier across the execution lifecycle.

    Example:
        >>> context = SimulationContext(settings=my_settings)
        >>> print(context.run_id)  # e.g., "a1b2c3d4-..."
        >>> context.metadata['experiment_name'] = 'high_volatility_test'
    """

    run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique identifier for this simulation run (auto-generated UUID)",
    )
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp when the simulation started",
    )
    settings: SimulationSettings = Field(
        default_factory=SimulationSettings,
        description="Configuration used for this simulation",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value pairs for tracking additional information",
    )

    model_config = {"arbitrary_types_allowed": True}


class SimulationResult(BaseModel):
    """Rich result object for simulation execution with metadata and error handling.

    Returned by SimulationService methods to provide comprehensive information
    about simulation outcomes, whether successful or failed.

    Uses a Result-type pattern where either `result` or `error` is set, but never both.
    This enables clean error handling and makes the success/failure state explicit.

    Example:
        >>> sim_result = service.run_simulation(settings)
        >>> if sim_result.error is None:
        ...     df = sim_result.result.get_order_book_l1()
        ... else:
        ...     print(f"Simulation {sim_result.context.run_id} failed: {sim_result.error}")
    """

    context: SimulationContext = Field(
        ...,
        description="Reference to the SimulationContext with run metadata",
    )
    duration_seconds: float = Field(
        ...,
        description="Wall-clock time taken to run the simulation",
        ge=0.0,
    )
    result: SimulationOutput | None = Field(
        default=None,
        description="The simulation output (None if failed)",
    )
    error: Exception | None = Field(
        default=None,
        description="Exception that caused failure (None if successful)",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=True,
        validate_default=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,  # To permit Exception and SimulationOutput types
    )

    def model_post_init(self, __context):
        """Validate that exactly one of result or error is set."""
        if self.result is not None and self.error is not None:
            raise ValueError("SimulationResult cannot have both result and error set")
        if self.result is None and self.error is None:
            raise ValueError("SimulationResult must have either result or error set")
