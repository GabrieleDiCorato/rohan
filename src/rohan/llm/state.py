"""LangGraph state schema for the refinement loop.

Defines :class:`RefinementState` — the shared state flowing through all
nodes in the strategy-refinement graph.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rohan.llm.models import (
    AggregatedFeedback,
    IterationSummary,
    ScenarioExplanation,
)

# ---------------------------------------------------------------------------
# Scenario configuration  (lightweight – serialisable in state)
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    """Overrides for a single scenario run."""

    name: str = "default"
    config_override: dict[str, Any] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Result of executing a single scenario."""

    scenario_name: str = "default"
    interpreter_prompt: str = ""
    strategy_pnl: float | None = None
    volatility_delta_pct: float | None = None
    spread_delta_pct: float | None = None
    trade_count: int = 0
    error: str | None = None

    # Base64-encoded PNG charts (populated by executor)
    price_chart_b64: str | None = None
    spread_chart_b64: str | None = None
    volume_chart_b64: str | None = None


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class RefinementState(TypedDict, total=False):
    """Shared state for the strategy-refinement LangGraph.

    Every node reads from and writes to this dict.  LangGraph handles
    serialisation / checkpointing automatically.
    """

    # --- Goal & config ---
    goal: str
    max_iterations: int
    scenarios: list[ScenarioConfig]

    # --- Current iteration ---
    current_code: str | None
    current_class_name: str | None
    current_reasoning: str | None
    validation_errors: list[str]
    validation_attempts: int

    # --- Execution results (current iteration) ---
    scenario_results: list[ScenarioResult]

    # --- Analysis (current iteration) ---
    explanations: list[ScenarioExplanation]
    aggregated_feedback: AggregatedFeedback | None

    # --- History ---
    iterations: list[IterationSummary]
    iteration_number: int

    # --- Control ---
    status: Literal[
        "writing",
        "validating",
        "executing",
        "explaining",
        "aggregating",
        "done",
        "failed",
    ]

    # LangGraph message list (for tracing / debugging)
    messages: Annotated[list, add_messages]
