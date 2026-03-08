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
# Reducers
# ---------------------------------------------------------------------------


def _concat_or_clear(existing: list, update: list | str) -> list:
    """LangGraph reducer for list fields used in map-reduce nodes.

    Accepts two kinds of updates:
    * A list  → concatenated onto *existing* (used by ``process_scenario_node``
      fan-out via ``Send``).
    * The string ``"clear"`` → returns an empty list (used by
      ``validator_node`` to reset results at the start of each new iteration).
    """
    if update == "clear":
        return []
    if isinstance(update, list):
        return existing + update
    return existing  # defensive: ignore unknown sentinel values


# ---------------------------------------------------------------------------
# Scenario configuration  (lightweight – serialisable in state)
# ---------------------------------------------------------------------------


class ScenarioConfig(BaseModel):
    """Overrides for a single scenario run."""

    name: str = "default"
    config_override: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = Field(
        default=None,
        description="Deterministic seed for this scenario.  Assigned by run_refinement() when not explicitly provided.",
    )


class ScenarioResult(BaseModel):
    """Result of executing a single scenario."""

    scenario_name: str = "default"
    interpreter_prompt: str = ""
    strategy_pnl: float | None = None
    volatility_delta_pct: float | None = None
    spread_delta_pct: float | None = None
    trade_count: int = 0
    fill_rate: float | None = None
    order_to_trade_ratio: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    end_inventory: int = 0
    inventory_std: float | None = None
    error: str | None = None

    # Simulation context (needed by deterministic scoring)
    starting_capital_cents: int = 10_000_000
    baseline_mean_spread: float | None = None
    baseline_traded_volume: float | None = None
    bid_liquidity_delta_pct: float | None = None
    ask_liquidity_delta_pct: float | None = None

    # Base64-encoded PNG charts (populated by executor)
    price_chart_b64: str | None = None
    spread_chart_b64: str | None = None
    volume_chart_b64: str | None = None

    # Rich analysis (Step 8)
    pnl_chart_b64: str | None = None
    inventory_chart_b64: str | None = None
    fill_scatter_b64: str | None = None
    rich_analysis_json: str | None = None

    # Regime context for explainer (Step 9 — forward-compatible with adversarial plan)
    regime_context: str = ""


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
    active_scenario: ScenarioConfig  # Used during mapped execution

    # --- Current iteration ---
    current_code: str | None
    current_class_name: str | None
    current_reasoning: str | None
    validation_errors: list[str]
    validation_attempts: int

    # --- Execution results (current iteration) ---
    # Annotated with _concat_or_clear so that:
    #   • process_scenario_node fan-out branches accumulate via operator.add semantics
    #   • validator_node can reset with the sentinel string "clear"
    scenario_results: Annotated[list[ScenarioResult], _concat_or_clear]

    # --- Analysis (current iteration) ---
    explanations: Annotated[list[ScenarioExplanation], _concat_or_clear]
    aggregated_feedback: AggregatedFeedback | None

    # --- History ---
    iterations: list[IterationSummary]
    iteration_number: int

    # --- Best-known tracking (for rollback) ---
    # Populated and updated by aggregator_node after every iteration.
    best_iteration_number: int
    best_code: str | None  # code that achieved the highest score so far
    best_score: float | None  # highest judge score seen so far
    rolled_back_from: dict | None  # snapshot of failed attempt; consumed by writer_node

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
