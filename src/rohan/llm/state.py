"""LangGraph state schema for the refinement loop.

Defines :class:`RefinementState` — the shared state flowing through all
nodes in the strategy-refinement graph.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from rohan.config.feature_flags import FeatureFlagSnapshot
from rohan.llm.models import (
    AggregatedFeedback,
    IterationSummary,
    ScenarioExplanation,
)

TerminalReason = Literal[
    "converged",
    "plateau_detected",
    "max_iterations_reached",
    "validation_budget_exhausted",
    "failed_execution",
]
FeatureFlagState = FeatureFlagSnapshot | dict[str, bool]

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
    vwap_cents: int | None = None
    avg_slippage_bps: float | None = None
    error: str | None = None

    # Simulation context (needed by deterministic scoring)
    starting_capital_cents: int = 10_000_000
    baseline_mean_spread: float | None = None
    baseline_traded_volume: float | None = None
    bid_liquidity_delta_pct: float | None = None
    ask_liquidity_delta_pct: float | None = None

    # Microstructure absolute values (from strategy run)
    vpin: float | None = None
    lob_imbalance_mean: float | None = None
    resilience_mean_ns: float | None = None
    market_ott_ratio: float | None = None
    pct_time_two_sided: float | None = None
    two_sided_delta_pct: float | None = None

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

    # Hasufel-native objects for persistence
    compiled_config: dict[str, Any] | None = None
    hasufel_summary: dict[str, Any] | None = None


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
    run_id: str
    max_iterations: int
    scenarios: list[ScenarioConfig]
    active_scenario: ScenarioConfig  # Used during mapped execution
    feature_flags: FeatureFlagSnapshot
    scenario_plan_reasoning: str  # Explanation from the scenario planner

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
    terminal_reason: TerminalReason | None
    terminal_iteration: int | None
    terminal_context: dict[str, Any]

    # LangGraph message list (for tracing / debugging)
    messages: Annotated[list, add_messages]


def is_feature_enabled(
    feature_flags: FeatureFlagState | None,
    flag: str,
    default: bool = True,
) -> bool:
    """Read a rollout flag from either the typed snapshot or legacy dict form."""
    if feature_flags is None:
        return default
    if isinstance(feature_flags, FeatureFlagSnapshot):
        return bool(getattr(feature_flags, flag, default))
    return bool(feature_flags.get(flag, default))


def terminal_metadata(
    feature_flags: FeatureFlagState | None,
    *,
    reason: TerminalReason,
    iteration: int,
    context: dict[str, Any],
) -> dict[str, TerminalReason | int | dict[str, Any] | None]:
    """Build terminal-state fields, respecting the rollout flag for explicit reasons."""
    if not is_feature_enabled(feature_flags, "explicit_terminal_reasons_v1", default=True):
        return {
            "terminal_reason": None,
            "terminal_iteration": None,
            "terminal_context": {},
        }
    return {
        "terminal_reason": reason,
        "terminal_iteration": iteration,
        "terminal_context": context,
    }
