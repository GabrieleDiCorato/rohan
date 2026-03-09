"""Pydantic response models for LLM-structured outputs.

These models are used with LangChain's ``with_structured_output`` to
guarantee that LLM responses parse into well-typed Python objects.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Writer Agent
# ---------------------------------------------------------------------------
class GeneratedStrategy(BaseModel):
    """Structured output from the Writer agent."""

    class_name: str = Field(description="Name of the strategy class")
    code: str = Field(description="Complete Python source code implementing StrategicAgent")
    reasoning: str = Field(description="Brief explanation of the strategy approach")


# ---------------------------------------------------------------------------
# Explainer Agent
# ---------------------------------------------------------------------------
class ScenarioExplanation(BaseModel):
    """Analysis for a single scenario run."""

    scenario_name: str = Field(default="default")
    strengths: list[str] = Field(default_factory=list, description="What the strategy did well")
    weaknesses: list[str] = Field(default_factory=list, description="What needs improvement")
    market_impact_assessment: str = Field(default="", description="Was the strategy stabilizing or destabilizing?")
    key_observations: list[str] = Field(default_factory=list, description="Notable patterns")
    recommendations: list[str] = Field(default_factory=list, description="Specific improvement suggestions")
    raw_analysis: str = Field(default="", description="Full-text LLM analysis")


# ---------------------------------------------------------------------------
# Aggregator / Judge
# ---------------------------------------------------------------------------
class JudgeVerdict(BaseModel):
    """Convergence assessment — scores are deterministic, reasoning is from LLM."""

    score: float = Field(ge=1, le=10, description="Weighted final score (1-10), computed by formula")
    comparison: Literal["better", "worse", "similar"] = Field(description="Relative to best iteration so far")
    reasoning: str = Field(description="Qualitative analysis from LLM")
    recommendation: Literal["continue", "stop_converged", "stop_plateau"] = Field(description="Deterministic stop/continue decision")

    # 6-axis sub-scores (all computed deterministically)
    profitability_score: float | None = Field(default=None, ge=1, le=10, description="Profitability sub-score (1-10)")
    risk_score: float | None = Field(default=None, ge=1, le=10, description="Risk-adjusted performance sub-score (1-10)")
    volatility_impact_score: float | None = Field(default=None, ge=1, le=10, description="Volatility impact sub-score (1-10)")
    spread_impact_score: float | None = Field(default=None, ge=1, le=10, description="Spread impact sub-score (1-10)")
    liquidity_impact_score: float | None = Field(default=None, ge=1, le=10, description="Liquidity impact sub-score (1-10)")
    execution_score: float | None = Field(default=None, ge=1, le=10, description="Execution quality sub-score (1-10)")


class QualitativeAnalysis(BaseModel):
    """LLM-produced qualitative analysis — no scoring, just reasoning."""

    reasoning: str = Field(description="Qualitative analysis explaining *why* the strategy scored as it did on each axis")
    strengths: list[str] = Field(default_factory=list, description="What the strategy did well")
    weaknesses: list[str] = Field(default_factory=list, description="What needs improvement")
    recommendations: list[str] = Field(default_factory=list, description="Actionable improvement suggestions")


class AggregatedFeedback(BaseModel):
    """Combined output from the Aggregator node."""

    verdict: JudgeVerdict
    cross_scenario_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns observed across all scenarios",
    )
    unified_feedback: str = Field(description="Consolidated feedback for the Writer agent")
    # Per-scenario structured feedback (from explainer ScenarioExplanation)
    scenario_weaknesses: list[tuple[str, list[str]]] = Field(
        default_factory=list,
        description="Per-scenario weaknesses: list of (scenario_name, weaknesses)",
    )
    scenario_recommendations: list[tuple[str, list[str]]] = Field(
        default_factory=list,
        description="Per-scenario recommendations: list of (scenario_name, recommendations)",
    )


# ---------------------------------------------------------------------------
# Iteration history
# ---------------------------------------------------------------------------
class ScenarioMetrics(BaseModel):
    """Lightweight metrics snapshot for a single scenario."""

    scenario_name: str
    total_pnl: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    fill_rate: float | None = None
    order_to_trade_ratio: float | None = None
    inventory_std: float | None = None
    end_inventory: int = 0
    trade_count: int = 0
    volatility_delta_pct: float | None = None
    spread_delta_pct: float | None = None

    # Base64-encoded PNG charts (carried from ScenarioResult)
    price_chart_b64: str | None = None
    spread_chart_b64: str | None = None
    volume_chart_b64: str | None = None
    pnl_chart_b64: str | None = None
    inventory_chart_b64: str | None = None
    fill_scatter_b64: str | None = None


class IterationSummary(BaseModel):
    """Summary of a single iteration for history tracking."""

    iteration_number: int
    strategy_code: str
    scenario_metrics: dict[str, ScenarioMetrics] = Field(default_factory=dict)
    aggregated_explanation: str = ""
    judge_score: float | None = None
    judge_reasoning: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    # True when this iteration scored below the previous best and triggered a rollback.
    rolled_back: bool = False
    # 6-axis sub-scores (1-10 each, or None when unavailable)
    profitability_score: float | None = None
    risk_score: float | None = None
    volatility_impact_score: float | None = None
    spread_impact_score: float | None = None
    liquidity_impact_score: float | None = None
    execution_score: float | None = None
    # Which weight profile was used (e.g. "default", "risk_focused")
    scoring_profile: str | None = None
