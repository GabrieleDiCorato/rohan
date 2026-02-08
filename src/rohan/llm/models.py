"""Pydantic response models for LLM-structured outputs.

These models are used with LangChain's ``with_structured_output`` to
guarantee that LLM responses parse into well-typed Python objects.
"""

from __future__ import annotations

from datetime import datetime
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
    """Convergence assessment from the LLM judge."""

    score: float = Field(ge=1, le=10, description="Quality score (1-10)")
    comparison: Literal["better", "worse", "similar"] = Field(description="Relative to previous iteration")
    reasoning: str = Field(description="Explanation of the score and comparison")
    recommendation: Literal["continue", "stop_converged", "stop_plateau"] = Field(description="Whether to keep iterating")


class AggregatedFeedback(BaseModel):
    """Combined output from the Aggregator node."""

    verdict: JudgeVerdict
    cross_scenario_patterns: list[str] = Field(
        default_factory=list,
        description="Patterns observed across all scenarios",
    )
    unified_feedback: str = Field(description="Consolidated feedback for the Writer agent")


# ---------------------------------------------------------------------------
# Iteration history
# ---------------------------------------------------------------------------
class ScenarioMetrics(BaseModel):
    """Lightweight metrics snapshot for a single scenario."""

    scenario_name: str
    total_pnl: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    trade_count: int = 0
    volatility_delta_pct: float | None = None
    spread_delta_pct: float | None = None


class IterationSummary(BaseModel):
    """Summary of a single iteration for history tracking."""

    iteration_number: int
    strategy_code: str
    scenario_metrics: dict[str, ScenarioMetrics] = Field(default_factory=dict)
    aggregated_explanation: str = ""
    judge_score: float | None = None
    judge_reasoning: str | None = None
    timestamp: datetime = Field(default_factory=datetime.now)
