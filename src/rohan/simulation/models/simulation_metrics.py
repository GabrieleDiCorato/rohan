from typing import Any

from pydantic import BaseModel, Field


class SimulationMetrics(BaseModel):
    """
    Summary metrics for a simulation run.
    """

    total_pnl: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    volatility: float
    traded_volume: int
    custom_metrics: dict[str, Any] = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    """Performance metrics for a specific agent."""

    agent_id: int
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    active_returns: float = 0.0  # Returns exceeding baseline/risk-free
    trade_count: int = 0
    fill_rate: float = 0.0
    start_inventory: int = 0
    end_inventory: int = 0
    avg_hold_time_ns: float | None = None


class MarketMetrics(BaseModel):
    """Market-wide statistics for baseline comparison."""

    volatility: float
    mean_spread: float
    avg_bid_liquidity: float
    avg_ask_liquidity: float
    traded_volume: int


class ComparisonResult(BaseModel):
    """Comparison between strategy run and baseline run."""

    strategy_metrics: AgentMetrics
    strategy_market_metrics: MarketMetrics
    baseline_metrics: MarketMetrics
    market_impact: dict[str, float]  # Percentage delta between runs


class RunSummary(BaseModel):
    """Structured summary for LLM interpretation.

    Provides a compact view of simulation results including metrics,
    optional visualizations, and execution metadata.
    """

    comparison: ComparisonResult
    price_chart: str | None = None  # Base64-encoded PNG
    spread_chart: str | None = None  # Base64-encoded PNG
    duration_seconds: float = 0.0
    error: str | None = None
