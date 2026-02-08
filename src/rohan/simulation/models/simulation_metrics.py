"""Simulation metrics models.

All monetary values are in **integer cents**, matching ABIDES conventions
(e.g. ``10050`` = $100.50).  Fields that have not yet been computed are
``None``, not ``0``, so that downstream consumers can distinguish
"not computed" from "actually zero."
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Market-wide metrics
# ---------------------------------------------------------------------------
class SimulationMetrics(BaseModel):
    """Aggregate market-level metrics for a simulation run."""

    volatility: float | None = Field(default=None, description="Annualised mid-price volatility (σ)")
    mean_spread: float | None = Field(default=None, description="Average quoted bid-ask spread (cents)")
    effective_spread: float | None = Field(default=None, description="Average effective spread 2×|trade−mid| (cents)")
    avg_bid_liquidity: float | None = Field(default=None, description="Mean quantity at best bid")
    avg_ask_liquidity: float | None = Field(default=None, description="Mean quantity at best ask")
    traded_volume: int | None = Field(default=None, description="Total shares traded")


class AgentMetrics(BaseModel):
    """Performance metrics for a specific agent.

    PnL is defined relative to the agent's initial state:
        ``total_pnl = (ending_cash + mark_to_market_inventory) − initial_cash``
    """

    agent_id: int

    # --- Capital ---
    initial_cash: int = Field(default=0, description="Cash at simulation start (cents)")
    ending_cash: int = Field(default=0, description="Cash at simulation end (cents)")

    # --- PnL ---
    total_pnl: float | None = Field(default=None, description="Net PnL = ending_value − initial_cash (cents, may be fractional from mid-price)")

    # --- Risk ---
    sharpe_ratio: float | None = Field(default=None, description="Annualised Sharpe ratio")
    max_drawdown: float | None = Field(default=None, description="Maximum peak-to-trough drawdown (cents)")
    inventory_std: float | None = Field(default=None, description="Std dev of position over time (shares)")

    # --- Execution quality ---
    trade_count: int = Field(default=0, description="Number of fills")
    fill_rate: float | None = Field(default=None, description="Fills / submissions")
    order_to_trade_ratio: float | None = Field(default=None, description="Submissions / fills (lower is better)")

    # --- Inventory ---
    start_inventory: int = Field(default=0, description="Position at simulation start")
    end_inventory: int = Field(default=0, description="Position at simulation end")
    avg_hold_time_ns: float | None = Field(default=None, description="Mean holding period (ns)")


class MarketMetrics(BaseModel):
    """Market-wide statistics for baseline comparison."""

    volatility: float | None = Field(default=None, description="Annualised mid-price volatility")
    mean_spread: float | None = Field(default=None, description="Average quoted spread (cents)")
    effective_spread: float | None = Field(default=None, description="Average effective spread (cents)")
    avg_bid_liquidity: float | None = Field(default=None, description="Mean qty at best bid")
    avg_ask_liquidity: float | None = Field(default=None, description="Mean qty at best ask")
    traded_volume: int | None = Field(default=None, description="Total shares traded")


# ---------------------------------------------------------------------------
# Market impact (typed, not a bare dict)
# ---------------------------------------------------------------------------
class MarketImpact(BaseModel):
    """Percentage deltas between strategy run and baseline run.

    Positive values mean the strategy *increased* the metric.
    """

    spread_delta_pct: float | None = Field(default=None, description="% change in mean spread")
    volatility_delta_pct: float | None = Field(default=None, description="% change in volatility")
    bid_liquidity_delta_pct: float | None = Field(default=None, description="% change in avg bid qty")
    ask_liquidity_delta_pct: float | None = Field(default=None, description="% change in avg ask qty")


class ComparisonResult(BaseModel):
    """Comparison between strategy run and baseline run."""

    strategy_metrics: AgentMetrics
    strategy_market_metrics: MarketMetrics
    baseline_metrics: MarketMetrics
    market_impact: MarketImpact


class RunSummary(BaseModel):
    """Structured summary for LLM interpretation.

    Provides a compact view of simulation results including metrics,
    optional visualizations, and execution metadata.
    """

    comparison: ComparisonResult
    price_chart: str | None = Field(default=None, description="Base64-encoded PNG")
    spread_chart: str | None = Field(default=None, description="Base64-encoded PNG")
    duration_seconds: float = 0.0
    error: str | None = None
