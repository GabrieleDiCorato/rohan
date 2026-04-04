"""Pydantic models for rich simulation analysis.

These models capture detailed, per-event simulation data that the
:class:`~rohan.framework.analysis_service.AnalysisService` extracts from
raw ABIDES output.  All models are fully JSON-serialisable (via
``.model_dump()`` / ``.model_dump_json()``) so they can be:

* stored as DB artifacts (``SessionRepository``)
* embedded in LangGraph state as a JSON string
* transmitted across container boundaries without ABIDES dependencies

Monetary values are in **integer cents** to match ABIDES conventions.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FillRecord(BaseModel):
    """A single fill (execution) event for an agent."""

    timestamp_ns: int = Field(description="Nanoseconds since midnight")
    side: str = Field(description="'BUY' or 'SELL'")
    price: int = Field(description="Fill price in cents")
    qty: int = Field(description="Fill quantity in shares")
    mid_at_fill: float | None = Field(default=None, description="Nearest L1 mid-price at fill time (cents)")
    slippage_bps: float | None = Field(default=None, description="Signed slippage: side_sign × (fill − mid) / mid × 10 000")
    counterparty_type: str | None = Field(default=None, description="Agent type of the counterparty (e.g. 'NoiseAgent')")


class PnLPoint(BaseModel):
    """A single mark-to-market PnL observation."""

    timestamp_ns: int = Field(description="Nanoseconds since midnight")
    mark_to_market_pnl: float = Field(description="MTM PnL = NAV − initial_NAV (cents)")


class InventoryPoint(BaseModel):
    """A single inventory observation (one per fill)."""

    timestamp_ns: int = Field(description="Nanoseconds since midnight")
    position: int = Field(description="Cumulative position after fill (shares)")


class OrderLifecycleRecord(BaseModel):
    """Lifecycle record for a single order."""

    order_id: int = Field(description="ABIDES order ID")
    submitted_at_ns: int = Field(description="Submission timestamp (ns since midnight)")
    status: str = Field(description="'filled', 'cancelled', or 'resting'")
    resting_time_ns: int | None = Field(default=None, description="Time from submission to resolution (ns)")
    filled_qty: int = Field(default=0, description="Quantity filled")
    submitted_qty: int = Field(default=0, description="Quantity at submission")


class CounterpartySummary(BaseModel):
    """Aggregated counterparty statistics for one agent type."""

    agent_type: str = Field(description="Type name, e.g. 'NoiseAgent', 'ValueAgent'")
    trade_count: int = Field(default=0, description="Number of fills against this type")
    avg_size: float = Field(default=0.0, description="Average fill quantity")
    total_volume: int = Field(default=0, description="Total shares traded against this type")


class MidPricePoint(BaseModel):
    """A single L1 mid-price observation."""

    timestamp_ns: int = Field(description="Nanoseconds since midnight")
    mid_price: float = Field(description="(bid + ask) / 2 in cents")


class L2Snapshot(BaseModel):
    """An L2 order-book snapshot at a single point in time."""

    timestamp_ns: int = Field(description="Nanoseconds since midnight")
    bids: list[tuple[int, int]] = Field(default_factory=list, description="List of (price_cents, qty) tuples, best first")
    asks: list[tuple[int, int]] = Field(default_factory=list, description="List of (price_cents, qty) tuples, best first")


class RichAnalysisBundle(BaseModel):
    """Complete rich-analysis output for one scenario execution.

    Designed to be stored as a single JSON blob on ``ScenarioResult``
    and in the ``artifacts`` DB table.  All fields are fully serialisable
    so that investigation tools can work from the JSON without needing
    a live ``SimulationOutput``.
    """

    fills: list[FillRecord] = Field(default_factory=list)
    pnl_curve: list[PnLPoint] = Field(default_factory=list)
    inventory_trajectory: list[InventoryPoint] = Field(default_factory=list)
    adverse_selection_bps: float | None = Field(default=None, description="Average mid-price move against fill direction within default look-ahead window (bps)")
    adverse_selection_by_window: dict[str, float] = Field(default_factory=dict, description="Adverse selection at multiple look-ahead windows, keyed by label (e.g. '100ms', '500ms', '1s', '5s')")
    counterparty_breakdown: list[CounterpartySummary] = Field(default_factory=list)
    order_lifecycle: list[OrderLifecycleRecord] = Field(default_factory=list)

    # Raw queryable data for investigation tools
    mid_price_series: list[MidPricePoint] = Field(default_factory=list, description="Full L1 mid-price time-series for recomputation")
    l2_snapshots: list[L2Snapshot] = Field(default_factory=list, description="Sampled L2 snapshots at fills, PnL turning points, and regular intervals")
