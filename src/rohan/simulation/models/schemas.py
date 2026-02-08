"""Pandera schemas for simulation DataFrame outputs.

These schemas serve as the **single source of truth** for the column contracts
of every DataFrame that crosses module boundaries.  They replace implicit
column-name conventions with explicit, validated definitions.

Usage:
    Producers (e.g. ``AbidesOutput``) call ``schema.validate(df)`` before
    returning data.  Consumers (e.g. ``AnalysisService``) can rely on type
    annotations (``DataFrame[OrderBookL1Schema]``) for documentation without
    re-validating.

Design rationale (see ``implementation_plan.md § 3.2``):
    * **Pydantic** is used for lightweight metadata/summaries (``SimulationMetrics``).
    * **Pandera** is used for bulk tabular data — it validates DataFrames
      in-place with near-zero overhead compared to row-by-row Pydantic parsing.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing.pandas import Series


# ---------------------------------------------------------------------------
# Level 1 Order Book
# ---------------------------------------------------------------------------
class OrderBookL1Schema(pa.DataFrameModel):
    """Level 1 (best bid/ask) order-book snapshots.

    Produced by ``SimulationOutput.get_order_book_l1()``.

    Columns
    -------
    time : int
        Nanoseconds since midnight (exchange local time).
    bid_price : float
        Best bid price.  May be ``NaN`` during crossed-book periods.
    bid_qty : float
        Quantity available at best bid.
    ask_price : float
        Best ask price.  May be ``NaN`` during crossed-book periods.
    ask_qty : float
        Quantity available at best ask.
    timestamp : datetime64[ns]
        Wall-clock timestamp derived from ``time``.
    """

    time: Series[int] = pa.Field(description="Nanoseconds since midnight")
    bid_price: Series[float] = pa.Field(nullable=True, description="Best bid price")
    bid_qty: Series[float] = pa.Field(nullable=True, description="Quantity at best bid")
    ask_price: Series[float] = pa.Field(nullable=True, description="Best ask price")
    ask_qty: Series[float] = pa.Field(nullable=True, description="Quantity at best ask")
    timestamp: Series[pa.DateTime] = pa.Field(description="Wall-clock timestamp")

    class Config:
        strict = False  # allow extra columns (e.g. downstream-computed 'mid_price')
        coerce = True


# ---------------------------------------------------------------------------
# Level 2 Order Book (tidy / long format)
# ---------------------------------------------------------------------------
class OrderBookL2Schema(pa.DataFrameModel):
    """Level 2 (depth-of-book) snapshots in tidy (long) format.

    Produced by ``SimulationOutput.get_order_book_l2(n_levels)``.

    Columns
    -------
    time : int
        Nanoseconds since midnight.
    level : int
        1-indexed depth level.
    side : str
        ``"bid"`` or ``"ask"``.
    price : float
        Price at this level.
    qty : float
        Quantity at this level.
    timestamp : datetime64[ns]
        Wall-clock timestamp derived from ``time``.
    """

    time: Series[int] = pa.Field(description="Nanoseconds since midnight")
    level: Series[int] = pa.Field(ge=1, description="1-indexed depth level")
    side: Series[str] = pa.Field(isin=["bid", "ask"], description="Order side")
    price: Series[float] = pa.Field(description="Price at level")
    qty: Series[float] = pa.Field(description="Quantity at level")
    timestamp: Series[pa.DateTime] = pa.Field(description="Wall-clock timestamp")

    class Config:
        strict = False
        coerce = True


# ---------------------------------------------------------------------------
# Agent Logs
# ---------------------------------------------------------------------------
class AgentLogsSchema(pa.DataFrameModel):
    """Parsed agent event logs.

    Produced by ``SimulationOutput.get_logs_df()``.

    .. note::

       The upstream ``abides_core.utils.parse_logs_df`` may emit additional
       columns depending on the agent types in the simulation.  This schema
       validates only the **guaranteed** columns.

    Columns
    -------
    agent_id : int
        Numeric agent identifier.
    agent_type : str
        Class name of the agent.
    EventType : str
        Event type string (e.g. ``"ORDER_SUBMITTED"``).
    time_placed : int | float
        Simulation timestamp of the event.
    """

    agent_id: Series[int] = pa.Field(description="Agent identifier")
    agent_type: Series[str] = pa.Field(description="Agent class name")
    EventType: Series[str] = pa.Field(description="Event type")

    class Config:
        strict = False  # upstream may add extra columns
        coerce = True
