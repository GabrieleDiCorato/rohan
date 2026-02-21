"""Pandera schemas for simulation DataFrame outputs.

These schemas serve as the **single source of truth** for the column contracts
of every DataFrame that crosses module boundaries.  They replace implicit
column-name conventions with explicit, validated definitions.

Validation philosophy
---------------------
Schemas are validated at the **production boundary** — i.e. inside
``AbidesOutput`` right after data is computed and before it is cached.
Consumers (e.g. ``AnalysisService``) rely on type annotations
(``DataFrame[OrderBookL1Schema]``) for documentation without re-validating.

``strict`` vs ``coerce``
------------------------
* ``OrderBookL1Schema``:  ``strict=False`` — downstream code may attach
  computed columns (e.g. ``mid_price``).  ``coerce=True`` because ABIDES
  emits mixed numeric types from C extensions; we normalise them here.
* ``OrderBookL2Schema``:  ``strict=True`` — no downstream columns are
  expected.  ``coerce=True`` for the same dtype-normalisation reason.
* ``AgentLogsSchema``:  ``strict=False`` — ``parse_logs_df`` may append
  extra event-specific columns depending on agent type.  ``coerce=True``
  to handle float agent IDs from ABIDES JSON round-trips.

Usage:
    Producers (e.g. ``AbidesOutput``) call ``schema.validate(df)`` before
    returning data.  Consumers (e.g. ``AnalysisService``) can rely on type
    annotations (``DataFrame[OrderBookL1Schema]``) for documentation without
    re-validating.
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
    bid_price: Series[float] = pa.Field(nullable=True, description="Best bid price in cents (float64 for NaN compat)")
    bid_qty: Series[float] = pa.Field(nullable=True, description="Quantity at best bid")
    ask_price: Series[float] = pa.Field(nullable=True, description="Best ask price in cents (float64 for NaN compat)")
    ask_qty: Series[float] = pa.Field(nullable=True, description="Quantity at best ask")
    timestamp: Series[pa.DateTime] = pa.Field(description="Wall-clock timestamp")

    class Config:
        # strict=False: downstream code may attach computed columns such as
        # `mid_price` or `spread` without triggering a validation error.
        # This is intentional — we validate at the production boundary and
        # consumers are free to add columns as needed.
        strict = False
        # coerce=True: ABIDES C extensions sometimes emit mixed numeric types
        # (e.g. int32/int64, float32/float64).  Coercion normalises them to
        # the declared dtypes before analysis code consumes the DataFrame.
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
    price: Series[float] = pa.Field(description="Price at level in cents (float64 for NaN compat)")
    qty: Series[float] = pa.Field(description="Quantity at level")
    timestamp: Series[pa.DateTime] = pa.Field(description="Wall-clock timestamp")

    class Config:
        # strict=True: unlike L1, no downstream code is expected to add
        # columns to L2 DataFrames.  Extra columns would indicate an upstream
        # change in ABIDES output format that we should be aware of.
        strict = True
        # coerce=True: normalise mixed numeric types from ABIDES C extensions.
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
        # strict=False: parse_logs_df appends extra event-specific columns
        # (e.g. `order_id`, `quantity`, `price`) depending on the agent type.
        # These are not guaranteed and vary across simulation configurations.
        strict = False
        # coerce=True: agent IDs can be float after JSON/CSV round-trips.
        coerce = True
