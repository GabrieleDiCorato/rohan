"""Adapter that wraps hasufel's ``SimulationResult`` as a Rohan ``SimulationOutput``.

Wraps hasufel's typed ``SimulationResult`` objects and converts
numpy-array snapshots and agent logs into Pandera-validated DataFrames.

The adapter converts hasufel's numpy-array L1/L2 snapshots and agent logs
into the Pandera-validated DataFrames that downstream consumers expect.

Limitations
-----------
* ``get_logs_by_agent()`` is unsupported (no production callers).
* ``end_state`` is unavailable — code that accesses ``result.end_state``
  (e.g. ``compute_agent_metrics()``) gracefully degrades via its existing
  ``hasattr(result, "end_state")`` guards.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, override

import numpy as np
import pandas as pd
from abides_core.utils import ns_date
from abides_markets.simulation.result import SimulationResult as HasufelResult
from pandera.typing.pandas import DataFrame

from rohan.simulation.models import SimulationOutput
from rohan.simulation.models.schemas import (
    AgentLogsSchema,
    OrderBookL1Schema,
    OrderBookL2Schema,
)

logger = logging.getLogger(__name__)


class HasufelOutput(SimulationOutput):
    """Wraps a hasufel ``SimulationResult`` to satisfy the ``SimulationOutput`` ABC."""

    def __init__(
        self,
        hasufel_result: HasufelResult,
        *,
        ticker: str = "ABM",
        strategic_agent_id: int | None = None,
    ):
        super().__init__()
        self._result = hasufel_result
        self._ticker = ticker
        self.strategic_agent_id = strategic_agent_id
        self._order_book_l2_cache: dict[int, DataFrame[OrderBookL2Schema]] = {}

    # ------------------------------------------------------------------
    # L1
    # ------------------------------------------------------------------
    @override
    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        return self._order_book_l1

    @functools.cached_property
    def _order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        market = self._result.markets.get(self._ticker)
        if market is None or market.l1_series is None:
            raise ValueError(f"No L1 series for ticker '{self._ticker}'. Ensure run_batch was called with a profile that includes L1_SERIES.")

        l1 = market.l1_series
        times_ns = np.asarray(l1.times_ns, dtype="int64")
        times_midnight = np.array([t - ns_date(t) for t in times_ns], dtype="int64")

        df = pd.DataFrame(
            {
                "time": times_midnight,
                "bid_price": np.asarray(l1.bid_prices, dtype="float64"),
                "bid_qty": np.asarray(l1.bid_quantities, dtype="float64"),
                "ask_price": np.asarray(l1.ask_prices, dtype="float64"),
                "ask_qty": np.asarray(l1.ask_quantities, dtype="float64"),
            }
        )
        df["timestamp"] = pd.to_datetime(times_midnight.astype("int64"), unit="ns")
        return OrderBookL1Schema.validate(df)  # pyright: ignore[reportReturnType]

    # ------------------------------------------------------------------
    # L2
    # ------------------------------------------------------------------
    @override
    def get_order_book_l2(self, n_levels: int) -> DataFrame[OrderBookL2Schema]:
        if n_levels not in self._order_book_l2_cache:
            self._order_book_l2_cache[n_levels] = self._build_l2(n_levels)
        return self._order_book_l2_cache[n_levels]

    def _build_l2(self, n_levels: int) -> DataFrame[OrderBookL2Schema]:
        market = self._result.markets.get(self._ticker)
        if market is None or market.l2_series is None:
            raise ValueError(f"No L2 series for ticker '{self._ticker}'. Ensure run_batch was called with a profile that includes L2_SERIES.")

        l2 = market.l2_series
        times_ns = np.asarray(l2.times_ns, dtype="int64")
        times_midnight = np.array([t - ns_date(t) for t in times_ns], dtype="int64")
        t_count = len(times_midnight)

        rows: list[dict[str, Any]] = []
        for t_idx in range(t_count):
            t = times_midnight[t_idx]
            bids = l2.bids[t_idx] if t_idx < len(l2.bids) else []
            asks = l2.asks[t_idx] if t_idx < len(l2.asks) else []
            for lvl in range(n_levels):
                if lvl < len(bids):
                    price, qty = bids[lvl]
                    rows.append({"time": t, "level": lvl + 1, "side": "bid", "price": float(price), "qty": float(qty)})
                if lvl < len(asks):
                    price, qty = asks[lvl]
                    rows.append({"time": t, "level": lvl + 1, "side": "ask", "price": float(price), "qty": float(qty)})

        if rows:
            df = pd.DataFrame(rows).sort_values(["time", "side", "level"]).reset_index(drop=True)
        else:
            df = pd.DataFrame(
                {"time": pd.Series(dtype="int64"), "level": pd.Series(dtype="int64"), "side": pd.Series(dtype="str"), "price": pd.Series(dtype="float64"), "qty": pd.Series(dtype="float64")}
            )

        df["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ns")
        return OrderBookL2Schema.validate(df)  # pyright: ignore[reportReturnType]

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------
    @override
    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        return self._logs_df

    @functools.cached_property
    def _logs_df(self) -> DataFrame[AgentLogsSchema]:
        if self._result.logs is None:
            raise ValueError("No agent logs available. Ensure run_batch was called with a profile that includes AGENT_LOGS.")
        validated = AgentLogsSchema.validate(self._result.logs)  # pyright: ignore[reportReturnType]

        if "EventType" in validated.columns:
            n_fills = int((validated["EventType"] == "ORDER_EXECUTED").sum())
            if n_fills == 0:
                logger.warning("Agent logs contain zero ORDER_EXECUTED events — fill-based metrics (effective spread, VPIN, volume) will be unavailable")

        return validated  # pyright: ignore[reportReturnType]

    @override
    def get_logs_by_agent(self) -> dict[int, Any]:
        raise NotImplementedError("get_logs_by_agent() is not supported by HasufelOutput. Use get_logs_df() and group by agent_id instead.")
