"""Wrapper helpers for ABIDES simulation end-state output.

This module provides a convenience `AbidesOutput` object that makes it
easier to extract commonly-used data structures from an ABIDES
simulation end-state. Methods perform potentially expensive
transformations (parsing logs, flattening L1/L2 order book snapshots),
so many are implemented lazily and cached on the instance.

Notes / pitfalls:
- Converting times: ABIDES provides timestamps as ns since epoch. We
    often convert those to "ns since midnight" using `ns_date()` so that
    snapshots align with a single trading day.

Caching strategy:
- Single-argument computed properties use ``@functools.cached_property``,
  which stores the result in the instance's ``__dict__``.  This is
  garbage-collected with the instance, avoiding the memory-leak footgun
  of ``@lru_cache`` on instance methods (which pins ``self`` in the
  function-level cache forever).
- The depth-parameterized ``get_order_book_l2(n_levels)`` uses an
  explicit ``dict`` initialised in ``__init__`` so the cache is clearly
  owned by the instance and does not require ``__hash__`` / ``__eq__``
  overrides that ``lru_cache`` would demand.
- All cache entries are populated **after** Pandera validation so that
  the "validate at production boundary" contract is preserved.
"""

import functools
import logging
from typing import Any, override

import numpy as np
import pandas as pd
from abides_core.utils import parse_logs_df
from abides_markets.agents import ExchangeAgent
from abides_markets.order_book import OrderBook, ns_date
from pandera.typing.pandas import DataFrame

from rohan.simulation.models import SimulationOutput
from rohan.simulation.models.schemas import (
    AgentLogsSchema,
    OrderBookL1Schema,
    OrderBookL2Schema,
)

logger = logging.getLogger(__name__)


class AbidesOutput(SimulationOutput):
    """Wrapper class for ABIDES simulation output.
    Most methods are computationally expensive and return static information,
    so they are cached for efficiency.
    """

    def __init__(
        self,
        end_state: dict[str, Any],
        *,
        strategic_agent_id: int | None = None,
    ):
        super().__init__()
        self.end_state = end_state
        self.strategic_agent_id = strategic_agent_id
        # Depth-parameterised L2 cache: {n_levels: validated_df}
        # Using an explicit dict owned by the instance avoids the memory-leak
        # footgun of @lru_cache on instance methods (see module docstring).
        self._order_book_l2_cache: dict[int, DataFrame[OrderBookL2Schema]] = {}

    @override
    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        """Returns the Level 1 order book data as a single DataFrame.
        Columns: time, bid_price, bid_qty, ask_price, ask_qty, timestamp
        """
        return self._order_book_l1

    @functools.cached_property
    def _order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        order_book = self.order_book
        df = AbidesOutput._compute_order_book_l1(order_book)
        return OrderBookL1Schema.validate(df)  # pyright: ignore[reportReturnType]

    @override
    def get_order_book_l2(self, n_levels: int) -> DataFrame[OrderBookL2Schema]:
        """Returns the Level 2 order book as a single tidy DataFrame.

        Preferred representation: long (tidy) format with one row per (time, level, side).
        Columns: time (ns from midnight), timestamp (pd.Timestamp), level (1-indexed),
                 side ('bid'|'ask'), price, qty
        """
        if n_levels not in self._order_book_l2_cache:
            order_book = self.order_book
            df = AbidesOutput._compute_order_book_l2(order_book, n_levels)
            self._order_book_l2_cache[n_levels] = OrderBookL2Schema.validate(df)  # pyright: ignore[reportArgumentType]
        return self._order_book_l2_cache[n_levels]

    @override
    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        """Returns a single DataFrame with the logs from all agents."""
        return self._logs_df

    @functools.cached_property
    def _logs_df(self) -> DataFrame[AgentLogsSchema]:
        # Parsing all agent logs can be expensive; cached_property ensures it
        # runs once and is garbage-collected with the instance.
        df = parse_logs_df(self.end_state)
        validated = AgentLogsSchema.validate(df)  # pyright: ignore[reportReturnType]

        # Warn if no ORDER_EXECUTED events exist — all fill-based metrics
        # (effective spread, VPIN, traded volume) will return None.
        if "EventType" in validated.columns:
            n_fills = int((validated["EventType"] == "ORDER_EXECUTED").sum())
            if n_fills == 0:
                logger.warning("Agent logs contain zero ORDER_EXECUTED events — fill-based metrics (effective spread, VPIN, volume) will be unavailable")

        return validated  # pyright: ignore[reportReturnType]

    @override
    def get_logs_by_agent(self) -> dict[int, Any]:
        """Returns a dictionary of agent logs."""
        return self._logs_by_agent

    @functools.cached_property
    def _logs_by_agent(self) -> dict[int, Any]:
        return {agent.id: agent.log for agent in self.end_state["agents"]}

    def get_exchange_agent(self) -> ExchangeAgent:
        """Returns the exchange agent from the end state."""
        return self.exchange_agent

    @functools.cached_property
    def exchange_agent(self) -> ExchangeAgent:
        # By convention in ABIDES setups used here, the exchange agent
        # appears as the first agent in the `end_state['agents']` list.
        return self.end_state["agents"][0]  # pyright: ignore[reportReturnType]

    def get_order_book(self) -> OrderBook:
        """Returns the order book data from the end state."""
        return self.order_book

    @functools.cached_property
    def order_book(self) -> OrderBook:
        return self.exchange_agent.order_books["ABM"]  # pyright: ignore[reportReturnType]

    @staticmethod
    def _compute_order_book_l1(order_book: OrderBook) -> pd.DataFrame:
        """Returns the Level 1 order book data as a single DataFrame.

        Prices are in integer cents, matching ABIDES conventions.  They are
        stored as ``float64`` in the DataFrame for NaN compatibility.

        Columns: time, bid_price, bid_qty, ask_price, ask_qty, timestamp
        """
        # `get_L1_snapshots()` returns lists of tuples like (time_ns, price,
        # qty). We convert to DataFrames and normalize column names so that
        # bids and asks can be merged on their snapshot times.
        l1: dict[str, Any] = order_book.get_L1_snapshots()

        best_bids = pd.DataFrame(columns=["time", "price", "qty"]) if len(l1["best_bids"]) == 0 else pd.DataFrame(l1["best_bids"], columns=["time", "price", "qty"])  # pyright: ignore[reportArgumentType]
        best_asks = pd.DataFrame(columns=["time", "price", "qty"]) if len(l1["best_asks"]) == 0 else pd.DataFrame(l1["best_asks"], columns=["time", "price", "qty"])  # pyright: ignore[reportArgumentType]

        # All times are provided as ns since epoch (1970). We subtract the
        # date component (ns at midnight for that date) so the `time`
        # column becomes ns since midnight. This keeps snapshots within a
        # single day's context and avoids large epoch values in the table.
        best_bids["time"] = best_bids["time"].apply(lambda x: x - ns_date(x))
        best_asks["time"] = best_asks["time"].apply(lambda x: x - ns_date(x))

        # normalize column names
        best_bids.columns = ["time", "bid_price", "bid_qty"]
        best_asks.columns = ["time", "ask_price", "ask_qty"]

        # get_L1_snapshots() iterates book_log2 once, producing two
        # parallel arrays of identical length and identical timestamps.
        # Positional concatenation is the correct join.  The previous
        # pd.merge(on="time") created an N×M cross product at duplicate
        # timestamps — a data explosion bug that corrupted all downstream
        # metrics (spreads, mid-prices, volatility).
        if best_bids.empty and best_asks.empty:
            df = pd.DataFrame(columns=["time", "bid_price", "bid_qty", "ask_price", "ask_qty"])  # pyright: ignore[reportArgumentType]
        elif best_asks.empty:
            logger.warning("L1 snapshot has bids but no asks — one-sided book for entire simulation")
            df = best_bids.copy()
            df["ask_price"] = np.nan
            df["ask_qty"] = np.nan
        elif best_bids.empty:
            logger.warning("L1 snapshot has asks but no bids — one-sided book for entire simulation")
            df = best_asks.copy()
            df["bid_price"] = np.nan
            df["bid_qty"] = np.nan
        else:
            if len(best_bids) != len(best_asks):
                raise ValueError(
                    f"L1 parallel-array invariant violated: len(best_bids)={len(best_bids)} != len(best_asks)={len(best_asks)}. This indicates a change in ABIDES get_L1_snapshots() return format."
                )
            df = best_bids.copy()
            df["ask_price"] = best_asks["ask_price"].values
            df["ask_qty"] = best_asks["ask_qty"].values

        df = df.sort_values("time").reset_index(drop=True)

        # Sanity check: after ns_date subtraction, times should be monotonically
        # non-decreasing.  A violation means either a cross-midnight simulation
        # or a bug in time normalisation.
        if not df.empty and not df["time"].is_monotonic_increasing:
            logger.warning("L1 time column is not monotonically increasing after ns_date subtraction — possible cross-midnight simulation or time normalisation bug")

        # Also provide a pandas Timestamp column (ns -> pd.Timestamp)
        df["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ns")

        return df

    @staticmethod
    def _compute_order_book_l2(order_book: OrderBook, n_levels: int) -> pd.DataFrame:
        """Returns the Level 2 order book as a single tidy DataFrame.

        Prices are in integer cents, matching ABIDES conventions.  They are
        stored as ``float64`` in the DataFrame for NaN compatibility.

        Columns: time (ns from midnight), timestamp (pd.Timestamp), level (1-indexed),
                 side ('bid'|'ask'), price, qty
        """
        l2 = order_book.get_L2_snapshots(n_levels)

        # Convert times (ns since epoch) -> ns since midnight for that day
        times = np.array([t - ns_date(t) for t in l2["times"]], dtype="int64")
        t_shape = times.shape[0]

        # `bids` and `asks` are expected as arrays shaped (T, levels, 2)
        # where the final axis is (price, qty). We use `np.asarray` to
        # tolerate list inputs and then flatten into a long/tidy format.
        bids = np.asarray(l2["bids"], dtype=float)  # shape (T, levels, 2) price, qty
        if bids.size == 0:
            bids = np.empty((0, n_levels, 2), dtype=float)
        asks = np.asarray(l2["asks"], dtype=float)  # shape (T, levels, 2)
        if asks.size == 0:
            asks = np.empty((0, n_levels, 2), dtype=float)

        # Validate shapes — ragged lists (some timestamps with fewer levels)
        # would produce an object array, not a 3D float array.  Catching this
        # early prevents cryptic IndexErrors downstream.
        if bids.size > 0 and (bids.ndim != 3 or bids.shape[1:] != (n_levels, 2)):
            raise ValueError(f"Unexpected bids shape: {bids.shape} (expected (T, {n_levels}, 2)). ABIDES may have returned ragged depth levels.")
        if asks.size > 0 and (asks.ndim != 3 or asks.shape[1:] != (n_levels, 2)):
            raise ValueError(f"Unexpected asks shape: {asks.shape} (expected (T, {n_levels}, 2)). ABIDES may have returned ragged depth levels.")

        # Repeat/time and level indices to match flattened price/qty arrays
        time_rep = np.repeat(times, n_levels)
        level_rep = np.tile(np.arange(1, n_levels + 1), t_shape)

        bids_price = bids[:, :, 0].ravel()
        bids_qty = bids[:, :, 1].ravel()
        asks_price = asks[:, :, 0].ravel()
        asks_qty = asks[:, :, 1].ravel()

        df_bids = pd.DataFrame(
            {
                "time": time_rep,
                "level": level_rep,
                "side": "bid",
                "price": bids_price,
                "qty": bids_qty,
            }
        )
        df_asks = pd.DataFrame(
            {
                "time": time_rep,
                "level": level_rep,
                "side": "ask",
                "price": asks_price,
                "qty": asks_qty,
            }
        )

        # Combine and sort into a tidy DataFrame. Sorting by `side`
        # (bid/ask) after time ensures a consistent ordering for
        # downstream consumers; `level` is 1-indexed to match human
        # expectations (level 1 = best bid/ask).
        df = pd.concat([df_bids, df_asks], ignore_index=True).sort_values(["time", "side", "level"]).reset_index(drop=True)

        df["timestamp"] = pd.to_datetime(df["time"].astype("int64"), unit="ns")

        # Sanity check: L2 times should be monotonically non-decreasing
        # (within each side, after sorting).
        if not df.empty:
            unique_times = pd.Series(times).sort_values()
            if not unique_times.is_monotonic_increasing:
                logger.warning("L2 time column is not monotonically increasing after ns_date subtraction — possible cross-midnight simulation or time normalisation bug")

        return df
