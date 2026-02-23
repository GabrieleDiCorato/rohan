"""Regression test: _vpin must not crash when L1 data has duplicate timestamps.

This was a production crash that 300+ unit tests missed because mocked data
never contained duplicate timestamps — a common occurrence in real ABIDES runs
where multiple agents act in the same nanosecond.

Root cause: ``mid_lookup.loc[nearest_t]`` returns a *Series* when the index
has duplicates, and ``float(series)`` raises ``TypeError``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import SimulationOutput
from rohan.simulation.models.schemas import (
    AgentLogsSchema,
    OrderBookL1Schema,
    OrderBookL2Schema,
)

# ---------------------------------------------------------------------------
# Timestamps: 09:30 on day 0 of the Unix epoch → ns_date() returns 0,
# so t_midnight == ts_ns.  Same convention as test_agent_metrics.
# ---------------------------------------------------------------------------
_ONE_MIN_NS: int = 60 * 10**9
_T0: int = 34_200 * 10**9  # 09:30:00 in ns since epoch


# ---------------------------------------------------------------------------
# MockOutput that supports get_logs_df (unlike test_agent_metrics.MockOutput)
# ---------------------------------------------------------------------------
class _MockOutput(SimulationOutput):
    def __init__(self, l1_df: pd.DataFrame, logs_df: pd.DataFrame) -> None:
        self.l1 = l1_df
        self._logs = logs_df

    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        return self.l1  # pyright: ignore[reportReturnType]

    def get_order_book_l2(self, n_levels: int) -> DataFrame[OrderBookL2Schema]:
        raise NotImplementedError

    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        return self._logs  # pyright: ignore[reportReturnType]

    def get_logs_by_agent(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_l1_with_dupes(n: int = 40) -> pd.DataFrame:
    """L1 data with *duplicate* 'time' values — the exact edge case that
    caused the production crash."""
    rng = np.random.RandomState(7)
    times = sorted(rng.randint(100, 5000, size=n).tolist())
    # Inject guaranteed duplicates
    for i in range(5, min(15, n)):
        times[i] = times[i - 1]

    mid = 10_000 + rng.randint(-200, 200, size=n).cumsum()
    half_spread = 50
    return pd.DataFrame(
        {
            "time": times,
            "bid_price": (mid - half_spread).astype(float),
            "ask_price": (mid + half_spread).astype(float),
            "bid_qty": np.full(n, 20.0),
            "ask_qty": np.full(n, 20.0),
            "timestamp": pd.date_range("1970-01-01 09:30", periods=n, freq="1min"),
        }
    )


def _make_fills_df(n: int = 30) -> pd.DataFrame:
    """Minimal fills DataFrame with enough rows to pass _MIN_FILLS_FOR_VPIN."""
    rng = np.random.RandomState(42)
    event_times = np.array([_T0 + i * _ONE_MIN_NS for i in range(n)], dtype=np.int64)
    return pd.DataFrame(
        {
            "agent_id": np.zeros(n, dtype=int),
            "agent_type": ["NoiseAgent"] * n,
            "EventType": ["ORDER_EXECUTED"] * n,
            "EventTime": event_times,
            "fill_price": rng.randint(9800, 10200, size=n).astype(float),
            "quantity": rng.randint(1, 5, size=n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestVpinDuplicateTimestamps:
    """Ensure _vpin handles duplicate L1 timestamps without crashing."""

    def test_vpin_no_type_error(self) -> None:
        """_vpin must return float | None, never raise TypeError."""
        l1 = _make_l1_with_dupes(40)
        fills = _make_fills_df(30)
        output = _MockOutput(l1, fills)

        # Prepare two_sided the same way compute_metrics does
        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()
        two_sided["mid_price"] = (two_sided["bid_price"] + two_sided["ask_price"]) / 2

        # This used to raise:
        # TypeError: cannot convert the series to <class 'float'>
        vpin = AnalysisService._vpin(output, two_sided)
        assert vpin is None or isinstance(vpin, float)

    def test_vpin_valid_range(self) -> None:
        """If VPIN is computed, it should be in [0, 1]."""
        l1 = _make_l1_with_dupes(80)
        fills = _make_fills_df(50)
        output = _MockOutput(l1, fills)

        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()
        two_sided["mid_price"] = (two_sided["bid_price"] + two_sided["ask_price"]) / 2

        vpin = AnalysisService._vpin(output, two_sided)
        if vpin is not None:
            assert 0.0 <= vpin <= 1.0, f"VPIN out of range: {vpin}"

    def test_effective_spread_with_dupes(self) -> None:
        """_effective_spread_and_volume must also handle duplicate timestamps."""
        l1 = _make_l1_with_dupes(40)
        fills = _make_fills_df(30)
        output = _MockOutput(l1, fills)

        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()
        two_sided["mid_price"] = (two_sided["bid_price"] + two_sided["ask_price"]) / 2

        # Should not raise
        eff_spread, volume = AnalysisService._effective_spread_and_volume(output, two_sided)
        if eff_spread is not None:
            assert eff_spread >= 0.0
        if volume is not None:
            assert volume >= 0
