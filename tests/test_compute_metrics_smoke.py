"""Smoke tests: compute_metrics must survive realistic (messy) data shapes.

These tests catch the *class* of bugs where individual metric helpers crash
on edge-case data that unit tests with clean mocks don't reproduce:

• **Duplicate timestamps** — multiple agents acting in the same nanosecond.
• **NaN bid/ask** — one-sided book states (genuine illiquidity events).
• **Single-row L1** — extremely short simulations.
• **Empty fills** — no ORDER_EXECUTED events at all.

Convention: L1 fixtures should ALWAYS include duplicate timestamps and NaN
sides to match real ABIDES output.  Never build test L1 with perfectly
unique, perfectly clean rows — that masks real bugs.
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
# Shared MockOutput — supports both get_order_book_l1 and get_logs_df
# ---------------------------------------------------------------------------
class _MockOutput(SimulationOutput):
    """Lightweight mock that satisfies the SimulationOutput interface."""

    def __init__(
        self,
        l1_df: pd.DataFrame,
        logs_df: pd.DataFrame | None = None,
    ) -> None:
        self.l1 = l1_df
        self._logs = logs_df if logs_df is not None else pd.DataFrame()

    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        return self.l1  # pyright: ignore[reportReturnType]

    def get_order_book_l2(self, n_levels: int) -> DataFrame[OrderBookL2Schema]:
        raise NotImplementedError

    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        return self._logs  # pyright: ignore[reportReturnType]

    def get_logs_by_agent(self) -> dict:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# L1 fixtures — deliberately messy
# ---------------------------------------------------------------------------
_ONE_MIN_NS: int = 60 * 10**9
_T0: int = 34_200 * 10**9  # 09:30:00 in ns since epoch


def _build_realistic_l1(n: int = 200, *, with_dupes: bool = True, with_nans: bool = True) -> pd.DataFrame:
    """L1 data that mimics real ABIDES output, including known edge cases.

    Args:
        n: Number of rows.
        with_dupes: Inject duplicate timestamps (very common in real runs).
        with_nans: Inject NaN bid/ask (one-sided book states).
    """
    rng = np.random.RandomState(99)
    times = sorted(rng.randint(100, 10000, size=n).tolist())

    if with_dupes:
        for i in range(5, min(20, n)):
            times[i] = times[i - 1]

    bid = 10_000 + rng.randint(-5, 5, size=n).cumsum()
    ask = bid + rng.randint(5, 15, size=n)

    bid_f = bid.astype(float)
    ask_f = ask.astype(float)

    if with_nans:
        # Inject some NaN sides (one-sided book states)
        for idx in rng.choice(n, size=min(10, n // 10), replace=False):
            if rng.random() > 0.5:
                bid_f[idx] = np.nan
            else:
                ask_f[idx] = np.nan

    return pd.DataFrame(
        {
            "time": times,
            "bid_price": bid_f,
            "ask_price": ask_f,
            "bid_qty": rng.randint(1, 50, size=n).astype(float),
            "ask_qty": rng.randint(1, 50, size=n).astype(float),
            "timestamp": pd.date_range("1970-01-01 09:30", periods=n, freq="1s"),
        }
    )


def _build_fills_df(n: int = 40) -> pd.DataFrame:
    """Fills log matching AgentLogsSchema columns."""
    rng = np.random.RandomState(42)
    event_times = np.array([_T0 + i * _ONE_MIN_NS for i in range(n)], dtype=np.int64)
    return pd.DataFrame(
        {
            "agent_id": np.zeros(n, dtype=int),
            "agent_type": ["NoiseAgent"] * n,
            "EventType": ["ORDER_EXECUTED"] * n,
            "EventTime": event_times,
            "fill_price": rng.randint(9950, 10050, size=n).astype(float),
            "quantity": rng.randint(1, 10, size=n).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestComputeMetricsSmoke:
    """End-to-end smoke tests for AnalysisService.compute_metrics."""

    def test_no_crash_realistic_data(self) -> None:
        """compute_metrics must not raise on realistic (messy) L1 + fills."""
        l1 = _build_realistic_l1(200, with_dupes=True, with_nans=True)
        fills = _build_fills_df(40)
        output = _MockOutput(l1, fills)

        metrics = AnalysisService.compute_metrics(output)

        assert metrics is not None
        # Spread and liquidity should always be computable when L1 is non-empty
        assert metrics.mean_spread is not None

    def test_no_crash_dupes_only(self) -> None:
        """compute_metrics with duplicate timestamps but no NaNs."""
        l1 = _build_realistic_l1(150, with_dupes=True, with_nans=False)
        fills = _build_fills_df(30)
        output = _MockOutput(l1, fills)

        metrics = AnalysisService.compute_metrics(output)
        assert metrics is not None

    def test_no_crash_nans_only(self) -> None:
        """compute_metrics with NaN bid/ask but unique timestamps."""
        l1 = _build_realistic_l1(150, with_dupes=False, with_nans=True)
        fills = _build_fills_df(30)
        output = _MockOutput(l1, fills)

        metrics = AnalysisService.compute_metrics(output)
        assert metrics is not None

    def test_empty_l1(self) -> None:
        """compute_metrics on empty L1 returns default SimulationMetrics."""
        output = _MockOutput(pd.DataFrame())

        metrics = AnalysisService.compute_metrics(output)
        assert metrics is not None
        assert metrics.volatility is None
        assert metrics.mean_spread is None

    def test_single_row_l1(self) -> None:
        """compute_metrics with a single L1 row must not crash."""
        l1 = pd.DataFrame(
            {
                "time": [100],
                "bid_price": [10_000.0],
                "ask_price": [10_050.0],
                "bid_qty": [10.0],
                "ask_qty": [10.0],
                "timestamp": pd.date_range("1970-01-01 09:30", periods=1, freq="1s"),
            }
        )
        output = _MockOutput(l1)

        metrics = AnalysisService.compute_metrics(output)
        assert metrics is not None

    def test_no_fills(self) -> None:
        """compute_metrics with L1 data but zero fills."""
        l1 = _build_realistic_l1(100, with_dupes=True, with_nans=False)
        empty_logs = pd.DataFrame(columns=["agent_id", "agent_type", "EventType", "EventTime", "fill_price", "quantity"])  # pyright: ignore[reportArgumentType]
        output = _MockOutput(l1, empty_logs)

        metrics = AnalysisService.compute_metrics(output)
        assert metrics is not None
        # No fills → volume should be None or 0, effective spread None
        assert metrics.effective_spread is None

    def test_all_nan_bid_ask(self) -> None:
        """L1 where every row has at least one NaN side → empty two_sided."""
        n = 20
        l1 = pd.DataFrame(
            {
                "time": list(range(n)),
                "bid_price": [np.nan] * n,
                "ask_price": [10_000.0 + i for i in range(n)],
                "bid_qty": [10.0] * n,
                "ask_qty": [10.0] * n,
                "timestamp": pd.date_range("1970-01-01 09:30", periods=n, freq="1s"),
            }
        )
        output = _MockOutput(l1)

        metrics = AnalysisService.compute_metrics(output)
        assert metrics is not None
        # No two-sided rows → most metrics None
        assert metrics.volatility is None

    def test_metrics_types(self) -> None:
        """All non-None metric values must be the correct Python type."""
        l1 = _build_realistic_l1(200, with_dupes=True, with_nans=True)
        fills = _build_fills_df(40)
        output = _MockOutput(l1, fills)

        metrics = AnalysisService.compute_metrics(output)

        float_fields = [
            "volatility",
            "mean_spread",
            "effective_spread",
            "avg_bid_liquidity",
            "avg_ask_liquidity",
            "lob_imbalance_mean",
            "lob_imbalance_std",
            "vpin",
            "resilience_mean_ns",
            "market_ott_ratio",
        ]
        for field in float_fields:
            val = getattr(metrics, field, None)
            if val is not None:
                assert isinstance(val, float), f"{field} should be float, got {type(val)}"

        if metrics.traded_volume is not None:
            assert isinstance(metrics.traded_volume, int)

    def test_volatility_not_none_with_heavy_dupes(self) -> None:
        """Volatility must be computed when there are enough distinct timestamps,
        even when most rows share the same timestamp (real ABIDES behaviour:
        multiple book state changes per nanosecond)."""
        rng = np.random.RandomState(123)
        # 50 distinct timestamps spanning ~100 seconds (above 60s minimum),
        # each repeated 2-5 times → ~175 rows
        base_ns = 34_200_000_000_000  # 09:30 in ns
        distinct_offsets = sorted((rng.random_sample(50) * 100_000_000_000).astype(int).tolist())  # 0..100s in ns
        times: list[int] = []
        for t in distinct_offsets:
            times.extend([base_ns + t] * rng.randint(2, 6))

        n = len(times)
        mid = 10_000 + rng.randint(-20, 20, size=n).cumsum()
        half_spread = 50

        l1 = pd.DataFrame(
            {
                "time": times,
                "bid_price": (mid - half_spread).astype(float),
                "ask_price": (mid + half_spread).astype(float),
                "bid_qty": np.full(n, 20.0),
                "ask_qty": np.full(n, 20.0),
                "timestamp": pd.date_range("1970-01-01 09:30", periods=n, freq="1s"),
            }
        )
        output = _MockOutput(l1)

        metrics = AnalysisService.compute_metrics(output)

        assert metrics is not None
        assert metrics.volatility is not None, "Volatility should be computable with 50+ distinct timestamps even when most timestamps are duplicated"
        assert metrics.volatility > 0
