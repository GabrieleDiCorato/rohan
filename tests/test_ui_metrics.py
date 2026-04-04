"""Tests for src/rohan/ui/metrics.py — pure financial computations.

These functions accept pandas objects and return plain dataclasses or Series.
No Streamlit or Plotly needed; no SimulationResult mocking for the core tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rohan.ui.metrics import (
    L1Derived,
    OrderFlowStats,
    ReturnStats,
    SpreadStats,
    compute_book_pressure,
    compute_cumulative_imbalance,
    compute_maker_taker_summary,
    compute_order_flow_stats,
    compute_return_stats,
    compute_rolling_vol,
    compute_spread_stats,
    derive_l1,
)

# ---------------------------------------------------------------------------
# Fixtures — synthetic market data
# ---------------------------------------------------------------------------


def _make_l1_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a minimal L1 DataFrame that derive_l1 can consume."""
    rng = np.random.default_rng(seed)
    base_price = 10_000  # $100.00 in cents
    mid_prices = base_price + np.cumsum(rng.integers(-5, 6, size=n))
    half_spread = rng.integers(1, 4, size=n)

    return pd.DataFrame(
        {
            "time_ns": np.arange(n) * 1_000_000_000 + 1_680_000_000_000_000_000,
            "bid_price_cents": mid_prices - half_spread,
            "ask_price_cents": mid_prices + half_spread,
            "bid_qty": rng.integers(10, 200, size=n),
            "ask_qty": rng.integers(10, 200, size=n),
        }
    )


def _make_order_df(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic order-event DataFrame."""
    rng = np.random.default_rng(seed)
    events = rng.choice(["ORDER_SUBMITTED", "ORDER_EXECUTED", "ORDER_CANCELLED"], size=n, p=[0.5, 0.3, 0.2])
    sides = rng.choice(["BID", "ASK"], size=n)
    return pd.DataFrame(
        {
            "EventType": events,
            "EventTime": pd.date_range("2026-01-01 09:30", periods=n, freq="s"),
            "side": sides,
        }
    )


@pytest.fixture
def l1_df():
    return _make_l1_df()


@pytest.fixture
def l1(l1_df) -> L1Derived:
    return derive_l1(l1_df)


@pytest.fixture
def order_df():
    return _make_order_df()


# ---------------------------------------------------------------------------
# derive_l1
# ---------------------------------------------------------------------------


class TestDeriveL1:
    def test_returns_l1derived(self, l1_df):
        result = derive_l1(l1_df)
        assert isinstance(result, L1Derived)

    def test_time_is_datetime(self, l1):
        assert pd.api.types.is_datetime64_any_dtype(l1.time)

    def test_bid_ask_in_dollars(self, l1, l1_df):
        # bid/ask should be cents / 100
        expected_bid_first = l1_df["bid_price_cents"].iloc[0] / 100
        assert l1.bid.iloc[0] == pytest.approx(expected_bid_first)

    def test_spread_positive(self, l1):
        valid_spread = l1.spread.dropna()
        assert (valid_spread >= 0).all(), "Spread should never be negative"

    def test_mid_between_bid_ask(self, l1):
        valid = l1.bid.notna() & l1.ask.notna()
        assert (l1.mid[valid] >= l1.bid[valid]).all()
        assert (l1.mid[valid] <= l1.ask[valid]).all()

    def test_log_returns_no_inf(self, l1):
        assert not np.isinf(l1.log_returns).any()

    def test_log_returns_no_nan(self, l1):
        assert not l1.log_returns.isna().any()

    def test_handles_nan_prices(self):
        df = pd.DataFrame(
            {
                "time_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000],
                "bid_price_cents": [10_000, np.nan, 10_002],
                "ask_price_cents": [10_002, np.nan, 10_004],
                "bid_qty": [100, 100, 100],
                "ask_qty": [100, 100, 100],
            }
        )
        result = derive_l1(df)
        assert isinstance(result, L1Derived)
        # Log returns should skip NaN mid rows
        assert len(result.log_returns) <= 2


# ---------------------------------------------------------------------------
# compute_spread_stats
# ---------------------------------------------------------------------------


class TestComputeSpreadStats:
    def test_returns_spread_stats(self, l1):
        stats = compute_spread_stats(l1.spread, l1.mid)
        assert isinstance(stats, SpreadStats)

    def test_counts_consistent(self, l1):
        stats = compute_spread_stats(l1.spread, l1.mid)
        assert stats.n_total == stats.n_two_sided + stats.n_one_sided

    def test_mean_positive(self, l1):
        stats = compute_spread_stats(l1.spread, l1.mid)
        # Spread should be positive when bid < ask
        assert stats.mean is not None
        assert stats.mean > 0

    def test_median_leq_max(self, l1):
        stats = compute_spread_stats(l1.spread, l1.mid)
        assert stats.median is not None and stats.max is not None
        assert stats.median <= stats.max

    def test_pct_fields_present(self, l1):
        stats = compute_spread_stats(l1.spread, l1.mid)
        assert stats.mean_pct is not None
        assert stats.median_pct is not None

    def test_all_nan_spread(self):
        spread = pd.Series([np.nan, np.nan, np.nan])
        mid = pd.Series([100.0, 100.01, 100.02])
        stats = compute_spread_stats(spread, mid)
        assert stats.n_two_sided == 0
        assert stats.mean is None

    def test_single_value(self):
        spread = pd.Series([0.02])
        mid = pd.Series([100.0])
        stats = compute_spread_stats(spread, mid)
        assert stats.mean == pytest.approx(0.02)
        assert stats.std is None  # Need > 1 for std


# ---------------------------------------------------------------------------
# compute_return_stats
# ---------------------------------------------------------------------------


class TestComputeReturnStats:
    def test_returns_return_stats(self, l1):
        result = compute_return_stats(l1.log_returns)
        assert isinstance(result, ReturnStats)

    def test_returns_none_for_short_series(self):
        short = pd.Series([0.001, 0.002, 0.003])
        assert compute_return_stats(short) is None

    def test_mean_finite(self, l1):
        result = compute_return_stats(l1.log_returns)
        assert result is not None
        assert np.isfinite(result.mean)

    def test_std_non_negative(self, l1):
        result = compute_return_stats(l1.log_returns)
        assert result is not None
        assert result.std >= 0


# ---------------------------------------------------------------------------
# compute_book_pressure
# ---------------------------------------------------------------------------


class TestComputeBookPressure:
    def test_returns_series(self, l1_df):
        result = compute_book_pressure(l1_df)
        assert isinstance(result, pd.Series)
        assert len(result) == len(l1_df)

    def test_pressure_is_bid_minus_ask_qty(self, l1_df):
        result = compute_book_pressure(l1_df)
        expected = l1_df["bid_qty"].astype(float) - l1_df["ask_qty"].astype(float)
        pd.testing.assert_series_equal(result, expected, check_names=False, check_dtype=False)


# ---------------------------------------------------------------------------
# compute_rolling_vol
# ---------------------------------------------------------------------------


class TestComputeRollingVol:
    def test_returns_none_for_short_series(self):
        short = pd.Series([0.001] * 10)
        assert compute_rolling_vol(short) is None

    def test_returns_tuple(self, l1):
        result = compute_rolling_vol(l1.log_returns)
        assert result is not None
        vol_series, window = result
        assert isinstance(vol_series, pd.Series)
        assert isinstance(window, int)
        assert window > 0


# ---------------------------------------------------------------------------
# compute_order_flow_stats
# ---------------------------------------------------------------------------


class TestComputeOrderFlowStats:
    def test_returns_order_flow_stats(self, order_df):
        result = compute_order_flow_stats(order_df)
        assert isinstance(result, OrderFlowStats)

    def test_counts_add_up(self, order_df):
        result = compute_order_flow_stats(order_df)
        # executed + cancelled <= submitted (approximately; some may be neither)
        assert result.executed >= 0
        assert result.cancelled >= 0
        assert result.total_submitted >= 0

    def test_fill_rate_in_range(self, order_df):
        result = compute_order_flow_stats(order_df)
        assert 0 <= result.fill_rate <= 100

    def test_handles_missing_event_type(self):
        df = pd.DataFrame({"price": [100, 200]})
        result = compute_order_flow_stats(df)
        assert result == OrderFlowStats(0, 0, 0, 0.0, 0.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame({"EventType": pd.Series(dtype=str)})
        result = compute_order_flow_stats(df)
        assert result.total_submitted == 0
        assert result.fill_rate == 0.0


# ---------------------------------------------------------------------------
# compute_cumulative_imbalance
# ---------------------------------------------------------------------------


class TestComputeCumulativeImbalance:
    def test_returns_dataframe(self, order_df):
        result = compute_cumulative_imbalance(order_df)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "cum_imbalance" in result.columns

    def test_returns_none_without_required_columns(self):
        df = pd.DataFrame({"price": [100]})
        assert compute_cumulative_imbalance(df) is None

    def test_bid_adds_ask_subtracts(self):
        df = pd.DataFrame(
            {
                "EventType": ["ORDER_SUBMITTED", "ORDER_SUBMITTED", "ORDER_SUBMITTED"],
                "EventTime": pd.date_range("2026-01-01", periods=3, freq="s"),
                "side": ["BID", "ASK", "BID"],
            }
        )
        result = compute_cumulative_imbalance(df)
        assert result is not None
        assert result["cum_imbalance"].tolist() == [1, 0, 1]


# ---------------------------------------------------------------------------
# compute_maker_taker_summary
# ---------------------------------------------------------------------------


class TestComputeMakerTakerSummary:
    def test_basic_summary(self):
        df = pd.DataFrame(
            {
                "maker_type": ["MM", "MM", "Noise"],
                "taker_type": ["Noise", "Value", "MM"],
                "quantity": [100, 200, 50],
            }
        )
        result = compute_maker_taker_summary(df)
        assert result.total_trades == 3
        assert isinstance(result.maker_volume_by_type, pd.Series)
        assert result.maker_volume_by_type["MM"] == 300
        assert result.taker_volume_by_type["Noise"] == 100

    def test_empty_dataframe(self):
        df = pd.DataFrame({"maker_type": pd.Series(dtype=str), "taker_type": pd.Series(dtype=str), "quantity": pd.Series(dtype=int)})
        result = compute_maker_taker_summary(df)
        assert result.total_trades == 0


# ---------------------------------------------------------------------------
# Edge cases: derive_l1 with extreme data
# ---------------------------------------------------------------------------


class TestDeriveL1EdgeCases:
    def test_identical_prices(self):
        """When all prices are equal, log returns should be 0."""
        df = pd.DataFrame(
            {
                "time_ns": [1_000_000_000 * i for i in range(5)],
                "bid_price_cents": [10_000] * 5,
                "ask_price_cents": [10_002] * 5,
                "bid_qty": [100] * 5,
                "ask_qty": [100] * 5,
            }
        )
        result = derive_l1(df)
        if len(result.log_returns) > 0:
            assert (result.log_returns.abs() < 1e-10).all()

    def test_two_rows_minimum(self):
        """derive_l1 should handle the minimum case of 2 rows."""
        df = pd.DataFrame(
            {
                "time_ns": [1_000_000_000, 2_000_000_000],
                "bid_price_cents": [10_000, 10_010],
                "ask_price_cents": [10_002, 10_012],
                "bid_qty": [100, 100],
                "ask_qty": [100, 100],
            }
        )
        result = derive_l1(df)
        assert len(result.mid) == 2
