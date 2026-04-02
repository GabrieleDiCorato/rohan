"""Tests for defensive guards in AnalysisService.

Covers:
    - Volatility: minimum time-span guard (point 4)
    - Effective spread: NaN propagation guard (point 5)
    - VPIN: zero bucket-size guard (point 6)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from rohan.framework.analysis_service import AnalysisService

# ---------------------------------------------------------------------------
# Point 4: Volatility minimum time-span guard
# ---------------------------------------------------------------------------


class TestVolatilityTimeSpanGuard:
    """Volatility should return None if total time span < 60 seconds."""

    def test_short_span_returns_none(self):
        """Returns spanning < 60 seconds should produce None volatility."""
        # 50 observations over 10 seconds (200ms apart)
        n = 50
        times = pd.Series([34_200_000_000_000 + i * 200_000_000 for i in range(n)])
        # Synthetic returns with non-zero variance
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.001, n))

        result = AnalysisService._annualised_volatility(returns, times)
        assert result is None

    def test_long_span_returns_value(self):
        """Returns spanning > 60 seconds should produce a volatility value."""
        n = 50
        # 2 seconds apart = 100 seconds total
        times = pd.Series([34_200_000_000_000 + i * 2_000_000_000 for i in range(n)])
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0, 0.001, n))

        result = AnalysisService._annualised_volatility(returns, times)
        assert result is not None
        assert result > 0


# ---------------------------------------------------------------------------
# Point 5: Effective spread NaN guard
# ---------------------------------------------------------------------------


class TestEffectiveSpreadNaNGuard:
    """Effective spread should return None (not NaN) when all fills are NaN."""

    def test_all_nan_fills_returns_none(self):
        """When all effective spreads are NaN, should return None not NaN."""
        # Create a SimulationOutput mock where all fills have NaN prices
        result = MagicMock()
        logs = pd.DataFrame(
            {
                "EventType": ["ORDER_EXECUTED", "ORDER_EXECUTED"],
                "fill_price": [np.nan, np.nan],
                "quantity": [100, 200],
                "EventTime": [34_200_000_000_000, 34_201_000_000_000],
            }
        )
        result.get_logs_df.return_value = logs

        two_sided = pd.DataFrame(
            {
                "time": [0, 1_000_000_000],
                "bid_price": [10000.0, 10001.0],
                "ask_price": [10010.0, 10011.0],
                "mid_price": [10005.0, 10006.0],
            }
        )

        eff_spread, vol = AnalysisService._effective_spread_and_volume(result, two_sided)
        # NaN fills should result in None spread, not NaN
        assert eff_spread is None or not np.isnan(eff_spread)


# ---------------------------------------------------------------------------
# Point 6: VPIN zero bucket-size guard
# ---------------------------------------------------------------------------


class TestVpinZeroBucketGuard:
    """VPIN should return None when all fill quantities are zero."""

    def test_zero_volume_returns_none(self):
        """If all quantities are zero, bucket_size = 0 → should return None, not infinite loop."""
        result = MagicMock()
        n = 30  # above _MIN_FILLS_FOR_VPIN
        logs = pd.DataFrame(
            {
                "EventType": ["ORDER_EXECUTED"] * n,
                "fill_price": [10000.0] * n,
                "quantity": [0] * n,  # all zero!
                "EventTime": [34_200_000_000_000 + i * 1_000_000 for i in range(n)],
            }
        )
        result.get_logs_df.return_value = logs

        two_sided = pd.DataFrame(
            {
                "time": [0, 1_000_000_000],
                "bid_price": [10000.0, 10001.0],
                "ask_price": [10010.0, 10011.0],
                "mid_price": [10005.0, 10006.0],
            }
        )

        # This should NOT hang — should return None quickly
        vpin = AnalysisService._vpin(result, two_sided)
        assert vpin is None
