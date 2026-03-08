"""Parametrized boundary tests for deterministic scoring (Step 7).

Each ``pytest.mark.parametrize`` vector targets a specific piecewise
boundary, edge case, or guard condition in the 6-axis scoring system
defined in ``rohan.llm.scoring``.
"""

import pytest

from rohan.llm.scoring import (
    AxisScores,
    ScoringWeights,
    _score_execution,
    _score_liquidity_impact,
    _score_negative_impact,
    _score_profitability,
    _score_risk,
    compute_axis_scores,
    compute_final_score,
)

# ═══════════════════════════════════════════════════════════════════════════
# _score_profitability — opportunity capture rate
# ═══════════════════════════════════════════════════════════════════════════

# Common baseline values (spread=10 cents, volume=100_000 → opportunity = 500_000)
_SPREAD = 10.0
_VOLUME = 100_000.0
_OPP = _SPREAD * _VOLUME / 2.0  # 500_000


def _pnl_for_rate(rate: float) -> float:
    """Convert capture rate to PnL given standard baseline."""
    return rate * _OPP


class TestScoreProfitability:
    """Boundary tests for _score_profitability."""

    @pytest.mark.parametrize(
        ("pnl", "trades", "spread", "volume", "expected_lo", "expected_hi", "label"),
        [
            # Guard conditions
            (None, 10, _SPREAD, _VOLUME, 1.0, 1.0, "pnl_is_none"),
            (100, 0, _SPREAD, _VOLUME, 1.0, 1.0, "zero_trades"),
            (100, 10, None, _VOLUME, 1.0, 1.0, "no_baseline_spread"),
            (100, 10, _SPREAD, None, 1.0, 1.0, "no_baseline_volume"),
            (100, 10, 0, _VOLUME, 1.0, 1.0, "zero_spread"),
            # Negative capture (<0 → 1–3)
            (_pnl_for_rate(-0.02), 10, _SPREAD, _VOLUME, 1.0, 1.0, "deep_negative"),
            (_pnl_for_rate(-0.01), 10, _SPREAD, _VOLUME, 1.0, 3.0, "mid_negative"),
            (_pnl_for_rate(-0.001), 10, _SPREAD, _VOLUME, 2.5, 3.0, "slight_negative"),
            # Boundary: exactly 0
            (0, 10, _SPREAD, _VOLUME, 3.0, 3.0, "exactly_zero"),
            # Minimal capture (0–0.001 → 3–5)
            (_pnl_for_rate(0.0005), 10, _SPREAD, _VOLUME, 3.5, 4.5, "low_capture"),
            (_pnl_for_rate(0.001), 10, _SPREAD, _VOLUME, 5.0, 5.0, "boundary_0.001"),
            # Solid capture (0.001–0.005 → 5–7)
            (_pnl_for_rate(0.003), 10, _SPREAD, _VOLUME, 5.5, 6.5, "solid_capture"),
            (_pnl_for_rate(0.005), 10, _SPREAD, _VOLUME, 7.0, 7.0, "boundary_0.005"),
            # Strong capture (0.005–0.02 → 7–9)
            (_pnl_for_rate(0.01), 10, _SPREAD, _VOLUME, 7.5, 8.5, "strong_capture"),
            (_pnl_for_rate(0.02), 10, _SPREAD, _VOLUME, 9.0, 9.0, "boundary_0.02"),
            # Exceptional (>0.02 → 9–10)
            (_pnl_for_rate(0.03), 10, _SPREAD, _VOLUME, 9.0, 10.0, "exceptional"),
            (_pnl_for_rate(0.05), 10, _SPREAD, _VOLUME, 10.0, 10.0, "extreme_positive"),
        ],
    )
    def test_profitability_boundaries(self, pnl, trades, spread, volume, expected_lo, expected_hi, label):
        score = _score_profitability(pnl, trades, spread, volume)
        assert expected_lo <= score <= expected_hi, f"[{label}] score={score:.2f}, expected [{expected_lo}, {expected_hi}]"

    def test_always_in_range(self):
        """Score is always clamped to [1, 10]."""
        for rate in [-1.0, -0.1, 0, 0.001, 0.01, 0.1, 1.0]:
            score = _score_profitability(_pnl_for_rate(rate), 50, _SPREAD, _VOLUME)
            assert 1.0 <= score <= 10.0


# ═══════════════════════════════════════════════════════════════════════════
# _score_risk — Sharpe ratio + drawdown penalty
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreRisk:
    """Boundary tests for _score_risk."""

    _CAP = 10_000_000  # 100k dollars in cents

    @pytest.mark.parametrize(
        ("sharpe", "trades", "drawdown", "capital", "expected_lo", "expected_hi", "label"),
        [
            # Sharpe is None
            (None, 3, None, _CAP, 3.0, 3.0, "none_sharpe_low_trades"),
            (None, 10, None, _CAP, 2.0, 2.0, "none_sharpe_high_trades"),
            # Sharpe extremes
            (-2.0, 10, None, _CAP, 1.0, 1.0, "very_negative_sharpe"),
            (-1.0, 10, None, _CAP, 1.0, 1.0, "sharpe_neg1_boundary"),
            (-0.5, 10, None, _CAP, 1.0, 3.0, "sharpe_neg_half"),
            (0.0, 10, None, _CAP, 3.0, 3.0, "sharpe_zero"),
            (0.5, 10, None, _CAP, 4.0, 4.0, "sharpe_half"),
            (1.0, 10, None, _CAP, 4.0, 6.0, "sharpe_one"),
            (1.5, 10, None, _CAP, 6.0, 6.0, "sharpe_1.5"),
            (3.0, 10, None, _CAP, 8.0, 8.0, "sharpe_3.0"),
            (5.0, 10, None, _CAP, 10.0, 10.0, "sharpe_5.0"),
            (10.0, 10, None, _CAP, 10.0, 10.0, "sharpe_very_high"),
            # Drawdown penalties
            (1.5, 10, -int(_CAP * 0.06), _CAP, 4.0, 4.0, "large_drawdown_penalty_2"),
            (1.5, 10, -int(_CAP * 0.03), _CAP, 5.0, 5.0, "moderate_drawdown_penalty_1"),
            (1.5, 10, -int(_CAP * 0.01), _CAP, 6.0, 6.0, "small_drawdown_no_penalty"),
            # Drawdown + low Sharpe: can't go below 1.0
            (-0.5, 10, -int(_CAP * 0.06), _CAP, 1.0, 1.0, "drawdown_clamp_to_1"),
        ],
    )
    def test_risk_boundaries(self, sharpe, trades, drawdown, capital, expected_lo, expected_hi, label):
        score = _score_risk(sharpe, trades, drawdown, capital)
        assert expected_lo <= score <= expected_hi, f"[{label}] score={score:.2f}, expected [{expected_lo}, {expected_hi}]"

    _CAP_CLASS = 10_000_000

    def test_always_in_range(self):
        for sharpe in [None, -5, -1, 0, 0.5, 1, 2, 5, 10]:
            score = _score_risk(sharpe, 10, None, self._CAP_CLASS)
            assert 1.0 <= score <= 10.0


# ═══════════════════════════════════════════════════════════════════════════
# _score_negative_impact — volatility / spread (lower = better)
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreNegativeImpact:
    """Boundary tests for _score_negative_impact (used for vol + spread)."""

    @pytest.mark.parametrize(
        ("delta", "expected_lo", "expected_hi", "label"),
        [
            (None, 5.5, 5.5, "none_neutral"),
            (0.15, 1.0, 1.0, "very_positive_bad"),
            (0.10, 1.0, 1.0, "boundary_10pct"),
            (0.075, 1.0, 4.0, "between_5_10"),
            (0.05, 4.0, 4.0, "boundary_5pct"),
            (0.0, 4.0, 7.0, "zero_change"),
            (-0.05, 7.0, 7.0, "boundary_neg5"),
            (-0.10, 7.0, 9.0, "between_neg5_neg15"),
            (-0.15, 9.0, 9.0, "boundary_neg15"),
            (-0.25, 9.0, 9.0, "very_negative_capped"),
        ],
    )
    def test_negative_impact_boundaries(self, delta, expected_lo, expected_hi, label):
        score = _score_negative_impact(delta)
        assert expected_lo <= score <= expected_hi, f"[{label}] score={score:.2f}, expected [{expected_lo}, {expected_hi}]"


# ═══════════════════════════════════════════════════════════════════════════
# _score_liquidity_impact — positive change is good
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreLiquidityImpact:
    """Boundary tests for _score_liquidity_impact."""

    @pytest.mark.parametrize(
        ("delta", "expected_lo", "expected_hi", "label"),
        [
            (None, 5.5, 5.5, "none_neutral"),
            (-0.20, 1.0, 1.0, "very_negative"),
            (-0.15, 1.0, 1.0, "boundary_neg15"),
            (-0.10, 1.0, 4.0, "between_neg15_neg5"),
            (-0.05, 4.0, 4.0, "boundary_neg5"),
            (0.0, 4.0, 7.0, "zero_change"),
            (0.05, 7.0, 7.0, "boundary_pos5"),
            (0.10, 7.0, 9.0, "between_5_15"),
            (0.15, 9.0, 9.0, "boundary_pos15"),
            (0.25, 9.0, 9.0, "very_positive_capped"),
        ],
    )
    def test_liquidity_impact_boundaries(self, delta, expected_lo, expected_hi, label):
        score = _score_liquidity_impact(delta)
        assert expected_lo <= score <= expected_hi, f"[{label}] score={score:.2f}, expected [{expected_lo}, {expected_hi}]"

    def test_mirror_of_negative_impact(self):
        """Liquidity is the mirror of negative-impact: positive = good."""
        # +5% liquidity should score the same as -5% volatility
        liq = _score_liquidity_impact(0.05)
        vol = _score_negative_impact(-0.05)
        assert liq == vol


# ═══════════════════════════════════════════════════════════════════════════
# _score_execution — fill rate + OTT penalty
# ═══════════════════════════════════════════════════════════════════════════


class TestScoreExecution:
    """Boundary tests for _score_execution."""

    @pytest.mark.parametrize(
        ("fill_rate", "ott", "expected_lo", "expected_hi", "label"),
        [
            # Fill rate boundaries
            (None, None, 1.0, 1.0, "none_fill_rate"),
            (0.0, None, 1.0, 1.0, "zero_fill"),
            (0.025, None, 1.0, 2.0, "very_low_fill"),
            (0.05, None, 2.0, 2.0, "boundary_5pct"),
            (0.10, None, 2.0, 4.0, "low_fill"),
            (0.15, None, 4.0, 4.0, "boundary_15pct"),
            (0.20, None, 4.0, 6.0, "moderate_fill"),
            (0.30, None, 6.0, 6.0, "boundary_30pct"),
            (0.40, None, 6.0, 8.0, "good_fill"),
            (0.50, None, 8.0, 8.0, "boundary_50pct"),
            (0.65, None, 8.0, 10.0, "high_fill"),
            (0.80, None, 10.0, 10.0, "very_high_fill"),
            (1.0, None, 10.0, 10.0, "perfect_fill"),
            # OTT penalties
            (0.50, 50, 8.0, 8.0, "ott_50_no_penalty"),
            (0.50, 150, 7.0, 7.0, "ott_150_penalty_1"),
            (0.50, 250, 6.0, 6.0, "ott_250_penalty_2"),
            # OTT penalty floored at 1.0
            (0.05, 250, 1.0, 1.0, "ott_penalty_clamped"),
        ],
    )
    def test_execution_boundaries(self, fill_rate, ott, expected_lo, expected_hi, label):
        score = _score_execution(fill_rate, ott)
        assert expected_lo <= score <= expected_hi, f"[{label}] score={score:.2f}, expected [{expected_lo}, {expected_hi}]"


# ═══════════════════════════════════════════════════════════════════════════
# compute_axis_scores — integration (liquidity averaging)
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeAxisScoresIntegration:
    """Integration tests for compute_axis_scores."""

    _BASE = {
        "strategy_pnl": _pnl_for_rate(0.003),
        "trade_count": 50,
        "sharpe_ratio": 1.0,
        "max_drawdown": -100_000,
        "fill_rate": 0.30,
        "order_to_trade_ratio": 50.0,
        "volatility_delta_pct": -0.02,
        "spread_delta_pct": 0.01,
        "bid_liquidity_delta_pct": 0.05,
        "ask_liquidity_delta_pct": 0.03,
        "starting_capital_cents": 10_000_000,
        "baseline_mean_spread": _SPREAD,
        "baseline_traded_volume": _VOLUME,
    }

    def test_all_axes_in_range(self):
        axes = compute_axis_scores(**self._BASE)
        for field in ("profitability", "risk_adjusted", "volatility_impact", "spread_impact", "liquidity_impact", "execution_quality"):
            val = getattr(axes, field)
            assert 1.0 <= val <= 10.0, f"{field}={val}"

    def test_liquidity_averages_bid_ask(self):
        """Liquidity score should use average of bid and ask deltas."""
        axes_both = compute_axis_scores(**{**self._BASE, "bid_liquidity_delta_pct": 0.10, "ask_liquidity_delta_pct": 0.06})
        # Average = 0.08 → between 5% and 15%
        assert axes_both.liquidity_impact >= 7.0

    def test_liquidity_bid_only(self):
        axes = compute_axis_scores(**{**self._BASE, "bid_liquidity_delta_pct": 0.10, "ask_liquidity_delta_pct": None})
        assert axes.liquidity_impact >= 7.0

    def test_liquidity_none_both(self):
        axes = compute_axis_scores(**{**self._BASE, "bid_liquidity_delta_pct": None, "ask_liquidity_delta_pct": None})
        assert axes.liquidity_impact == 5.5  # neutral


# ═══════════════════════════════════════════════════════════════════════════
# compute_final_score — weighted average + rounding
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeFinalScore:
    """Tests for compute_final_score rounding and clamping."""

    _UNIFORM_WEIGHTS = ScoringWeights(
        profitability=1 / 6,
        risk_adjusted=1 / 6,
        volatility_impact=1 / 6,
        spread_impact=1 / 6,
        liquidity_impact=1 / 6,
        execution_quality=1 / 6,
    )

    def test_all_tens(self):
        axes = AxisScores(10, 10, 10, 10, 10, 10)
        assert compute_final_score(axes, self._UNIFORM_WEIGHTS) == 10.0

    def test_all_ones(self):
        axes = AxisScores(1, 1, 1, 1, 1, 1)
        assert compute_final_score(axes, self._UNIFORM_WEIGHTS) == 1.0

    def test_rounds_to_half(self):
        # All 5.0 → raw=5.0 → rounds to 5.0
        axes = AxisScores(5, 5, 5, 5, 5, 5)
        assert compute_final_score(axes, self._UNIFORM_WEIGHTS) == 5.0

    def test_weighted_score(self):
        """Heavy profitability weight with high profitability score."""
        from rohan.llm.scoring import WEIGHT_PROFILES

        axes = AxisScores(
            profitability=9.0,
            risk_adjusted=3.0,
            volatility_impact=5.0,
            spread_impact=5.0,
            liquidity_impact=5.0,
            execution_quality=7.0,
        )
        score = compute_final_score(axes, WEIGHT_PROFILES["default"])
        assert 5.0 <= score <= 8.0

    def test_result_clamped(self):
        # Even with all 1s and a weird weight, result is clamped to [1, 10]
        axes = AxisScores(1, 1, 1, 1, 1, 1)
        assert compute_final_score(axes, self._UNIFORM_WEIGHTS) >= 1.0
