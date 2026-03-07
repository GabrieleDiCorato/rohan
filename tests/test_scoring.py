"""Tests for deterministic multi-axis scoring module."""

import pytest

from rohan.llm.scoring import (
    WEIGHT_PROFILES,
    AxisScores,
    ScoringWeights,
    _clamp,
    _lerp,
    classify_goal_weights,
    compute_axis_scores,
    compute_final_score,
)

# ── Helpers ──────────────────────────────────────────────────────────────


class TestLerp:
    def test_midpoint(self):
        assert _lerp(5.0, 0.0, 10.0, 0.0, 100.0) == pytest.approx(50.0)

    def test_at_low_boundary(self):
        assert _lerp(0.0, 0.0, 10.0, 0.0, 100.0) == pytest.approx(0.0)

    def test_at_high_boundary(self):
        assert _lerp(10.0, 0.0, 10.0, 0.0, 100.0) == pytest.approx(100.0)

    def test_clamps_below(self):
        assert _lerp(-5.0, 0.0, 10.0, 0.0, 100.0) == pytest.approx(0.0)

    def test_clamps_above(self):
        assert _lerp(15.0, 0.0, 10.0, 0.0, 100.0) == pytest.approx(100.0)

    def test_inverted_output_range(self):
        """Output range can be decreasing (out_lo > out_hi)."""
        assert _lerp(5.0, 0.0, 10.0, 100.0, 0.0) == pytest.approx(50.0)

    def test_zero_width_input(self):
        assert _lerp(5.0, 5.0, 5.0, 0.0, 10.0) == pytest.approx(5.0)


class TestClamp:
    def test_within_range(self):
        assert _clamp(5.0) == 5.0

    def test_below(self):
        assert _clamp(-1.0) == 1.0

    def test_above(self):
        assert _clamp(15.0) == 10.0

    def test_at_boundaries(self):
        assert _clamp(1.0) == 1.0
        assert _clamp(10.0) == 10.0


# ── ScoringWeights ───────────────────────────────────────────────────────


class TestScoringWeights:
    def test_valid_weights(self):
        w = ScoringWeights(
            profitability=0.35,
            risk_adjusted=0.20,
            volatility_impact=0.07,
            spread_impact=0.08,
            liquidity_impact=0.05,
            execution_quality=0.25,
        )
        assert w.profitability == 0.35

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(
                profitability=0.5,
                risk_adjusted=0.5,
                volatility_impact=0.5,
                spread_impact=0.5,
                liquidity_impact=0.5,
                execution_quality=0.5,
            )

    def test_all_profiles_sum_to_one(self):
        for name, w in WEIGHT_PROFILES.items():
            total = w.profitability + w.risk_adjusted + w.volatility_impact + w.spread_impact + w.liquidity_impact + w.execution_quality
            assert abs(total - 1.0) < 1e-6, f"Profile {name!r} sums to {total}"

    def test_weights_are_frozen(self):
        w = WEIGHT_PROFILES["default"]
        with pytest.raises(AttributeError):
            w.profitability = 0.99  # type: ignore[misc]


# ── classify_goal_weights ────────────────────────────────────────────────


class TestClassifyGoalWeights:
    def test_default_profile(self):
        w = classify_goal_weights("Make a profitable market maker")
        assert w == WEIGHT_PROFILES["default"]

    def test_impact_profile_vpin(self):
        w = classify_goal_weights("Minimise VPIN and reduce market impact")
        assert w == WEIGHT_PROFILES["impact_focused"]

    def test_impact_profile_spread(self):
        w = classify_goal_weights("Tighten the spread and stabilize volatility")
        assert w == WEIGHT_PROFILES["impact_focused"]

    def test_risk_profile_sharpe(self):
        w = classify_goal_weights("Maximise the Sharpe ratio with low drawdown")
        assert w == WEIGHT_PROFILES["risk_focused"]

    def test_risk_profile_hedge(self):
        w = classify_goal_weights("Hedge risk-adjusted returns with protection")
        assert w == WEIGHT_PROFILES["risk_focused"]

    def test_execution_profile(self):
        w = classify_goal_weights("Improve fill rate and reduce slippage")
        assert w == WEIGHT_PROFILES["execution_focused"]

    def test_execution_profile_ott(self):
        w = classify_goal_weights("Lower order-to-trade ratio and improve execution")
        assert w == WEIGHT_PROFILES["execution_focused"]

    def test_empty_goal_returns_default(self):
        w = classify_goal_weights("")
        assert w == WEIGHT_PROFILES["default"]

    def test_case_insensitive(self):
        w = classify_goal_weights("MINIMISE VPIN")
        assert w == WEIGHT_PROFILES["impact_focused"]

    def test_tie_returns_default(self):
        """When multiple categories have equal hits, fall back to default."""
        w = classify_goal_weights("risk and impact")
        assert w == WEIGHT_PROFILES["default"]


# ── compute_axis_scores — Profitability ──────────────────────────────────

# Shared defaults for fields not under test
_DEFAULTS = {
    "sharpe_ratio": None,
    "max_drawdown": None,
    "fill_rate": None,
    "order_to_trade_ratio": None,
    "volatility_delta_pct": None,
    "spread_delta_pct": None,
    "bid_liquidity_delta_pct": None,
    "ask_liquidity_delta_pct": None,
    "starting_capital_cents": 10_000_000,
    "baseline_mean_spread": 200.0,  # 200 cents = $2 spread
    "baseline_traded_volume": 50_000.0,  # 50K shares
}


class TestProfitabilityScoring:
    """Tests for the profitability axis (opportunity capture rate)."""

    def _score(self, pnl: float | None, trades: int = 100, **overrides) -> float:
        kw = {**_DEFAULTS, "strategy_pnl": pnl, "trade_count": trades, **overrides}
        return compute_axis_scores(**kw).profitability

    def test_zero_trades_floor(self):
        assert self._score(1000.0, trades=0) == 1.0

    def test_none_pnl(self):
        assert self._score(None) == 1.0

    def test_no_baseline_spread(self):
        assert self._score(1000.0, baseline_mean_spread=None) == 1.0

    def test_no_baseline_volume(self):
        assert self._score(1000.0, baseline_traded_volume=None) == 1.0

    def test_zero_baseline_volume(self):
        assert self._score(1000.0, baseline_traded_volume=0.0) == 1.0

    def test_negative_pnl_deep(self):
        """Very negative PnL should score near 1.0."""
        # opportunity = 200 * 50000 / 2 = 5,000,000
        # capture_rate = -200_000 / 5_000_000 = -0.04 → capped at -0.02 → 1.0
        score = self._score(-200_000.0)
        assert score == pytest.approx(1.0)

    def test_negative_pnl_moderate(self):
        """Moderate negative PnL → between 1 and 3."""
        # capture_rate = -50_000 / 5_000_000 = -0.01
        score = self._score(-50_000.0)
        assert 1.0 <= score <= 3.0

    def test_zero_pnl(self):
        """Zero PnL should score 3.0 (boundary of losing/minimal)."""
        score = self._score(0.0)
        assert score == pytest.approx(3.0)

    def test_minimal_capture(self):
        """capture_rate = 0.0005 → should be in 3–5 range."""
        # opportunity = 5,000,000. pnl = 5_000_000 * 0.0005 = 2500
        score = self._score(2500.0)
        assert 3.0 <= score <= 5.0

    def test_solid_capture(self):
        """capture_rate = 0.003 → should be in 5–7 range."""
        # pnl = 5_000_000 * 0.003 = 15_000
        score = self._score(15_000.0)
        assert 5.0 <= score <= 7.0

    def test_strong_capture(self):
        """capture_rate = 0.01 → should be in 7–9 range."""
        # pnl = 5_000_000 * 0.01 = 50_000
        score = self._score(50_000.0)
        assert 7.0 <= score <= 9.0

    def test_exceptional_capture(self):
        """capture_rate = 0.03 → should be in 9–10 range."""
        # pnl = 5_000_000 * 0.03 = 150_000
        score = self._score(150_000.0)
        assert 9.0 <= score <= 10.0

    def test_capture_rate_at_boundary_0001(self):
        """capture_rate = 0.001 exactly → should be 5.0."""
        pnl = 5_000_000 * 0.001  # = 5000
        score = self._score(pnl)
        assert score == pytest.approx(5.0)

    def test_capture_rate_at_boundary_0005(self):
        """capture_rate = 0.005 exactly → should be 7.0."""
        pnl = 5_000_000 * 0.005  # = 25_000
        score = self._score(pnl)
        assert score == pytest.approx(7.0)

    def test_capture_rate_at_boundary_002(self):
        """capture_rate = 0.02 exactly → should be 9.0."""
        pnl = 5_000_000 * 0.02  # = 100_000
        score = self._score(pnl)
        assert score == pytest.approx(9.0)


# ── compute_axis_scores — Risk ───────────────────────────────────────────


class TestRiskScoring:
    """Tests for the risk-adjusted performance axis."""

    def _score(self, sharpe: float | None = None, trades: int = 100, drawdown: float | None = None, capital: int = 10_000_000) -> float:
        kw = {
            **_DEFAULTS,
            "strategy_pnl": 1000.0,
            "trade_count": trades,
            "sharpe_ratio": sharpe,
            "max_drawdown": drawdown,
            "starting_capital_cents": capital,
        }
        return compute_axis_scores(**kw).risk_adjusted

    def test_none_sharpe_few_trades(self):
        """< 5 trades with None Sharpe → 3.0."""
        assert self._score(sharpe=None, trades=3) == pytest.approx(3.0)

    def test_none_sharpe_many_trades(self):
        """≥ 5 trades with None Sharpe → 2.0."""
        assert self._score(sharpe=None, trades=10) == pytest.approx(2.0)

    def test_sharpe_very_negative(self):
        assert self._score(sharpe=-2.0) == pytest.approx(1.0)

    def test_sharpe_negative(self):
        """Sharpe = -0.5 → between 1 and 3."""
        score = self._score(sharpe=-0.5)
        assert 1.0 <= score <= 3.0

    def test_sharpe_zero(self):
        assert self._score(sharpe=0.0) == pytest.approx(3.0)

    def test_sharpe_low_positive(self):
        """Sharpe = 0.25 → between 3 and 4."""
        score = self._score(sharpe=0.25)
        assert 3.0 <= score <= 4.0

    def test_sharpe_moderate(self):
        """Sharpe = 1.0 → between 4 and 6."""
        score = self._score(sharpe=1.0)
        assert 4.0 <= score <= 6.0

    def test_sharpe_good(self):
        """Sharpe = 2.0 → between 6 and 8."""
        score = self._score(sharpe=2.0)
        assert 6.0 <= score <= 8.0

    def test_sharpe_excellent(self):
        """Sharpe = 4.0 → between 8 and 10."""
        score = self._score(sharpe=4.0)
        assert 8.0 <= score <= 10.0

    def test_drawdown_severe_penalty(self):
        """Drawdown > 5% of capital → -2 penalty."""
        # No drawdown
        base = self._score(sharpe=2.0, drawdown=None)
        # 6% drawdown
        penalized = self._score(sharpe=2.0, drawdown=600_000, capital=10_000_000)
        assert base - penalized == pytest.approx(2.0)

    def test_drawdown_moderate_penalty(self):
        """Drawdown 2-5% of capital → -1 penalty."""
        base = self._score(sharpe=2.0, drawdown=None)
        # 3% drawdown
        penalized = self._score(sharpe=2.0, drawdown=300_000, capital=10_000_000)
        assert base - penalized == pytest.approx(1.0)

    def test_drawdown_small_no_penalty(self):
        """Drawdown < 2% of capital → no penalty."""
        base = self._score(sharpe=2.0, drawdown=None)
        # 1% drawdown
        penalized = self._score(sharpe=2.0, drawdown=100_000, capital=10_000_000)
        assert base == penalized

    def test_drawdown_clamps_to_floor(self):
        """Even with a big penalty, score cannot go below 1.0."""
        # Sharpe = -0.5 → ~2.0, minus 2 drawdown → clamped at 1.0
        score = self._score(sharpe=-0.5, drawdown=600_000)
        assert score == pytest.approx(1.0)


# ── compute_axis_scores — Impact axes ────────────────────────────────────


class TestVolatilityImpactScoring:
    def _score(self, delta: float | None) -> float:
        kw = {**_DEFAULTS, "strategy_pnl": 1000.0, "trade_count": 100, "volatility_delta_pct": delta}
        return compute_axis_scores(**kw).volatility_impact

    def test_none_delta_neutral(self):
        assert self._score(None) == pytest.approx(5.5)

    def test_strongly_destabilizing(self):
        assert self._score(0.15) == pytest.approx(1.0)

    def test_destabilizing(self):
        score = self._score(0.07)
        assert 1.0 <= score <= 4.0

    def test_neutral(self):
        score = self._score(0.0)
        assert 4.0 <= score <= 7.0

    def test_zero_delta_midpoint(self):
        assert self._score(0.0) == pytest.approx(5.5)

    def test_stabilizing(self):
        score = self._score(-0.10)
        assert 7.0 <= score <= 9.0

    def test_strongly_stabilizing(self):
        assert self._score(-0.20) == pytest.approx(9.0)

    def test_at_boundary_plus_10(self):
        """At +10%, the piecewise just enters the 'strongly destabilizing' zone."""
        assert self._score(0.10) == pytest.approx(1.0)

    def test_at_boundary_plus_5(self):
        assert self._score(0.05) == pytest.approx(4.0)

    def test_at_boundary_minus_5(self):
        assert self._score(-0.05) == pytest.approx(7.0)

    def test_at_boundary_minus_15(self):
        assert self._score(-0.15) == pytest.approx(9.0)


class TestSpreadImpactScoring:
    """Spread impact uses the same formula as volatility impact."""

    def _score(self, delta: float | None) -> float:
        kw = {**_DEFAULTS, "strategy_pnl": 1000.0, "trade_count": 100, "spread_delta_pct": delta}
        return compute_axis_scores(**kw).spread_impact

    def test_widening(self):
        assert self._score(0.12) == pytest.approx(1.0)

    def test_tightening(self):
        score = self._score(-0.10)
        assert 7.0 <= score <= 9.0

    def test_neutral(self):
        assert self._score(0.0) == pytest.approx(5.5)


class TestLiquidityImpactScoring:
    """Liquidity impact is inverted: positive delta = good."""

    def _score(self, bid_delta: float | None, ask_delta: float | None) -> float:
        kw = {
            **_DEFAULTS,
            "strategy_pnl": 1000.0,
            "trade_count": 100,
            "bid_liquidity_delta_pct": bid_delta,
            "ask_liquidity_delta_pct": ask_delta,
        }
        return compute_axis_scores(**kw).liquidity_impact

    def test_both_none_neutral(self):
        assert self._score(None, None) == pytest.approx(5.5)

    def test_strongly_draining(self):
        assert self._score(-0.20, -0.20) == pytest.approx(1.0)

    def test_strongly_improving(self):
        assert self._score(0.20, 0.20) == pytest.approx(9.0)

    def test_neutral(self):
        assert self._score(0.0, 0.0) == pytest.approx(5.5)

    def test_one_side_only_bid(self):
        """Only bid liquidity delta provided."""
        score = self._score(0.10, None)
        assert 7.0 <= score <= 9.0

    def test_one_side_only_ask(self):
        """Only ask liquidity delta provided."""
        score = self._score(None, -0.10)
        assert 1.0 <= score <= 4.0

    def test_asymmetric_averaging(self):
        """Bid +10%, Ask -10% → average 0% → neutral."""
        assert self._score(0.10, -0.10) == pytest.approx(5.5)

    def test_at_boundary_minus_15(self):
        assert self._score(-0.15, -0.15) == pytest.approx(1.0)

    def test_at_boundary_minus_5(self):
        assert self._score(-0.05, -0.05) == pytest.approx(4.0)

    def test_at_boundary_plus_5(self):
        assert self._score(0.05, 0.05) == pytest.approx(7.0)

    def test_at_boundary_plus_15(self):
        assert self._score(0.15, 0.15) == pytest.approx(9.0)


# ── compute_axis_scores — Execution ─────────────────────────────────────


class TestExecutionScoring:
    def _score(self, fill_rate: float | None = 0.20, ott: float | None = None) -> float:
        kw = {
            **_DEFAULTS,
            "strategy_pnl": 1000.0,
            "trade_count": 100,
            "fill_rate": fill_rate,
            "order_to_trade_ratio": ott,
        }
        return compute_axis_scores(**kw).execution_quality

    def test_none_fill_rate(self):
        assert self._score(fill_rate=None) == pytest.approx(1.0)

    def test_very_low_fill_rate(self):
        score = self._score(fill_rate=0.02)
        assert 1.0 <= score <= 2.0

    def test_low_fill_rate(self):
        score = self._score(fill_rate=0.10)
        assert 2.0 <= score <= 4.0

    def test_moderate_fill_rate(self):
        score = self._score(fill_rate=0.20)
        assert 4.0 <= score <= 6.0

    def test_good_fill_rate(self):
        score = self._score(fill_rate=0.40)
        assert 6.0 <= score <= 8.0

    def test_excellent_fill_rate(self):
        score = self._score(fill_rate=0.60)
        assert 8.0 <= score <= 10.0

    def test_ott_extreme_penalty(self):
        """OTT > 200 → -2 points."""
        base = self._score(fill_rate=0.40, ott=None)
        penalized = self._score(fill_rate=0.40, ott=250.0)
        assert base - penalized == pytest.approx(2.0)

    def test_ott_moderate_penalty(self):
        """OTT 100–200 → -1 point."""
        base = self._score(fill_rate=0.40, ott=None)
        penalized = self._score(fill_rate=0.40, ott=150.0)
        assert base - penalized == pytest.approx(1.0)

    def test_ott_below_threshold_no_penalty(self):
        """OTT < 100 → no penalty."""
        base = self._score(fill_rate=0.40, ott=None)
        no_penalty = self._score(fill_rate=0.40, ott=50.0)
        assert base == no_penalty

    def test_ott_penalty_clamps_floor(self):
        """Fill rate 2% (score ~1.4) with OTT 250 → clamped to 1.0."""
        score = self._score(fill_rate=0.02, ott=250.0)
        assert score == pytest.approx(1.0)

    def test_fill_rate_boundary_5pct(self):
        assert self._score(fill_rate=0.05) == pytest.approx(2.0)

    def test_fill_rate_boundary_15pct(self):
        assert self._score(fill_rate=0.15) == pytest.approx(4.0)

    def test_fill_rate_boundary_30pct(self):
        assert self._score(fill_rate=0.30) == pytest.approx(6.0)

    def test_fill_rate_boundary_50pct(self):
        assert self._score(fill_rate=0.50) == pytest.approx(8.0)


# ── compute_final_score ──────────────────────────────────────────────────


class TestComputeFinalScore:
    def test_uniform_scores(self):
        """All axes = 5.0 → final = 5.0 regardless of weights."""
        axes = AxisScores(5.0, 5.0, 5.0, 5.0, 5.0, 5.0)
        score = compute_final_score(axes, WEIGHT_PROFILES["default"])
        assert score == pytest.approx(5.0)

    def test_rounds_to_nearest_half(self):
        """Score should round to nearest 0.5."""
        # Construct scores that produce a non-0.5 average
        axes = AxisScores(5.1, 5.1, 5.1, 5.1, 5.1, 5.1)
        score = compute_final_score(axes, WEIGHT_PROFILES["default"])
        assert score == 5.0  # 5.1 rounds to 5.0

    def test_rounds_up_past_midpoint(self):
        """5.3 → raw*2 = 10.6 → round = 11 → 11/2 = 5.5."""
        axes = AxisScores(5.3, 5.3, 5.3, 5.3, 5.3, 5.3)
        score = compute_final_score(axes, WEIGHT_PROFILES["default"])
        assert score == 5.5

    def test_with_default_profile_weight_emphasis(self):
        """Default profile weights profitability (35%) and execution (25%) most."""
        axes = AxisScores(
            profitability=10.0,  # 35%
            risk_adjusted=1.0,  # 20%
            volatility_impact=1.0,  # 7%
            spread_impact=1.0,  # 8%
            liquidity_impact=1.0,  # 5%
            execution_quality=1.0,  # 25%
        )
        score = compute_final_score(axes, WEIGHT_PROFILES["default"])
        # = 10*0.35 + 1*0.20 + 1*0.07 + 1*0.08 + 1*0.05 + 1*0.25
        # = 3.5 + 0.65 = 4.15 → rounds to 4.0
        assert score == pytest.approx(4.0)

    def test_clamped_to_10(self):
        """Perfect scores should not exceed 10.0."""
        axes = AxisScores(10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        score = compute_final_score(axes, WEIGHT_PROFILES["default"])
        assert score == 10.0

    def test_clamped_to_1(self):
        """Floor scores should not go below 1.0."""
        axes = AxisScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        score = compute_final_score(axes, WEIGHT_PROFILES["default"])
        assert score == 1.0

    def test_impact_focused_emphasizes_impact_axes(self):
        """Impact profile gives 20+20+15 = 55% to impact axes."""
        axes = AxisScores(
            profitability=1.0,
            risk_adjusted=1.0,
            volatility_impact=10.0,
            spread_impact=10.0,
            liquidity_impact=10.0,
            execution_quality=1.0,
        )
        score = compute_final_score(axes, WEIGHT_PROFILES["impact_focused"])
        # = 1*0.15 + 1*0.10 + 10*0.20 + 10*0.20 + 10*0.15 + 1*0.20
        # = 0.15 + 0.10 + 2.0 + 2.0 + 1.5 + 0.20 = 5.95 → 6.0
        assert score == pytest.approx(6.0)

    def test_all_profiles_produce_valid_range(self):
        """Verify all profiles produce scores within [1, 10] for extreme inputs."""
        low = AxisScores(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        high = AxisScores(10.0, 10.0, 10.0, 10.0, 10.0, 10.0)
        for profile_name, weights in WEIGHT_PROFILES.items():
            lo_score = compute_final_score(low, weights)
            hi_score = compute_final_score(high, weights)
            assert 1.0 <= lo_score <= 10.0, f"{profile_name} low out of range"
            assert 1.0 <= hi_score <= 10.0, f"{profile_name} high out of range"


# ── Integration: full compute_axis_scores ────────────────────────────────


class TestComputeAxisScoresIntegration:
    """End-to-end tests combining all axes."""

    def test_all_scores_in_valid_range(self):
        """Every axis score must be in [1.0, 10.0]."""
        scores = compute_axis_scores(
            strategy_pnl=5000.0,
            trade_count=100,
            sharpe_ratio=1.5,
            max_drawdown=150_000,
            fill_rate=0.25,
            order_to_trade_ratio=8.0,
            volatility_delta_pct=0.03,
            spread_delta_pct=-0.02,
            bid_liquidity_delta_pct=0.05,
            ask_liquidity_delta_pct=0.03,
            starting_capital_cents=10_000_000,
            baseline_mean_spread=200.0,
            baseline_traded_volume=50_000.0,
        )
        for field_name in ("profitability", "risk_adjusted", "volatility_impact", "spread_impact", "liquidity_impact", "execution_quality"):
            val = getattr(scores, field_name)
            assert 1.0 <= val <= 10.0, f"{field_name} = {val} out of range"

    def test_all_none_metrics(self):
        """Strategy that did nothing — all optional metrics are None."""
        scores = compute_axis_scores(
            strategy_pnl=None,
            trade_count=0,
            sharpe_ratio=None,
            max_drawdown=None,
            fill_rate=None,
            order_to_trade_ratio=None,
            volatility_delta_pct=None,
            spread_delta_pct=None,
            bid_liquidity_delta_pct=None,
            ask_liquidity_delta_pct=None,
            starting_capital_cents=10_000_000,
            baseline_mean_spread=200.0,
            baseline_traded_volume=50_000.0,
        )
        assert scores.profitability == 1.0
        assert scores.risk_adjusted == 3.0  # None Sharpe + < 5 trades
        assert scores.volatility_impact == 5.5  # no data → neutral
        assert scores.spread_impact == 5.5
        assert scores.liquidity_impact == 5.5
        assert scores.execution_quality == 1.0

    def test_excellent_strategy(self):
        """A strategy that aces everything should score high on all axes."""
        scores = compute_axis_scores(
            strategy_pnl=150_000.0,  # capture_rate = 0.03 → 9-10 range
            trade_count=500,
            sharpe_ratio=4.0,  # excellent
            max_drawdown=50_000,  # 0.5% of capital → no penalty
            fill_rate=0.60,  # excellent
            order_to_trade_ratio=5.0,  # no penalty
            volatility_delta_pct=-0.10,  # stabilizing
            spread_delta_pct=-0.10,  # tightening
            bid_liquidity_delta_pct=0.10,  # improving
            ask_liquidity_delta_pct=0.10,
            starting_capital_cents=10_000_000,
            baseline_mean_spread=200.0,
            baseline_traded_volume=50_000.0,
        )
        assert scores.profitability >= 9.0
        assert scores.risk_adjusted >= 8.0
        assert scores.volatility_impact >= 7.0
        assert scores.spread_impact >= 7.0
        assert scores.liquidity_impact >= 7.0
        assert scores.execution_quality >= 8.0
