"""Tests for multi-axis scoring module."""

import pytest

from rohan.llm.scoring import (
    WEIGHT_PROFILES,
    ScoringWeights,
    build_scoring_rubric,
    classify_goal_weights,
)


class TestScoringWeights:
    def test_valid_weights(self):
        w = ScoringWeights(
            profitability=0.4,
            risk_adjusted=0.25,
            market_impact=0.15,
            execution_quality=0.20,
        )
        assert w.profitability == 0.4

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(
                profitability=0.5,
                risk_adjusted=0.5,
                market_impact=0.5,
                execution_quality=0.5,
            )

    def test_all_profiles_sum_to_one(self):
        for name, w in WEIGHT_PROFILES.items():
            total = w.profitability + w.risk_adjusted + w.market_impact + w.execution_quality
            assert abs(total - 1.0) < 1e-6, f"Profile {name!r} sums to {total}"

    def test_weights_are_frozen(self):
        w = WEIGHT_PROFILES["default"]
        with pytest.raises(AttributeError):
            w.profitability = 0.99  # type: ignore[misc]


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
        # Both have 1 hit each — tie → default
        assert w == WEIGHT_PROFILES["default"]


class TestBuildScoringRubric:
    def test_rubric_contains_all_axes(self):
        rubric = build_scoring_rubric(WEIGHT_PROFILES["default"])
        assert "Profitability" in rubric
        assert "Risk-Adjusted" in rubric
        assert "Market Impact" in rubric
        assert "Execution Quality" in rubric

    def test_rubric_contains_weights(self):
        rubric = build_scoring_rubric(WEIGHT_PROFILES["default"])
        assert "40%" in rubric  # profitability weight
        assert "25%" in rubric  # risk weight
        assert "15%" in rubric  # impact weight
        assert "20%" in rubric  # execution weight

    def test_rubric_contains_formula(self):
        rubric = build_scoring_rubric(WEIGHT_PROFILES["default"])
        assert "final_score" in rubric
        assert "0.40" in rubric

    def test_impact_focused_rubric_weights(self):
        rubric = build_scoring_rubric(WEIGHT_PROFILES["impact_focused"])
        assert "45%" in rubric  # market_impact weight
        assert "20%" in rubric  # profitability weight

    def test_rubric_is_string(self):
        rubric = build_scoring_rubric(WEIGHT_PROFILES["default"])
        assert isinstance(rubric, str)
        assert len(rubric) > 100  # non-trivial content
