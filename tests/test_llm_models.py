"""Tests for LLM Pydantic response models (Phase 2.1)."""

import pytest
from pydantic import ValidationError

from rohan.llm.models import (
    AggregatedFeedback,
    GeneratedStrategy,
    IterationSummary,
    JudgeVerdict,
    ScenarioExplanation,
    ScenarioMetrics,
)


class TestGeneratedStrategy:
    def test_minimal_creation(self):
        gs = GeneratedStrategy(
            class_name="MyStrategy",
            code="class MyStrategy: pass",
            reasoning="Simple test",
        )
        assert gs.class_name == "MyStrategy"
        assert "class MyStrategy" in gs.code
        assert gs.reasoning == "Simple test"

    def test_required_fields(self):
        with pytest.raises(ValidationError):
            GeneratedStrategy()  # type: ignore[call-arg]


class TestScenarioExplanation:
    def test_defaults(self):
        exp = ScenarioExplanation()
        assert exp.scenario_name == "default"
        assert exp.strengths == []
        assert exp.weaknesses == []
        assert exp.recommendations == []
        assert exp.key_observations == []
        assert exp.raw_analysis == ""

    def test_full_creation(self):
        exp = ScenarioExplanation(
            scenario_name="volatile",
            strengths=["Good PnL"],
            weaknesses=["High drawdown"],
            market_impact_assessment="Stabilizing",
            key_observations=["Trades correlate with volume spikes"],
            recommendations=["Reduce position size"],
            raw_analysis="Detailed analysis...",
        )
        assert exp.scenario_name == "volatile"
        assert len(exp.strengths) == 1
        assert len(exp.recommendations) == 1


class TestJudgeVerdict:
    def test_valid_verdict(self):
        v = JudgeVerdict(
            score=7.5,
            comparison="better",
            reasoning="Improved PnL",
            recommendation="continue",
        )
        assert v.score == 7.5
        assert v.comparison == "better"
        assert v.recommendation == "continue"

    def test_score_bounds(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(score=0, comparison="better", reasoning="x", recommendation="continue")
        with pytest.raises(ValidationError):
            JudgeVerdict(score=11, comparison="better", reasoning="x", recommendation="continue")

    def test_invalid_comparison(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(score=5, comparison="unknown", reasoning="x", recommendation="continue")  # type: ignore[arg-type]

    def test_invalid_recommendation(self):
        with pytest.raises(ValidationError):
            JudgeVerdict(score=5, comparison="better", reasoning="x", recommendation="invalid")  # type: ignore[arg-type]

    def test_all_recommendations(self):
        for rec in ("continue", "stop_converged", "stop_plateau"):
            v = JudgeVerdict(score=5, comparison="similar", reasoning="ok", recommendation=rec)  # type: ignore[arg-type]
            assert v.recommendation == rec


class TestAggregatedFeedback:
    def test_creation(self):
        verdict = JudgeVerdict(score=6, comparison="better", reasoning="Improving", recommendation="continue")
        feedback = AggregatedFeedback(
            verdict=verdict,
            cross_scenario_patterns=["Pattern 1"],
            unified_feedback="Keep going",
        )
        assert feedback.verdict.score == 6
        assert len(feedback.cross_scenario_patterns) == 1
        assert feedback.unified_feedback == "Keep going"


class TestScenarioMetrics:
    def test_defaults(self):
        sm = ScenarioMetrics(scenario_name="test")
        assert sm.total_pnl is None
        assert sm.trade_count == 0

    def test_full(self):
        sm = ScenarioMetrics(
            scenario_name="volatile",
            total_pnl=-500.0,
            sharpe_ratio=-0.3,
            max_drawdown=1000.0,
            trade_count=42,
            volatility_delta_pct=0.05,
            spread_delta_pct=-0.02,
        )
        assert sm.total_pnl == -500.0
        assert sm.trade_count == 42


class TestIterationSummary:
    def test_creation(self):
        summary = IterationSummary(
            iteration_number=1,
            strategy_code="class X: pass",
        )
        assert summary.iteration_number == 1
        assert summary.judge_score is None
        assert summary.scenario_metrics == {}
        assert summary.timestamp is not None

    def test_with_metrics(self):
        sm = ScenarioMetrics(scenario_name="default", total_pnl=100.0)
        summary = IterationSummary(
            iteration_number=2,
            strategy_code="class Y: pass",
            scenario_metrics={"default": sm},
            judge_score=7.0,
            judge_reasoning="Good progress",
        )
        assert "default" in summary.scenario_metrics
        assert summary.judge_score == 7.0
