"""Tests for LangGraph state models (Phase 2.3)."""

from rohan.llm.state import RefinementState, ScenarioConfig, ScenarioResult


class TestScenarioConfig:
    def test_defaults(self):
        sc = ScenarioConfig()
        assert sc.name == "default"
        assert sc.config_override == {}

    def test_with_overrides(self):
        sc = ScenarioConfig(
            name="volatile",
            config_override={"seed": 42, "start_time": "09:30:00"},
        )
        assert sc.name == "volatile"
        assert sc.config_override["seed"] == 42

    def test_serializable(self):
        sc = ScenarioConfig(name="test", config_override={"x": 1})
        d = sc.model_dump()
        restored = ScenarioConfig(**d)
        assert restored.name == "test"
        assert restored.config_override == {"x": 1}


class TestScenarioResult:
    def test_defaults(self):
        sr = ScenarioResult()
        assert sr.scenario_name == "default"
        assert sr.error is None
        assert sr.strategy_pnl is None
        assert sr.trade_count == 0

    def test_success_result(self):
        sr = ScenarioResult(
            scenario_name="stable",
            interpreter_prompt="Prompt text here",
            strategy_pnl=500.0,
            volatility_delta_pct=-0.03,
            spread_delta_pct=0.01,
            trade_count=15,
        )
        assert sr.strategy_pnl == 500.0
        assert sr.error is None

    def test_error_result(self):
        sr = ScenarioResult(
            scenario_name="crash",
            error="Simulation timed out",
        )
        assert sr.error == "Simulation timed out"
        assert sr.strategy_pnl is None


class TestRefinementState:
    def test_can_construct_initial_state(self):
        """RefinementState is a TypedDict — verify it works as a plain dict."""
        state: RefinementState = {
            "goal": "make money",
            "max_iterations": 3,
            "scenarios": [ScenarioConfig(name="default")],
            "current_code": None,
            "current_class_name": None,
            "current_reasoning": None,
            "validation_errors": [],
            "validation_attempts": 0,
            "scenario_results": [],
            "explanations": [],
            "aggregated_feedback": None,
            "iterations": [],
            "iteration_number": 1,
            "status": "writing",
            "messages": [],
        }
        assert state["goal"] == "make money"
        assert state["status"] == "writing"
        assert state["iteration_number"] == 1

    def test_partial_state(self):
        """total=False means not all keys are required."""
        state: RefinementState = {"goal": "test"}  # type: ignore[typeddict-item]
        assert state["goal"] == "test"
