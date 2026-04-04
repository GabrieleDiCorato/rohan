"""Tests for the scenario planner (3-tier fallback).

All LLM calls are mocked — these tests verify the fallback logic,
keyword heuristic, structured output parsing, and public API without
requiring API keys.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rohan.llm.models import PlannedScenario, ScenarioPlan
from rohan.llm.planner import (
    _keyword_fallback,
    _structured_fallback,
    _to_scenario_config,
    plan_scenarios,
)
from rohan.llm.state import ScenarioConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_settings(max_adversarial: int = 2) -> MagicMock:
    """Create a mock LLMSettings with the given max_adversarial_scenarios."""
    settings = MagicMock()
    settings.max_adversarial_scenarios = max_adversarial
    return settings


def _make_user_scenarios(*names: str) -> list[ScenarioConfig]:
    return [ScenarioConfig(name=n) for n in names]


# ---------------------------------------------------------------------------
# Tier 3: Keyword heuristic
# ---------------------------------------------------------------------------


class TestKeywordFallback:
    """_keyword_fallback is deterministic — no LLM needed."""

    def test_market_making_goal(self):
        results = _keyword_fallback("Build a market-making strategy", max_adversarial=2)
        assert len(results) >= 1
        assert any(s.name == "volatile_stress" for s in results)

    def test_spread_goal(self):
        results = _keyword_fallback("Optimise the bid-ask spread capture", max_adversarial=2)
        assert any(s.name == "wide_spread" for s in results)

    def test_profit_goal(self):
        results = _keyword_fallback("Maximise profit in all conditions", max_adversarial=2)
        assert any(s.name == "thin_liquidity" for s in results)

    def test_risk_goal(self):
        results = _keyword_fallback("Minimise risk exposure", max_adversarial=2)
        assert any(s.name == "volatile_stress" for s in results)

    def test_vpin_goal(self):
        results = _keyword_fallback("Reduce VPIN impact on fills", max_adversarial=2)
        assert any(s.name == "informed_flow" for s in results)

    def test_execution_goal(self):
        results = _keyword_fallback("Improve execution quality", max_adversarial=2)
        assert any(s.name == "fast_moving" for s in results)

    def test_respects_max_adversarial(self):
        # A goal that matches multiple keywords
        results = _keyword_fallback("market-making spread profit risk", max_adversarial=1)
        assert len(results) == 1

    def test_deduplicates_by_name(self):
        # "market-mak" and "risk" both map to "volatile_stress"
        results = _keyword_fallback("market-making risk strategy", max_adversarial=5)
        names = [s.name for s in results]
        assert len(names) == len(set(names)), "Duplicate scenario names returned"

    def test_unrecognised_goal_gets_default(self):
        results = _keyword_fallback("do something completely unrelated", max_adversarial=2)
        assert len(results) == 1
        assert results[0].name == "volatile_stress"
        assert results[0].rationale.startswith("Default adversarial")

    def test_empty_goal_gets_default(self):
        results = _keyword_fallback("", max_adversarial=1)
        assert len(results) == 1

    def test_max_zero_returns_empty(self):
        results = _keyword_fallback("market-making", max_adversarial=0)
        assert results == []

    def test_all_results_are_planned_scenarios(self):
        results = _keyword_fallback("market-making spread profit", max_adversarial=5)
        for s in results:
            assert isinstance(s, PlannedScenario)
            assert s.name
            assert s.template_name
            assert s.rationale


# ---------------------------------------------------------------------------
# Tier 2: Structured LLM fallback
# ---------------------------------------------------------------------------

# Patch targets for structured fallback (deferred imports from rohan.llm.factory)
_PATCH_PLANNER_MODEL = "rohan.llm.factory.get_planner_model"
_PATCH_STRUCTURED_MODEL = "rohan.llm.factory.get_structured_model"
_PATCH_LLM_SETTINGS = "rohan.config.llm_settings.LLMSettings"


class TestStructuredFallback:
    @patch(_PATCH_PLANNER_MODEL)
    @patch(_PATCH_STRUCTURED_MODEL)
    def test_successful_structured_output(self, mock_get_structured, _mock_get_planner):
        plan = ScenarioPlan(
            scenarios=[
                PlannedScenario(name="stress_vol", template_name="rmsc04", rationale="volatility stress"),
            ],
            reasoning="Added volatility stress",
        )
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = {"parsed": plan}
        mock_get_structured.return_value = mock_structured_model

        settings = _make_settings()
        result = _structured_fallback("market-making", ["default"], 2, settings)
        assert len(result) == 1
        assert result[0].name == "stress_vol"

    @patch(_PATCH_PLANNER_MODEL)
    @patch(_PATCH_STRUCTURED_MODEL)
    def test_structured_returns_none_falls_to_keywords(self, mock_get_structured, _mock_get_planner):
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = {"parsed": None}
        mock_get_structured.return_value = mock_structured_model

        settings = _make_settings()
        result = _structured_fallback("market-making", ["default"], 2, settings)
        # Should fall back to keyword heuristic
        assert len(result) >= 1
        assert any(s.name == "volatile_stress" for s in result)

    @patch(_PATCH_PLANNER_MODEL)
    @patch(_PATCH_STRUCTURED_MODEL)
    def test_structured_exception_falls_to_keywords(self, mock_get_structured, _mock_get_planner):
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.side_effect = RuntimeError("API error")
        mock_get_structured.return_value = mock_structured_model

        settings = _make_settings()
        result = _structured_fallback("profit maximisation", ["default"], 1, settings)
        assert len(result) == 1

    @patch(_PATCH_PLANNER_MODEL)
    @patch(_PATCH_STRUCTURED_MODEL)
    def test_structured_respects_max_adversarial(self, mock_get_structured, _mock_get_planner):
        plan = ScenarioPlan(
            scenarios=[
                PlannedScenario(name="s1", rationale="r1"),
                PlannedScenario(name="s2", rationale="r2"),
                PlannedScenario(name="s3", rationale="r3"),
            ],
            reasoning="Three scenarios",
        )
        mock_structured_model = MagicMock()
        mock_structured_model.invoke.return_value = {"parsed": plan}
        mock_get_structured.return_value = mock_structured_model

        settings = _make_settings()
        result = _structured_fallback("goal", [], 2, settings)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Helper: _to_scenario_config
# ---------------------------------------------------------------------------


class TestToScenarioConfig:
    def test_basic_conversion(self):
        planned = PlannedScenario(
            name="vol_stress",
            template_name="stress_test",
            regime_tags=["volatile"],
            config_override={"seed": 123},
            rationale="Test volatility",
        )
        config = _to_scenario_config(planned)
        assert isinstance(config, ScenarioConfig)
        assert config.name == "vol_stress"
        assert config.config_override["template"] == "stress_test"
        assert config.config_override["regime_tags"] == ["volatile"]
        assert config.config_override["seed"] == 123

    def test_empty_override(self):
        planned = PlannedScenario(name="basic", template_name="rmsc04")
        config = _to_scenario_config(planned)
        assert config.name == "basic"
        assert config.config_override["template"] == "rmsc04"

    def test_no_template_no_tags(self):
        planned = PlannedScenario(name="bare", template_name="", regime_tags=[])
        config = _to_scenario_config(planned)
        # Empty template and tags should not be added to override
        assert "template" not in config.config_override or config.config_override.get("template") == ""


# ---------------------------------------------------------------------------
# Public API: plan_scenarios
# ---------------------------------------------------------------------------


class TestPlanScenarios:
    def test_disabled_planner_returns_user_scenarios_only(self):
        settings = _make_settings(max_adversarial=0)

        user = _make_user_scenarios("baseline", "volatile")
        result, reasoning = plan_scenarios("market-making", user, settings)
        assert len(result) == 2
        assert result[0].name == "baseline"
        assert "disabled" in reasoning.lower()

    @patch("rohan.llm.planner._react_planner")
    def test_react_success_adds_adversarial(self, mock_react):
        plan = ScenarioPlan(
            scenarios=[
                PlannedScenario(name="adv_1", rationale="adversarial 1"),
                PlannedScenario(name="adv_2", rationale="adversarial 2"),
            ],
            reasoning="ReAct selected two adversarial scenarios",
        )
        mock_react.return_value = plan

        settings = _make_settings(max_adversarial=2)
        user = _make_user_scenarios("baseline")
        result, reasoning = plan_scenarios("market-making", user, settings)

        assert len(result) == 3  # 1 user + 2 adversarial
        assert result[0].name == "baseline"
        assert "ReAct selected" in reasoning

    @patch("rohan.llm.planner._react_planner")
    def test_react_deduplicates_user_scenarios(self, mock_react):
        plan = ScenarioPlan(
            scenarios=[
                PlannedScenario(name="baseline", rationale="same as user"),
                PlannedScenario(name="adv_new", rationale="new one"),
            ],
            reasoning="Planned",
        )
        mock_react.return_value = plan

        settings = _make_settings(max_adversarial=2)
        user = _make_user_scenarios("baseline")
        result, _ = plan_scenarios("goal", user, settings)

        # "baseline" from planner should be filtered out
        names = [s.name for s in result]
        assert names.count("baseline") == 1
        assert "adv_new" in names

    @patch("rohan.llm.planner._react_planner", side_effect=RuntimeError("LLM down"))
    @patch("rohan.llm.planner._structured_fallback")
    def test_react_fail_falls_to_structured(self, mock_structured, _mock_react):
        mock_structured.return_value = [
            PlannedScenario(name="struct_1", rationale="from structured"),
        ]

        settings = _make_settings(max_adversarial=1)
        user = _make_user_scenarios("baseline")
        result, reasoning = plan_scenarios("goal", user, settings)

        assert len(result) == 2  # 1 user + 1 adversarial
        mock_structured.assert_called_once()
        assert "structured" in reasoning.lower() or "fallback" in reasoning.lower()

    @patch("rohan.llm.planner._react_planner", side_effect=RuntimeError("LLM down"))
    @patch("rohan.llm.planner._structured_fallback", side_effect=RuntimeError("Also down"))
    def test_full_fallback_to_keywords(self, _mock_structured, _mock_react):
        settings = _make_settings(max_adversarial=2)
        user = _make_user_scenarios("baseline")
        result, reasoning = plan_scenarios("market-making strategy", user, settings)

        # Should have user + keyword-heuristic scenarios
        assert len(result) >= 2
        assert result[0].name == "baseline"
        assert "keyword" in reasoning.lower() or "heuristic" in reasoning.lower()

    @patch("rohan.llm.planner._react_planner", side_effect=RuntimeError("LLM down"))
    @patch("rohan.llm.planner._structured_fallback", side_effect=RuntimeError("Also down"))
    def test_full_fallback_empty_goal(self, _mock_structured, _mock_react):
        settings = _make_settings(max_adversarial=1)
        user = _make_user_scenarios("baseline")
        result, _ = plan_scenarios("", user, settings)

        # Even empty goal should get a default adversarial
        assert len(result) == 2

    def test_user_scenarios_never_removed(self):
        """User scenarios are always preserved, regardless of planner outcome."""
        settings = _make_settings(max_adversarial=0)
        user = _make_user_scenarios("a", "b", "c")
        result, _ = plan_scenarios("goal", user, settings)
        assert [s.name for s in result] == ["a", "b", "c"]

    @patch("rohan.llm.planner._react_planner")
    def test_react_respects_max_adversarial(self, mock_react):
        plan = ScenarioPlan(
            scenarios=[PlannedScenario(name=f"adv_{i}", rationale=f"r{i}") for i in range(5)],
            reasoning="Many scenarios",
        )
        mock_react.return_value = plan

        settings = _make_settings(max_adversarial=2)
        user = _make_user_scenarios("baseline")
        result, _ = plan_scenarios("goal", user, settings)

        # 1 user + at most 2 adversarial
        assert len(result) <= 3

    @patch("rohan.llm.planner._react_planner", side_effect=RuntimeError("down"))
    @patch("rohan.llm.planner._structured_fallback", side_effect=RuntimeError("down"))
    @patch(_PATCH_LLM_SETTINGS)
    def test_loads_default_settings_when_none(self, mock_settings_cls, _mock_struct, _mock_react):
        mock_settings = _make_settings(max_adversarial=1)
        mock_settings_cls.return_value = mock_settings

        result, _ = plan_scenarios("market-making", [], None)
        mock_settings_cls.assert_called_once()
        assert len(result) >= 1  # At least keyword-fallback result
