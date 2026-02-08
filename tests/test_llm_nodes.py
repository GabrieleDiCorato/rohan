"""Tests for LangGraph agent nodes (Phase 2.2).

All LLM calls are mocked — these tests verify node logic, state
transitions, and error handling without requiring API keys.
"""

from unittest.mock import MagicMock, patch

from rohan.llm.models import (
    AggregatedFeedback,
    GeneratedStrategy,
    IterationSummary,
    JudgeVerdict,
    ScenarioExplanation,
    ScenarioMetrics,
)
from rohan.llm.nodes import (
    _build_history_table,
    _format_explanations,
    aggregator_node,
    explainer_node,
    validator_node,
    writer_node,
)
from rohan.llm.state import RefinementState, ScenarioConfig, ScenarioResult

# -- Valid strategy code for tests (mirrors test_iteration_pipeline.py) -----
VALID_STRATEGY = """\
from rohan.simulation.models.strategy_api import (
    AgentConfig, MarketState, OrderAction, OrderType, Side, Order,
)

class TestStrategy:
    def initialize(self, config: AgentConfig) -> None:
        self.config = config

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_order_update(self, update: Order) -> list[OrderAction]:
        return []
"""

INVALID_STRATEGY = """\
import os

class BadStrategy:
    def hack(self):
        os.system("rm -rf /")
"""

INCOMPLETE_STRATEGY = """\
class MissingMethods:
    def initialize(self, config):
        pass
"""


def _base_state(**overrides) -> RefinementState:
    """Create a minimal valid state dict."""
    state: RefinementState = {
        "goal": "Create a simple strategy",
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
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


# ═══════════════════════════════════════════════════════════════════════════
# Writer Node
# ═══════════════════════════════════════════════════════════════════════════


class TestWriterNode:
    @patch("rohan.llm.nodes.get_codegen_model")
    def test_produces_code(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GeneratedStrategy(
            class_name="MyStrategy",
            code=VALID_STRATEGY,
            reasoning="Simple passive strategy",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state()
        result = writer_node(state)

        assert result["current_code"] == VALID_STRATEGY
        assert result["current_class_name"] == "MyStrategy"
        assert result["status"] == "validating"
        assert result["validation_errors"] == []

    @patch("rohan.llm.nodes.get_codegen_model")
    def test_includes_validation_errors_as_feedback(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GeneratedStrategy(
            class_name="Fixed",
            code="class Fixed: pass",
            reasoning="Fixed the errors",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state(
            current_code="class Broken: pass",
            validation_errors=["Import forbidden: 'os'"],
        )
        result = writer_node(state)
        assert result["current_code"] == "class Fixed: pass"

    @patch("rohan.llm.nodes.get_codegen_model")
    def test_includes_aggregated_feedback(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = GeneratedStrategy(
            class_name="V2",
            code="class V2: pass",
            reasoning="Improved version",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        verdict = JudgeVerdict(
            score=5.0,
            comparison="similar",
            reasoning="Needs work",
            recommendation="continue",
        )
        feedback = AggregatedFeedback(
            verdict=verdict,
            cross_scenario_patterns=["Pattern A"],
            unified_feedback="Improve PnL",
        )
        state = _base_state(
            current_code="class V1: pass",
            aggregated_feedback=feedback,
            iteration_number=2,
        )
        result = writer_node(state)
        assert result["current_code"] == "class V2: pass"


# ═══════════════════════════════════════════════════════════════════════════
# Validator Node
# ═══════════════════════════════════════════════════════════════════════════


class TestValidatorNode:
    def test_valid_code_passes(self):
        state = _base_state(current_code=VALID_STRATEGY)
        result = validator_node(state)
        assert result["validation_errors"] == []
        assert result["status"] == "executing"
        assert result["validation_attempts"] == 1

    def test_forbidden_import_fails(self):
        state = _base_state(current_code=INVALID_STRATEGY)
        result = validator_node(state)
        assert len(result["validation_errors"]) > 0
        assert result["status"] == "validating"

    def test_missing_methods_fails(self):
        state = _base_state(current_code=INCOMPLETE_STRATEGY)
        result = validator_node(state)
        assert len(result["validation_errors"]) > 0
        assert "missing required method" in result["validation_errors"][0]

    def test_no_class_fails(self):
        state = _base_state(current_code="x = 1")
        result = validator_node(state)
        assert "No class definition" in result["validation_errors"][0]

    def test_empty_code_fails(self):
        state = _base_state(current_code="")
        result = validator_node(state)
        assert "No class definition" in result["validation_errors"][0]

    def test_attempt_counter_increments(self):
        state = _base_state(current_code=INVALID_STRATEGY, validation_attempts=2)
        result = validator_node(state)
        assert result["validation_attempts"] == 3


# ═══════════════════════════════════════════════════════════════════════════
# Explainer Node
# ═══════════════════════════════════════════════════════════════════════════


class TestExplainerNode:
    @patch("rohan.llm.nodes.get_analysis_model")
    def test_error_scenario_produces_explanation(self, mock_get_model):
        # Even error scenarios need the model imported (it's at top of function)
        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        state = _base_state(
            scenario_results=[
                ScenarioResult(scenario_name="broken", error="Simulation crashed"),
            ],
        )
        result = explainer_node(state)
        assert len(result["explanations"]) == 1
        assert "Scenario failed" in result["explanations"][0].weaknesses[0]
        assert result["status"] == "aggregating"

    @patch("rohan.llm.nodes.get_analysis_model")
    def test_successful_scenario(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = ScenarioExplanation(
            scenario_name="default",
            strengths=["Good PnL"],
            weaknesses=["High impact"],
            recommendations=["Reduce size"],
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state(
            scenario_results=[
                ScenarioResult(
                    scenario_name="default",
                    interpreter_prompt="Some prompt",
                    strategy_pnl=100.0,
                ),
            ],
        )
        result = explainer_node(state)
        assert len(result["explanations"]) == 1
        assert result["explanations"][0].strengths == ["Good PnL"]

    @patch("rohan.llm.nodes.get_analysis_model")
    def test_llm_failure_produces_fallback(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("API down")
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state(
            scenario_results=[
                ScenarioResult(scenario_name="default", interpreter_prompt="P"),
            ],
        )
        result = explainer_node(state)
        assert len(result["explanations"]) == 1
        assert "Analysis failed" in result["explanations"][0].weaknesses[0]


# ═══════════════════════════════════════════════════════════════════════════
# Aggregator Node
# ═══════════════════════════════════════════════════════════════════════════


class TestAggregatorNode:
    @patch("rohan.llm.nodes.get_judge_model")
    def test_produces_feedback_and_history(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = JudgeVerdict(
            score=6.0,
            comparison="better",
            reasoning="Some improvement",
            recommendation="continue",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state(
            current_code="class X: pass",
            iteration_number=1,
            explanations=[
                ScenarioExplanation(
                    scenario_name="default",
                    strengths=["OK"],
                    key_observations=["Pattern 1"],
                    recommendations=["Do better"],
                ),
            ],
            scenario_results=[
                ScenarioResult(scenario_name="default", strategy_pnl=50.0),
            ],
        )
        result = aggregator_node(state)

        assert result["aggregated_feedback"] is not None
        assert result["aggregated_feedback"].verdict.score == 6.0
        assert len(result["iterations"]) == 1
        assert result["iterations"][0].iteration_number == 1
        assert result["iteration_number"] == 2
        assert result["status"] == "writing"  # continue

    @patch("rohan.llm.nodes.get_judge_model")
    def test_stop_converged(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = JudgeVerdict(
            score=9.0,
            comparison="similar",
            reasoning="Converged",
            recommendation="stop_converged",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state(
            current_code="class X: pass",
            iteration_number=2,
            explanations=[ScenarioExplanation()],
            scenario_results=[ScenarioResult()],
        )
        result = aggregator_node(state)
        assert result["status"] == "done"

    @patch("rohan.llm.nodes.get_judge_model")
    def test_max_iterations_reached(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = JudgeVerdict(
            score=5.0,
            comparison="better",
            reasoning="Still improving",
            recommendation="continue",
        )
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state(
            current_code="class X: pass",
            iteration_number=3,
            max_iterations=3,
            explanations=[ScenarioExplanation()],
            scenario_results=[ScenarioResult()],
        )
        result = aggregator_node(state)
        assert result["status"] == "done"  # max reached

    @patch("rohan.llm.nodes.get_judge_model")
    def test_llm_failure_defaults_to_continue(self, mock_get_model):
        mock_model = MagicMock()
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = RuntimeError("API error")
        mock_model.with_structured_output.return_value = mock_structured
        mock_get_model.return_value = mock_model

        state = _base_state(
            current_code="class X: pass",
            iteration_number=1,
            explanations=[ScenarioExplanation()],
            scenario_results=[ScenarioResult()],
        )
        result = aggregator_node(state)
        assert result["aggregated_feedback"].verdict.score == 5.0
        assert result["status"] == "writing"  # fallback continues


# ═══════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildHistoryTable:
    def test_empty_history(self):
        result = _build_history_table([])
        assert "No previous iterations" in result

    def test_single_iteration(self):
        sm = ScenarioMetrics(scenario_name="default", total_pnl=500.0)
        summary = IterationSummary(
            iteration_number=1,
            strategy_code="class A: pass",
            scenario_metrics={"default": sm},
            judge_score=7.0,
            aggregated_explanation="Good",
        )
        result = _build_history_table([summary])
        assert "| 1 |" in result
        assert "7.0" in result


class TestFormatExplanations:
    def test_empty(self):
        result = _format_explanations([])
        assert result == ""

    def test_single_explanation(self):
        exp = ScenarioExplanation(
            scenario_name="test",
            strengths=["Good"],
            weaknesses=["Bad"],
            recommendations=["Fix"],
            market_impact_assessment="Neutral",
        )
        result = _format_explanations([exp])
        assert "### test" in result
        assert "Good" in result
        assert "Bad" in result
        assert "Fix" in result
        assert "Neutral" in result
