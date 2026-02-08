"""Tests for the LangGraph refinement graph (Phase 2.3).

Tests routing logic and graph compilation. Full end-to-end tests
are mocked since they require LLM API keys.
"""

from rohan.llm.graph import (
    MAX_VALIDATION_RETRIES,
    build_refinement_graph,
    should_continue,
    validation_router,
)
from rohan.llm.state import RefinementState


def _state(**overrides) -> RefinementState:
    """Minimal state for router tests."""
    base: RefinementState = {
        "goal": "test",
        "max_iterations": 3,
        "scenarios": [],
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
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


# ═══════════════════════════════════════════════════════════════════════════
# Routing Functions
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationRouter:
    def test_no_errors_executes(self):
        assert validation_router(_state(validation_errors=[])) == "execute"

    def test_errors_with_retries_left(self):
        result = validation_router(_state(validation_errors=["err"], validation_attempts=1))
        assert result == "retry"

    def test_errors_at_max_retries_fails(self):
        result = validation_router(
            _state(
                validation_errors=["err"],
                validation_attempts=MAX_VALIDATION_RETRIES,
            )
        )
        assert result == "fail"

    def test_errors_beyond_max_retries_fails(self):
        result = validation_router(
            _state(
                validation_errors=["err"],
                validation_attempts=MAX_VALIDATION_RETRIES + 1,
            )
        )
        assert result == "fail"


class TestShouldContinue:
    def test_writing_continues(self):
        assert should_continue(_state(status="writing")) == "continue"

    def test_done_stops(self):
        assert should_continue(_state(status="done")) == "done"

    def test_failed_stops(self):
        assert should_continue(_state(status="failed")) == "done"

    def test_default_stops(self):
        # Missing status should default to "done"
        state = _state()
        del state["status"]
        assert should_continue(state) == "done"


# ═══════════════════════════════════════════════════════════════════════════
# Graph Compilation
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildGraph:
    def test_graph_compiles(self):
        """Graph should compile without errors."""
        graph = build_refinement_graph()
        assert graph is not None

    def test_graph_has_expected_nodes(self):
        graph = build_refinement_graph()
        # LangGraph compiled graph stores nodes in .nodes
        node_names = set(graph.nodes.keys())
        assert "writer" in node_names
        assert "validator" in node_names
        assert "executor" in node_names
        assert "explainer" in node_names
        assert "aggregator" in node_names
