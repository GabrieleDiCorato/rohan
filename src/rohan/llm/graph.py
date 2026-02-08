"""LangGraph refinement graph — wires agent nodes into the strategy loop.

Usage::

    from rohan.llm.graph import build_refinement_graph, run_refinement

    # Quick start
    final_state = run_refinement(
        goal="Create a market-making strategy that profits from the spread",
        max_iterations=3,
    )
    print(final_state["current_code"])

    # Or build the graph manually for more control
    graph = build_refinement_graph()
    for event in graph.stream(initial_state):
        print(event)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from rohan.llm.nodes import (
    aggregator_node,
    explainer_node,
    scenario_executor_node,
    validator_node,
    writer_node,
)
from rohan.llm.state import RefinementState, ScenarioConfig

logger = logging.getLogger(__name__)

# Maximum validation retries before giving up
MAX_VALIDATION_RETRIES = 3


# ── Routing functions ─────────────────────────────────────────────────────


def validation_router(state: RefinementState) -> Literal["retry", "execute", "fail"]:
    """Route after validation: retry, execute, or fail."""
    errors = state.get("validation_errors", [])
    attempts = state.get("validation_attempts", 0)

    if not errors:
        return "execute"
    if attempts >= MAX_VALIDATION_RETRIES:
        logger.warning("Max validation retries (%d) exceeded", MAX_VALIDATION_RETRIES)
        return "fail"
    return "retry"


def should_continue(state: RefinementState) -> Literal["continue", "done"]:
    """Route after aggregation: continue refining or finish."""
    status = state.get("status", "done")
    if status == "writing":
        return "continue"
    return "done"


# ── Graph builder ─────────────────────────────────────────────────────────


def build_refinement_graph() -> Any:
    """Build and compile the strategy-refinement LangGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``.invoke()`` or ``.stream()``.
    """
    graph = StateGraph(RefinementState)

    # ── Nodes ──
    graph.add_node("writer", writer_node)
    graph.add_node("validator", validator_node)
    graph.add_node("executor", scenario_executor_node)
    graph.add_node("explainer", explainer_node)
    graph.add_node("aggregator", aggregator_node)

    # ── Edges ──
    graph.add_edge("writer", "validator")

    graph.add_conditional_edges(
        "validator",
        validation_router,
        {
            "retry": "writer",  # Invalid + retries left → regenerate
            "execute": "executor",  # Valid → run simulation
            "fail": END,  # Max retries exceeded → abort
        },
    )

    graph.add_edge("executor", "explainer")
    graph.add_edge("explainer", "aggregator")

    graph.add_conditional_edges(
        "aggregator",
        should_continue,
        {
            "continue": "writer",  # More iterations needed
            "done": END,  # Converged or max iterations
        },
    )

    # ── Entry point ──
    graph.set_entry_point("writer")

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────


def run_refinement(
    goal: str,
    max_iterations: int = 3,
    scenarios: list[ScenarioConfig] | None = None,
) -> RefinementState:
    """Run the full refinement loop and return the final state.

    Parameters
    ----------
    goal:
        Natural language description of the desired trading strategy.
    max_iterations:
        Maximum number of write→validate→execute→explain→judge cycles.
    scenarios:
        List of scenario configurations.  Defaults to a single default
        scenario.

    Returns
    -------
    RefinementState
        The final graph state containing the best strategy code,
        iteration history, and convergence verdict.
    """
    if scenarios is None:
        scenarios = [ScenarioConfig(name="default")]

    initial_state: RefinementState = {
        "goal": goal,
        "max_iterations": max_iterations,
        "scenarios": scenarios,
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

    graph = build_refinement_graph()
    logger.info("Starting refinement loop: goal=%r, max_iterations=%d", goal, max_iterations)

    final_state = graph.invoke(initial_state)

    # Log summary
    iterations = final_state.get("iterations", [])
    if iterations:
        last = iterations[-1]
        logger.info(
            "Refinement complete: %d iterations, final score=%.1f, recommendation=%s",
            len(iterations),
            last.judge_score or 0,
            final_state.get("aggregated_feedback", {}).verdict.recommendation if final_state.get("aggregated_feedback") else "N/A",
        )
    else:
        logger.warning("Refinement ended with no completed iterations (validation failures?)")

    from typing import cast

    return cast(RefinementState, final_state)
