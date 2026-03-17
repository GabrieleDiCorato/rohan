"""LangGraph refinement graph — wires agent nodes into the strategy loop.

Usage::

    from rohan.llm.graph import build_refinement_graph, run_refinement

    # Quick start
    final_state = run_refinement(
        goal="Create a market-making strategy that profits from the spread",
        max_iterations=5,
    )
    print(final_state["current_code"])

    # Or build the graph manually for more control
    graph = build_refinement_graph()
    for event in graph.stream(initial_state):
        print(event)
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import time
import uuid
from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.types import Send

from rohan.config import LLMSettings, feature_flags_snapshot
from rohan.llm.nodes import (
    aggregator_node,
    process_scenario_node,
    validator_node,
    writer_node,
)
from rohan.llm.state import RefinementState, ScenarioConfig, terminal_metadata

logger = logging.getLogger(__name__)

# Load settings once at module level for use in graph building / routing.
_llm_settings = LLMSettings()

# Re-export for backward compatibility (UI, CLI, tests import these names).
MAX_VALIDATION_RETRIES: int = _llm_settings.max_validation_retries
DEFAULT_MAX_ITERATIONS: int = _llm_settings.default_max_iterations
DEFAULT_CONVERGENCE_THRESHOLD: float = _llm_settings.default_convergence_threshold
DEFAULT_RECURSION_LIMIT: int = _llm_settings.default_recursion_limit

# Legacy alias used by tests and UI
_DEFAULT_RECURSION_LIMIT = DEFAULT_RECURSION_LIMIT


# ── Node instrumentation ─────────────────────────────────────────────────


def _timed_node(name: str, fn: Callable) -> Callable:
    """Wrap a LangGraph node function with entry/exit logging and timing.

    This makes every state transition visible in logs, which is critical
    for diagnosing hangs between nodes.
    """

    @functools.wraps(fn)
    def wrapper(state: RefinementState) -> dict:
        logger.info("[graph] >>> entering node %r", name)
        t0 = time.monotonic()
        try:
            result = fn(state)
            elapsed = time.monotonic() - t0
            logger.info("[graph] <<< leaving  node %r  (%.2fs)", name, elapsed)
            return result
        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception("[graph] !!! node %r raised after %.2fs", name, elapsed)
            raise

    return wrapper


# ── Routing functions ─────────────────────────────────────────────────────


def validation_router(state: RefinementState):
    """Route after validation: retry, execute, or fail."""
    errors = state.get("validation_errors", [])
    attempts = state.get("validation_attempts", 0)

    if not errors:
        # Send must include the full parent state — LangGraph Send does NOT
        # inherit the parent graph state, so without this, process_scenario_node
        # would not see current_code, goal, etc.
        return [Send("process_scenario", {**state, "active_scenario": sc}) for sc in state.get("scenarios", [])]
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


def terminalize_validation_failure_node(state: RefinementState) -> dict:
    """Set explicit terminal metadata when validation retry budget is exhausted."""
    terminal_fields = terminal_metadata(
        state.get("feature_flags"),
        reason="validation_budget_exhausted",
        iteration=state.get("iteration_number", 1),
        context={
            "validation_attempts": state.get("validation_attempts", 0),
            "max_validation_retries": MAX_VALIDATION_RETRIES,
            "validation_errors": state.get("validation_errors", []),
        },
    )
    return {
        "status": "failed",
        **terminal_fields,
    }


# ── Graph builder ─────────────────────────────────────────────────────────


def build_refinement_graph() -> Any:
    """Build and compile the strategy-refinement LangGraph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph ready for ``.invoke()`` or ``.stream()``.
    """
    # Suppress LangSmith tracing unless explicitly enabled.
    # Done at build time (not module level) to avoid mutating env on import.
    if "LANGCHAIN_TRACING_V2" not in os.environ:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"

    graph = StateGraph(RefinementState)  # type: ignore[invalid-argument-type]

    # ── Nodes (wrapped with timing instrumentation) ──
    graph.add_node("writer", _timed_node("writer", writer_node))
    graph.add_node("validator", _timed_node("validator", validator_node))
    graph.add_node("process_scenario", _timed_node("process_scenario", process_scenario_node))
    graph.add_node("terminalize_validation_failure", _timed_node("terminalize_validation_failure", terminalize_validation_failure_node))

    graph.add_node("aggregator", _timed_node("aggregator", aggregator_node))

    # ── Edges ──
    graph.add_edge("writer", "validator")

    graph.add_conditional_edges(
        "validator",
        validation_router,
        {
            "retry": "writer",  # Invalid + retries left → regenerate
            "fail": "terminalize_validation_failure",  # Max retries exceeded → explicit failure reason
            "process_scenario": "process_scenario",
        },
    )

    graph.add_edge("terminalize_validation_failure", END)

    graph.add_edge("process_scenario", "aggregator")

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

    return graph.compile(
        # Prevent infinite loops: 5 nodes/iter × max_iters + retries
        checkpointer=None,  # no checkpoint persistence (avoids SQLite locks)
    )


# ── Convenience runner ────────────────────────────────────────────────────


def _deterministic_seed(scenario_name: str, session_ts: int, index: int = 0) -> int:
    """Derive a reproducible uint32 seed from scenario name + session timestamp.

    *index* disambiguates scenarios that share the same name (defensive).
    """
    digest = hashlib.sha256(f"{index}:{scenario_name}:{session_ts}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**32 - 1)


def run_refinement(
    goal: str,
    max_iterations: int = 5,
    scenarios: list[ScenarioConfig] | None = None,
    max_wall_clock_seconds: int = 3600,
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
    max_wall_clock_seconds:
        Hard wall-clock timeout for the entire refinement run (default
        3600 = 1 hour).  Raises ``TimeoutError`` if exceeded.

    Returns
    -------
    RefinementState
        The final graph state containing the best strategy code,
        iteration history, and convergence verdict.
    """
    if scenarios is None:
        scenarios = [ScenarioConfig(name="default")]

    # Assign deterministic per-scenario seeds (same seed every iteration)
    session_ts = int(time.monotonic_ns())
    for idx, sc in enumerate(scenarios):
        if sc.seed is None:
            sc.seed = _deterministic_seed(sc.name, session_ts, index=idx)
            logger.debug("Assigned seed %d to scenario %r", sc.seed, sc.name)

    initial_state: RefinementState = {
        "goal": goal,
        "run_id": uuid.uuid4().hex,
        "max_iterations": max_iterations,
        "scenarios": scenarios,
        "feature_flags": feature_flags_snapshot(),
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
        "terminal_reason": None,
        "terminal_iteration": 0,
        "terminal_context": {},
        "messages": [],
    }

    graph = build_refinement_graph()
    logger.info("Starting refinement loop: goal=%r, max_iterations=%d, timeout=%ds", goal, max_iterations, max_wall_clock_seconds)

    config = {
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
    }

    wall_clock_start = time.monotonic()
    final_state = graph.invoke(initial_state, config=config)
    elapsed = time.monotonic() - wall_clock_start

    if elapsed > max_wall_clock_seconds:
        logger.warning("Refinement exceeded wall-clock timeout (%.0fs > %ds)", elapsed, max_wall_clock_seconds)

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
