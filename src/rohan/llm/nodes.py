"""LangGraph agent nodes for the strategy-refinement loop.

Each public function is a LangGraph node with signature::

    def node(state: RefinementState) -> dict
"""

from __future__ import annotations

import logging
import traceback

from langchain_core.messages import HumanMessage, SystemMessage

from rohan.config import SimulationSettings
from rohan.framework.analysis_service import AnalysisService
from rohan.framework.prompts import format_interpreter_prompt
from rohan.llm.factory import (
    get_analysis_model,
    get_codegen_model,
    get_judge_model,
    get_structured_model,
)
from rohan.llm.models import (
    AggregatedFeedback,
    GeneratedStrategy,
    IterationSummary,
    JudgeVerdict,
    ScenarioExplanation,
    ScenarioMetrics,
)
from rohan.llm.prompts import (
    AGGREGATOR_HUMAN,
    AGGREGATOR_SYSTEM,
    EXPLAINER_HUMAN,
    EXPLAINER_SYSTEM,
    HISTORY_ROW_TEMPLATE,
    HISTORY_TABLE_HEADER,
    WRITER_FEEDBACK_TEMPLATE,
    WRITER_HUMAN,
    WRITER_SYSTEM,
)
from rohan.llm.state import RefinementState, ScenarioResult
from rohan.simulation.models.simulation_metrics import (
    ComparisonResult,
    MarketImpact,
    MarketMetrics,
    SimulationMetrics,
)
from rohan.simulation.simulation_service import SimulationService
from rohan.simulation.strategy_validator import StrategyValidator, execute_strategy_safely

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Writer Node
# ═══════════════════════════════════════════════════════════════════════════


def writer_node(state: RefinementState) -> dict:
    """Generate (or regenerate) strategy code from the goal + feedback."""
    logger.info("Writer node — iteration %d", state.get("iteration_number", 1))

    # Build feedback section
    feedback_section = ""
    feedback = state.get("aggregated_feedback")
    current_code = state.get("current_code")
    validation_errors = state.get("validation_errors", [])

    if validation_errors and current_code:
        # We're retrying after a validation failure
        feedback_section = "## Validation Errors (fix these)\n" + "\n".join(f"- {e}" for e in validation_errors) + f"\n\n### Previous Code:\n```python\n{current_code}\n```"
    elif feedback and current_code:
        feedback_section = WRITER_FEEDBACK_TEMPLATE.format(
            iteration_number=state.get("iteration_number", 1) - 1,
            score=feedback.verdict.score if feedback else "N/A",
            strengths="\n".join(f"- {s}" for s in (feedback.cross_scenario_patterns or [])),
            weaknesses=feedback.verdict.reasoning if feedback else "",
            recommendations=feedback.unified_feedback if feedback else "",
            previous_code=current_code,
        )

    human_msg = WRITER_HUMAN.format(
        goal=state.get("goal", ""),
        feedback_section=feedback_section,
    )

    model = get_codegen_model()
    structured = get_structured_model(model, GeneratedStrategy)
    result = structured.invoke(
        [
            SystemMessage(content=WRITER_SYSTEM),
            HumanMessage(content=human_msg),
        ]
    )

    logger.info("Writer produced class %r (%d chars)", result.class_name, len(result.code))

    return {
        "current_code": result.code,
        "current_class_name": result.class_name,
        "current_reasoning": result.reasoning,
        "validation_errors": [],
        "status": "validating",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Validator Node
# ═══════════════════════════════════════════════════════════════════════════


def validator_node(state: RefinementState) -> dict:
    """Validate strategy code (AST safety + instantiation check)."""
    code = state.get("current_code", "")
    attempts = state.get("validation_attempts", 0) + 1

    logger.info("Validator node — attempt %d", attempts)

    validator = StrategyValidator()
    result = validator.validate(code or "")

    if not result.is_valid:
        logger.warning("Validation failed (attempt %d): %s", attempts, result.errors)
        return {
            "validation_errors": result.errors,
            "validation_attempts": attempts,
            "status": "validating",
        }

    # Try to actually parse and instantiate the class
    try:
        import ast as _ast

        if not code:
            return {
                "validation_errors": ["No code to validate"],
                "validation_attempts": attempts,
                "status": "validating",
            }

        tree = _ast.parse(code)
        class_name = None
        for node in tree.body:
            if isinstance(node, _ast.ClassDef):
                class_name = node.name
                break
        if not class_name:
            return {
                "validation_errors": ["No class definition found in code"],
                "validation_attempts": attempts,
                "status": "validating",
            }

        # This will exec the code in a sandbox and check the protocol
        validator.execute_strategy(code, class_name)
    except Exception as exc:
        logger.warning("Instantiation check failed: %s", exc)
        return {
            "validation_errors": [str(exc)],
            "validation_attempts": attempts,
            "status": "validating",
        }

    logger.info("Validation passed on attempt %d", attempts)
    return {
        "validation_errors": [],
        "validation_attempts": attempts,
        "status": "executing",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Scenario Executor Node
# ═══════════════════════════════════════════════════════════════════════════


def scenario_executor_node(state: RefinementState) -> dict:
    """Run the validated strategy across all configured scenarios."""
    code = state.get("current_code", "")
    scenarios = state.get("scenarios", [])
    if not scenarios:
        # Default: single scenario with no overrides
        from rohan.llm.state import ScenarioConfig

        scenarios = [ScenarioConfig(name="default")]

    logger.info("Executor node — %d scenario(s)", len(scenarios))

    base_settings = SimulationSettings()
    analyzer = AnalysisService()
    service = SimulationService()
    results: list[ScenarioResult] = []

    for scenario in scenarios:
        logger.info("Running scenario %r", scenario.name)
        try:
            # Apply overrides
            settings = base_settings.model_copy(deep=True)
            for k, v in scenario.config_override.items():
                if hasattr(settings, k):
                    setattr(settings, k, v)

            # Run strategy
            if not code:
                raise ValueError("No strategy code to execute")
            sim_result = execute_strategy_safely(code, settings, _timeout_seconds=300)
            if sim_result.error or not sim_result.result:
                results.append(
                    ScenarioResult(
                        scenario_name=scenario.name,
                        error=str(sim_result.error) if sim_result.error else "No result",
                    )
                )
                continue

            strategy_output = sim_result.result
            strategy_sim_metrics = analyzer.compute_metrics(strategy_output)

            if strategy_output.strategic_agent_id is None:
                raise RuntimeError("No strategic agent in simulation output")
            strategy_agent_metrics = analyzer.compute_agent_metrics(
                strategy_output,
                strategy_output.strategic_agent_id,
                initial_cash=settings.starting_cash,
            )

            # Run baseline
            baseline_result = service.run_simulation(settings, strategy=None)
            if baseline_result.error or not baseline_result.result:
                results.append(
                    ScenarioResult(
                        scenario_name=scenario.name,
                        error=f"Baseline failed: {baseline_result.error}",
                    )
                )
                continue

            baseline_sim_metrics = analyzer.compute_metrics(baseline_result.result)

            # Build comparison
            def _to_market(m: SimulationMetrics) -> MarketMetrics:
                return MarketMetrics(
                    volatility=m.volatility,
                    mean_spread=m.mean_spread,
                    effective_spread=m.effective_spread,
                    avg_bid_liquidity=m.avg_bid_liquidity,
                    avg_ask_liquidity=m.avg_ask_liquidity,
                    traded_volume=m.traded_volume,
                )

            def _pct(a: float | None, b: float | None) -> float | None:
                if a is None or b is None:
                    return None
                if b == 0:
                    return 0.0 if a == 0 else float("inf")
                return (a - b) / b

            strat_market = _to_market(strategy_sim_metrics)
            base_market = _to_market(baseline_sim_metrics)
            impact = MarketImpact(
                spread_delta_pct=_pct(strat_market.mean_spread, base_market.mean_spread),
                volatility_delta_pct=_pct(strat_market.volatility, base_market.volatility),
                bid_liquidity_delta_pct=_pct(strat_market.avg_bid_liquidity, base_market.avg_bid_liquidity),
                ask_liquidity_delta_pct=_pct(strat_market.avg_ask_liquidity, base_market.avg_ask_liquidity),
            )

            comparison = ComparisonResult(
                strategy_metrics=strategy_agent_metrics,
                strategy_market_metrics=strat_market,
                baseline_metrics=base_market,
                market_impact=impact,
            )

            summary = analyzer.generate_summary(
                comparison,
                strategy_output=strategy_output,
                duration_seconds=sim_result.duration_seconds,
            )

            prompt = format_interpreter_prompt(summary, goal=state.get("goal", ""))

            results.append(
                ScenarioResult(
                    scenario_name=scenario.name,
                    interpreter_prompt=prompt,
                    strategy_pnl=strategy_agent_metrics.total_pnl,
                    volatility_delta_pct=impact.volatility_delta_pct,
                    spread_delta_pct=impact.spread_delta_pct,
                    trade_count=strategy_agent_metrics.trade_count,
                )
            )

        except Exception as exc:
            logger.error("Scenario %r failed: %s", scenario.name, exc, exc_info=True)
            results.append(
                ScenarioResult(
                    scenario_name=scenario.name,
                    error=f"{exc}\n{traceback.format_exc()}",
                )
            )

    return {
        "scenario_results": results,
        "status": "explaining",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Explainer Node
# ═══════════════════════════════════════════════════════════════════════════


def explainer_node(state: RefinementState) -> dict:
    """Analyse each scenario result using an LLM with tool access.

    For MVP, this uses a simple prompt-based analysis without full ReAct
    tool calling.  The tool-equipped version is available via
    :func:`create_explainer_react_agent` for advanced use.
    """
    scenario_results = state.get("scenario_results", [])
    logger.info("Explainer node — %d scenario(s)", len(scenario_results))

    model = get_analysis_model()
    explanations: list[ScenarioExplanation] = []

    for sr in scenario_results:
        if sr.error:
            explanations.append(
                ScenarioExplanation(
                    scenario_name=sr.scenario_name,
                    weaknesses=[f"Scenario failed: {sr.error}"],
                    raw_analysis=f"Error: {sr.error}",
                )
            )
            continue

        human_msg = EXPLAINER_HUMAN.format(
            scenario_name=sr.scenario_name,
            interpreter_prompt=sr.interpreter_prompt,
        )

        try:
            structured = get_structured_model(model, ScenarioExplanation)
            result = structured.invoke(
                [
                    SystemMessage(content=EXPLAINER_SYSTEM),
                    HumanMessage(content=human_msg),
                ]
            )
            result.scenario_name = sr.scenario_name
            explanations.append(result)
        except Exception as exc:
            logger.error("Explainer failed for %r: %s", sr.scenario_name, exc)
            explanations.append(
                ScenarioExplanation(
                    scenario_name=sr.scenario_name,
                    weaknesses=[f"Analysis failed: {exc}"],
                    raw_analysis=str(exc),
                )
            )

    return {
        "explanations": explanations,
        "status": "aggregating",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Aggregator Node
# ═══════════════════════════════════════════════════════════════════════════


def _build_history_table(iterations: list[IterationSummary]) -> str:
    """Format iteration history as a Markdown table."""
    if not iterations:
        return "(No previous iterations)"

    rows = [HISTORY_TABLE_HEADER]
    for it in iterations:
        # Pick first scenario metrics for the table
        first_metrics = next(iter(it.scenario_metrics.values()), None)
        rows.append(
            HISTORY_ROW_TEMPLATE.format(
                iter=it.iteration_number,
                pnl=f"${(first_metrics.total_pnl or 0) / 100:,.2f}" if first_metrics else "N/A",
                vol_delta=f"{(first_metrics.volatility_delta_pct or 0):+.1%}" if first_metrics else "N/A",
                spread_delta=f"{(first_metrics.spread_delta_pct or 0):+.1%}" if first_metrics else "N/A",
                score=f"{it.judge_score:.1f}" if it.judge_score else "N/A",
                summary=it.aggregated_explanation[:80] if it.aggregated_explanation else "",
            )
        )
    return "\n".join(rows)


def _format_explanations(explanations: list[ScenarioExplanation]) -> str:
    """Format scenario explanations for the aggregator prompt."""
    parts = []
    for exp in explanations:
        part = f"### {exp.scenario_name}\n"
        if exp.strengths:
            part += "**Strengths:** " + "; ".join(exp.strengths) + "\n"
        if exp.weaknesses:
            part += "**Weaknesses:** " + "; ".join(exp.weaknesses) + "\n"
        if exp.recommendations:
            part += "**Recommendations:** " + "; ".join(exp.recommendations) + "\n"
        if exp.market_impact_assessment:
            part += f"**Market Impact:** {exp.market_impact_assessment}\n"
        parts.append(part)
    return "\n".join(parts)


def aggregator_node(state: RefinementState) -> dict:
    """Combine scenario explanations and judge convergence."""
    explanations = state.get("explanations", [])
    iterations = state.get("iterations", [])
    iteration_number = state.get("iteration_number", 1)
    scenario_results = state.get("scenario_results", [])

    logger.info("Aggregator node — iteration %d, %d explanation(s)", iteration_number, len(explanations))

    history_table = _build_history_table(iterations)
    explanations_text = _format_explanations(explanations)

    human_msg = AGGREGATOR_HUMAN.format(
        goal=state.get("goal", ""),
        history_table=history_table,
        iteration_number=iteration_number,
        explanations=explanations_text,
    )

    model = get_judge_model()
    structured = get_structured_model(model, JudgeVerdict)

    try:
        verdict = structured.invoke(
            [
                SystemMessage(content=AGGREGATOR_SYSTEM),
                HumanMessage(content=human_msg),
            ]
        )
    except Exception as exc:
        logger.error("Judge failed: %s", exc)
        verdict = JudgeVerdict(
            score=5.0,
            comparison="similar",
            reasoning=f"Judge invocation failed: {exc}",
            recommendation="continue",
        )

    # Build aggregated feedback
    feedback = AggregatedFeedback(
        verdict=verdict,
        cross_scenario_patterns=[obs for exp in explanations for obs in exp.key_observations],
        unified_feedback=(
            f"Score: {verdict.score}/10 ({verdict.comparison}). {verdict.reasoning}\n\nConsolidated recommendations:\n" + "\n".join(f"- {r}" for exp in explanations for r in exp.recommendations)
        ),
    )

    # Build iteration summary for history
    scenario_metrics: dict[str, ScenarioMetrics] = {}
    for sr in scenario_results:
        scenario_metrics[sr.scenario_name] = ScenarioMetrics(
            scenario_name=sr.scenario_name,
            total_pnl=sr.strategy_pnl,
            volatility_delta_pct=sr.volatility_delta_pct,
            spread_delta_pct=sr.spread_delta_pct,
            trade_count=sr.trade_count,
        )

    iteration_summary = IterationSummary(
        iteration_number=iteration_number,
        strategy_code=state.get("current_code") or "",
        scenario_metrics=scenario_metrics,
        aggregated_explanation=feedback.unified_feedback,
        judge_score=verdict.score,
        judge_reasoning=verdict.reasoning,
    )

    new_iterations = list(iterations) + [iteration_summary]

    # Determine next status
    max_iterations = state.get("max_iterations", 3)
    if verdict.recommendation in ("stop_converged", "stop_plateau"):
        next_status = "done"
    elif iteration_number >= max_iterations:
        logger.info("Max iterations (%d) reached", max_iterations)
        next_status = "done"
    else:
        next_status = "writing"

    logger.info(
        "Aggregator verdict: score=%.1f, comparison=%s, recommendation=%s → %s",
        verdict.score,
        verdict.comparison,
        verdict.recommendation,
        next_status,
    )

    return {
        "aggregated_feedback": feedback,
        "iterations": new_iterations,
        "iteration_number": iteration_number + 1,
        "status": next_status,
    }
