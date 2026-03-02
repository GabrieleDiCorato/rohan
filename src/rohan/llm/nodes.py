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
    WRITER_ROLLBACK_SECTION,
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
    iterations = state.get("iterations", [])
    rolled_back_from: dict | None = state.get("rolled_back_from")

    if validation_errors and current_code:
        # Retrying after a validation / smoke-test failure
        feedback_section = "## Validation Errors (fix these)\n" + "\n".join(f"- {e}" for e in validation_errors) + f"\n\n### Previous Code:\n```python\n{current_code}\n```"
    elif feedback and current_code:
        # Build full history table (excluding the last entry which IS current_code)
        history_text = _build_history_table(iterations[:-1]) if len(iterations) > 1 else "(No previous iterations)"

        # Build concise metrics summary from the most recent scenario results
        scenario_results: list[ScenarioResult] = state.get("scenario_results", [])
        metrics_lines: list[str] = []
        for sr in scenario_results:
            if sr.error:
                metrics_lines.append(f"- **{sr.scenario_name}:** ERROR")
            else:
                pnl_str = f"${(sr.strategy_pnl or 0) / 100:,.2f}"
                fill_str = f"{(sr.fill_rate or 0):.1%}" if sr.fill_rate is not None else "N/A"
                ott_str = f"{sr.order_to_trade_ratio:.1f}" if sr.order_to_trade_ratio is not None else "N/A"
                sharpe_str = f"{sr.sharpe_ratio:.2f}" if sr.sharpe_ratio is not None else "N/A"
                drawdown_str = f"${(sr.max_drawdown or 0) / 100:,.2f}" if sr.max_drawdown is not None else "N/A"
                inv_std_str = f"{sr.inventory_std:.1f}" if sr.inventory_std is not None else "N/A"
                vol_str = f"{(sr.volatility_delta_pct or 0):+.1%}"
                metrics_lines.append(
                    f"- **{sr.scenario_name}:** PnL={pnl_str}, Trades={sr.trade_count}, "
                    f"Fill Rate={fill_str}, OTT={ott_str}, Sharpe={sharpe_str}, "
                    f"Max Drawdown={drawdown_str}, End Inventory={sr.end_inventory}, "
                    f"Inventory Std={inv_std_str}, Vol Δ={vol_str}"
                )
        metrics_summary = "\n".join(metrics_lines) if metrics_lines else "(no simulation results available)"

        feedback_section = WRITER_FEEDBACK_TEMPLATE.format(
            iteration_number=state.get("iteration_number", 1) - 1,
            score=feedback.verdict.score if feedback else "N/A",
            metrics_summary=metrics_summary,
            strengths="\n".join(f"- {s}" for s in (feedback.cross_scenario_patterns or [])),
            weaknesses=feedback.verdict.reasoning if feedback else "",
            recommendations=feedback.unified_feedback if feedback else "",
            previous_code=current_code,
            iteration_history=history_text,
        )

        # Append rollback context when recovering from a regression
        if rolled_back_from:
            feedback_section += "\n\n" + WRITER_ROLLBACK_SECTION.format(
                failed_iteration=rolled_back_from.get("iteration_number", "?"),
                failed_score=rolled_back_from.get("score", "?"),
                best_score=rolled_back_from.get("best_score", "?"),
                failure_reasoning=rolled_back_from.get("judge_reasoning", ""),
                failed_metrics=rolled_back_from.get("metrics_text", ""),
                failed_code=rolled_back_from.get("code", ""),
            )

    human_msg = WRITER_HUMAN.format(
        goal=state.get("goal", ""),
        feedback_section=feedback_section,
    )

    messages = [
        SystemMessage(content=WRITER_SYSTEM),
        HumanMessage(content=human_msg),
    ]

    # Retry the LLM call up to 3 times before giving up — some models
    # occasionally fail to invoke the function-calling tool on the first
    # attempt, especially with long prompts.
    _max_llm_retries = 3
    result = None
    for attempt in range(1, _max_llm_retries + 1):
        try:
            model = get_codegen_model()
            structured = get_structured_model(model, GeneratedStrategy)
            result = structured.invoke(messages)
        except Exception as exc:
            logger.warning("Writer LLM call raised on attempt %d: %s", attempt, exc)
            result = None

        if result is not None:
            break

        logger.warning(
            "Writer LLM returned None (attempt %d/%d) — model did not invoke tool schema",
            attempt,
            _max_llm_retries,
        )
        if attempt < _max_llm_retries:
            # On retry, strip rollback/history bulk from the prompt to reduce
            # token pressure. Keep only the goal + compact error note.
            retry_note = (
                "\n\n**NOTE:** Your previous response did not use the required"
                " structured-output format. You MUST respond using the tool schema"
                " (fields: `class_name`, `code`, `reasoning`). Do not return plain text."
            )
            messages = [
                SystemMessage(content=WRITER_SYSTEM),
                HumanMessage(
                    content=WRITER_HUMAN.format(
                        goal=state.get("goal", ""),
                        feedback_section=(feedback_section or "") + retry_note,
                    )
                ),
            ]

    if result is None:
        logger.error(
            "Writer node: all %d LLM attempts returned None — clearing code to force fresh generation",
            _max_llm_retries,
        )
        return {
            "current_code": None,
            "current_class_name": None,
            "current_reasoning": None,
            "validation_errors": ["Writer LLM failed to produce structured output after all retries"],
            "rolled_back_from": None,
            "status": "validating",
        }

    logger.info("Writer produced class %r (%d chars)", result.class_name, len(result.code))

    return {
        "current_code": result.code,
        "current_class_name": result.class_name,
        "current_reasoning": result.reasoning,
        "validation_errors": [],
        "rolled_back_from": None,  # consume rollback context so it doesn't bleed
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
        strategy_class = validator.execute_strategy(code, class_name)

        # Smoke-test: instantiate + call initialize/on_market_data/on_tick
        # to catch runtime errors before spending minutes on a full sim.
        validator.smoke_test(strategy_class)
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
            # Apply overrides — supports both full-config and partial diffs
            merged = base_settings.model_dump()
            merged.update(scenario.config_override)
            settings = SimulationSettings.model_validate(merged)

            # Log seed for reproducibility (2.7.9)
            logger.info("Scenario %r  seed=%d", scenario.name, settings.seed)

            # Run strategy
            if not code:
                raise ValueError("No strategy code to execute")
            sim_result = execute_strategy_safely(code, settings, timeout_seconds=300)
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
                    lob_imbalance_mean=m.lob_imbalance_mean,
                    lob_imbalance_std=m.lob_imbalance_std,
                    vpin=m.vpin,
                    resilience_mean_ns=m.resilience_mean_ns,
                    market_ott_ratio=m.market_ott_ratio,
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
                lob_imbalance_delta_pct=_pct(strat_market.lob_imbalance_mean, base_market.lob_imbalance_mean),
                vpin_delta_pct=_pct(strat_market.vpin, base_market.vpin),
                resilience_delta_pct=_pct(strat_market.resilience_mean_ns, base_market.resilience_mean_ns),
                ott_ratio_delta_pct=_pct(strat_market.market_ott_ratio, base_market.market_ott_ratio),
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

            # Generate charts for the UI
            import matplotlib.pyplot as plt

            price_chart_b64: str | None = None
            spread_chart_b64: str | None = None
            volume_chart_b64: str | None = None
            try:
                fig = analyzer.plot_price_series(strategy_output, title=f"{scenario.name} — Price")
                price_chart_b64 = analyzer.figure_to_base64(fig)
                plt.close(fig)
            except Exception:
                pass
            try:
                fig = analyzer.plot_spread(strategy_output, title=f"{scenario.name} — Spread")
                spread_chart_b64 = analyzer.figure_to_base64(fig)
                plt.close(fig)
            except Exception:
                pass
            try:
                fig = analyzer.plot_volume(strategy_output, title=f"{scenario.name} — Volume")
                volume_chart_b64 = analyzer.figure_to_base64(fig)
                plt.close(fig)
            except Exception:
                pass

            results.append(
                ScenarioResult(
                    scenario_name=scenario.name,
                    interpreter_prompt=prompt,
                    strategy_pnl=strategy_agent_metrics.total_pnl,
                    volatility_delta_pct=impact.volatility_delta_pct,
                    spread_delta_pct=impact.spread_delta_pct,
                    trade_count=strategy_agent_metrics.trade_count,
                    fill_rate=strategy_agent_metrics.fill_rate,
                    order_to_trade_ratio=strategy_agent_metrics.order_to_trade_ratio,
                    sharpe_ratio=strategy_agent_metrics.sharpe_ratio,
                    max_drawdown=strategy_agent_metrics.max_drawdown,
                    end_inventory=strategy_agent_metrics.end_inventory,
                    inventory_std=strategy_agent_metrics.inventory_std,
                    price_chart_b64=price_chart_b64,
                    spread_chart_b64=spread_chart_b64,
                    volume_chart_b64=volume_chart_b64,
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
            strategy_code=state.get("current_code") or "(code unavailable)",
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
            if result is None:
                raise ValueError("LLM failed to return a structured explanation schema")
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
        summary = it.aggregated_explanation[:100] if it.aggregated_explanation else ""
        if it.rolled_back:
            summary = "⚠️ ROLLED BACK — " + summary
        rows.append(
            HISTORY_ROW_TEMPLATE.format(
                iter=it.iteration_number,
                pnl=f"${(first_metrics.total_pnl or 0) / 100:,.2f}" if first_metrics else "N/A",
                trades=str(first_metrics.trade_count) if first_metrics else "N/A",
                fill_rate=f"{(first_metrics.fill_rate or 0):.1%}" if first_metrics and first_metrics.fill_rate is not None else "N/A",
                vol_delta=f"{(first_metrics.volatility_delta_pct or 0):+.1%}" if first_metrics else "N/A",
                spread_delta=f"{(first_metrics.spread_delta_pct or 0):+.1%}" if first_metrics else "N/A",
                score=f"{it.judge_score:.1f}" if it.judge_score else "N/A",
                summary=summary,
            )
        )
    return "\n".join(rows)


def _format_explanations(
    explanations: list[ScenarioExplanation],
    scenario_results: list[ScenarioResult] | None = None,
) -> str:
    """Format scenario explanations for the aggregator prompt.

    When *scenario_results* is provided the output includes a
    **Factual Metrics** block for each scenario so the judge has
    ground-truth numbers to work with, not just the explainer's
    (potentially hallucinated) text.
    """
    # Build a quick lookup: scenario_name → ScenarioResult
    sr_map: dict[str, ScenarioResult] = {}
    if scenario_results:
        for sr in scenario_results:
            sr_map[sr.scenario_name] = sr

    parts = []
    for exp in explanations:
        part = f"### {exp.scenario_name}\n"

        # ── Inject ground-truth metrics when available ──
        sr = sr_map.get(exp.scenario_name)
        if sr and not sr.error:
            pnl_str = f"${(sr.strategy_pnl or 0) / 100:,.2f}"
            vol_str = f"{(sr.volatility_delta_pct or 0):+.1%}"
            spread_str = f"{(sr.spread_delta_pct or 0):+.1%}"
            fill_str = f"{(sr.fill_rate or 0):.1%}" if sr.fill_rate is not None else "N/A"
            ott_str = f"{sr.order_to_trade_ratio:.1f}" if sr.order_to_trade_ratio is not None else "N/A"
            sharpe_str = f"{sr.sharpe_ratio:.2f}" if sr.sharpe_ratio is not None else "N/A"
            drawdown_str = f"${(sr.max_drawdown or 0) / 100:,.2f}" if sr.max_drawdown is not None else "N/A"
            inv_std_str = f"{sr.inventory_std:.1f}" if sr.inventory_std is not None else "N/A"
            part += (
                f"**Factual Metrics (from simulation):** "
                f"PnL={pnl_str}, Trades={sr.trade_count}, Fill Rate={fill_str}, OTT={ott_str}, "
                f"Sharpe={sharpe_str}, Max Drawdown={drawdown_str}, "
                f"End Inventory={sr.end_inventory}, Inventory Std={inv_std_str}, "
                f"Vol Δ={vol_str}, Spread Δ={spread_str}\n"
            )
        elif sr and sr.error:
            part += f"**Simulation Error:** {sr.error[:120]}\n"

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
    """Combine scenario explanations, judge convergence, and handle rollbacks."""
    explanations = state.get("explanations", [])
    iterations = state.get("iterations", [])
    iteration_number = state.get("iteration_number", 1)
    scenario_results = state.get("scenario_results", [])

    # Pull best-tracking state from previous aggregations
    best_score: float = state.get("best_score") or 0.0
    best_code: str | None = state.get("best_code")
    best_iteration_number: int = state.get("best_iteration_number") or 0

    logger.info("Aggregator node — iteration %d, %d explanation(s)", iteration_number, len(explanations))

    history_table = _build_history_table(iterations)
    explanations_text = _format_explanations(explanations, scenario_results)

    # Best-iteration context line for the judge
    if best_score > 0.0:
        best_iteration_line = f"Iteration {best_iteration_number} — Score {best_score:.1f}/10 (this is the benchmark for the `comparison` field)"
    else:
        best_iteration_line = "None yet (this is the first iteration)"

    # Build concise current-iteration metrics block (ground-truth for the judge)
    current_metrics_lines: list[str] = []
    for sr in scenario_results:
        if sr.error:
            current_metrics_lines.append(f"- **{sr.scenario_name}:** ERROR — {sr.error[:100]}")
        else:
            pnl_str = f"${(sr.strategy_pnl or 0) / 100:,.2f}"
            vol_str = f"{(sr.volatility_delta_pct or 0):+.1%}"
            spread_str = f"{(sr.spread_delta_pct or 0):+.1%}"
            fill_str = f"{(sr.fill_rate or 0):.1%}" if sr.fill_rate is not None else "N/A"
            ott_str = f"{sr.order_to_trade_ratio:.1f}" if sr.order_to_trade_ratio is not None else "N/A"
            sharpe_str = f"{sr.sharpe_ratio:.2f}" if sr.sharpe_ratio is not None else "N/A"
            drawdown_str = f"${(sr.max_drawdown or 0) / 100:,.2f}" if sr.max_drawdown is not None else "N/A"
            inv_std_str = f"{sr.inventory_std:.1f}" if sr.inventory_std is not None else "N/A"
            current_metrics_lines.append(
                f"- **{sr.scenario_name}:** PnL={pnl_str}, Trades={sr.trade_count}, "
                f"Fill Rate={fill_str}, OTT={ott_str}, Sharpe={sharpe_str}, "
                f"Max Drawdown={drawdown_str}, End Inventory={sr.end_inventory}, "
                f"Inventory Std={inv_std_str}, Vol Δ={vol_str}, Spread Δ={spread_str}"
            )
    current_metrics_block = "\n".join(current_metrics_lines) if current_metrics_lines else "(no results)"

    human_msg = AGGREGATOR_HUMAN.format(
        goal=state.get("goal", ""),
        best_iteration_line=best_iteration_line,
        history_table=history_table,
        iteration_number=iteration_number,
        current_metrics=current_metrics_block,
        explanations=explanations_text,
    )

    model = get_judge_model()
    structured = get_structured_model(model, JudgeVerdict)

    try:
        verdict_result = structured.invoke(
            [
                SystemMessage(content=AGGREGATOR_SYSTEM),
                HumanMessage(content=human_msg),
            ]
        )
        if verdict_result is None:
            raise ValueError("LLM failed to return a structured verdict schema")
        verdict = verdict_result
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
        cross_scenario_patterns=[s for exp in explanations for s in exp.strengths],
        unified_feedback=("Consolidated recommendations:\n" + "\n".join(f"- {r}" for exp in explanations for r in exp.recommendations)),
    )

    # Build per-scenario metrics snapshot for history
    scenario_metrics: dict[str, ScenarioMetrics] = {}
    for sr in scenario_results:
        scenario_metrics[sr.scenario_name] = ScenarioMetrics(
            scenario_name=sr.scenario_name,
            total_pnl=sr.strategy_pnl,
            sharpe_ratio=sr.sharpe_ratio,
            max_drawdown=sr.max_drawdown,
            fill_rate=sr.fill_rate,
            order_to_trade_ratio=sr.order_to_trade_ratio,
            inventory_std=sr.inventory_std,
            end_inventory=sr.end_inventory,
            volatility_delta_pct=sr.volatility_delta_pct,
            spread_delta_pct=sr.spread_delta_pct,
            trade_count=sr.trade_count,
            price_chart_b64=sr.price_chart_b64,
            spread_chart_b64=sr.spread_chart_b64,
            volume_chart_b64=sr.volume_chart_b64,
        )

    # ── Regression detection & hard rollback ─────────────────────────────
    current_code = state.get("current_code") or ""
    # A regression is a score drop of more than 0.5 below the historical best.
    is_regression = best_score > 0.0 and verdict.score < best_score - 0.5
    rolled_back_from: dict | None = None
    next_code = current_code  # what the writer will receive as current_code

    if is_regression:
        logger.warning(
            "Regression detected: score %.1f < best %.1f (iter %d) — rolling back",
            verdict.score,
            best_score,
            best_iteration_number,
        )
        rolled_back_from = {
            "iteration_number": iteration_number,
            "score": verdict.score,
            "best_score": best_score,
            "judge_reasoning": verdict.reasoning,
            "metrics_text": current_metrics_block,
            "code": current_code,
        }
        next_code = best_code  # revert to the best-known code

    # Update best-tracking if this iteration improved
    new_best_score = best_score
    new_best_code = best_code
    new_best_iteration = best_iteration_number
    if verdict.score > best_score:
        new_best_score = verdict.score
        new_best_code = current_code
        new_best_iteration = iteration_number

    iteration_summary = IterationSummary(
        iteration_number=iteration_number,
        strategy_code=current_code,
        scenario_metrics=scenario_metrics,
        aggregated_explanation=verdict.reasoning,  # judge reasoning, not recommendations boilerplate
        judge_score=verdict.score,
        judge_reasoning=verdict.reasoning,
        rolled_back=is_regression,
    )

    new_iterations = list(iterations) + [iteration_summary]

    # ── Rule-based stop guard ─────────────────────────────────────────────
    # The LLM's recommendation is advisory only; we gate it against hard rules.
    max_iterations = state.get("max_iterations", 3)
    lm_wants_stop = verdict.recommendation in ("stop_converged", "stop_plateau")

    if lm_wants_stop:
        all_scores = [it.judge_score for it in new_iterations if it.judge_score is not None]

        if verdict.recommendation == "stop_plateau":
            # Require 3+ consecutive iterations all within ±0.5 of each other
            consecutive_similar = 0
            for i in range(len(all_scores) - 1, 0, -1):
                if abs(all_scores[i] - all_scores[i - 1]) <= 0.5:
                    consecutive_similar += 1
                else:
                    break
            if consecutive_similar < 2:
                logger.info(
                    "LLM recommended stop_plateau but rule-based guard vetoed: need 3+ consecutive similar scores, found only %d pair(s)",
                    consecutive_similar,
                )
                lm_wants_stop = False

        elif verdict.recommendation == "stop_converged":
            # Require score >= 7.0 for a convergence stop
            if verdict.score < 7.0:
                logger.info(
                    "LLM recommended stop_converged but rule-based guard vetoed: score %.1f < 7.0",
                    verdict.score,
                )
                lm_wants_stop = False

        # Regression always overrides any stop recommendation
        if is_regression:
            logger.info("LLM recommended stop but regression detected — forcing continue")
            lm_wants_stop = False

    if lm_wants_stop:
        next_status = "done"
    elif iteration_number >= max_iterations:
        logger.info("Max iterations (%d) reached", max_iterations)
        next_status = "done"
    else:
        next_status = "writing"

    logger.info(
        "Aggregator verdict: score=%.1f, best=%.1f, comparison=%s, recommendation=%s → %s%s",
        verdict.score,
        new_best_score,
        verdict.comparison,
        verdict.recommendation,
        next_status,
        " [REGRESSION→ROLLBACK]" if is_regression else "",
    )

    return {
        "aggregated_feedback": feedback,
        "iterations": new_iterations,
        "iteration_number": iteration_number + 1,
        "status": next_status,
        # Rollback / best-tracking state
        "current_code": next_code,
        "best_score": new_best_score,
        "best_code": new_best_code,
        "best_iteration_number": new_best_iteration,
        "rolled_back_from": rolled_back_from,
    }
