"""LangGraph agent nodes for the strategy-refinement loop.

Each public function is a LangGraph node with signature::

    def node(state: RefinementState) -> dict
"""

from __future__ import annotations

import logging
import traceback

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

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
    QualitativeAnalysis,
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
from rohan.llm.scoring import WEIGHT_PROFILES, classify_goal_weights, compute_axis_scores, compute_final_score
from rohan.llm.state import RefinementState, ScenarioResult
from rohan.llm.tools import make_investigation_tools
from rohan.simulation.models.simulation_metrics import (
    ComparisonResult,
    MarketImpact,
)
from rohan.simulation.simulation_service import SimulationService
from rohan.simulation.strategy_validator import StrategyValidator, execute_strategy_safely
from rohan.simulation.utils import _pct_change, _to_market_metrics

logger = logging.getLogger(__name__)


# ─── Formatting helpers (consistent None→"N/A", proper negative signs) ────


def _fmt_dollar(cents: float | None) -> str:
    """Format cents as ``$X.XX`` with proper negative sign, or ``N/A``."""
    if cents is None:
        return "N/A"
    dollars = cents / 100.0
    sign = "-" if dollars < 0 else ""
    return f"{sign}${abs(dollars):,.2f}"


def _fmt_pct(value: float | None, signed: bool = False) -> str:
    """Format a fractional value as ``X.X%``, or ``N/A``."""
    if value is None:
        return "N/A"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def _fmt_float(value: float | None, fmt: str = ".2f") -> str:
    """Format a float, or ``N/A``."""
    if value is None:
        return "N/A"
    return f"{value:{fmt}}"


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
                pnl_str = _fmt_dollar(sr.strategy_pnl)
                fill_str = _fmt_pct(sr.fill_rate)
                ott_str = _fmt_float(sr.order_to_trade_ratio, ".1f")
                sharpe_str = _fmt_float(sr.sharpe_ratio)
                drawdown_str = _fmt_dollar(sr.max_drawdown)
                inv_std_str = _fmt_float(sr.inventory_std, ".1f")
                vol_str = _fmt_pct(sr.volatility_delta_pct, signed=True)
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
            # Use deterministic per-scenario seed when available
            if scenario.seed is not None:
                merged["seed"] = scenario.seed
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
            strat_market = _to_market_metrics(strategy_sim_metrics)
            base_market = _to_market_metrics(baseline_sim_metrics)
            impact = MarketImpact(
                spread_delta_pct=_pct_change(strat_market.mean_spread, base_market.mean_spread),
                volatility_delta_pct=_pct_change(strat_market.volatility, base_market.volatility),
                bid_liquidity_delta_pct=_pct_change(strat_market.avg_bid_liquidity, base_market.avg_bid_liquidity),
                ask_liquidity_delta_pct=_pct_change(strat_market.avg_ask_liquidity, base_market.avg_ask_liquidity),
                lob_imbalance_delta_pct=_pct_change(strat_market.lob_imbalance_mean, base_market.lob_imbalance_mean),
                vpin_delta_pct=_pct_change(strat_market.vpin, base_market.vpin),
                resilience_delta_pct=_pct_change(strat_market.resilience_mean_ns, base_market.resilience_mean_ns),
                ott_ratio_delta_pct=_pct_change(strat_market.market_ott_ratio, base_market.market_ott_ratio),
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

            # Generate charts for the UI
            price_chart_b64: str | None = None
            spread_chart_b64: str | None = None
            volume_chart_b64: str | None = None
            try:
                fig = analyzer.plot_price_series(strategy_output, title=f"{scenario.name} — Price")
                price_chart_b64 = analyzer.figure_to_base64(fig)
            except Exception:
                logger.debug("Price chart generation failed for %r", scenario.name, exc_info=True)
            try:
                fig = analyzer.plot_spread(strategy_output, title=f"{scenario.name} — Spread")
                spread_chart_b64 = analyzer.figure_to_base64(fig)
            except Exception:
                logger.debug("Spread chart generation failed for %r", scenario.name, exc_info=True)
            try:
                fig = analyzer.plot_volume(strategy_output, title=f"{scenario.name} — Volume")
                volume_chart_b64 = analyzer.figure_to_base64(fig)
            except Exception:
                logger.debug("Volume chart generation failed for %r", scenario.name, exc_info=True)

            # --- Rich analysis (Step 8) ---
            rich_bundle = None
            pnl_chart_b64: str | None = None
            inventory_chart_b64: str | None = None
            fill_scatter_b64: str | None = None
            rich_analysis_json: str | None = None
            try:
                rich_bundle = analyzer.compute_rich_analysis(
                    strategy_output,
                    strategy_output.strategic_agent_id,
                    initial_cash=settings.starting_cash,
                )
                rich_analysis_json = rich_bundle.model_dump_json()
            except Exception:
                logger.warning("Rich analysis failed for scenario %r", scenario.name, exc_info=True)

            # Generate prompt AFTER rich analysis so it includes execution quality
            prompt = format_interpreter_prompt(summary, goal=state.get("goal", ""), rich_analysis_json=rich_analysis_json)

            if rich_bundle is not None:
                try:
                    fig = analyzer.plot_pnl_curve(rich_bundle.pnl_curve, title=f"{scenario.name} — PnL")
                    pnl_chart_b64 = analyzer.figure_to_base64(fig)
                except Exception:
                    logger.debug("PnL chart generation failed for %r", scenario.name, exc_info=True)
                try:
                    fig = analyzer.plot_inventory(rich_bundle.inventory_trajectory, title=f"{scenario.name} — Inventory")
                    inventory_chart_b64 = analyzer.figure_to_base64(fig)
                except Exception:
                    logger.debug("Inventory chart generation failed for %r", scenario.name, exc_info=True)
                try:
                    fig = analyzer.plot_fills_vs_mid(rich_bundle.fills, title=f"{scenario.name} — Fills")
                    fill_scatter_b64 = analyzer.figure_to_base64(fig)
                except Exception:
                    logger.debug("Fill scatter chart generation failed for %r", scenario.name, exc_info=True)

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
                    starting_capital_cents=settings.starting_cash,
                    baseline_mean_spread=base_market.mean_spread,
                    baseline_traded_volume=base_market.traded_volume,
                    bid_liquidity_delta_pct=impact.bid_liquidity_delta_pct,
                    ask_liquidity_delta_pct=impact.ask_liquidity_delta_pct,
                    price_chart_b64=price_chart_b64,
                    spread_chart_b64=spread_chart_b64,
                    volume_chart_b64=volume_chart_b64,
                    pnl_chart_b64=pnl_chart_b64,
                    inventory_chart_b64=inventory_chart_b64,
                    fill_scatter_b64=fill_scatter_b64,
                    rich_analysis_json=rich_analysis_json,
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


def _error_explanation(scenario_name: str, error: str) -> ScenarioExplanation:
    """Create a placeholder explanation for a scenario that errored."""
    return ScenarioExplanation(
        scenario_name=scenario_name,
        weaknesses=[f"Scenario failed: {error}"],
        raw_analysis=f"Error: {error}",
    )


def _fallback_structured_explanation(
    model: BaseChatModel,
    sr: ScenarioResult,
    state: RefinementState,
) -> ScenarioExplanation:
    """Single structured-output call (no tools) used when ReAct fails."""
    human_msg = EXPLAINER_HUMAN.format(
        scenario_name=sr.scenario_name,
        strategy_code=state.get("current_code") or "(code unavailable)",
        interpreter_prompt=sr.interpreter_prompt,
        regime_context=sr.regime_context or "",
    )
    structured = get_structured_model(model, ScenarioExplanation)
    result = structured.invoke(
        [
            SystemMessage(content=EXPLAINER_SYSTEM),
            HumanMessage(content=human_msg),
        ]
    )
    if result is None:
        raise ValueError("LLM returned None for structured explanation")
    result.scenario_name = sr.scenario_name
    return result


_REACT_RECURSION_LIMIT = 25


def explainer_node(state: RefinementState) -> dict:
    """Analyse each scenario result using a ReAct agent with investigation tools.

    For each non-error scenario the node:
    1. Builds parameterised investigation tools from the serialised
       ``RichAnalysisBundle`` attached to the scenario result.
    2. Creates a ``create_react_agent`` with those tools, the explainer
       system prompt, and ``ScenarioExplanation`` as a structured output.
    3. Invokes the agent, letting it call tools in a loop until it
       produces the final structured response.

    If the ReAct agent fails for any reason the node falls back to a
    single structured-output LLM call (no tools).
    """
    scenario_results = state.get("scenario_results", [])
    logger.info("Explainer node — %d scenario(s)", len(scenario_results))

    model = get_analysis_model()
    explanations: list[ScenarioExplanation] = []

    for sr in scenario_results:
        # ── Error scenarios ──────────────────────────────────────────
        if sr.error:
            explanations.append(_error_explanation(sr.scenario_name, sr.error))
            continue

        # ── Build human message ──────────────────────────────────────
        human_msg = EXPLAINER_HUMAN.format(
            scenario_name=sr.scenario_name,
            strategy_code=state.get("current_code") or "(code unavailable)",
            interpreter_prompt=sr.interpreter_prompt,
            regime_context=sr.regime_context or "",
        )

        # ── ReAct agent path ─────────────────────────────────────────
        try:
            tools = make_investigation_tools(sr.rich_analysis_json)
            agent = create_react_agent(
                model,
                tools=tools,
                prompt=EXPLAINER_SYSTEM,
                response_format=ScenarioExplanation,
            )
            agent_output = agent.invoke(
                {"messages": [HumanMessage(content=human_msg)]},
                config={"recursion_limit": _REACT_RECURSION_LIMIT},
            )
            result: ScenarioExplanation = agent_output["structured_response"]
            result.scenario_name = sr.scenario_name
            explanations.append(result)
        except Exception as exc:
            logger.warning(
                "ReAct agent failed for %r, falling back to structured call: %s",
                sr.scenario_name,
                exc,
            )
            # ── Fallback: single structured-output call ──────────────
            try:
                explanations.append(_fallback_structured_explanation(model, sr, state))
            except Exception as fallback_exc:
                logger.error(
                    "Fallback also failed for %r: %s",
                    sr.scenario_name,
                    fallback_exc,
                )
                explanations.append(
                    ScenarioExplanation(
                        scenario_name=sr.scenario_name,
                        weaknesses=[f"Analysis failed: {fallback_exc}"],
                        raw_analysis=str(fallback_exc),
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
                pnl=_fmt_dollar(first_metrics.total_pnl) if first_metrics else "N/A",
                trades=str(first_metrics.trade_count) if first_metrics else "N/A",
                fill_rate=_fmt_pct(first_metrics.fill_rate) if first_metrics else "N/A",
                vol_delta=_fmt_pct(first_metrics.volatility_delta_pct, signed=True) if first_metrics else "N/A",
                spread_delta=_fmt_pct(first_metrics.spread_delta_pct, signed=True) if first_metrics else "N/A",
                score=_fmt_float(it.judge_score, ".1f"),
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
            pnl_str = _fmt_dollar(sr.strategy_pnl)
            vol_str = _fmt_pct(sr.volatility_delta_pct, signed=True)
            spread_str = _fmt_pct(sr.spread_delta_pct, signed=True)
            fill_str = _fmt_pct(sr.fill_rate)
            ott_str = _fmt_float(sr.order_to_trade_ratio, ".1f")
            sharpe_str = _fmt_float(sr.sharpe_ratio)
            drawdown_str = _fmt_dollar(sr.max_drawdown)
            inv_std_str = _fmt_float(sr.inventory_std, ".1f")
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
    """Combine scenario explanations, score deterministically, and handle rollbacks.

    Scoring is fully deterministic via ``compute_axis_scores`` /
    ``compute_final_score``.  The LLM is called only for qualitative
    analysis (reasoning, strengths, weaknesses, recommendations).
    """
    explanations = state.get("explanations", [])
    iterations = state.get("iterations", [])
    iteration_number = state.get("iteration_number", 1)
    scenario_results = state.get("scenario_results", [])

    # Pull best-tracking state from previous aggregations
    best_score: float = state.get("best_score") or 0.0
    best_code: str | None = state.get("best_code")
    best_iteration_number: int = state.get("best_iteration_number") or 0

    logger.info("Aggregator node — iteration %d, %d explanation(s)", iteration_number, len(explanations))

    # ── Deterministic scoring ────────────────────────────────────────────
    goal = state.get("goal", "")
    weights = classify_goal_weights(goal)

    # Compute per-scenario axis scores, then average
    all_axis = []
    for sr in scenario_results:
        if sr.error:
            continue
        axis = compute_axis_scores(
            strategy_pnl=sr.strategy_pnl,
            trade_count=sr.trade_count,
            sharpe_ratio=sr.sharpe_ratio,
            max_drawdown=sr.max_drawdown,
            fill_rate=sr.fill_rate,
            order_to_trade_ratio=sr.order_to_trade_ratio,
            volatility_delta_pct=sr.volatility_delta_pct,
            spread_delta_pct=sr.spread_delta_pct,
            bid_liquidity_delta_pct=sr.bid_liquidity_delta_pct,
            ask_liquidity_delta_pct=sr.ask_liquidity_delta_pct,
            starting_capital_cents=sr.starting_capital_cents,
            baseline_mean_spread=sr.baseline_mean_spread,
            baseline_traded_volume=sr.baseline_traded_volume,
        )
        all_axis.append(axis)

    if all_axis:
        from rohan.llm.scoring import AxisScores

        avg_axis = AxisScores(
            profitability=sum(a.profitability for a in all_axis) / len(all_axis),
            risk_adjusted=sum(a.risk_adjusted for a in all_axis) / len(all_axis),
            volatility_impact=sum(a.volatility_impact for a in all_axis) / len(all_axis),
            spread_impact=sum(a.spread_impact for a in all_axis) / len(all_axis),
            liquidity_impact=sum(a.liquidity_impact for a in all_axis) / len(all_axis),
            execution_quality=sum(a.execution_quality for a in all_axis) / len(all_axis),
        )
        final_score = compute_final_score(avg_axis, weights)
    else:
        from rohan.llm.scoring import AxisScores

        avg_axis = AxisScores(1.0, 1.0, 5.5, 5.5, 5.5, 1.0)
        final_score = 1.0

    # ── Deterministic comparison ─────────────────────────────────────────
    if best_score <= 0:
        comparison: str = "similar"
    elif final_score > best_score:
        comparison = "better"
    elif final_score < best_score - 0.5:
        comparison = "worse"
    else:
        comparison = "similar"

    # ── Deterministic recommendation ─────────────────────────────────────
    all_scores = [it.judge_score for it in iterations if it.judge_score is not None]
    all_scores.append(final_score)

    recommendation = "continue"
    if final_score >= 7.0 and len(all_scores) >= 3:
        # Check plateau: last 3 scores within ±0.5 of each other
        last3 = all_scores[-3:]
        if max(last3) - min(last3) <= 0.5:
            recommendation = "stop_converged"
    elif len(all_scores) >= 3:
        last3 = all_scores[-3:]
        if max(last3) - min(last3) <= 0.5:
            recommendation = "stop_plateau"

    # ── LLM qualitative analysis (optional — enriches feedback) ──────────
    history_table = _build_history_table(iterations)
    explanations_text = _format_explanations(explanations, scenario_results)

    # Build concise current-iteration metrics block (ground-truth for the LLM)
    current_metrics_lines: list[str] = []
    for sr in scenario_results:
        if sr.error:
            current_metrics_lines.append(f"- **{sr.scenario_name}:** ERROR — {sr.error[:100]}")
        else:
            pnl_str = _fmt_dollar(sr.strategy_pnl)
            vol_str = _fmt_pct(sr.volatility_delta_pct, signed=True)
            spread_str = _fmt_pct(sr.spread_delta_pct, signed=True)
            fill_str = _fmt_pct(sr.fill_rate)
            ott_str = _fmt_float(sr.order_to_trade_ratio, ".1f")
            sharpe_str = _fmt_float(sr.sharpe_ratio)
            drawdown_str = _fmt_dollar(sr.max_drawdown)
            inv_std_str = _fmt_float(sr.inventory_std, ".1f")
            current_metrics_lines.append(
                f"- **{sr.scenario_name}:** PnL={pnl_str}, Trades={sr.trade_count}, "
                f"Fill Rate={fill_str}, OTT={ott_str}, Sharpe={sharpe_str}, "
                f"Max Drawdown={drawdown_str}, End Inventory={sr.end_inventory}, "
                f"Inventory Std={inv_std_str}, Vol Δ={vol_str}, Spread Δ={spread_str}"
            )
    current_metrics_block = "\n".join(current_metrics_lines) if current_metrics_lines else "(no results)"

    # Scores block for the LLM to reason about
    scores_block = (
        f"**Deterministic Scores (computed, not for LLM to change):**\n"
        f"- Profitability: {avg_axis.profitability:.1f}/10\n"
        f"- Risk-Adjusted: {avg_axis.risk_adjusted:.1f}/10\n"
        f"- Volatility Impact: {avg_axis.volatility_impact:.1f}/10\n"
        f"- Spread Impact: {avg_axis.spread_impact:.1f}/10\n"
        f"- Liquidity Impact: {avg_axis.liquidity_impact:.1f}/10\n"
        f"- Execution Quality: {avg_axis.execution_quality:.1f}/10\n"
        f"- **Weighted Final: {final_score:.1f}/10** (profile: "
        f"{next((k for k, v in WEIGHT_PROFILES.items() if v == weights), 'custom')})"
    )

    best_iteration_line = f"Iteration {best_iteration_number} — Score {best_score:.1f}/10" if best_score > 0.0 else "None yet (this is the first iteration)"

    human_msg = AGGREGATOR_HUMAN.format(
        goal=goal,
        best_iteration_line=best_iteration_line,
        history_table=history_table,
        iteration_number=iteration_number,
        current_metrics=current_metrics_block,
        explanations=explanations_text,
    )
    # Prepend scores block to human message so LLM can explain them
    human_msg = scores_block + "\n\n" + human_msg

    model = get_judge_model()
    structured = get_structured_model(model, QualitativeAnalysis)

    qualitative_reasoning = ""
    try:
        analysis_result = structured.invoke(
            [
                SystemMessage(content=AGGREGATOR_SYSTEM),
                HumanMessage(content=human_msg),
            ]
        )
        if analysis_result is not None:
            qualitative_reasoning = analysis_result.reasoning
    except Exception as exc:
        logger.error("Qualitative analysis LLM call failed: %s", exc)
        qualitative_reasoning = f"Qualitative analysis unavailable: {exc}"

    # ── Build verdict (deterministic + qualitative) ──────────────────────
    verdict = JudgeVerdict(
        score=final_score,
        comparison=comparison,  # type: ignore[arg-type]
        reasoning=qualitative_reasoning,
        recommendation=recommendation,  # type: ignore[arg-type]
        profitability_score=avg_axis.profitability,
        risk_score=avg_axis.risk_adjusted,
        volatility_impact_score=avg_axis.volatility_impact,
        spread_impact_score=avg_axis.spread_impact,
        liquidity_impact_score=avg_axis.liquidity_impact,
        execution_score=avg_axis.execution_quality,
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
            pnl_chart_b64=sr.pnl_chart_b64,
            inventory_chart_b64=sr.inventory_chart_b64,
            fill_scatter_b64=sr.fill_scatter_b64,
        )

    # ── Regression detection & hard rollback ─────────────────────────────
    current_code = state.get("current_code") or ""
    # Deterministic scores eliminate noise — rollback on any drop > 1.0
    score_drop = best_score - final_score if best_score > 0 else 0.0
    is_regression = best_score > 0 and score_drop > 1.0
    rolled_back_from: dict | None = None
    next_code = current_code

    if is_regression:
        logger.warning(
            "Regression detected: score %.1f < best %.1f (iter %d) — rolling back",
            final_score,
            best_score,
            best_iteration_number,
        )
        rolled_back_from = {
            "iteration_number": iteration_number,
            "score": final_score,
            "best_score": best_score,
            "judge_reasoning": qualitative_reasoning,
            "metrics_text": current_metrics_block,
            "code": current_code,
        }
        next_code = best_code  # revert to the best-known code

    # Update best-tracking if this iteration improved
    new_best_score = best_score
    new_best_code = best_code
    new_best_iteration = best_iteration_number
    if final_score > best_score:
        new_best_score = final_score
        new_best_code = current_code
        new_best_iteration = iteration_number

    iteration_summary = IterationSummary(
        iteration_number=iteration_number,
        strategy_code=current_code,
        scenario_metrics=scenario_metrics,
        aggregated_explanation=qualitative_reasoning,
        judge_score=final_score,
        judge_reasoning=qualitative_reasoning,
        rolled_back=is_regression,
        profitability_score=avg_axis.profitability,
        risk_score=avg_axis.risk_adjusted,
        volatility_impact_score=avg_axis.volatility_impact,
        spread_impact_score=avg_axis.spread_impact,
        liquidity_impact_score=avg_axis.liquidity_impact,
        execution_score=avg_axis.execution_quality,
        scoring_profile=next((k for k, v in WEIGHT_PROFILES.items() if v == weights), "custom"),
    )

    new_iterations = list(iterations) + [iteration_summary]

    # ── Stop conditions (deterministic) ──────────────────────────────────
    max_iterations = state.get("max_iterations", 3)

    # Regression always forces continue
    if is_regression:
        recommendation = "continue"

    if recommendation in ("stop_converged", "stop_plateau") and not is_regression:
        next_status = "done"
    elif iteration_number >= max_iterations:
        logger.info("Max iterations (%d) reached", max_iterations)
        next_status = "done"
    else:
        next_status = "writing"

    logger.info(
        "Aggregator verdict: score=%.1f, best=%.1f, comparison=%s, recommendation=%s → %s%s",
        final_score,
        new_best_score,
        comparison,
        recommendation,
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
