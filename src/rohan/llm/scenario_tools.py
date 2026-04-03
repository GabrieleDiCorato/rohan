"""Agentic tools for scenario generation, config validation, and metric explanation.

These are standalone LangChain tools (not bound to a ``RichAnalysisBundle``).
They expose hasufel's discoverability APIs and Rohan's diagnostic patterns
to the ReAct agent framework.

The factory :func:`make_scenario_tools` returns the tools ready to be
passed to ``create_react_agent``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Maximum characters per tool response.
_MAX_RESPONSE_CHARS = 4000


def _truncate(text: str, limit: int = _MAX_RESPONSE_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_scenario_tools() -> list[Any]:
    """Return LangChain tools for scenario generation, validation, and explanation.

    Returns four tools:

    - ``list_scenarios``: enumerate available templates with regime tags
    - ``build_scenario``: template-based scenario assembly from regime tags
    - ``validate_scenario``: structural + cross-agent config validation
    - ``explain_metrics``: rule-based diagnostic interpretation of comparison results
    """

    # ── Tool 0: Template discovery ────────────────────────────────

    @tool
    def list_scenarios() -> str:
        """List all available simulation scenario templates with their regime tags and descriptions.

        Use this to discover templates before calling ``build_scenario``.
        """
        from abides_markets.config_system import list_templates

        templates = list_templates()
        entries = []
        for t in templates:
            entries.append(
                {
                    "name": t["name"],
                    "is_overlay": t.get("is_overlay", False),
                    "regime_tags": t.get("regime_tags", []),
                    "agent_types": t.get("agent_types", []),
                    "description": t.get("scenario_description", t.get("description", "")),
                }
            )
        return _truncate(json.dumps(entries, indent=2))

    # ── Tool 1: Template-based scenario generation (R11) ──────────

    @tool
    def build_scenario(
        regime_tags: list[str] | None = None,
        template_name: str | None = None,
        ticker: str = "ABM",
        seed: int = 42,
        start_time: str = "09:30:00",
        end_time: str = "10:00:00",
        overrides: dict[str, Any] | None = None,
    ) -> str:
        """Build a simulation scenario from template and/or regime tags.

        Selects a base template by matching *regime_tags* against the
        template metadata, or uses *template_name* directly.  Returns a
        JSON config dict ready for simulation.

        Parameters
        ----------
        regime_tags : optional
            Filter templates by regime tags (e.g. ["volatile", "thin"]).
            Selects the template with the most matching tags.
        template_name : optional
            Specific template name (e.g. "rmsc04", "stress_test").
            Takes priority over *regime_tags*.
        ticker : str
            Ticker symbol (default "ABM").
        seed : int
            Random seed.
        start_time, end_time : str
            Simulation time window (HH:MM:SS).
        overrides : optional
            Additional builder overrides as ``{method_name: value}``.
        """
        from abides_markets.config_system import SimulationBuilder, list_templates

        templates = list_templates()
        selected: str | None = template_name

        if selected is None and regime_tags:
            # Score templates by number of matching tags
            best_score = -1
            for t in templates:
                t_tags = set(t.get("regime_tags", []))
                score = len(t_tags & set(regime_tags))
                if score > best_score and not t.get("is_overlay", False):
                    best_score = score
                    selected = t["name"]

        if selected is None:
            selected = "rmsc04"

        try:
            builder = SimulationBuilder()
            builder.from_template(selected)
            builder.seed(seed)
            builder.market(ticker=ticker, start_time=start_time, end_time=end_time)

            if overrides:
                for method, value in overrides.items():
                    fn = getattr(builder, method, None)
                    if fn and callable(fn):
                        fn(value) if not isinstance(value, dict) else fn(**value)

            config = builder.build()
            config_dict = config.model_dump(mode="json")

            # Summarise selected template
            tmpl_info = next((t for t in templates if t["name"] == selected), {})
            overlays = [t["name"] for t in templates if t.get("is_overlay", False)]
            summary = {
                "template": selected,
                "regime_tags": tmpl_info.get("regime_tags", []),
                "scenario_description": tmpl_info.get("scenario_description", ""),
                "default_risk_guards": tmpl_info.get("default_risk_guards", {}),
                "agent_types": tmpl_info.get("agent_types", []),
                "available_overlays": overlays,
                "config": config_dict,
            }
            return _truncate(json.dumps(summary, indent=2, default=str))

        except Exception as e:
            return f"Error building scenario: {e}"

    # ── Tool 2: Config validation (§7.2) ──────────────────────────

    @tool
    def validate_scenario(config_json: str) -> str:
        """Validate a proposed simulation configuration.

        Takes a JSON string of a config dict and returns structured
        validation errors and warnings with field paths.

        Parameters
        ----------
        config_json : str
            JSON string of the config dict to validate.
        """
        from abides_markets.config_system import validate_config

        try:
            config_dict = json.loads(config_json)
        except json.JSONDecodeError as e:
            return f"Invalid JSON: {e}"

        result = validate_config(config_dict)

        lines = [f"Valid: {result.valid}"]
        if result.errors:
            lines.append(f"\nErrors ({len(result.errors)}):")
            for issue in result.errors:
                path = getattr(issue, "field_path", "(root)")
                msg = getattr(issue, "message", str(issue))
                suggestion = getattr(issue, "suggestion", None)
                lines.append(f"  [{path}] {msg}")
                if suggestion:
                    lines.append(f"    Suggestion: {suggestion}")

        if result.warnings:
            lines.append(f"\nWarnings ({len(result.warnings)}):")
            for issue in result.warnings:
                path = getattr(issue, "field_path", "(root)")
                msg = getattr(issue, "message", str(issue))
                lines.append(f"  [{path}] {msg}")

        if result.valid:
            lines.append("\nConfiguration is valid and ready for simulation.")

        return _truncate("\n".join(lines))

    # ── Tool 3: Rule-based metric explanation (R12) ───────────────

    @tool
    def explain_metrics(comparison_json: str) -> str:
        """Explain simulation comparison results using diagnostic rules.

        Takes a JSON-serialised ``ComparisonResult`` and returns a
        structured interpretation with actionable recommendations.

        Parameters
        ----------
        comparison_json : str
            JSON string from ``ComparisonResult.model_dump_json()``.
        """
        from rohan.simulation.models.simulation_metrics import ComparisonResult

        try:
            comparison = ComparisonResult.model_validate_json(comparison_json)
        except Exception as e:
            return f"Failed to parse comparison result: {e}"

        sm = comparison.strategy_metrics
        impact = comparison.market_impact

        findings: list[str] = []
        recommendations: list[str] = []

        # --- Alpha generation ---
        if sm.total_pnl is not None:
            if sm.total_pnl > 0:
                findings.append(f"Strategy is profitable (PnL: {sm.total_pnl:+.0f} cents)")
            else:
                findings.append(f"Strategy is unprofitable (PnL: {sm.total_pnl:+.0f} cents)")

        if sm.sharpe_ratio is not None:
            if sm.sharpe_ratio > 1.0:
                findings.append(f"Strong risk-adjusted returns (Sharpe: {sm.sharpe_ratio:.2f})")
            elif sm.sharpe_ratio < 0:
                findings.append(f"Negative risk-adjusted returns (Sharpe: {sm.sharpe_ratio:.2f})")
                recommendations.append("Review trade timing and sizing")

        # --- High PnL + high volatility delta: profitable but destabilising ---
        if sm.total_pnl is not None and sm.total_pnl > 0 and impact is not None and impact.volatility_delta_pct is not None and impact.volatility_delta_pct > 5.0:
            findings.append(f"Profitable but destabilising — volatility increased {impact.volatility_delta_pct:.1f}%")
            recommendations.append("Reduce order size or frequency to limit market impact")

        # --- Low fill rate + low PnL: too passive ---
        if sm.fill_rate is not None and sm.fill_rate < 0.3 and sm.total_pnl is not None and sm.total_pnl <= 0:
            findings.append(f"Strategy is too passive — fill rate {sm.fill_rate:.1%} with negative PnL")
            recommendations.append("Consider crossing the spread or using more aggressive limit orders")

        # --- High VPIN: aggressive directional flow ---
        bm = comparison.baseline_metrics
        if bm.vpin is not None and bm.vpin > 0.7:
            findings.append(f"High VPIN ({bm.vpin:.2f}) — significant informed trading pressure")
            recommendations.append("Reduce order aggressiveness or size to lower adverse selection risk")

        # --- High inventory risk ---
        if sm.inventory_std is not None and sm.max_drawdown is not None and sm.inventory_std > 50 and abs(sm.max_drawdown) > 5000:
            findings.append(f"Large unhedged positions — inventory σ={sm.inventory_std:.0f}, max drawdown={sm.max_drawdown:+.0f} cents")
            recommendations.append("Implement tighter inventory limits or position sizing")

        # --- Resilience degradation ---
        if impact is not None and impact.resilience_delta_pct is not None and impact.resilience_delta_pct > 10.0:
            findings.append(f"Strategy slows spread recovery — resilience worsened by {impact.resilience_delta_pct:.1f}%")
            recommendations.append("Reduce order frequency or improve order placement timing")

        # --- Spread impact ---
        if impact is not None and impact.spread_delta_pct is not None:
            if impact.spread_delta_pct > 5.0:
                findings.append(f"Strategy widens spreads by {impact.spread_delta_pct:.1f}%")
                recommendations.append("Consider using smaller orders or passive limit orders")
            elif impact.spread_delta_pct < -5.0:
                findings.append(f"Strategy tightens spreads by {abs(impact.spread_delta_pct):.1f}% — beneficial market making")

        # --- Market availability ---
        if impact is not None and impact.two_sided_delta_pct is not None and impact.two_sided_delta_pct < -5.0:
            findings.append(f"Strategy reduces market availability by {abs(impact.two_sided_delta_pct):.1f}%")
            recommendations.append("Avoid consuming all resting liquidity on one side of the book")

        # --- Execution quality ---
        if sm.vwap_cents is not None and sm.trade_count > 0:
            findings.append(f"Agent VWAP: {sm.vwap_cents} cents over {sm.trade_count} trades")

        # Format output
        lines = ["=== Strategy Diagnostic Report ===", ""]
        if findings:
            lines.append("Findings:")
            for i, f in enumerate(findings, 1):
                lines.append(f"  {i}. {f}")
        else:
            lines.append("No significant patterns detected.")

        if recommendations:
            lines.append("\nRecommendations:")
            for i, r in enumerate(recommendations, 1):
                lines.append(f"  {i}. {r}")

        return "\n".join(lines)

    return [list_scenarios, build_scenario, validate_scenario, explain_metrics]
