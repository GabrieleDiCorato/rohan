"""Prompt templates for LLM agents in the Rohan framework."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rohan.simulation import RunSummary

INTERPRETER_PROMPT_TEMPLATE = """You are analyzing the results of a trading strategy simulation.

## Goal
{goal}

## Strategy Performance

### Agent Metrics
- **Total PnL**: ${total_pnl}
- **Initial Cash**: ${initial_cash}
- **Ending Cash**: ${ending_cash}
- **Trade Count**: {trade_count}
- **Fill Rate**: {fill_rate}
- **End Inventory**: {end_inventory}

### Market Impact (vs Baseline)
- **Volatility Change**: {volatility_delta_pct}
- **Spread Change**: {spread_delta_pct}
- **Bid Liquidity Change**: {bid_liquidity_delta_pct}
- **Ask Liquidity Change**: {ask_liquidity_delta_pct}

{charts_section}

## Analysis Task
1. Summarize the strategy's overall performance
2. Identify key strengths and weaknesses
3. Assess the market impact (was the strategy stabilizing or destabilizing?)
4. Suggest specific improvements for the next iteration
"""


def _fmt_dollar(v: int | float | None) -> str:
    """Format a value in cents as a dollar string for human display."""
    return f"{v / 100:,.2f}" if v is not None else "N/A"


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.1%}" if v is not None else "N/A"


def _fmt_rate(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "N/A"


def format_interpreter_prompt(summary: "RunSummary", goal: str = "") -> str:
    """Format the interpreter prompt with RunSummary data.

    Args:
        summary: RunSummary from generate_summary().
        goal: Optional strategy goal description.

    Returns:
        Formatted prompt string.
    """

    comp = summary.comparison
    agent = comp.strategy_metrics
    impact = comp.market_impact

    charts_section = ""
    if summary.price_chart:
        charts_section += "## Charts\n[Price Chart attached]\n"
    if summary.spread_chart:
        charts_section += "[Spread Chart attached]\n"

    return INTERPRETER_PROMPT_TEMPLATE.format(
        goal=goal,
        total_pnl=_fmt_dollar(agent.total_pnl),
        initial_cash=_fmt_dollar(agent.initial_cash),
        ending_cash=_fmt_dollar(agent.ending_cash),
        trade_count=agent.trade_count,
        fill_rate=_fmt_rate(agent.fill_rate),
        end_inventory=agent.end_inventory,
        volatility_delta_pct=_fmt_pct(impact.volatility_delta_pct),
        spread_delta_pct=_fmt_pct(impact.spread_delta_pct),
        bid_liquidity_delta_pct=_fmt_pct(impact.bid_liquidity_delta_pct),
        ask_liquidity_delta_pct=_fmt_pct(impact.ask_liquidity_delta_pct),
        charts_section=charts_section,
    )
