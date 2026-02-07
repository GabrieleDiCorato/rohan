"""Prompt templates for LLM agents in the Rohan framework."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rohan.simulation import RunSummary

INTERPRETER_PROMPT_TEMPLATE = """You are analyzing the results of a trading strategy simulation.

## Goal
{goal}

## Strategy Performance

### Agent Metrics
- **Total PnL**: ${total_pnl:,.2f}
- **Realized PnL**: ${realized_pnl:,.2f}
- **Unrealized PnL**: ${unrealized_pnl:,.2f}
- **Trade Count**: {trade_count}
- **Fill Rate**: {fill_rate:.1%}
- **End Inventory**: {end_inventory}

### Market Impact (vs Baseline)
- **Volatility Change**: {volatility_pct_change:+.1%}
- **Spread Change**: {spread_pct_change:+.1%}
- **Bid Liquidity Change**: {liquidity_bid_pct_change:+.1%}
- **Ask Liquidity Change**: {liquidity_ask_pct_change:+.1%}

{charts_section}

## Analysis Task
1. Summarize the strategy's overall performance
2. Identify key strengths and weaknesses
3. Assess the market impact (was the strategy stabilizing or destabilizing?)
4. Suggest specific improvements for the next iteration
"""


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
        total_pnl=agent.total_pnl,
        realized_pnl=agent.realized_pnl,
        unrealized_pnl=agent.unrealized_pnl,
        trade_count=agent.trade_count,
        fill_rate=agent.fill_rate,
        end_inventory=agent.end_inventory,
        volatility_pct_change=impact.get("volatility_pct_change", 0),
        spread_pct_change=impact.get("spread_pct_change", 0),
        liquidity_bid_pct_change=impact.get("liquidity_bid_pct_change", 0),
        liquidity_ask_pct_change=impact.get("liquidity_ask_pct_change", 0),
        charts_section=charts_section,
    )
