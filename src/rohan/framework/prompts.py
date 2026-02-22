"""Prompt templates for LLM agents in the Rohan framework."""

from typing import TYPE_CHECKING

from rohan.utils.formatting import fmt_dollar_metric

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
- **Sharpe Ratio**: {sharpe_ratio}
- **Max Drawdown**: {max_drawdown}
- **Inventory Std**: {inventory_std}

### Market Impact (vs Baseline)
- **Volatility Change**: {volatility_delta_pct}
- **Spread Change**: {spread_delta_pct}
- **Bid Liquidity Change**: {bid_liquidity_delta_pct}
- **Ask Liquidity Change**: {ask_liquidity_delta_pct}

### Microstructure Impact (vs Baseline)
- **LOB Imbalance Change**: {lob_imbalance_delta_pct}
- **VPIN Change**: {vpin_delta_pct}
- **Resilience Change**: {resilience_delta_pct}
- **OTT Ratio Change**: {ott_ratio_delta_pct}

### Market Microstructure (Strategy Run)
- **Volatility (ann.)**: {volatility}
- **Mean Spread**: {mean_spread}
- **Effective Spread**: {effective_spread}
- **Traded Volume**: {traded_volume}
- **LOB Imbalance (mean)**: {lob_imbalance_mean}
- **LOB Imbalance (std)**: {lob_imbalance_std}
- **VPIN**: {vpin}
- **Resilience (mean)**: {resilience_mean}
- **Market OTT Ratio**: {market_ott_ratio}

{charts_section}

## Analysis Task
1. Summarize the strategy's overall performance
2. Identify key strengths and weaknesses
3. Assess the market impact (was the strategy stabilizing or destabilizing?)
4. Comment on microstructure quality (VPIN, LOB imbalance, resilience, OTT)
5. Suggest specific improvements for the next iteration
"""


def _fmt_pct(v: float | None) -> str:
    return f"{v:+.1%}" if v is not None else "N/A"


def _fmt_rate(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "N/A"


def _fmt_float(v: float | None, fmt: str = ".4f") -> str:
    return f"{v:{fmt}}" if v is not None else "N/A"


def _fmt_ns(v: float | None) -> str:
    """Format a nanosecond duration as milliseconds."""
    if v is None:
        return "N/A"
    return f"{v / 1e6:.2f} ms"


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
    strat_market = comp.strategy_market_metrics

    charts_section = ""
    if summary.price_chart:
        charts_section += "## Charts\n[Price Chart attached]\n"
    if summary.spread_chart:
        charts_section += "[Spread Chart attached]\n"

    return INTERPRETER_PROMPT_TEMPLATE.format(
        goal=goal,
        # Agent metrics
        total_pnl=fmt_dollar_metric(agent.total_pnl),
        initial_cash=fmt_dollar_metric(agent.initial_cash),
        ending_cash=fmt_dollar_metric(agent.ending_cash),
        trade_count=agent.trade_count,
        fill_rate=_fmt_rate(agent.fill_rate),
        end_inventory=agent.end_inventory,
        sharpe_ratio=_fmt_float(agent.sharpe_ratio, ".3f"),
        max_drawdown=fmt_dollar_metric(agent.max_drawdown),
        inventory_std=_fmt_float(agent.inventory_std, ".1f"),
        # Market impact deltas
        volatility_delta_pct=_fmt_pct(impact.volatility_delta_pct),
        spread_delta_pct=_fmt_pct(impact.spread_delta_pct),
        bid_liquidity_delta_pct=_fmt_pct(impact.bid_liquidity_delta_pct),
        ask_liquidity_delta_pct=_fmt_pct(impact.ask_liquidity_delta_pct),
        lob_imbalance_delta_pct=_fmt_pct(impact.lob_imbalance_delta_pct),
        vpin_delta_pct=_fmt_pct(impact.vpin_delta_pct),
        resilience_delta_pct=_fmt_pct(impact.resilience_delta_pct),
        ott_ratio_delta_pct=_fmt_pct(impact.ott_ratio_delta_pct),
        # Absolute microstructure values for the strategy run
        volatility=_fmt_float(strat_market.volatility, ".6f"),
        mean_spread=fmt_dollar_metric(strat_market.mean_spread),
        effective_spread=fmt_dollar_metric(strat_market.effective_spread),
        traded_volume=str(strat_market.traded_volume) if strat_market.traded_volume is not None else "N/A",
        lob_imbalance_mean=_fmt_float(strat_market.lob_imbalance_mean, ".4f"),
        lob_imbalance_std=_fmt_float(strat_market.lob_imbalance_std, ".4f"),
        vpin=_fmt_float(strat_market.vpin, ".4f"),
        resilience_mean=_fmt_ns(strat_market.resilience_mean_ns),
        market_ott_ratio=_fmt_float(strat_market.market_ott_ratio, ".2f"),
        charts_section=charts_section,
    )
