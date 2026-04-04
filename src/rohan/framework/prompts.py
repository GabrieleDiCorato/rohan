"""Prompt templates for LLM agents in the Rohan framework."""

from typing import TYPE_CHECKING

from rohan.utils.formatting import fmt_dollar_metric, fmt_float, fmt_ns, fmt_pct, fmt_rate

if TYPE_CHECKING:
    from rohan.simulation import RunSummary

INTERPRETER_PROMPT_TEMPLATE = """You are analyzing the results of a trading strategy simulation.

## Goal
{goal}

## Strategy Performance

### Agent Metrics
- **Total PnL**: ${total_pnl}
- **Ending Cash**: ${ending_cash}
- **Trade Count**: {trade_count}
- **Fill Rate**: {fill_rate}
- **End Inventory**: {end_inventory}
- **Sharpe Ratio**: {sharpe_ratio}
- **Max Drawdown**: {max_drawdown}
- **Inventory Std**: {inventory_std}

### Execution Quality
- **VWAP**: {vwap}
- **Avg Fill Slippage**: {avg_slippage}
- **Adverse Selection**: {adverse_selection}
- **Adverse Selection (multi-window)**: {adverse_selection_windows}
- **Counterparty Mix**: {counterparty_mix}
- **Order Lifecycle**: {order_lifecycle}

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
- **Availability Change**: {two_sided_delta_pct}

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
- **Market Availability**: {pct_time_two_sided}

{charts_section}

## Analysis Task
1. Summarize the strategy's overall performance
2. Identify key strengths and weaknesses
3. Assess the market impact (was the strategy stabilizing or destabilizing?)
4. Comment on microstructure quality (VPIN, LOB imbalance, resilience, OTT, market availability)
5. Suggest specific improvements for the next iteration
"""


# Local aliases for backward compatibility
_fmt_pct = lambda v: fmt_pct(v, signed=True)  # noqa: E731
_fmt_rate = fmt_rate
_fmt_float = fmt_float
_fmt_ns = fmt_ns


def format_interpreter_prompt(summary: "RunSummary", goal: str = "", rich_analysis_json: str | None = None) -> str:
    """Format the interpreter prompt with RunSummary data.

    Args:
        summary: RunSummary from generate_summary().
        goal: Optional strategy goal description.
        rich_analysis_json: Optional JSON string from RichAnalysisBundle.

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

    # --- Rich analysis summary ---
    adverse_selection = "N/A"
    adverse_selection_windows = "N/A"
    counterparty_mix = "N/A"
    avg_slippage = "N/A"
    order_lifecycle = "N/A"
    vwap = "N/A"

    # VWAP from agent metrics (available even without rich analysis)
    if agent.vwap_cents is not None:
        vwap = fmt_dollar_metric(agent.vwap_cents)

    if rich_analysis_json:
        try:
            import json

            bundle = json.loads(rich_analysis_json)
            if bundle.get("adverse_selection_bps") is not None:
                adverse_selection = f"{bundle['adverse_selection_bps']:.1f} bps"
            # Multi-window adverse selection
            as_windows = bundle.get("adverse_selection_by_window", {})
            if as_windows:
                parts = [f"{label}={val:.1f}bps" for label, val in as_windows.items()]
                adverse_selection_windows = ", ".join(parts)
            cps = bundle.get("counterparty_breakdown", [])
            if cps:
                parts = [f"{cp['agent_type']}({cp['trade_count']})" for cp in cps]
                counterparty_mix = ", ".join(parts)
            fills = bundle.get("fills", [])
            slippages = [f["slippage_bps"] for f in fills if f.get("slippage_bps") is not None]
            if slippages:
                avg_slippage = f"{sum(slippages) / len(slippages):.1f} bps"
            # Order lifecycle summary
            lifecycles = bundle.get("order_lifecycle", [])
            if lifecycles:
                total = len(lifecycles)
                filled = sum(1 for o in lifecycles if o.get("status") == "filled")
                cancelled = sum(1 for o in lifecycles if o.get("status") == "cancelled")
                resting = total - filled - cancelled
                order_lifecycle = f"{filled} filled, {cancelled} cancelled, {resting} resting (of {total})"
        except Exception:
            pass

    return INTERPRETER_PROMPT_TEMPLATE.format(
        goal=goal,
        # Agent metrics
        total_pnl=fmt_dollar_metric(agent.total_pnl),
        ending_cash=fmt_dollar_metric(agent.ending_cash),
        trade_count=agent.trade_count,
        fill_rate=_fmt_rate(agent.fill_rate),
        end_inventory=agent.end_inventory,
        sharpe_ratio=_fmt_float(agent.sharpe_ratio, ".3f"),
        max_drawdown=fmt_dollar_metric(agent.max_drawdown),
        inventory_std=_fmt_float(agent.inventory_std, ".1f"),
        # Execution quality (rich analysis)
        adverse_selection=adverse_selection,
        adverse_selection_windows=adverse_selection_windows,
        counterparty_mix=counterparty_mix,
        avg_slippage=avg_slippage,
        order_lifecycle=order_lifecycle,
        vwap=vwap,
        # Market impact deltas
        volatility_delta_pct=_fmt_pct(impact.volatility_delta_pct),
        spread_delta_pct=_fmt_pct(impact.spread_delta_pct),
        bid_liquidity_delta_pct=_fmt_pct(impact.bid_liquidity_delta_pct),
        ask_liquidity_delta_pct=_fmt_pct(impact.ask_liquidity_delta_pct),
        lob_imbalance_delta_pct=_fmt_pct(impact.lob_imbalance_delta_pct),
        vpin_delta_pct=_fmt_pct(impact.vpin_delta_pct),
        resilience_delta_pct=_fmt_pct(impact.resilience_delta_pct),
        ott_ratio_delta_pct=_fmt_pct(impact.ott_ratio_delta_pct),
        two_sided_delta_pct=_fmt_pct(impact.two_sided_delta_pct),
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
        pct_time_two_sided=_fmt_rate(strat_market.pct_time_two_sided),
        charts_section=charts_section,
    )
