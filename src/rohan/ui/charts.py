"""Plotly figure builders for the Rohan financial dashboard.

Each function accepts pre-computed data (Series, DataFrames, scalars) and
returns a ``plotly.graph_objects.Figure``.  No Streamlit imports here.

Section A: Market microstructure charts (ported from abides-ui)
Section B: Rohan-specific analysis charts
Section C: Refinement Lab charts
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from rohan.ui.utils.theme import HEIGHT_PRIMARY, HEIGHT_SECONDARY, PALETTE, SERIES_COLORS, apply_fin_theme

# ── Section A: Market Overview ────────────────────────────────────────────────


def price_series(time: pd.Series, bid: pd.Series, ask: pd.Series, mid: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=bid, mode="lines", name="Best Bid", line={"color": PALETTE["market"], "width": 1}))
    fig.add_trace(go.Scatter(x=time, y=ask, mode="lines", name="Best Ask", line={"color": PALETTE["hft"], "width": 1}))
    fig.add_trace(go.Scatter(x=time, y=mid, mode="lines", name="Mid Price", line={"color": PALETTE["institutional"], "width": 2}))
    fig.update_layout(title="Price Series (Bid / Ask / Mid)", xaxis_title="Time", yaxis_title="Price ($)", height=HEIGHT_PRIMARY)
    return apply_fin_theme(fig)


def spread_over_time(time: pd.Series, spread: pd.Series, avg_spread: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=spread, mode="lines", name="Spread", fill="tozeroy", line={"color": PALETTE["warning"], "width": 1}, fillcolor="rgba(255, 165, 0, 0.12)"))
    fig.add_hline(y=avg_spread, line_dash="dash", line_color=PALETTE["text_dim"], annotation_text=f"Mean: ${avg_spread:.4f}")
    fig.update_layout(title="Bid-Ask Spread Over Time", xaxis_title="Time", yaxis_title="Spread ($)", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


# ── Section A: Microstructure ─────────────────────────────────────────────────


def rolling_volatility(ret_time: pd.Series, rolling_vol: pd.Series, window: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ret_time, y=rolling_vol, mode="lines", name=f"Rolling σ ({window}-tick)", line={"color": "#9467bd", "width": 1.5}))
    fig.update_layout(title=f"Rolling Realized Volatility ({window}-tick window)", xaxis_title="Time", yaxis_title="σ (log returns)", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


def book_pressure(time: pd.Series, pressure: pd.Series) -> go.Figure:
    colors = [PALETTE["market"] if v >= 0 else PALETTE["hft"] for v in pressure]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=time, y=pressure, name="Bid − Ask Qty", marker_color=colors))
    fig.update_layout(title="Order Book Pressure (Bid Qty − Ask Qty)", xaxis_title="Time", yaxis_title="Qty Imbalance", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


def returns_histogram(log_returns: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=log_returns, nbinsx=50, name="Log Returns", marker_color=PALETTE["institutional"], opacity=0.7))
    fig.update_layout(title="Distribution of Log Returns (Mid-Price)", xaxis_title="Log Return", yaxis_title="Frequency", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


# ── Section A: Order Flow ─────────────────────────────────────────────────────


def event_type_pie(event_counts: pd.Series) -> go.Figure:
    fig = go.Figure(data=[go.Pie(labels=event_counts.index.tolist(), values=event_counts.values.tolist(), hole=0.4, marker={"colors": SERIES_COLORS[: len(event_counts)]})])
    fig.update_layout(title="Order Event Types", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


def side_balance(side_counts: pd.Series) -> go.Figure:
    fig = go.Figure(data=[go.Bar(x=side_counts.index.tolist(), y=side_counts.values.tolist(), marker_color=[PALETTE["market"] if "BID" in s else PALETTE["hft"] for s in side_counts.index])])
    fig.update_layout(title="Order Side Balance (Submitted)", xaxis_title="Side", yaxis_title="Count", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


def cumulative_imbalance(flow_time: pd.Series, cum_imbalance: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=flow_time, y=cum_imbalance, mode="lines", name="Cumulative Imbalance", line={"color": "#17becf", "width": 1.5}, fill="tozeroy", fillcolor="rgba(23, 190, 207, 0.10)"))
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["text_dim"])
    fig.update_layout(title="Cumulative Order Flow Imbalance (Buy − Sell)", xaxis_title="Time", yaxis_title="Cumulative Imbalance", height=HEIGHT_PRIMARY)
    return apply_fin_theme(fig)


def volume_by_agent_type(vol_by_type: pd.Series) -> go.Figure:
    fig = go.Figure(data=[go.Bar(x=vol_by_type.values.tolist(), y=vol_by_type.index.tolist(), orientation="h", marker_color=PALETTE["institutional"])])
    fig.update_layout(
        title="Executed Volume by Agent Type",
        xaxis_title="Total Quantity Executed",
        yaxis_title="Agent Type",
        height=max(250, len(vol_by_type) * 50),
        margin={"l": 150, "r": 16, "t": 44, "b": 32},
    )
    return apply_fin_theme(fig)


# ── Section A: Agent Analytics ────────────────────────────────────────────────


def pnl_box_plot(agent_df: pd.DataFrame) -> go.Figure:
    agent_type_list = sorted(agent_df["Type"].unique())
    fig = go.Figure()
    for i, atype in enumerate(agent_type_list):
        subset = agent_df[agent_df["Type"] == atype]["P&L ($)"]
        fig.add_trace(go.Box(y=subset, name=atype, marker_color=SERIES_COLORS[i % len(SERIES_COLORS)], boxmean="sd"))
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["text_dim"])
    fig.update_layout(title="P&L Distribution by Agent Type", yaxis_title="P&L ($)", height=HEIGHT_PRIMARY)
    return apply_fin_theme(fig)


# ── Section A: Execution Analytics ───────────────────────────────────────────


def equity_curve(ec_df: pd.DataFrame, agent_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ec_df["time"], y=ec_df["Peak NAV ($)"], mode="lines", name="Peak NAV", line={"color": "rgba(100,100,100,0.4)", "width": 1, "dash": "dot"}))
    fig.add_trace(go.Scatter(x=ec_df["time"], y=ec_df["NAV ($)"], mode="lines", name="NAV", line={"color": PALETTE["institutional"], "width": 2}, fill="tonexty", fillcolor="rgba(255, 59, 63, 0.06)"))
    fig.update_layout(title=f"Equity Curve — {agent_name}", xaxis_title="Time", yaxis_title="NAV ($)", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


def slippage_comparison(exec_agents_data: list[dict[str, Any]]) -> go.Figure:
    names = [d["name"] for d in exec_agents_data]
    slippages = [d["vwap_slippage_bps"] for d in exec_agents_data]
    colors = [PALETTE["market"] if s <= 0 else PALETTE["hft"] for s in slippages]
    fig = go.Figure(data=[go.Bar(x=names, y=slippages, marker_color=colors)])
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["text_dim"])
    fig.update_layout(title="VWAP Slippage by Execution Agent", xaxis_title="Agent", yaxis_title="Slippage (bps)", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


# ── Section A: Trade Attribution ──────────────────────────────────────────────


def maker_taker_volume(maker_vol: pd.Series, taker_vol: pd.Series) -> go.Figure:
    all_types = sorted(set(maker_vol.index) | set(taker_vol.index))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=all_types, y=[int(maker_vol.get(t, 0)) for t in all_types], name="Maker (passive)", marker_color=PALETTE["market"]))
    fig.add_trace(go.Bar(x=all_types, y=[int(taker_vol.get(t, 0)) for t in all_types], name="Taker (aggressive)", marker_color=PALETTE["hft"]))
    fig.update_layout(title="Trade Volume: Maker vs Taker by Agent Type", xaxis_title="Agent Type", yaxis_title="Volume (shares)", barmode="group", height=HEIGHT_PRIMARY)
    return apply_fin_theme(fig)


def trade_price_scatter(attr_df: pd.DataFrame) -> go.Figure:
    buys = attr_df[attr_df["side"].str.upper().str.contains("BID|BUY")]
    sells = attr_df[~attr_df.index.isin(buys.index)]
    fig = go.Figure()
    if len(buys) > 0:
        fig.add_trace(go.Scattergl(x=buys["time"], y=buys["price ($)"], mode="markers", name="Buy", marker={"color": PALETTE["market"], "size": 3, "opacity": 0.6}))
    if len(sells) > 0:
        fig.add_trace(go.Scattergl(x=sells["time"], y=sells["price ($)"], mode="markers", name="Sell", marker={"color": PALETTE["hft"], "size": 3, "opacity": 0.6}))
    fig.update_layout(title="Trade Prices Over Time (by Side)", xaxis_title="Time", yaxis_title="Price ($)", height=HEIGHT_PRIMARY)
    return apply_fin_theme(fig)


def trade_attribution_heatmap(attr_df: pd.DataFrame) -> go.Figure:
    """Heatmap of trade volume flowing from maker type → taker type."""
    if attr_df.empty or "maker_type" not in attr_df.columns or "taker_type" not in attr_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Trade Attribution Heatmap", height=HEIGHT_PRIMARY)
        return apply_fin_theme(fig)

    pivot = attr_df.pivot_table(index="maker_type", columns="taker_type", values="quantity", aggfunc="sum", fill_value=0)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="Blues",
            text=pivot.values,
            texttemplate="%{text:,.0f}",
            hovertemplate="Maker: %{y}<br>Taker: %{x}<br>Volume: %{z:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Trade Flow: Maker → Taker (Volume)",
        xaxis_title="Taker (Aggressive)",
        yaxis_title="Maker (Passive)",
        height=HEIGHT_PRIMARY,
    )
    return apply_fin_theme(fig)


# ── Section A: L2 Order Book Depth ────────────────────────────────────────────


def l2_depth_heatmap(l2_df: pd.DataFrame, mid: pd.Series | None = None) -> go.Figure:
    bids = l2_df[l2_df["side"] == "bid"].copy()
    asks = l2_df[l2_df["side"] == "ask"].copy()

    for df in (bids, asks):
        df["time"] = pd.to_datetime(df["time_ns"], unit="ns")
        df["price ($)"] = df["price_cents"] / 100

    bid_pivot = bids.pivot_table(index="price ($)", columns="time", values="qty", aggfunc="sum", fill_value=0)
    ask_pivot = asks.pivot_table(index="price ($)", columns="time", values="qty", aggfunc="sum", fill_value=0)

    fig = go.Figure()
    if len(bid_pivot) > 0:
        fig.add_trace(
            go.Heatmap(
                x=bid_pivot.columns,
                y=bid_pivot.index,
                z=bid_pivot.values,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, PALETTE["market"]]],
                name="Bids",
                showscale=False,
                hovertemplate="Time: %{x}<br>Price: $%{y:.2f}<br>Qty: %{z}<extra>Bid</extra>",
            )
        )
    if len(ask_pivot) > 0:
        fig.add_trace(
            go.Heatmap(
                x=ask_pivot.columns,
                y=ask_pivot.index,
                z=ask_pivot.values,
                colorscale=[[0, "rgba(0,0,0,0)"], [1, PALETTE["hft"]]],
                name="Asks",
                showscale=False,
                hovertemplate="Time: %{x}<br>Price: $%{y:.2f}<br>Qty: %{z}<extra>Ask</extra>",
            )
        )
    if mid is not None and len(mid.dropna()) > 0:
        mid_time = pd.to_datetime(np.sort(l2_df["time_ns"].unique()), unit="ns")
        mid_resampled = mid.iloc[: len(mid_time)] if len(mid) >= len(mid_time) else mid
        fig.add_trace(go.Scatter(x=mid_time[: len(mid_resampled)], y=mid_resampled.values, mode="lines", name="Mid Price", line={"color": PALETTE["institutional"], "width": 2}))

    fig.update_layout(title="Order Book Depth Heatmap (L2)", xaxis_title="Time", yaxis_title="Price ($)", height=HEIGHT_PRIMARY + 80)
    return apply_fin_theme(fig)


def l2_depth_profile(l2_df: pd.DataFrame) -> go.Figure:
    bids = l2_df[l2_df["side"] == "bid"]
    asks = l2_df[l2_df["side"] == "ask"]
    bid_levels = bids.groupby("level")["qty"].mean().sort_index()
    ask_levels = asks.groupby("level")["qty"].mean().sort_index()

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[-v for v in bid_levels.values], y=[f"Bid L{i}" for i in bid_levels.index], orientation="h", name="Bid Depth", marker_color=PALETTE["market"]))
    fig.add_trace(go.Bar(x=ask_levels.values.tolist(), y=[f"Ask L{i}" for i in ask_levels.index], orientation="h", name="Ask Depth", marker_color=PALETTE["hft"]))
    fig.update_layout(title="Average Depth Profile by Level", xaxis_title="Avg Quantity (negative = bids)", yaxis_title="Book Level", height=HEIGHT_SECONDARY, barmode="relative")
    return apply_fin_theme(fig)


# ── Section A: Rich Metrics ───────────────────────────────────────────────────


def adverse_selection_by_window(fill_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of mean adverse selection at each look-ahead window.

    Expects columns like ``AS 100ms (bps)``, ``AS 500ms (bps)``, etc.
    """
    as_cols = [c for c in fill_df.columns if c.startswith("AS ") and c.endswith("(bps)")]
    if not as_cols:
        fig = go.Figure()
        fig.update_layout(title="Adverse Selection by Window", height=HEIGHT_SECONDARY)
        return apply_fin_theme(fig)

    means = {c: fill_df[c].dropna().mean() for c in as_cols}
    medians = {c: fill_df[c].dropna().median() for c in as_cols}
    labels = [c.replace("AS ", "").replace(" (bps)", "") for c in as_cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[means[c] for c in as_cols], name="Mean", marker_color=PALETTE["warning"]))
    fig.add_trace(go.Bar(x=labels, y=[medians[c] for c in as_cols], name="Median", marker_color=PALETTE["institutional"]))
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["text_dim"])
    fig.update_layout(
        title="Adverse Selection by Look-Ahead Window",
        xaxis_title="Window",
        yaxis_title="Adverse Selection (bps)",
        barmode="group",
        height=HEIGHT_SECONDARY,
    )
    return apply_fin_theme(fig)


def fill_slippage_histogram(fill_df: pd.DataFrame) -> go.Figure:
    """Histogram of per-fill slippage in basis points."""
    if "slippage (bps)" not in fill_df.columns:
        fig = go.Figure()
        fig.update_layout(title="Fill Slippage Distribution", height=HEIGHT_SECONDARY)
        return apply_fin_theme(fig)

    slippage = fill_df["slippage (bps)"].dropna()
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=slippage, nbinsx=40, name="Slippage (bps)", marker_color=PALETTE["warning"], opacity=0.75))
    fig.add_vline(x=0, line_dash="dash", line_color=PALETTE["text_dim"])
    if len(slippage) > 0:
        mean_slip = float(slippage.mean())
        fig.add_vline(x=mean_slip, line_dash="dot", line_color=PALETTE["hft"], annotation_text=f"Mean: {mean_slip:.1f} bps")
    fig.update_layout(title="Per-Fill Slippage Distribution", xaxis_title="Slippage (bps)", yaxis_title="Frequency", height=HEIGHT_SECONDARY)
    return apply_fin_theme(fig)


def rich_agent_comparison(rich_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart comparing key metrics across agent types."""
    if rich_df.empty:
        fig = go.Figure()
        fig.update_layout(title="Agent Metrics Comparison", height=HEIGHT_SECONDARY)
        return apply_fin_theme(fig)

    agg_cols: dict[str, tuple[str, str]] = {}
    if "Sharpe" in rich_df.columns:
        agg_cols["Sharpe"] = ("Sharpe", "mean")
    if "OTT Ratio" in rich_df.columns:
        agg_cols["OTT Ratio"] = ("OTT Ratio", "mean")
    if "Inventory σ" in rich_df.columns:
        agg_cols["Inventory σ"] = ("Inventory σ", "mean")
    if "Trade Count" in rich_df.columns:
        agg_cols["Trades"] = ("Trade Count", "sum")
    if not agg_cols:
        fig = go.Figure()
        fig.update_layout(title="Agent Metrics Comparison", height=HEIGHT_SECONDARY)
        return apply_fin_theme(fig)

    grouped = rich_df.groupby("Type").agg(**agg_cols).reset_index()
    fig = go.Figure()
    bar_colors = [PALETTE["institutional"], PALETTE["market"], PALETTE["hft"], PALETTE["warning"]]
    for i, col in enumerate(c for c in agg_cols if c in grouped.columns):
        fig.add_trace(go.Bar(x=grouped["Type"], y=grouped[col], name=col, marker_color=bar_colors[i % len(bar_colors)]))
    fig.update_layout(title="Rich Agent Metrics by Type", xaxis_title="Agent Type", barmode="group", height=HEIGHT_PRIMARY)
    return apply_fin_theme(fig)


# ── Section B: Rohan Price / Returns ─────────────────────────────────────────


def price_returns_chart(price_df: pd.DataFrame, historical_df: pd.DataFrame | None = None) -> go.Figure:
    """Price series (bid/ask/mid) with returns panel and optional historical overlay.

    *price_df* must have columns: bid_price, ask_price, mid_price, returns (index = time).
    *historical_df* (optional) must have columns: timestamp, historical_price_cents.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Bid, Ask, and Mid Prices", "Price Returns"),
        row_heights=[0.7, 0.3],
    )

    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["bid_price"], name="Bid", line={"color": PALETTE["market"], "width": 1}, opacity=0.7), row=1, col=1)

    if historical_df is not None and not historical_df.empty:
        fig.add_trace(
            go.Scatter(x=historical_df["timestamp"], y=historical_df["historical_price_cents"], name="Historical", line={"color": PALETTE["warning"], "width": 2, "dash": "dot"}),
            row=1,
            col=1,
        )

    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["ask_price"], name="Ask", line={"color": PALETTE["hft"], "width": 1}, opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["mid_price"], name="Mid", line={"color": PALETTE["institutional"], "width": 2}), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_df.index, y=price_df["returns"], name="Returns", line={"color": PALETTE["warning"], "width": 1}, fill="tozeroy", opacity=0.6), row=2, col=1)

    fig.update_xaxes(title_text="Time Index", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Returns", row=2, col=1)
    fig.update_layout(height=700)
    return apply_fin_theme(fig)


# ── Section B: Rohan Volume / Imbalance ──────────────────────────────────────


def volume_imbalance_chart(volume_df: pd.DataFrame) -> go.Figure:
    """Bid/ask volume and volume imbalance in two stacked panels.

    *volume_df* must have columns: bid_qty, ask_qty, volume_imbalance (index = time).
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Bid and Ask Volume", "Volume Imbalance"),
        row_heights=[0.6, 0.4],
    )

    fig.add_trace(go.Scatter(x=volume_df.index, y=volume_df["bid_qty"], name="Bid Volume", line={"color": PALETTE["market"], "width": 1.5}, fill="tozeroy", opacity=0.6), row=1, col=1)
    fig.add_trace(go.Scatter(x=volume_df.index, y=volume_df["ask_qty"], name="Ask Volume", line={"color": PALETTE["hft"], "width": 1.5}, fill="tozeroy", opacity=0.6), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=volume_df.index, y=volume_df["volume_imbalance"], name="Volume Imbalance", line={"color": PALETTE["institutional"], "width": 2}, fill="tozeroy", opacity=0.7),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["text_dim"], row=2, col=1)  # pyright: ignore[reportArgumentType]

    fig.update_xaxes(title_text="Time Index", row=2, col=1)
    fig.update_yaxes(title_text="Quantity", row=1, col=1)
    fig.update_yaxes(title_text="Imbalance", row=2, col=1)
    fig.update_layout(height=700)
    return apply_fin_theme(fig)


# ── Section B: Rohan Spread Analysis ─────────────────────────────────────────


def spread_analysis_chart(spread_df: pd.DataFrame) -> go.Figure:
    """Four-panel spread analysis: time series, distribution, bps, scatter vs price.

    *spread_df* must have columns: spread, spread_bps, mid_price (index = time).
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Spread Over Time", "Spread Distribution", "Spread (Basis Points)", "Spread vs Mid Price"),
        specs=[[{"type": "scatter"}, {"type": "histogram"}], [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    fig.add_trace(go.Scatter(x=spread_df.index, y=spread_df["spread"], name="Spread", line={"color": PALETTE["institutional"], "width": 1.5}, fill="tozeroy", opacity=0.7), row=1, col=1)
    fig.add_trace(go.Histogram(x=spread_df["spread"], name="Distribution", marker={"color": PALETTE["warning"]}, opacity=0.7, nbinsx=50), row=1, col=2)
    fig.add_trace(go.Scatter(x=spread_df.index, y=spread_df["spread_bps"], name="Spread (bps)", line={"color": PALETTE["market"], "width": 1.5}, fill="tozeroy", opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=spread_df["mid_price"], y=spread_df["spread"], mode="markers", name="Spread vs Price", marker={"color": PALETTE["hft"], "size": 3, "opacity": 0.5}), row=2, col=2)

    fig.update_layout(height=800, showlegend=False)
    return apply_fin_theme(fig)


# ── Section B: CSV Preview ────────────────────────────────────────────────────


def csv_preview_chart(preview_df: pd.DataFrame, price_col: str) -> go.Figure:
    """Minimal price sparkline used in the sidebar data-source preview.

    *preview_df* must be indexed by timestamp; *price_col* names the price column.
    """
    price_series_data = preview_df[price_col] / 100 if price_col == "price_cents" else preview_df[price_col]
    price_label = "Price ($)" if price_col == "price_cents" else "Price"

    fig = go.Figure(data=[go.Scatter(x=preview_df.index, y=price_series_data, mode="lines", name="Historical", line={"color": PALETTE["institutional"], "width": 1.4})])
    fig.update_layout(height=220, margin={"l": 8, "r": 8, "t": 8, "b": 8}, showlegend=False)
    fig.update_xaxes(showgrid=False, showticklabels=False)
    fig.update_yaxes(title_text=price_label, autorange=True, rangemode="normal", fixedrange=False)
    return apply_fin_theme(fig)


# ── Section C: Refinement Lab — Score Progression ────────────────────────────


def score_progression_chart(scores: list[float], scored_iterations: list[Any], convergence_threshold: float) -> go.Figure:
    """Composite + sub-score line chart for the refinement lab results tab.

    *scored_iterations* are IterationResult objects (must have profitability_score,
    risk_score, volatility_impact_score, spread_impact_score, liquidity_impact_score,
    execution_score attributes).
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, len(scores) + 1)), y=scores, mode="lines+markers", name="Composite", line={"color": PALETTE["institutional"], "width": 3}, marker={"size": 12}))

    _sub_cfg = [
        ("profitability_score", "Profitability", PALETTE["market"]),
        ("risk_score", "Risk", PALETTE["hft"]),
        ("volatility_impact_score", "Vol Impact", PALETTE["warning"]),
        ("spread_impact_score", "Spread Impact", "#E67E22"),
        ("liquidity_impact_score", "Liq Impact", "#1ABC9C"),
        ("execution_score", "Execution", "#9B59B6"),
    ]
    for attr, label, color in _sub_cfg:
        vals = [getattr(it, attr, None) for it in scored_iterations]
        if any(v is not None for v in vals):
            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(vals) + 1)),
                    y=vals,
                    mode="lines+markers",
                    name=label,
                    line={"color": color, "width": 1.5, "dash": "dot"},
                    marker={"size": 6, "color": color},
                    opacity=0.7,
                )
            )

    fig.add_hline(
        y=convergence_threshold,
        line_dash="dash",
        line_color=PALETTE["market"],
        annotation_text=f"Convergence ({convergence_threshold:.0f}/10)",
        annotation_position="top right",
        annotation_font_color=PALETTE["market"],
    )

    fig.update_layout(
        height=400,
        xaxis={"title": "Iteration", "dtick": 1},
        yaxis={"title": "Score", "range": [0, 10.5]},
        margin={"l": 60, "r": 30, "t": 30, "b": 50},
    )
    return apply_fin_theme(fig)


# ── Section C: Refinement Lab — Scoring Radar ────────────────────────────────


def scoring_radar_chart(axis_labels: list[str], axis_values: list[float]) -> go.Figure:
    """Polar radar chart showing the latest iteration's sub-score breakdown.

    *axis_labels* and *axis_values* must have the same length (≥ 3).
    The polygon is closed automatically.
    """
    labels_closed = axis_labels + [axis_labels[0]]
    values_closed = axis_values + [axis_values[0]]

    radar_fig = go.Figure()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            fillcolor="rgba(0, 112, 255, 0.15)",
            line={"color": PALETTE["institutional"], "width": 2},
            marker={"size": 8, "color": PALETTE["institutional"]},
            name="Latest",
        )
    )
    radar_fig.update_layout(
        height=400,
        polar={
            "bgcolor": PALETTE["surface_raised"],
            "radialaxis": {"visible": True, "range": [0, 10], "gridcolor": PALETTE["border"], "color": PALETTE["text_secondary"]},
            "angularaxis": {"gridcolor": PALETTE["border"], "color": PALETTE["text"]},
        },
        showlegend=False,
        margin={"l": 60, "r": 60, "t": 30, "b": 30},
    )
    return apply_fin_theme(radar_fig)
