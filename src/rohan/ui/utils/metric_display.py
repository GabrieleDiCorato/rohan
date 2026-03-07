"""Metric display configuration and delta helpers for the Terminal UI.

Maps each SimulationMetrics field to its display properties:
- help:        tooltip text for Streamlit's ``st.metric(help=...)``
- delta_color: "normal" (green when ↑), "inverse" (green when ↓), "off" (grey)

The semantics are grounded in market microstructure:
- Spreads, VPIN, volatility → lower is better for market quality → "inverse"
- Liquidity, volume → higher is better → "normal"
- Imbalance, OTT → context-dependent → "off"
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pandas as pd

DeltaColor = Literal["normal", "inverse", "off"]

# ---------------------------------------------------------------------------
# Display properties per SimulationMetrics field
# ---------------------------------------------------------------------------

METRIC_CONFIG: dict[str, dict[str, str]] = {
    "volatility": {
        "delta_color": "inverse",
        "help": (
            "Annualised mid-price volatility (σ).\n\n"
            "**Formula:** σ = std(log returns) × √(trading periods per year).\n\n"
            "**Reading it:** Lower → more stable market. "
            "A spike suggests sudden price swings or thin liquidity. "
            "Typical equity-sim values: 0.05–0.40."
        ),
    },
    "mean_spread": {
        "delta_color": "inverse",
        "help": (
            "Average quoted bid-ask spread, in cents.\n\n"
            "**Formula:** mean(best ask − best bid) across all snapshots.\n\n"
            "**Reading it:** Lower → tighter market, cheaper to trade. "
            "Wider spreads mean higher implicit transaction costs for every participant."
        ),
    },
    "effective_spread": {
        "delta_color": "inverse",
        "help": (
            "Average effective spread, in cents.\n\n"
            "**Formula:** 2 × |trade price − midpoint| averaged over all fills.\n\n"
            "**Reading it:** Captures the *actual* cost of executing a trade "
            "(including market impact), whereas the quoted spread only shows "
            "resting orders. Lower is better for market quality."
        ),
    },
    "avg_bid_liquidity": {
        "delta_color": "normal",
        "help": (
            "Mean quantity resting at the best bid level.\n\n**Reading it:** Higher → more depth available to sellers, reducing slippage on sell orders. A drop may indicate market-maker withdrawal."
        ),
    },
    "avg_ask_liquidity": {
        "delta_color": "normal",
        "help": (
            "Mean quantity resting at the best ask level.\n\n**Reading it:** Higher → more depth available to buyers, reducing slippage on buy orders. A drop may indicate market-maker withdrawal."
        ),
    },
    "traded_volume": {
        "delta_color": "normal",
        "help": (
            "Total number of shares traded during the simulation.\n\n"
            "**Reading it:** Higher → more market activity and liquidity turnover. "
            "Very low volume may indicate the order book is too thin or agents "
            "are not interacting."
        ),
    },
    "lob_imbalance_mean": {
        "delta_color": "off",
        "help": (
            "Mean Level-1 order-book imbalance.\n\n"
            "**Formula:** mean[(Qbid − Qask) / (Qbid + Qask)].\n\n"
            "**Range:** −1 (all asks) to +1 (all bids); 0 = perfectly balanced. "
            "Persistent positive values hint at buying pressure; persistent "
            "negative values hint at selling pressure."
        ),
    },
    "lob_imbalance_std": {
        "delta_color": "off",
        "help": (
            "Standard deviation of L1 order-book imbalance over time.\n\n"
            "**Reading it:** Lower → the bid/ask balance stays more stable "
            "throughout the session. High values suggest the book "
            "oscillates heavily between buy- and sell-heavy states."
        ),
    },
    "vpin": {
        "delta_color": "inverse",
        "help": (
            "Volume-Synchronised Probability of Informed Trading.\n\n"
            "**Reference:** Easley, López de Prado & O'Hara (2012).\n\n"
            "**Range:** 0 (no informed flow) to 1 (fully informed). "
            "Higher → more toxic / informed order flow. A reading above "
            "~0.5 in equity markets is considered elevated."
        ),
    },
    "resilience_mean_ns": {
        "delta_color": "inverse",
        "help": (
            "Mean spread-recovery time after a shock event (displayed in ms).\n\n"
            "**Reading it:** Lower → the market snaps back to normal spreads "
            "faster after a large trade or sweep. Higher → sluggish recovery, "
            "indicative of thin liquidity or slow market-maker replenishment."
        ),
    },
    "market_ott_ratio": {
        "delta_color": "off",
        "help": (
            "Market-wide Order-to-Trade ratio.\n\n"
            "**Formula:** total order submissions / total fills.\n\n"
            "**Reading it:** An excessively high OTT (≫ 4) may signal "
            "noise/spam quoting; very low OTT (≈ 1) means almost every order "
            "fills immediately. MiFID II RTS 9 caps OTT at 4:1 for certain "
            "instruments."
        ),
    },
}

# ---------------------------------------------------------------------------
# Scoring-axis descriptions (LLM judge sub-scores)
# ---------------------------------------------------------------------------

SCORING_AXIS_CONFIG: dict[str, dict[str, str]] = {
    "profitability": {
        "help": (
            "**Profitability** — absolute PnL on $100K starting capital.\n\n"
            "Default weight: **40%** of composite score.\n\n"
            "• 1–2: Net loss or zero trades\n"
            "• 3–4: $0 – $200 profit\n"
            "• 5–6: $200 – $1,500\n"
            "• 7–8: $1,500 – $5,000\n"
            "• 9–10: > $5,000"
        ),
    },
    "risk": {
        "help": (
            "**Risk-Adjusted Performance** — Sharpe ratio, drawdown & inventory risk.\n\n"
            "Default weight: **25%** of composite score.\n\n"
            "• 1–2: Sharpe < 0 or max drawdown > 50%\n"
            "• 3–4: Sharpe 0 – 0.5\n"
            "• 5–6: Sharpe 0.5 – 1.5\n"
            "• 7–8: Sharpe 1.5 – 3.0\n"
            "• 9–10: Sharpe > 3.0, drawdown < 10%"
        ),
    },
    "impact": {
        "help": (
            "**Market Impact** — stabilising vs destabilising effect on the market.\n\n"
            "Default weight: **15%** of composite score.\n\n"
            "• 1–2: Volatility ↑ > 10% AND spread ↑ > 10%\n"
            "• 3–4: Slight destabilisation (5–10%)\n"
            "• 5–6: Neutral (± 5%)\n"
            "• 7–8: Mildly stabilising (5–15% reduction)\n"
            "• 9–10: Strongly stabilising (> 15% reduction)"
        ),
    },
    "execution": {
        "help": (
            "**Execution Quality** — fill rate, OTT ratio & order management.\n\n"
            "Default weight: **20%** of composite score.\n\n"
            "• 1–2: Fill rate < 5% or OTT > 50\n"
            "• 3–4: Fill rate 5–15%, OTT 20–50\n"
            "• 5–6: Fill rate 15–30%, OTT 10–20\n"
            "• 7–8: Fill rate 30–50%, OTT 5–10\n"
            "• 9–10: Fill rate > 50%, OTT < 5"
        ),
    },
}


# ---------------------------------------------------------------------------
# Delta computation helpers
# ---------------------------------------------------------------------------


def metric_delta(
    current: float | int | None,
    previous: float | int | None,
) -> float | None:
    """Raw delta between two metric values.  ``None`` if either is missing."""
    if current is None or previous is None:
        return None
    return current - previous  # type: ignore[return-value]


def pct_delta(
    current: float | int | None,
    previous: float | int | None,
) -> float | None:
    """Percentage change ``(current − previous) / previous``.

    Returns ``None`` when undefined (either value missing, or previous is
    zero and current is nonzero).
    """
    if current is None or previous is None:
        return None
    if previous == 0:
        return 0.0 if current == 0 else None
    return (current - previous) / previous  # type: ignore[operator]


def fmt_pct(pct: float | None) -> str | None:
    """Format a percentage for ``st.metric(delta=...)``, e.g. ``'+12.3%'``."""
    if pct is None:
        return None
    return f"{pct:+.1%}"


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_delta_color(field: str) -> DeltaColor:
    """Delta-color for a metric field (``'normal'``, ``'inverse'``, or ``'off'``)."""
    value = METRIC_CONFIG.get(field, {}).get("delta_color", "off")
    assert value in ("normal", "inverse", "off")  # noqa: S101
    return value  # type: ignore[return-value]


def get_help(field: str) -> str | None:
    """Tooltip help text for a metric field, or ``None`` if unknown."""
    return METRIC_CONFIG.get(field, {}).get("help")


def get_scoring_help(axis: str) -> str | None:
    """Tooltip help text for a scoring sub-axis (profitability/risk/impact/execution)."""
    return SCORING_AXIS_CONFIG.get(axis, {}).get("help")


# ---------------------------------------------------------------------------
# Baseline comparison table builder
# ---------------------------------------------------------------------------

# Ordered display entries:  (field_name, display_label, format_spec)
_COMPARISON_FIELDS: list[tuple[str, str, str]] = [
    ("volatility", "Volatility (σ)", ".4f"),
    ("mean_spread", "Mean Spread (¢)", ".4f"),
    ("effective_spread", "Eff. Spread (¢)", ".4f"),
    ("avg_bid_liquidity", "Avg Bid Liquidity", ".2f"),
    ("avg_ask_liquidity", "Avg Ask Liquidity", ".2f"),
    ("traded_volume", "Traded Volume", ",.0f"),
    ("vpin", "VPIN", ".4f"),
]


def build_comparison_table(
    current_metrics: object,
    baseline_metrics: object,
    *,
    fields: list[tuple[str, str, str]] | None = None,
) -> pd.DataFrame:
    """Build a Current vs Baseline comparison DataFrame.

    Parameters
    ----------
    current_metrics, baseline_metrics:
        Objects whose attributes match the *field* names (e.g.
        ``SimulationMetrics``).
    fields:
        Override the default list of ``(field, label, fmt)`` tuples.

    Returns a DataFrame with columns **Metric**, **Current**, **Baseline**,
    **Δ%**.
    """
    import pandas as pd

    entries = fields or _COMPARISON_FIELDS
    rows: list[dict[str, str]] = []
    for field, label, fmt in entries:
        cur_val = getattr(current_metrics, field, None)
        bl_val = getattr(baseline_metrics, field, None)

        cur_str = f"{cur_val:{fmt}}" if cur_val is not None else "N/A"
        bl_str = f"{bl_val:{fmt}}" if bl_val is not None else "N/A"
        pct = pct_delta(cur_val, bl_val)
        pct_str: str = fmt_pct(pct) or "N/A"

        rows.append({"Metric": label, "Current": cur_str, "Baseline": bl_str, "Δ%": pct_str})

    return pd.DataFrame(rows)
