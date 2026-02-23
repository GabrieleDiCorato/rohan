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

from typing import Literal

DeltaColor = Literal["normal", "inverse", "off"]

# ---------------------------------------------------------------------------
# Display properties per SimulationMetrics field
# ---------------------------------------------------------------------------

METRIC_CONFIG: dict[str, dict[str, str]] = {
    "volatility": {
        "delta_color": "inverse",
        "help": ("Annualised mid-price volatility (σ). Lower values indicate a more stable market."),
    },
    "mean_spread": {
        "delta_color": "inverse",
        "help": ("Average quoted bid-ask spread (cents). Lower means tighter markets — cheaper for traders to cross."),
    },
    "effective_spread": {
        "delta_color": "inverse",
        "help": ("Average effective spread: 2×|trade−mid| (cents). Measures the actual cost of executing a trade, including market impact."),
    },
    "avg_bid_liquidity": {
        "delta_color": "normal",
        "help": ("Mean quantity resting at best bid. Higher means more depth available to sellers."),
    },
    "avg_ask_liquidity": {
        "delta_color": "normal",
        "help": ("Mean quantity resting at best ask. Higher means more depth available to buyers."),
    },
    "traded_volume": {
        "delta_color": "normal",
        "help": ("Total shares traded during the simulation. Higher indicates more market activity."),
    },
    "lob_imbalance_mean": {
        "delta_color": "off",
        "help": ("Mean L1 order-book imbalance: (Qbid−Qask)/(Qbid+Qask). 0 = balanced book, +1 = all bids, −1 = all asks."),
    },
    "lob_imbalance_std": {
        "delta_color": "off",
        "help": ("Std dev of L1 order-book imbalance over time. Lower means the bid/ask balance is more stable."),
    },
    "vpin": {
        "delta_color": "inverse",
        "help": ("Volume-synchronised probability of informed trading (Easley, López de Prado & O'Hara, 2012). Higher suggests more toxic/informed order flow. Range [0, 1]."),
    },
    "resilience_mean_ns": {
        "delta_color": "inverse",
        "help": ("Mean time for the spread to recover after a shock event. Lower means the market is more resilient."),
    },
    "market_ott_ratio": {
        "delta_color": "off",
        "help": ("Market-wide order-to-trade ratio (submissions / fills). MiFID II RTS 9 caps this at 4:1 for certain instruments."),
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
