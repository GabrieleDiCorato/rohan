"""Carbon Dark theme for the Rohan Streamlit UI.

Centralises all visual constants — colour palette, chart heights, CSS injection
and the per-figure Plotly theme helper — so every page stays presentation-agnostic.
"""

from __future__ import annotations

from typing import Literal

import plotly.graph_objects as go
import streamlit as st

# ── Semantic trading palette ──────────────────────────────────────────────────

PALETTE: dict[str, str] = {
    "hft": "#FF3B3F",
    "institutional": "#0070FF",
    "market": "#00C805",
    "warning": "#FFA500",
    "neutral": "#6B7280",
    "text": "#E0E0E0",
    "text_secondary": "#A0A8B4",
    "text_dim": "#6B7280",
    "bg": "#05070A",
    "surface": "#0B0E14",
    "surface_raised": "#111519",
    "border": "#1C2128",
}

# Categorical palette for multi-series charts (agent types, etc.)
SERIES_COLORS: list[str] = [
    "#0070FF",  # institutional blue
    "#FF3B3F",  # hft red
    "#00C805",  # market green
    "#FFA500",  # warning orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
]

# ── Chart height constants ────────────────────────────────────────────────────

HEIGHT_PRIMARY = 380  # full-width hero charts
HEIGHT_SECONDARY = 320  # half-width / supporting charts

# ── Legacy COLORS dict (bridges to the new palette so existing references work)

COLORS: dict[str, str] = {
    "background": PALETTE["bg"],
    "secondary_bg": PALETTE["surface"],
    "card_bg": PALETTE["surface_raised"],
    "primary": PALETTE["institutional"],
    "secondary": PALETTE["warning"],
    "success": PALETTE["market"],
    "danger": PALETTE["hft"],
    "text": PALETTE["text"],
    "text_muted": PALETTE["text_dim"],
    "border": PALETTE["border"],
}


# ── Global CSS ────────────────────────────────────────────────────────────────

CARBON_DARK_CSS: str = """
/* ── Google Fonts ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

/* ── Font overrides ───────────────────────────────────────────────────── */
html, body, .stApp {
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stWidgetLabel"], label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
}
input, textarea, code, pre, [data-testid="stCode"],
.stDataFrame, [data-testid="stDataFrame"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.mono-value {
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Layout tweaks ────────────────────────────────────────────────────── */
.stApp > header { background-color: transparent !important; }
.block-container, [data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 100% !important;
}

/* ── Sidebar ──────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    border-right: 1px solid #1C2128 !important;
}

/* ── Tabs ─────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: #0B0E14;
    border-bottom: 1px solid #1C2128;
    border-radius: 6px 6px 0 0;
    padding: 0 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500;
    color: #6B7280 !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: 4px 4px 0 0;
    border: none !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #0070FF !important;
    border-bottom: 2px solid #0070FF !important;
    background: rgba(0, 112, 255, 0.06) !important;
}

/* ── Expander ─────────────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    border: 1px solid #1C2128 !important;
    border-radius: 6px !important;
}

/* ── Dividers ─────────────────────────────────────────────────────────── */
hr { border-color: #1C2128 !important; opacity: 0.5; }

/* ── Scrollbar ────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #05070A; }
::-webkit-scrollbar-thumb { background: #1C2128; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6B7280; }

/* ── Hide Streamlit chrome ────────────────────────────────────────────── */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
"""

# ── Theme helpers ─────────────────────────────────────────────────────────────


def apply_theme() -> None:
    """Inject Carbon Dark CSS into the Streamlit app."""
    st.markdown(f"<style>{CARBON_DARK_CSS}</style>", unsafe_allow_html=True)


# ── Plotly theme ──────────────────────────────────────────────────────────────


def apply_fin_theme(fig: go.Figure) -> go.Figure:
    """Apply institutional-grade dark theme to any Plotly figure."""
    _grid = "rgba(255, 255, 255, 0.07)"
    _spike = "#6B7280"

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "JetBrains Mono, monospace", "color": "#E0E0E0", "size": 11},
        title_font={"family": "Inter, sans-serif", "color": "#E0E0E0", "size": 13},
        hovermode="x unified",
        hoverlabel={
            "bgcolor": "rgba(11, 14, 20, 0.92)",
            "bordercolor": "#1C2128",
            "font": {"family": "JetBrains Mono, monospace", "size": 11, "color": "#E0E0E0"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "bgcolor": "rgba(11, 14, 20, 0.6)",
            "bordercolor": "rgba(255, 255, 255, 0.06)",
            "borderwidth": 1,
            "font": {"size": 10, "color": "#8A919B"},
        },
        margin={"l": 50, "r": 16, "t": 44, "b": 32},
    )

    _axis_common = {
        "gridcolor": _grid,
        "gridwidth": 1,
        "zerolinecolor": "rgba(255, 255, 255, 0.12)",
        "zerolinewidth": 1,
        "showspikes": True,
        "spikemode": "across",
        "spikethickness": 1,
        "spikedash": "dot",
        "spikecolor": _spike,
        "spikesnap": "cursor",
        "title_font": {"family": "Inter, sans-serif", "size": 11, "color": "#8A919B"},
        "tickfont": {"family": "JetBrains Mono, monospace", "size": 10, "color": "#8A919B"},
    }
    fig.update_xaxes(**_axis_common)
    fig.update_yaxes(**_axis_common)

    return fig


# ── Backward-compatible helpers ───────────────────────────────────────────────


def create_metric_card(
    label: str,
    value: str,
    delta: str | None = None,
    delta_color: Literal["normal", "inverse", "off", "red", "orange", "yellow", "green", "blue", "violet", "gray", "grey", "primary"] = "normal",
) -> None:
    """Render a Streamlit metric widget (kept for backward compatibility)."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def create_header(title: str, subtitle: str | None = None) -> None:
    """Render a styled page header with optional subtitle."""
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(
            f"<p style='color: {COLORS['text_muted']}; font-size: 1.1rem; margin-top: -10px;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
