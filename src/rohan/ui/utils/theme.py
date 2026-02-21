"""Bloomberg Terminal-inspired theme for Streamlit application."""

from typing import Literal

import streamlit as st

# Bloomberg Terminal Color Palette
COLORS = {
    "background": "#0A0E27",
    "secondary_bg": "#131829",
    "card_bg": "#1A1F3A",
    "primary": "#00D9FF",  # Cyan
    "secondary": "#FFB800",  # Amber
    "success": "#00FF88",
    "danger": "#FF3366",
    "text": "#E8E8E8",
    "text_muted": "#8B92A8",
    "border": "#2A3150",
}


def apply_theme():
    """Apply Bloomberg Terminal-inspired theme to Streamlit app."""
    st.markdown(
        f"""
        <style>
        /* Global Styles */
        .stApp {{
            background-color: {COLORS["background"]};
            color: {COLORS["text"]};
            font-family: 'Courier New', monospace;
        }}

        /* Sidebar */
        [data-testid="stSidebar"] {{
            background-color: {COLORS["secondary_bg"]};
            border-right: 2px solid {COLORS["border"]};
        }}

        [data-testid="stSidebar"] .stMarkdown {{
            color: {COLORS["text"]};
        }}

        /* Headers */
        h1, h2, h3 {{
            color: {COLORS["primary"]};
            font-family: 'Courier New', monospace;
            font-weight: 700;
            letter-spacing: 1px;
        }}

        h1 {{
            border-bottom: 3px solid {COLORS["primary"]};
            padding-bottom: 10px;
        }}

        /* Cards/Containers */
        .element-container {{
            color: {COLORS["text"]};
        }}

        /* Metrics */
        [data-testid="stMetricValue"] {{
            color: {COLORS["primary"]};
            font-size: 2rem;
            font-weight: bold;
        }}

        [data-testid="stMetricLabel"] {{
            color: {COLORS["text_muted"]};
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 1px;
        }}

        /* Buttons */
        .stButton > button {{
            background-color: {COLORS["primary"]};
            color: {COLORS["background"]};
            border: none;
            border-radius: 4px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
            transition: all 0.3s ease;
        }}

        .stButton > button:hover {{
            background-color: {COLORS["secondary"]};
            box-shadow: 0 0 20px {COLORS["primary"]};
        }}

        /* Input Fields */
        .stTextInput > div > div > input,
        .stNumberInput > div > div > input,
        .stSelectbox > div > div > select,
        .stTextArea > div > div > textarea {{
            background-color: {COLORS["card_bg"]};
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            font-family: 'Courier New', monospace;
        }}

        .stTextInput > div > div > input:focus,
        .stNumberInput > div > div > input:focus,
        .stSelectbox > div > div > select:focus,
        .stTextArea > div > div > textarea:focus {{
            border-color: {COLORS["primary"]};
            box-shadow: 0 0 10px {COLORS["primary"]}33;
        }}

        /* Labels */
        .stTextInput > label,
        .stNumberInput > label,
        .stSelectbox > label,
        .stTextArea > label,
        .stCheckbox > label {{
            color: {COLORS["text_muted"]};
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            background-color: {COLORS["secondary_bg"]};
        }}

        .stTabs [data-baseweb="tab"] {{
            background-color: {COLORS["card_bg"]};
            color: {COLORS["text_muted"]};
            border-radius: 4px 4px 0 0;
            padding: 10px 20px;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: {COLORS["primary"]};
            color: {COLORS["background"]};
        }}

        /* Expander */
        .streamlit-expanderHeader {{
            background-color: {COLORS["card_bg"]};
            color: {COLORS["primary"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .streamlit-expanderContent {{
            background-color: {COLORS["secondary_bg"]};
            border: 1px solid {COLORS["border"]};
            border-top: none;
        }}

        /* DataFrames */
        .dataframe {{
            background-color: {COLORS["card_bg"]};
            color: {COLORS["text"]};
            font-family: 'Courier New', monospace;
            font-size: 0.85rem;
        }}

        .dataframe th {{
            background-color: {COLORS["secondary_bg"]};
            color: {COLORS["primary"]};
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: bold;
        }}

        /* Success/Error Messages */
        .stSuccess {{
            background-color: {COLORS["success"]}22;
            color: {COLORS["success"]};
            border-left: 4px solid {COLORS["success"]};
        }}

        .stError {{
            background-color: {COLORS["danger"]}22;
            color: {COLORS["danger"]};
            border-left: 4px solid {COLORS["danger"]};
        }}

        .stWarning {{
            background-color: {COLORS["secondary"]}22;
            color: {COLORS["secondary"]};
            border-left: 4px solid {COLORS["secondary"]};
        }}

        .stInfo {{
            background-color: {COLORS["primary"]}22;
            color: {COLORS["primary"]};
            border-left: 4px solid {COLORS["primary"]};
        }}

        /* Progress Bar */
        .stProgress > div > div > div > div {{
            background-color: {COLORS["primary"]};
        }}

        /* Divider */
        hr {{
            border-color: {COLORS["border"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def create_metric_card(
    label: str, value: str, delta: str | None = None, delta_color: Literal["normal", "inverse", "off", "red", "orange", "yellow", "green", "blue", "violet", "gray", "grey", "primary"] = "normal"
):
    """Create a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def create_header(title: str, subtitle: str | None = None):
    """Create a styled header with optional subtitle."""
    st.markdown(f"# {title}")
    if subtitle:
        st.markdown(
            f"<p style='color: {COLORS['text_muted']}; font-size: 1.1rem; margin-top: -10px;'>{subtitle}</p>",
            unsafe_allow_html=True,
        )
