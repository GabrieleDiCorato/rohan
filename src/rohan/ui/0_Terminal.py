"""ABIDES Simulation Terminal - Optimized Single Page Application.

Performance optimizations:
- @st.cache_data for expensive data operations
- @st.fragment for isolated component updates
- Lazy loading of analysis components
- Conditional rendering
"""

import html
import logging
import traceback
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from rohan.config import SimulationSettings
from rohan.config.agent_settings import (
    AdaptiveMarketMakerSettings,
    AgentSettings,
    ApiHistoricalProviderSettings,
    CsvHistoricalProviderSettings,
    DatabaseHistoricalProviderSettings,
    ExchangeAgentSettings,
    HistoricalOracleSettings,
    InterpolationMode,
    MomentumAgentSettings,
    NoiseAgentSettings,
    OracleSettings,
    OracleType,
    PriceUnit,
    ProviderType,
    ValueAgentSettings,
)
from rohan.config.latency_settings import LatencyModelSettings, LatencyType
from rohan.exceptions import BaselineComparisonError
from rohan.framework.analysis_service import AnalysisService
from rohan.framework.scenario_repository import ScenarioRepository
from rohan.simulation.simulation_service import SimulationService
from rohan.ui.utils.baseline_comparison import (
    build_baseline_context_table,
    ensure_baseline_comparable,
    get_baseline_compatibility_issues,
)
from rohan.ui.utils.metric_display import DeltaColor, build_comparison_table, get_delta_color, get_help, metric_delta
from rohan.ui.utils.presets import get_preset_config, get_preset_names
from rohan.ui.utils.startup import ensure_db_initialized
from rohan.ui.utils.theme import COLORS, apply_theme
from rohan.utils.formatting import fmt_dollar

_logger = logging.getLogger(__name__)

# Ensure DB tables exist (once per session — avoids noisy re-creation
# logs on every Streamlit rerun).
ensure_db_initialized()

_scenario_repo = ScenarioRepository()

# Page configuration
st.set_page_config(
    page_title="ABIDES Simulation Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply theme
apply_theme()

# Initialize session state
if "simulation_config" not in st.session_state:
    st.session_state.simulation_config = None  # Applied configuration (shown in Execute tab)

if "draft_config" not in st.session_state:
    st.session_state.draft_config = SimulationSettings()  # Draft configuration (being edited in sidebar)

if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None

if "simulation_metrics" not in st.session_state:
    st.session_state.simulation_metrics = None

if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = False

if "baseline_comparison" not in st.session_state:
    st.session_state.baseline_comparison = None

if "baseline_scenario_id" not in st.session_state:
    st.session_state.baseline_scenario_id = None

if "config_reset_counter" not in st.session_state:
    st.session_state.config_reset_counter = 0

# ============================================================================
# CACHED DATA FUNCTIONS
# ============================================================================


@st.cache_data
def get_l1_data(_result):
    """Cache L1 order book data retrieval."""
    return _result.get_order_book_l1()


@st.cache_data
def get_logs_data(_result):
    """Cache logs dataframe retrieval."""
    return _result.get_logs_df()


@st.cache_data
def compute_price_data(_l1_df):
    """Cache price calculations."""
    df = _l1_df.copy()
    df = df.dropna(subset=["bid_price", "ask_price"])
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["returns"] = df["mid_price"].pct_change().fillna(0)
    return df


@st.cache_data
def compute_volume_data(_l1_df):
    """Cache volume calculations."""
    df = _l1_df.copy()
    df = df.dropna(subset=["bid_price", "ask_price"])
    df["volume_imbalance"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"])
    return df


@st.cache_data
def compute_spread_data(_l1_df):
    """Cache spread calculations."""
    df = _l1_df.copy()
    df = df.dropna(subset=["bid_price", "ask_price"])
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread_bps"] = (df["spread"] / df["mid_price"]) * 10000
    return df


@st.cache_data
def load_historical_csv_preview(csv_path: str, file_mtime: float, file_size: int) -> tuple[pd.DataFrame, str, int]:
    """Load historical CSV and downsample for responsive sidebar preview.

    The mtime and size are part of the cache key so edits invalidate stale data.
    """
    _ = (file_mtime, file_size)  # Included for cache invalidation only.

    header = pd.read_csv(csv_path, nrows=0)
    columns = set(header.columns)
    price_col = "price_cents" if "price_cents" in columns else "price" if "price" in columns else ""

    if "timestamp" not in columns or not price_col:
        raise ValueError("CSV must contain 'timestamp' and either 'price_cents' or 'price' columns")

    def should_load_column(column_name: Hashable) -> bool:
        return column_name in {"timestamp", price_col}

    df = pd.read_csv(csv_path, usecols=should_load_column)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", price_col]).sort_values("timestamp")

    total_rows = len(df)
    max_points = 2000
    if total_rows > max_points:
        step = max(1, total_rows // max_points)
        sampled = df.iloc[::step].copy()
        if sampled.index[-1] != df.index[-1]:
            sampled = pd.concat([sampled, df.iloc[[-1]]], ignore_index=False)
        df = sampled

    return df.reset_index(drop=True), price_col, total_rows


@st.cache_data
def load_historical_series_for_analysis(
    csv_path: str,
    file_mtime: float,
    file_size: int,
    symbol: str,
    sim_date: str,
    start_time: str,
    end_time: str,
    price_unit: str,
    source_timezone: str,
) -> pd.DataFrame:
    """Load and downsample historical series for the analysis price chart."""
    _ = (file_mtime, file_size)  # Included for cache invalidation only.

    from rohan.simulation.data.csv_provider import CsvDataProvider

    provider = CsvDataProvider(
        path=csv_path,
        symbol=symbol,
        price_unit=PriceUnit(price_unit),
        source_timezone=source_timezone,
    )

    day_ns = int(pd.to_datetime(sim_date).value)
    start_ns = day_ns + int(pd.to_timedelta(start_time).value)
    end_ns = day_ns + int(pd.to_timedelta(end_time).value)

    series = provider.get_fundamental_series(symbol, start_ns, end_ns)
    if series.empty:
        return pd.DataFrame(columns=pd.Index(["timestamp", "historical_price_cents"]))

    historical_df = series.rename("historical_price_cents").reset_index()
    historical_df.columns = ["timestamp", "historical_price_cents"]

    max_points = 4000
    if len(historical_df) > max_points:
        step = max(1, len(historical_df) // max_points)
        historical_df = historical_df.iloc[::step].copy()

    return historical_df.reset_index(drop=True)


class MetricItem(TypedDict):
    label: str
    value: str
    delta: str | None
    delta_color: DeltaColor
    help: str | None


def _metric_item(
    label: str,
    value: str,
    *,
    delta: str | None,
    field: str,
) -> MetricItem:
    """Create a metric payload for compact column rendering."""
    return {
        "label": label,
        "value": value,
        "delta": delta,
        "delta_color": get_delta_color(field),
        "help": get_help(field),
    }


def _render_metric_columns(metric_items: list[MetricItem], *, column_count: int = 2) -> None:
    """Render metrics in balanced vertical columns for easier scanning."""
    if not metric_items:
        return

    columns = st.columns(column_count)
    chunk_size = (len(metric_items) + column_count - 1) // column_count

    for index, column in enumerate(columns):
        start = index * chunk_size
        end = start + chunk_size
        with column:
            for metric in metric_items[start:end]:
                st.metric(
                    metric["label"],
                    metric["value"],
                    delta=metric["delta"],
                    delta_color=metric["delta_color"],
                    help=metric["help"],
                )


def _render_comparison_context(current_config: SimulationSettings, baseline_config: SimulationSettings) -> None:
    """Render fairness-critical comparison metadata."""
    context_df = build_baseline_context_table(current_config, baseline_config)
    st.caption("Fair comparison check: date, time window, duration, and seed should align before comparing market metrics.")
    st.dataframe(context_df, width="stretch", hide_index=True)


# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================


def compact_input(label, widget_type, key, **kwargs):
    """Create a compact key-value input with label and widget on the same line."""
    # Append reset counter to key to force widget recreation when loading presets
    actual_key = f"{key}_{st.session_state.config_reset_counter}"

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"**{label}:**")
    with col2:
        if widget_type == "text":
            return st.text_input(label, key=actual_key, label_visibility="collapsed", **kwargs)
        if widget_type == "number":
            return st.number_input(label, key=actual_key, label_visibility="collapsed", **kwargs)
        if widget_type == "selectbox":
            return st.selectbox(label, key=actual_key, label_visibility="collapsed", **kwargs)
        if widget_type == "checkbox":
            return st.checkbox(label, key=actual_key, label_visibility="collapsed", **kwargs)
        raise ValueError(f"Unknown widget type: {widget_type}")


def _clear_config_widget_keys():
    """Clear all config widget keys from session state so they re-initialize from draft_config."""
    st.session_state.config_reset_counter += 1


@st.fragment
def render_sidebar_config():
    """Render configuration sidebar as a fragment for isolated updates."""
    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: {COLORS["primary"]}; font-size: 1.8rem; margin: 0;'>
                ABIDES-ROHAN
            </h1>
            <p style='color: {COLORS["secondary"]}; font-size: 1.2rem; margin: 5px 0 0 0; letter-spacing: 2px;'>
                TERMINAL
            </p>
            <hr style='border-color: {COLORS["border"]}; margin: 15px 0;'>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("## ⚙️ Configuration")

    # Preset selection
    st.markdown("### 📋 Presets")
    preset_name = st.selectbox(
        "Select preset",
        options=["Custom"] + get_preset_names(),
        help="Choose a preset configuration",
        key="preset_selector",
    )

    if st.button("Load Preset", type="primary", width="stretch") and preset_name != "Custom":
        new_config = get_preset_config(preset_name)
        st.session_state.draft_config = new_config.model_copy(deep=True)
        _clear_config_widget_keys()

        st.success(f"✅ Loaded preset into draft: {preset_name}")
        st.rerun()

    st.markdown("---")

    # ── Saved scenarios from DB ────────────────────────────────────
    st.markdown("### 💾 Saved Scenarios")

    @st.dialog("Load Saved Scenario", width="large")
    def load_scenario_dialog():
        try:
            _saved_list = _scenario_repo.list_scenarios()
        except Exception:
            _saved_list = []

        if not _saved_list:
            st.info("No saved scenarios yet. Save one from the Execute tab.")
            return

        st.markdown(f"**{len(_saved_list)} saved scenario(s)**")
        st.markdown("---")

        for _sc in _saved_list:
            _cfg = _sc.full_config or {}
            _agents = _cfg.get("agents", {})
            _noise_n = _agents.get("noise", {}).get("num_agents", "?")
            _value_n = _agents.get("value", {}).get("num_agents", "?")
            _mm_n = _agents.get("adaptive_market_maker", {}).get("num_agents", "?")
            _mom_n = _agents.get("momentum", {}).get("num_agents", "?")

            _sc_col1, _sc_col2, _sc_col3 = st.columns([5, 1, 1])
            with _sc_col1:
                st.markdown(f"**{_sc.name}**")
                if _sc.description:
                    st.caption(_sc.description)
                _detail_parts = [
                    f"📅 {_cfg.get('date', '?')}",
                    f"🕐 {_cfg.get('start_time', '?')} – {_cfg.get('end_time', '?')}",
                    f"🏷️ {_cfg.get('ticker', '?')}",
                ]
                st.caption(" · ".join(_detail_parts))
                st.caption(f"👥 Noise: {_noise_n} · Value: {_value_n} · MM: {_mm_n} · Momentum: {_mom_n}   | Created: {_sc.created_at:%Y-%m-%d %H:%M}")
            with _sc_col2:
                if st.button("Load", key=f"load_sc_{_sc.scenario_id}", width="stretch", type="primary"):
                    from rohan.config import SimulationSettings as _SimSettings

                    st.session_state.draft_config = _SimSettings.model_validate(_sc.full_config)
                    _clear_config_widget_keys()

                    st.toast(f"✅ Loaded into draft: {_sc.name}")
                    st.rerun()
            with _sc_col3:
                if st.button("🗑️", key=f"del_sc_{_sc.scenario_id}", width="stretch", help=f"Delete '{_sc.name}'"):
                    _scenario_repo.delete_scenario(_sc.scenario_id)
                    st.rerun()
            st.divider()

    if st.button("Load Configuration", width="stretch"):
        load_scenario_dialog()

    st.markdown("---")

    # Use draft config for sidebar inputs (being edited)
    config = st.session_state.draft_config

    # Simulation Parameters
    with st.expander("🎯 SIMULATION", expanded=True):
        date = compact_input("Date (YYYYMMDD)", "text", "cfg_date", value=config.date)
        start_time = compact_input("Start Time", "text", "cfg_start_time", value=config.start_time)
        end_time = compact_input("End Time", "text", "cfg_end_time", value=config.end_time)
        seed = compact_input("Random Seed", "number", "cfg_seed", value=config.seed, min_value=0)
        ticker = compact_input("Ticker", "text", "cfg_ticker", value=config.ticker)
        starting_cash = compact_input(
            "Starting Cash",
            "number",
            "cfg_starting_cash",
            value=config.starting_cash,
            min_value=0,
            step=1000000,
        )
        stdout_log_level = compact_input(
            "Log Level",
            "selectbox",
            "cfg_log_level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
        )
        log_orders = compact_input("Log Orders", "checkbox", "cfg_log_orders", value=config.log_orders)
        computation_delay_ns = compact_input(
            "Comp Delay (ns)",
            "number",
            "cfg_computation_delay_ns",
            value=config.computation_delay_ns,
            min_value=0,
        )

    # Exchange Agent
    with st.expander("🏦 EXCHANGE"):
        exchange_book_logging = compact_input(
            "Book Logging",
            "checkbox",
            "cfg_exchange_book_logging",
            value=config.agents.exchange.book_logging,
        )
        exchange_book_log_depth = compact_input(
            "Book Log Depth",
            "number",
            "cfg_exchange_book_log_depth",
            value=config.agents.exchange.book_log_depth,
            min_value=1,
        )
        exchange_stream_history = compact_input(
            "Stream History",
            "number",
            "cfg_exchange_stream_history",
            value=config.agents.exchange.stream_history_length,
            min_value=1,
        )
        exchange_log_orders = compact_input(
            "Log Orders",
            "checkbox",
            "cfg_exchange_log_orders",
            value=config.agents.exchange.exchange_log_orders,
        )
        exchange_pipeline_delay = compact_input(
            "Pipeline Delay (ns)",
            "number",
            "cfg_exchange_pipeline_delay",
            value=config.agents.exchange.pipeline_delay_ns,
            min_value=0,
        )
        exchange_computation_delay = compact_input(
            "Comp Delay (ns)",
            "number",
            "cfg_exchange_computation_delay",
            value=config.agents.exchange.computation_delay_ns,
            min_value=0,
        )

    # Noise Agents
    with st.expander("📢 NOISE AGENTS"):
        noise_num_agents = compact_input(
            "Number of Agents",
            "number",
            "cfg_noise_num_agents",
            value=config.agents.noise.num_agents,
            min_value=0,
        )

    # Value Agents
    with st.expander("💎 VALUE AGENTS"):
        value_num_agents = compact_input(
            "Number of Agents",
            "number",
            "cfg_value_num_agents",
            value=config.agents.value.num_agents,
            min_value=0,
        )
        value_r_bar = compact_input(
            "R Bar",
            "number",
            "cfg_value_r_bar",
            value=config.agents.value.r_bar,
            min_value=0,
        )
        value_kappa = compact_input(
            "Kappa",
            "number",
            "cfg_value_kappa",
            value=config.agents.value.kappa,
            format="%.2e",
        )
        value_lambda_a = compact_input(
            "Lambda A",
            "number",
            "cfg_value_lambda_a",
            value=config.agents.value.lambda_a,
            format="%.2e",
        )

    # Adaptive Market Makers
    with st.expander("🎯 MARKET MAKERS"):
        amm_num_agents = compact_input(
            "Number of Agents",
            "number",
            "cfg_amm_num_agents",
            value=config.agents.adaptive_market_maker.num_agents,
            min_value=0,
        )
        amm_window_size = compact_input(
            "Window Size",
            "text",
            "cfg_amm_window_size",
            value=str(config.agents.adaptive_market_maker.window_size),
        )
        amm_pov = compact_input(
            "POV",
            "number",
            "cfg_amm_pov",
            value=config.agents.adaptive_market_maker.pov,
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
        )
        amm_num_ticks = compact_input(
            "Num Ticks",
            "number",
            "cfg_amm_num_ticks",
            value=config.agents.adaptive_market_maker.num_ticks,
            min_value=1,
        )
        amm_wake_up_freq = compact_input(
            "Wake Up Freq",
            "text",
            "cfg_amm_wake_up_freq",
            value=config.agents.adaptive_market_maker.wake_up_freq,
        )
        amm_poisson = compact_input(
            "Poisson Arrival",
            "checkbox",
            "cfg_amm_poisson",
            value=config.agents.adaptive_market_maker.poisson_arrival,
        )
        amm_min_order_size = compact_input(
            "Min Order Size",
            "number",
            "cfg_amm_min_order_size",
            value=config.agents.adaptive_market_maker.min_order_size,
            min_value=1,
        )
        amm_level_spacing = compact_input(
            "Level Spacing",
            "number",
            "cfg_amm_level_spacing",
            value=config.agents.adaptive_market_maker.level_spacing,
            min_value=1,
        )
        amm_skew_beta = compact_input(
            "Skew Beta",
            "number",
            "cfg_amm_skew_beta",
            value=config.agents.adaptive_market_maker.skew_beta,
        )
        amm_price_skew = compact_input(
            "Price Skew",
            "number",
            "cfg_amm_price_skew",
            value=config.agents.adaptive_market_maker.price_skew,
        )
        amm_spread_alpha = compact_input(
            "Spread Alpha",
            "number",
            "cfg_amm_spread_alpha",
            value=config.agents.adaptive_market_maker.spread_alpha,
            min_value=0.0,
            max_value=1.0,
            format="%.2f",
        )
        amm_backstop_qty = compact_input(
            "Backstop Qty",
            "number",
            "cfg_amm_backstop_qty",
            value=config.agents.adaptive_market_maker.backstop_quantity,
            min_value=0,
        )

    # Momentum Agents
    with st.expander("📈 MOMENTUM AGENTS"):
        momentum_num_agents = compact_input(
            "Number of Agents",
            "number",
            "cfg_momentum_num_agents",
            value=config.agents.momentum.num_agents,
            min_value=0,
        )
        momentum_min_size = compact_input(
            "Min Size",
            "number",
            "cfg_momentum_min_size",
            value=config.agents.momentum.min_size,
            min_value=1,
        )
        momentum_max_size = compact_input(
            "Max Size",
            "number",
            "cfg_momentum_max_size",
            value=config.agents.momentum.max_size,
            min_value=1,
        )
        momentum_wake_up_freq = compact_input(
            "Wake Up Freq",
            "text",
            "cfg_momentum_wake_up_freq",
            value=config.agents.momentum.wake_up_freq,
        )
        momentum_poisson = compact_input(
            "Poisson Arrival",
            "checkbox",
            "cfg_momentum_poisson",
            value=config.agents.momentum.poisson_arrival,
        )

    # Oracle Settings
    with st.expander("🔮 ORACLE"):
        oracle_type_str = st.radio(
            "Oracle Type",
            options=[OracleType.SYNTHETIC.value, OracleType.HISTORICAL.value],
            index=0 if config.agents.oracle.oracle_type == OracleType.SYNTHETIC else 1,
            horizontal=True,
            key=f"cfg_oracle_type_{st.session_state.config_reset_counter}",
        )
        oracle_type = OracleType(oracle_type_str)
        historical_provider_type = config.agents.oracle.historical.provider_type
        historical_csv_path: str | None = config.agents.oracle.historical.csv.csv_path
        historical_db_dataset_id: str | None = config.agents.oracle.historical.database.dataset_id
        historical_api_symbol: str | None = config.agents.oracle.historical.api.symbol
        historical_api_key: str | None = config.agents.oracle.historical.api.api_key
        historical_api_provider = config.agents.oracle.historical.api.provider_name
        historical_price_unit = config.agents.oracle.historical.csv.price_unit
        historical_source_timezone = config.agents.oracle.historical.csv.source_timezone

        if oracle_type == OracleType.SYNTHETIC:
            oracle_kappa = compact_input(
                "Kappa",
                "number",
                "cfg_oracle_kappa",
                value=config.agents.oracle.kappa,
                format="%.2e",
            )
            oracle_sigma_s = compact_input(
                "Sigma S",
                "number",
                "cfg_oracle_sigma_s",
                value=config.agents.oracle.sigma_s,
                format="%.2e",
            )
            oracle_fund_vol = compact_input(
                "Fund Vol",
                "number",
                "cfg_oracle_fund_vol",
                value=config.agents.oracle.fund_vol,
                format="%.2e",
            )
            oracle_megashock_lambda = compact_input(
                "Megashock Lambda",
                "number",
                "cfg_oracle_megashock_lambda",
                value=config.agents.oracle.megashock_lambda_a,
                format="%.2e",
            )
            oracle_megashock_mean = compact_input(
                "Megashock Mean",
                "number",
                "cfg_oracle_megashock_mean",
                value=config.agents.oracle.megashock_mean,
                min_value=0,
            )
            oracle_megashock_var = compact_input(
                "Megashock Var",
                "number",
                "cfg_oracle_megashock_var",
                value=config.agents.oracle.megashock_var,
                min_value=0,
            )
        else:
            from rohan.simulation.data.csv_provider import CsvDataProvider
            from rohan.simulation.data.database_provider import DatabaseDataProvider

            provider_type_str = compact_input(
                "Provider",
                "selectbox",
                "cfg_oracle_historical_provider",
                options=[pt.value for pt in ProviderType],
                index=[pt.value for pt in ProviderType].index(config.agents.oracle.historical.provider_type.value),
            )
            historical_provider_type = ProviderType(provider_type_str)

            datasets_dir = Path("hist_data")
            datasets_dir.mkdir(parents=True, exist_ok=True)

            if historical_provider_type == ProviderType.CSV:
                available_csvs = CsvDataProvider.list_available(datasets_dir)

                if not available_csvs:
                    st.warning("No CSV datasets found in hist_data/")
                    historical_csv = None
                else:
                    historical_csv = st.selectbox(
                        "Historical Dataset",
                        options=available_csvs,
                        help="Select a CSV dataset. Look inside hist_data/",
                        key=f"cfg_oracle_historical_csv_{st.session_state.config_reset_counter}",
                    )
                    if historical_csv:
                        historical_csv_path = str(datasets_dir / f"{historical_csv}.csv")

                csv_unit_str = compact_input(
                    "CSV Price Unit",
                    "selectbox",
                    "cfg_oracle_historical_csv_unit",
                    options=[PriceUnit.CENTS.value, PriceUnit.DOLLARS.value],
                    index=0 if historical_price_unit == PriceUnit.CENTS else 1,
                )
                historical_price_unit = PriceUnit(csv_unit_str)
                historical_source_timezone = compact_input(
                    "Source Timezone",
                    "text",
                    "cfg_oracle_historical_csv_timezone",
                    value=historical_source_timezone,
                )
                historical_source_timezone = str(historical_source_timezone)

            elif historical_provider_type == ProviderType.DATABASE:
                available_datasets = DatabaseDataProvider.list_available()
                if not available_datasets:
                    st.warning("No historical datasets found in database table fundamental_datasets")
                else:
                    historical_db_dataset_id = st.selectbox(
                        "Database Dataset ID",
                        options=available_datasets,
                        key=f"cfg_oracle_historical_db_dataset_{st.session_state.config_reset_counter}",
                    )

            else:
                historical_api_provider = str(
                    compact_input(
                        "API Provider",
                        "selectbox",
                        "cfg_oracle_historical_api_provider",
                        options=["alpaca", "polygon"],
                        index=0 if historical_api_provider == "alpaca" else 1,
                    )
                )
                historical_api_symbol_value = compact_input(
                    "API Symbol",
                    "text",
                    "cfg_oracle_historical_api_symbol",
                    value=historical_api_symbol or config.ticker,
                )
                historical_api_symbol = str(historical_api_symbol_value) or None
                historical_api_key_value = compact_input(
                    "API Key",
                    "text",
                    "cfg_oracle_historical_api_key",
                    value=historical_api_key or "",
                )
                historical_api_key = str(historical_api_key_value) or None
                api_unit_str = compact_input(
                    "API Price Unit",
                    "selectbox",
                    "cfg_oracle_historical_api_unit",
                    options=[PriceUnit.CENTS.value, PriceUnit.DOLLARS.value],
                    index=1,
                )
                historical_price_unit = PriceUnit(api_unit_str)
                historical_source_timezone = str(
                    compact_input(
                        "API Source Timezone",
                        "text",
                        "cfg_oracle_historical_api_timezone",
                        value=historical_source_timezone,
                    )
                )

            historical_interpolation = compact_input(
                "Interpolation",
                "selectbox",
                "cfg_oracle_historical_interp",
                options=[m.value for m in InterpolationMode],
                index=([m.value for m in InterpolationMode].index(config.agents.oracle.historical.interpolation.value)),
            )

            historical_recenter = compact_input(
                "Recenter to R_Bar",
                "checkbox",
                "cfg_oracle_historical_recenter",
                value=config.agents.oracle.historical.recenter_r_bar,
            )

            if historical_provider_type == ProviderType.CSV and historical_csv_path:
                csv_path = Path(historical_csv_path)
                try:
                    preview_df, price_col, total_rows = load_historical_csv_preview(
                        str(csv_path),
                        csv_path.stat().st_mtime,
                        csv_path.stat().st_size,
                    )
                    preview_df = preview_df.set_index("timestamp")

                    st.markdown("**Preview**")
                    preview_series = preview_df[price_col] / 100 if price_col == "price_cents" else preview_df[price_col]
                    preview_label = "Price ($)" if price_col == "price_cents" else "Price"

                    preview_fig = go.Figure(
                        data=[
                            go.Scatter(
                                x=preview_df.index,
                                y=preview_series,
                                mode="lines",
                                name="Historical",
                                line={"color": COLORS["primary"], "width": 1.4},
                            )
                        ]
                    )
                    preview_fig.update_layout(
                        height=220,
                        margin={"l": 8, "r": 8, "t": 8, "b": 8},
                        plot_bgcolor=COLORS["background"],
                        paper_bgcolor=COLORS["background"],
                        font={"color": COLORS["text"], "family": "Courier New", "size": 11},
                        showlegend=False,
                    )
                    preview_fig.update_xaxes(showgrid=False, showticklabels=False)
                    preview_fig.update_yaxes(
                        title_text=preview_label,
                        gridcolor=COLORS["border"],
                        showgrid=True,
                        autorange=True,
                        rangemode="normal",
                        fixedrange=False,
                    )
                    st.plotly_chart(preview_fig, width="stretch", config={"displayModeBar": False})

                    if total_rows > len(preview_df):
                        st.caption(f"Previewing {len(preview_df):,} sampled points from {total_rows:,} rows for UI responsiveness.")

                    # Inform the user if the CSV date differs from the simulation
                    # date. The provider rebases the series automatically, so the
                    # simulation will still run correctly.
                    csv_first_ts = preview_df["timestamp"].iloc[0]
                    if not pd.isna(csv_first_ts):
                        csv_first_timestamp = pd.to_datetime(csv_first_ts, errors="coerce")
                        if isinstance(csv_first_timestamp, pd.Timestamp):
                            csv_date_str = csv_first_timestamp.strftime("%Y%m%d")
                            sim_date_str = str(config.date)
                            if csv_date_str != sim_date_str:
                                st.info(f"ℹ️ CSV date ({csv_date_str}) differs from simulation date ({sim_date_str}). The intraday price pattern will be rebased to the simulation date automatically.")
                except Exception as e:
                    st.error(f"Cannot preview dataset: {e}")

    # Latency Model
    with st.expander("⏱️ LATENCY"):
        latency_type = compact_input(
            "Type",
            "selectbox",
            "cfg_latency_type",
            options=[lt.value for lt in LatencyType],
            index=1,
        )
        if latency_type == LatencyType.CUBIC.value:
            latency_jitter = compact_input(
                "Jitter",
                "number",
                "cfg_latency_jitter",
                value=config.latency.jitter,
                min_value=0.0,
            )
            latency_jitter_clip = compact_input(
                "Jitter Clip",
                "number",
                "cfg_latency_jitter_clip",
                value=config.latency.jitter_clip,
                min_value=0.0,
            )
            latency_jitter_unit = compact_input(
                "Jitter Unit",
                "number",
                "cfg_latency_jitter_unit",
                value=config.latency.jitter_unit,
                min_value=0.0,
            )
        else:
            latency_jitter = config.latency.jitter
            latency_jitter_clip = config.latency.jitter_clip
            latency_jitter_unit = config.latency.jitter_unit

    st.markdown("---")

    # Save configuration button
    if st.button("🚀 Apply Configuration", type="primary", width="stretch"):
        try:
            # Build configuration
            agents = AgentSettings(
                exchange=ExchangeAgentSettings(
                    book_logging=bool(exchange_book_logging),
                    book_log_depth=int(exchange_book_log_depth),  # pyright: ignore[reportArgumentType]
                    stream_history_length=int(exchange_stream_history),  # pyright: ignore[reportArgumentType]
                    exchange_log_orders=bool(exchange_log_orders),
                    pipeline_delay_ns=int(exchange_pipeline_delay),  # pyright: ignore[reportArgumentType]
                    computation_delay_ns=int(exchange_computation_delay),  # pyright: ignore[reportArgumentType]
                ),
                noise=NoiseAgentSettings(num_agents=int(noise_num_agents)),  # pyright: ignore[reportArgumentType]
                value=ValueAgentSettings(
                    num_agents=int(value_num_agents),  # pyright: ignore[reportArgumentType]
                    r_bar=int(value_r_bar),  # pyright: ignore[reportArgumentType]
                    kappa=float(value_kappa),  # pyright: ignore[reportArgumentType]
                    lambda_a=float(value_lambda_a),  # pyright: ignore[reportArgumentType]
                ),
                adaptive_market_maker=AdaptiveMarketMakerSettings(
                    num_agents=int(amm_num_agents),  # pyright: ignore[reportArgumentType]
                    window_size=(amm_window_size if amm_window_size == "adaptive" else int(amm_window_size)),  # pyright: ignore[reportArgumentType]
                    pov=float(amm_pov),  # pyright: ignore[reportArgumentType]
                    num_ticks=int(amm_num_ticks),  # pyright: ignore[reportArgumentType]
                    wake_up_freq=str(amm_wake_up_freq),
                    poisson_arrival=bool(amm_poisson),
                    min_order_size=int(amm_min_order_size),  # pyright: ignore[reportArgumentType]
                    skew_beta=int(amm_skew_beta),  # pyright: ignore[reportArgumentType]
                    price_skew=int(amm_price_skew),  # pyright: ignore[reportArgumentType]
                    level_spacing=int(amm_level_spacing),  # pyright: ignore[reportArgumentType]
                    spread_alpha=float(amm_spread_alpha),  # pyright: ignore[reportArgumentType]
                    backstop_quantity=int(amm_backstop_qty),  # pyright: ignore[reportArgumentType]
                ),
                momentum=MomentumAgentSettings(
                    num_agents=int(momentum_num_agents),  # pyright: ignore[reportArgumentType]
                    min_size=int(momentum_min_size),  # pyright: ignore[reportArgumentType]
                    max_size=int(momentum_max_size),  # pyright: ignore[reportArgumentType]
                    wake_up_freq=str(momentum_wake_up_freq),
                    poisson_arrival=bool(momentum_poisson),
                ),
                oracle=(
                    OracleSettings(
                        oracle_type=OracleType.SYNTHETIC,
                        kappa=float(oracle_kappa),  # pyright: ignore[reportArgumentType]
                        sigma_s=float(oracle_sigma_s),  # pyright: ignore[reportArgumentType]
                        fund_vol=float(oracle_fund_vol),  # pyright: ignore[reportArgumentType]
                        megashock_lambda_a=float(oracle_megashock_lambda),  # pyright: ignore[reportArgumentType]
                        megashock_mean=int(oracle_megashock_mean),  # pyright: ignore[reportArgumentType]
                        megashock_var=int(oracle_megashock_var),  # pyright: ignore[reportArgumentType]
                    )
                    if oracle_type == OracleType.SYNTHETIC
                    else OracleSettings(
                        oracle_type=OracleType.HISTORICAL,
                        historical=HistoricalOracleSettings(
                            provider_type=historical_provider_type,
                            interpolation=InterpolationMode(str(historical_interpolation)),
                            recenter_r_bar=bool(historical_recenter),
                            csv=CsvHistoricalProviderSettings(
                                csv_path=historical_csv_path,
                                price_unit=historical_price_unit,
                                source_timezone=historical_source_timezone,
                            ),
                            database=DatabaseHistoricalProviderSettings(
                                dataset_id=historical_db_dataset_id,
                            ),
                            api=ApiHistoricalProviderSettings(
                                provider_name=historical_api_provider,
                                symbol=historical_api_symbol,
                                api_key=historical_api_key,
                                price_unit=historical_price_unit,
                                source_timezone=historical_source_timezone,
                            ),
                        ),
                    )
                ),
            )

            latency = LatencyModelSettings(
                type=LatencyType(latency_type),
                jitter=float(latency_jitter),  # pyright: ignore[reportArgumentType]
                jitter_clip=float(latency_jitter_clip),  # pyright: ignore[reportArgumentType]
                jitter_unit=float(latency_jitter_unit),  # pyright: ignore[reportArgumentType]
            )

            new_config = SimulationSettings(
                date=str(date),
                start_time=str(start_time),
                end_time=str(end_time),
                seed=int(seed),  # pyright: ignore[reportArgumentType]
                ticker=str(ticker),
                starting_cash=int(starting_cash),  # pyright: ignore[reportArgumentType]
                stdout_log_level=str(stdout_log_level),
                log_orders=bool(log_orders),
                computation_delay_ns=int(computation_delay_ns),  # pyright: ignore[reportArgumentType]
                agents=agents,
                latency=latency,
            )

            st.session_state.simulation_config = new_config
            st.session_state.draft_config = new_config.model_copy(deep=True)
            # Clear stale results — the config has changed
            st.session_state.simulation_result = None
            st.session_state.simulation_metrics = None
            st.session_state.baseline_comparison = None
            st.session_state.pop("simulation_timestamp", None)
            st.session_state.pop("simulation_duration", None)
            st.session_state.pop("simulation_seed", None)
            st.success("✅ Configuration applied!")
            st.rerun()  # Force full page refresh to update Execute tab

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Reset button
    if st.button("🔄 Reset to Default", width="stretch"):
        st.session_state.draft_config = SimulationSettings()
        _clear_config_widget_keys()
        st.success("✅ Draft reset to defaults!")
        st.rerun()


# Render sidebar
with st.sidebar:
    render_sidebar_config()

    # 2.7.9 — Show the seed used by the last completed run so results are
    # reproducible: the user can copy the seed back into the seed field to
    # replay an exact scenario.
    if "simulation_seed" in st.session_state:
        st.markdown("---")
        st.markdown("**🎲 Last Run Seed**")
        st.code(str(st.session_state.simulation_seed), language=None)
        st.caption("Set this seed in Configuration to reproduce results.")

# ============================================================================
# MAIN PANEL: HEADER
# ============================================================================

st.markdown(
    f"""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: {COLORS["primary"]}; font-size: 2.5rem; margin: 0;'>
            ABIDES-Markets Simulation Terminal
        </h1>
        <p style='color: {COLORS["text_muted"]}; font-size: 1.1rem; margin: 10px 0 0 0;'>
            Agent-Based Interactive Discrete Event Simulation
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================================
# MAIN PANEL: TABS
# ============================================================================

tab1, tab2 = st.tabs(["▶️ Execute", "📊 Analyze"])

# ============================================================================
# TAB 1: EXECUTE
# ============================================================================


def render_execute_tab():
    """Render execute tab as a fragment for isolated updates."""
    # Check if configuration exists
    if st.session_state.simulation_config is None:
        st.warning("⚠️ No configuration applied. Please configure the simulation in the sidebar first.")
        return

    config = st.session_state.simulation_config

    # ── Save Scenario ──────────────────────────────────────────────
    st.markdown("### 💾 Save as Scenario")
    save_sc_col1, save_sc_col2 = st.columns([3, 1])
    with save_sc_col1:
        _save_name = st.text_input(
            "Scenario name",
            placeholder="e.g. High-Vol Stress Test",
            key="save_scenario_name",
            label_visibility="collapsed",
        )
    with save_sc_col2:
        st.markdown("<div style='height: 2px'></div>", unsafe_allow_html=True)
        _save_clicked = st.button("💾 Save", width="stretch", disabled=not bool(_save_name and _save_name.strip()))
    if _save_clicked and _save_name and _save_name.strip():
        try:
            _scenario_repo.save_scenario(
                name=_save_name.strip(),
                full_config=config.model_dump(),
            )
            st.toast(f"✅ Scenario '{_save_name.strip()}' saved!")
        except Exception as _save_err:
            st.error(f"Could not save scenario: {_save_err}")

    st.markdown("---")

    # Configuration Summary
    st.markdown("### 📋 Applied Configuration")
    st.info("ℹ️ This is the configuration that will be used for simulation runs. Modify settings in the sidebar and click 'Apply Configuration' to update.")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["primary"]};'>
                <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>DATE</p>
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{html.escape(str(config.date))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["secondary"]};'>
                <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>TIME RANGE</p>
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{html.escape(str(config.start_time))} - {html.escape(str(config.end_time))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["success"]};'>
                <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>TICKER</p>
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{html.escape(str(config.ticker))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["danger"]};'>
                <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>SEED</p>
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{html.escape(str(config.seed))}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Agent Summary
    st.markdown("### 🤖 Agent Configuration")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Exchange", config.agents.exchange.num_agents)

    with col2:
        st.metric("Noise", config.agents.noise.num_agents)

    with col3:
        st.metric("Value", config.agents.value.num_agents)

    with col4:
        st.metric("Market Makers", config.agents.adaptive_market_maker.num_agents)

    with col5:
        st.metric("Momentum", config.agents.momentum.num_agents)

    total_agents = (
        config.agents.exchange.num_agents + config.agents.noise.num_agents + config.agents.value.num_agents + config.agents.adaptive_market_maker.num_agents + config.agents.momentum.num_agents
    )

    st.info(f"ℹ️ Total agents: **{total_agents}**")

    # Detailed configuration in expander
    with st.expander("🔍 View Full Configuration Details", expanded=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Simulation Settings**")
            st.markdown(f"- **Starting Cash:** ${config.starting_cash:,}")
            st.markdown(f"- **Log Level:** {config.stdout_log_level}")
            st.markdown(f"- **Log Orders:** {config.log_orders}")
            st.markdown(f"- **Computation Delay:** {config.computation_delay_ns} ns")
            st.markdown("")

            st.markdown("**Exchange Settings**")
            st.markdown(f"- **Book Logging:** {config.agents.exchange.book_logging}")
            st.markdown(f"- **Book Log Depth:** {config.agents.exchange.book_log_depth}")
            st.markdown(f"- **Stream History:** {config.agents.exchange.stream_history_length}")
            st.markdown(f"- **Pipeline Delay:** {config.agents.exchange.pipeline_delay_ns} ns")
            st.markdown("")

            st.markdown("**Latency Model**")
            st.markdown(f"- **Type:** {config.latency.type.value}")
            if config.latency.type.value == "cubic":
                st.markdown(f"- **Jitter:** {config.latency.jitter}")
                st.markdown(f"- **Jitter Clip:** {config.latency.jitter_clip}")

        with col2:
            st.markdown("**Oracle Settings**")
            st.markdown(f"- **Type:** {config.agents.oracle.oracle_type.value}")
            if config.agents.oracle.oracle_type == OracleType.SYNTHETIC:
                st.markdown(f"- **Kappa:** {config.agents.oracle.kappa:.2e}")
                st.markdown(f"- **Sigma S:** {config.agents.oracle.sigma_s:.2e}")
                st.markdown(f"- **Fund Vol:** {config.agents.oracle.fund_vol:.2e}")
                st.markdown(f"- **Megashock Lambda:** {config.agents.oracle.megashock_lambda_a:.2e}")
            else:
                historical = config.agents.oracle.historical
                st.markdown(f"- **Provider:** {historical.provider_type.value}")
                if historical.provider_type == ProviderType.CSV:
                    path_str = Path(historical.csv.csv_path).stem if historical.csv.csv_path else "None"
                    st.markdown(f"- **Dataset:** {path_str}")
                elif historical.provider_type == ProviderType.DATABASE:
                    st.markdown(f"- **Dataset ID:** {historical.database.dataset_id}")
                else:
                    st.markdown(f"- **API Provider:** {historical.api.provider_name}")
                    st.markdown(f"- **API Symbol:** {historical.api.symbol}")
                st.markdown(f"- **Interpolation:** {historical.interpolation.value}")
                st.markdown(f"- **Recenter:** {historical.recenter_r_bar}")
            st.markdown("")

            st.markdown("**Value Agents**")
            st.markdown(f"- **Count:** {config.agents.value.num_agents}")
            st.markdown(f"- **R Bar:** {config.agents.value.r_bar}")
            st.markdown(f"- **Kappa:** {config.agents.value.kappa:.2e}")
            st.markdown("")

            st.markdown("**Market Makers**")
            st.markdown(f"- **Count:** {config.agents.adaptive_market_maker.num_agents}")
            st.markdown(f"- **Window Size:** {config.agents.adaptive_market_maker.window_size}")
            st.markdown(f"- **POV:** {config.agents.adaptive_market_maker.pov:.4f}")

    st.markdown("---")

    # ── Comparison Options ──────────────────────────────────────────
    st.markdown("### ⚖️ Comparison Options")

    # Baseline scenario selector
    try:
        _saved_scenarios = _scenario_repo.list_scenarios()
    except Exception:
        _saved_scenarios = []

    _baseline_options: list[str] = ["None"]
    _baseline_id_map: dict[str, str] = {}  # display label → scenario_id
    _baseline_scenario_map = {}
    for _sc in _saved_scenarios:
        _label = _sc.name
        _baseline_options.append(_label)
        _baseline_id_map[_label] = str(_sc.scenario_id)
        _baseline_scenario_map[_label] = _sc

    _baseline_choice = st.selectbox(
        "Baseline scenario",
        options=_baseline_options,
        index=0,
        help="Pick a saved scenario to run alongside your config. Metrics will be compared automatically.",
        key="baseline_selector",
    )
    st.session_state.baseline_scenario_id = _baseline_id_map.get(_baseline_choice)  # None when "None"

    _selected_baseline = _baseline_scenario_map.get(_baseline_choice)
    _baseline_preview_config: SimulationSettings | None = None
    _baseline_preview_error: str | None = None
    _baseline_issues: list[str] = []

    if _selected_baseline is None:
        st.caption("Optional: pick a saved scenario to run as a baseline against the current applied configuration.")
    elif _selected_baseline.full_config is None:
        _baseline_preview_error = "Selected baseline has no saved configuration snapshot."
    else:
        try:
            _baseline_preview_config = SimulationSettings.model_validate(_selected_baseline.full_config)
            _baseline_issues = get_baseline_compatibility_issues(config, _baseline_preview_config)
        except Exception as _preview_err:
            _baseline_preview_error = f"Could not load selected baseline: {_preview_err}"

    if _selected_baseline is not None:
        if _baseline_preview_error is not None:
            st.error(_baseline_preview_error)
        elif _baseline_preview_config is not None:
            st.caption(f"Selected baseline: **{_selected_baseline.name}** · Created: {_selected_baseline.created_at:%Y-%m-%d %H:%M}")
            _render_comparison_context(config, _baseline_preview_config)

            if _baseline_issues:
                st.warning("Comparison will be skipped until these fairness checks match:\n" + "\n".join(f"- {issue}" for issue in _baseline_issues))
            else:
                st.success("Baseline is eligible for comparison. The Terminal will run it after the current simulation completes.")

    st.markdown("---")

    # Run Simulation
    st.markdown("### 🚀 Run Simulation")

    run_button = st.button(
        "▶️ RUN SIMULATION",
        type="primary",
        width="stretch",
        disabled=st.session_state.simulation_running,
    )

    if run_button:
        st.session_state.simulation_running = True

        # Create status container
        status_container = st.status("🔄 Running Simulation...", expanded=True)

        try:
            with status_container:
                # Step 1: Validate configuration
                st.write("✓ Configuration validated")

                # Step 2: Initialize simulation service
                st.write("⏳ Initializing simulation engine...")
                service = SimulationService()
                st.write("✓ Simulation engine initialized")

                # Step 3: Run simulation
                st.write("⏳ Running simulation (this may take a few minutes)...")
                start_time = datetime.now()

                sim_result = service.run_simulation(config)

                # Check if simulation succeeded
                if sim_result.error is not None:
                    raise sim_result.error

                result = sim_result.result
                assert result is not None, "Result must not be None when error is None"

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                st.write(f"✓ Simulation completed in {duration:.2f} seconds")

                # Step 4: Process results
                st.write("⏳ Processing results...")

                # Compute metrics
                metrics = AnalysisService.compute_metrics(result)

                st.write("✓ Results processed successfully")

                # Always track previous run for delta display
                st.session_state.previous_metrics = st.session_state.get("last_run_metrics")

                # Save to session state
                st.session_state.simulation_result = result
                st.session_state.simulation_metrics = metrics
                st.session_state.last_run_metrics = metrics
                st.session_state.simulation_duration = duration
                st.session_state.simulation_timestamp = datetime.now()
                st.session_state.simulation_seed = config.seed  # 2.7.9 reproducibility

                # Clear caches to force fresh data
                get_l1_data.clear()
                get_logs_data.clear()
                compute_price_data.clear()
                compute_volume_data.clear()
                compute_spread_data.clear()

            status_container.update(label="✅ Simulation Complete!", state="complete", expanded=False)

            st.success(f"🎉 Simulation completed successfully in {duration:.2f} seconds!")

            # ── Baseline run (if a saved scenario was selected) ─────
            if _selected_baseline is not None:
                if _baseline_preview_error is not None:
                    st.warning(f"Baseline comparison skipped: {_baseline_preview_error}")
                    st.session_state.baseline_comparison = None
                elif _baseline_preview_config is None:
                    st.warning("Baseline comparison skipped: baseline configuration is unavailable.")
                    st.session_state.baseline_comparison = None
                elif _baseline_issues:
                    st.warning("Baseline comparison skipped because the selected scenario is not comparable with the current run:\n" + "\n".join(f"- {issue}" for issue in _baseline_issues))
                    st.session_state.baseline_comparison = None
                else:
                    bl_status = st.status(f"🔄 Running baseline: {_selected_baseline.name}…", expanded=True)
                    try:
                        with bl_status:
                            st.write(f"⏳ Running baseline scenario **{_selected_baseline.name}**…")

                            baseline_config = _baseline_preview_config
                            ensure_baseline_comparable(config, baseline_config)
                            bl_service = SimulationService()
                            bl_start = datetime.now()
                            bl_result = bl_service.run_simulation(baseline_config)

                            if bl_result.error is not None:
                                raise bl_result.error

                            bl_output = bl_result.result
                            assert bl_output is not None

                            bl_duration = (datetime.now() - bl_start).total_seconds()
                            st.write(f"✓ Baseline completed in {bl_duration:.2f}s")

                            st.write("⏳ Computing comparison…")
                            bl_metrics = AnalysisService.compute_metrics(bl_output)

                            st.session_state.baseline_comparison = {
                                "name": _selected_baseline.name,
                                "metrics": bl_metrics,
                                "timestamp": datetime.now(),
                                "config_snapshot": {
                                    "current": config.model_dump(),
                                    "baseline": baseline_config.model_dump(),
                                },
                            }
                            st.write("✓ Comparison ready.")

                        bl_status.update(label="✅ Baseline Complete!", state="complete", expanded=False)

                    except BaselineComparisonError as bl_err:
                        bl_status.update(label="⚠️ Baseline Comparison Skipped", state="error", expanded=True)
                        st.warning(f"Baseline comparison skipped: {bl_err}")
                        st.session_state.baseline_comparison = None
                    except Exception as bl_err:
                        bl_status.update(label="❌ Baseline Failed", state="error", expanded=True)
                        st.error(f"Baseline comparison failed: {bl_err}")
            else:
                st.session_state.baseline_comparison = None

            # Show quick metrics
            st.markdown("### 📊 Quick Metrics")

            prev = st.session_state.get("previous_metrics")

            def _m(v: float | None, fmt: str = ".6f") -> str:
                return f"{v:{fmt}}" if v is not None else "N/A"

            quick_metrics: list[MetricItem] = []

            d = metric_delta(metrics.volatility, prev.volatility if prev else None)
            quick_metrics.append(_metric_item("Volatility", _m(metrics.volatility, ".4f"), delta=f"{d:+.4f}" if d is not None else None, field="volatility"))

            mean_spread_val = fmt_dollar(metrics.mean_spread, precision=4) if metrics.mean_spread is not None else "N/A"
            d = metric_delta(metrics.mean_spread, prev.mean_spread if prev else None)
            mean_spread_delta = (("+" if d > 0 else "") + fmt_dollar(d, precision=4)) if d is not None else None
            quick_metrics.append(_metric_item("Mean Spread", mean_spread_val, delta=mean_spread_delta, field="mean_spread"))

            d = metric_delta(metrics.avg_bid_liquidity, prev.avg_bid_liquidity if prev else None)
            quick_metrics.append(_metric_item("Avg Bid Liquidity", _m(metrics.avg_bid_liquidity, ".2f"), delta=f"{d:+.2f}" if d is not None else None, field="avg_bid_liquidity"))

            d = metric_delta(metrics.avg_ask_liquidity, prev.avg_ask_liquidity if prev else None)
            quick_metrics.append(_metric_item("Avg Ask Liquidity", _m(metrics.avg_ask_liquidity, ".2f"), delta=f"{d:+.2f}" if d is not None else None, field="avg_ask_liquidity"))

            # Add microstructure metrics when available.
            if any(v is not None for v in [metrics.vpin, metrics.lob_imbalance_mean, metrics.market_ott_ratio]):
                d = metric_delta(metrics.vpin, prev.vpin if prev else None)
                quick_metrics.append(_metric_item("VPIN", _m(metrics.vpin, ".4f"), delta=f"{d:+.4f}" if d is not None else None, field="vpin"))

                d = metric_delta(metrics.lob_imbalance_mean, prev.lob_imbalance_mean if prev else None)
                quick_metrics.append(_metric_item("LOB Imbalance", _m(metrics.lob_imbalance_mean, ".4f"), delta=f"{d:+.4f}" if d is not None else None, field="lob_imbalance_mean"))

                volume_val = f"{metrics.traded_volume:,}" if metrics.traded_volume is not None else "N/A"
                d = metric_delta(metrics.traded_volume, prev.traded_volume if prev else None)
                quick_metrics.append(_metric_item("Traded Volume", volume_val, delta=f"{d:+,}" if d is not None else None, field="traded_volume"))

                d = metric_delta(metrics.market_ott_ratio, prev.market_ott_ratio if prev else None)
                quick_metrics.append(_metric_item("Market OTT", _m(metrics.market_ott_ratio, ".2f"), delta=f"{d:+.2f}" if d is not None else None, field="market_ott_ratio"))

                availability_val = f"{metrics.pct_time_two_sided:.1%}" if metrics.pct_time_two_sided is not None else "N/A"
                d = metric_delta(metrics.pct_time_two_sided, prev.pct_time_two_sided if prev else None)
                quick_metrics.append(_metric_item("Availability", availability_val, delta=f"{d:+.1%}" if d is not None else None, field="pct_time_two_sided"))

            _render_metric_columns(quick_metrics, column_count=2)

        except Exception as e:
            status_container.update(label="❌ Simulation Failed", state="error", expanded=True)
            st.error(f"❌ Simulation failed: {str(e)}")

            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

            st.warning("💡 **Troubleshooting Tips:**")
            st.markdown(
                """
                - Check that all agent counts are valid (non-negative)
                - Ensure time range is at least 5 minutes
                - Verify that wake-up frequencies are in correct format (e.g., '60s')
                - Check that all numeric values are within reasonable ranges
                """
            )

        finally:
            st.session_state.simulation_running = False

    # ── Baseline summary (compact, shown outside the run block) ────
    bl_data = st.session_state.baseline_comparison
    if bl_data is not None and st.session_state.simulation_metrics is not None:
        st.markdown("---")
        bl_name = bl_data.get("name", "baseline")
        st.markdown(f"### ⚖️ Current Run  vs  Baseline: *{bl_name}*")
        st.caption(f"Side-by-side comparison of your **current run** against the saved baseline scenario **{bl_name}**. A positive Δ% means the current value is higher.")

        bm = bl_data["metrics"]
        cur = st.session_state.simulation_metrics
        snapshots = bl_data.get("config_snapshot")

        if snapshots is not None:
            current_config = SimulationSettings.model_validate(snapshots["current"])
            baseline_config = SimulationSettings.model_validate(snapshots["baseline"])
            _render_comparison_context(current_config, baseline_config)

        cmp_df = build_comparison_table(cur, bm)
        st.dataframe(cmp_df.set_index("Metric"), width="stretch")

    # Execution History
    if "simulation_timestamp" in st.session_state:
        st.markdown("---")
        st.markdown("### 📜 Last Execution")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                f"""
                <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px;'>
                    <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>TIMESTAMP</p>
                    <p style='color: {COLORS["text"]}; margin: 5px 0 0 0;'>{st.session_state.simulation_timestamp.strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px;'>
                    <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>DURATION</p>
                    <p style='color: {COLORS["text"]}; margin: 5px 0 0 0;'>{st.session_state.simulation_duration:.2f} seconds</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


with tab1:
    render_execute_tab()

# ============================================================================
# TAB 2: ANALYZE (with lazy loading and fragments)
# ============================================================================

with tab2:
    # Check if results exist
    if st.session_state.simulation_result is None:
        st.warning("⚠️ No simulation results found. Please run a simulation first.")
    else:
        result = st.session_state.simulation_result
        metrics = st.session_state.simulation_metrics

        # Sub-tabs for different analysis views
        (
            analysis_tab1,
            analysis_tab2,
            analysis_tab3,
            analysis_tab4,
            analysis_tab5,
        ) = st.tabs(["📈 Metrics", "💹 Price Charts", "📊 Volume", "📉 Spread", "📋 Logs"])

        # Metrics Dashboard
        def render_metrics_tab():
            """Render metrics tab as a fragment."""
            st.markdown("### 📊 Key Metrics")

            prev = st.session_state.get("previous_metrics")

            def _mv(v: float | None, fmt: str = ".6f") -> str:
                return f"{v:{fmt}}" if v is not None else "N/A"

            if metrics is None:
                st.warning("Metrics not available.")
                return

            key_metrics: list[MetricItem] = []

            d = metric_delta(metrics.volatility, prev.volatility if prev else None)
            key_metrics.append(_metric_item("Volatility", _mv(metrics.volatility, ".4f"), delta=f"{d:+.4f}" if d is not None else None, field="volatility"))

            val = fmt_dollar(metrics.mean_spread, precision=4) if metrics.mean_spread is not None else "N/A"
            d = metric_delta(metrics.mean_spread, prev.mean_spread if prev else None)
            d_str = (("+" if d > 0 else "") + fmt_dollar(d, precision=4)) if d is not None else None
            key_metrics.append(_metric_item("Mean Spread", val, delta=d_str, field="mean_spread"))

            val = fmt_dollar(metrics.effective_spread, precision=4) if metrics.effective_spread is not None else "N/A"
            d = metric_delta(metrics.effective_spread, prev.effective_spread if prev else None)
            d_str = (("+" if d > 0 else "") + fmt_dollar(d, precision=4)) if d is not None else None
            key_metrics.append(_metric_item("Effective Spread", val, delta=d_str, field="effective_spread"))

            d = metric_delta(metrics.avg_bid_liquidity, prev.avg_bid_liquidity if prev else None)
            key_metrics.append(_metric_item("Avg Bid Liquidity", _mv(metrics.avg_bid_liquidity, ".2f"), delta=f"{d:+.2f}" if d is not None else None, field="avg_bid_liquidity"))

            d = metric_delta(metrics.avg_ask_liquidity, prev.avg_ask_liquidity if prev else None)
            key_metrics.append(_metric_item("Avg Ask Liquidity", _mv(metrics.avg_ask_liquidity, ".2f"), delta=f"{d:+.2f}" if d is not None else None, field="avg_ask_liquidity"))

            vol_str = f"{metrics.traded_volume:,}" if metrics.traded_volume is not None else "N/A"
            d = metric_delta(metrics.traded_volume, prev.traded_volume if prev else None)
            key_metrics.append(_metric_item("Traded Volume", vol_str, delta=f"{d:+,}" if d is not None else None, field="traded_volume"))

            _render_metric_columns(key_metrics, column_count=2)

            st.markdown("---")

            # Microstructure metrics
            st.markdown("### 🔬 Microstructure")
            microstructure_metrics: list[MetricItem] = []
            d = metric_delta(metrics.lob_imbalance_mean, prev.lob_imbalance_mean if prev else None)
            microstructure_metrics.append(_metric_item("LOB Imbalance", _mv(metrics.lob_imbalance_mean, ".4f"), delta=f"{d:+.4f}" if d is not None else None, field="lob_imbalance_mean"))

            d = metric_delta(metrics.lob_imbalance_std, prev.lob_imbalance_std if prev else None)
            microstructure_metrics.append(_metric_item("LOB Imb. σ", _mv(metrics.lob_imbalance_std, ".4f"), delta=f"{d:+.4f}" if d is not None else None, field="lob_imbalance_std"))

            d = metric_delta(metrics.vpin, prev.vpin if prev else None)
            microstructure_metrics.append(_metric_item("VPIN", _mv(metrics.vpin, ".4f"), delta=f"{d:+.4f}" if d is not None else None, field="vpin"))

            if metrics.resilience_mean_ns is not None:
                res_ms = metrics.resilience_mean_ns / 1e6
                d_ns = metric_delta(metrics.resilience_mean_ns, prev.resilience_mean_ns if prev else None)
                d_str = f"{d_ns / 1e6:+.1f}ms" if d_ns is not None else None
                microstructure_metrics.append(_metric_item("Resilience", f"{res_ms:.1f}ms", delta=d_str, field="resilience_mean_ns"))
            else:
                microstructure_metrics.append(_metric_item("Resilience", "N/A", delta=None, field="resilience_mean_ns"))

            d = metric_delta(metrics.market_ott_ratio, prev.market_ott_ratio if prev else None)
            microstructure_metrics.append(_metric_item("Market OTT", _mv(metrics.market_ott_ratio, ".2f"), delta=f"{d:+.2f}" if d is not None else None, field="market_ott_ratio"))

            avail_val = f"{metrics.pct_time_two_sided:.1%}" if metrics.pct_time_two_sided is not None else "N/A"
            d = metric_delta(metrics.pct_time_two_sided, prev.pct_time_two_sided if prev else None)
            microstructure_metrics.append(_metric_item("Availability", avail_val, delta=f"{d:+.1%}" if d is not None else None, field="pct_time_two_sided"))

            _render_metric_columns(microstructure_metrics, column_count=2)

            # ── Baseline Comparison (if available) ──────────────────
            bl_data = st.session_state.get("baseline_comparison")
            if bl_data is not None:
                bl_name = bl_data.get("name", "baseline")
                st.markdown("---")
                st.markdown(f"### ⚖️ Current Run  vs  Baseline: *{bl_name}*")
                st.caption(f"Side-by-side comparison of your **current run** against the saved baseline scenario **{bl_name}**. A positive Δ% means the current value is higher.")

                bm = bl_data["metrics"]
                snapshots = bl_data.get("config_snapshot")
                if snapshots is not None:
                    current_config = SimulationSettings.model_validate(snapshots["current"])
                    baseline_config = SimulationSettings.model_validate(snapshots["baseline"])
                    _render_comparison_context(current_config, baseline_config)
                cmp_df = build_comparison_table(metrics, bm)
                st.dataframe(cmp_df.set_index("Metric"), width="stretch")

            st.markdown("---")

            # Order book L1 summary
            st.markdown("### 📖 Order Book Summary (L1)")

            try:
                l1_df = get_l1_data(result)

                if not l1_df.empty:
                    st.dataframe(l1_df.head(20), width="stretch")

                    csv = l1_df.to_csv(index=True)
                    st.download_button(
                        label="⬇️ Download L1 Data (CSV)",
                        data=csv,
                        file_name="order_book_l1.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("No L1 order book data available")
            except Exception as e:
                st.error(f"Error loading L1 data: {str(e)}")

        with analysis_tab1:
            render_metrics_tab()

        # Price Charts
        def render_price_charts():
            """Render price charts as a fragment."""
            st.markdown("### 💹 Price Evolution")

            try:
                l1_df = get_l1_data(result)
                config = st.session_state.simulation_config

                if not l1_df.empty:
                    price_df = compute_price_data(l1_df)
                    historical_df: pd.DataFrame | None = None

                    if (
                        config is not None
                        and config.agents.oracle.oracle_type == OracleType.HISTORICAL
                        and config.agents.oracle.historical.provider_type == ProviderType.CSV
                        and config.agents.oracle.historical.csv.csv_path
                    ):
                        try:
                            csv_path = Path(config.agents.oracle.historical.csv.csv_path)
                            historical_df = load_historical_series_for_analysis(
                                csv_path=str(csv_path),
                                file_mtime=csv_path.stat().st_mtime,
                                file_size=csv_path.stat().st_size,
                                symbol=config.ticker,
                                sim_date=str(config.date),
                                start_time=str(config.start_time),
                                end_time=str(config.end_time),
                                price_unit=config.agents.oracle.historical.csv.price_unit.value,
                                source_timezone=config.agents.oracle.historical.csv.source_timezone,
                            )
                        except Exception as hist_err:
                            st.warning(f"Could not load historical series for overlay: {hist_err}")

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Bid, Ask, and Mid Prices", "Price Returns"),
                        row_heights=[0.7, 0.3],
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=price_df.index,
                            y=price_df["bid_price"],
                            name="Bid",
                            line={"color": COLORS["success"], "width": 1},
                            opacity=0.7,
                        ),
                        row=1,
                        col=1,
                    )

                    if historical_df is not None and not historical_df.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=historical_df["timestamp"],
                                y=historical_df["historical_price_cents"],
                                name="Historical",
                                line={"color": "#F59E0B", "width": 2, "dash": "dot"},
                            ),
                            row=1,
                            col=1,
                        )

                    fig.add_trace(
                        go.Scatter(
                            x=price_df.index,
                            y=price_df["ask_price"],
                            name="Ask",
                            line={"color": COLORS["danger"], "width": 1},
                            opacity=0.7,
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=price_df.index,
                            y=price_df["mid_price"],
                            name="Mid",
                            line={"color": COLORS["primary"], "width": 2},
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=price_df.index,
                            y=price_df["returns"],
                            name="Returns",
                            line={"color": COLORS["secondary"], "width": 1},
                            fill="tozeroy",
                            opacity=0.6,
                        ),
                        row=2,
                        col=1,
                    )

                    fig.update_layout(
                        height=700,
                        plot_bgcolor=COLORS["background"],
                        paper_bgcolor=COLORS["background"],
                        font={"color": COLORS["text"], "family": "Courier New"},
                        hovermode="x unified",
                        showlegend=True,
                        legend={
                            "bgcolor": COLORS["card_bg"],
                            "bordercolor": COLORS["border"],
                            "borderwidth": 1,
                        },
                    )

                    fig.update_xaxes(
                        gridcolor=COLORS["border"],
                        showgrid=True,
                        title_text="Time Index",
                        row=2,
                        col=1,
                    )

                    fig.update_yaxes(
                        gridcolor=COLORS["border"],
                        showgrid=True,
                        title_text="Price",
                        row=1,
                        col=1,
                    )

                    fig.update_yaxes(
                        gridcolor=COLORS["border"],
                        showgrid=True,
                        title_text="Returns",
                        row=2,
                        col=1,
                    )

                    chart_col, data_col = st.columns([3, 2])

                    with chart_col:
                        st.plotly_chart(fig, width="stretch")

                    with data_col:
                        st.markdown("#### Historical Series")
                        if historical_df is not None and not historical_df.empty:
                            st.dataframe(
                                historical_df.rename(columns={"historical_price_cents": "historical_price"}).head(120),
                                width="stretch",
                                hide_index=True,
                            )
                            st.caption("Showing sampled historical points used for chart overlay.")
                        else:
                            st.caption("Historical series overlay is available when Oracle Type is Historical + CSV provider.")

                    # Price statistics
                    st.markdown("### 📊 Price Statistics")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Min Bid", f"{price_df['bid_price'].min():.2f}")
                        st.metric("Max Bid", f"{price_df['bid_price'].max():.2f}")

                    with col2:
                        st.metric("Min Ask", f"{price_df['ask_price'].min():.2f}")
                        st.metric("Max Ask", f"{price_df['ask_price'].max():.2f}")

                    with col3:
                        st.metric("Min Mid", f"{price_df['mid_price'].min():.2f}")
                        st.metric("Max Mid", f"{price_df['mid_price'].max():.2f}")

                    with col4:
                        st.metric("Mean Mid", f"{price_df['mid_price'].mean():.2f}")
                        st.metric("Std Mid", f"{price_df['mid_price'].std():.2f}")

                else:
                    st.info("No price data available")

            except Exception as e:
                st.error(f"Error creating price charts: {str(e)}")

        with analysis_tab2:
            render_price_charts()

        # Volume Analysis
        def render_volume_analysis():
            """Render volume analysis as a fragment."""
            st.markdown("### 📊 Volume Analysis")

            try:
                l1_df = get_l1_data(result)

                if not l1_df.empty:
                    volume_df = compute_volume_data(l1_df)

                    fig = make_subplots(
                        rows=2,
                        cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=("Bid and Ask Volume", "Volume Imbalance"),
                        row_heights=[0.6, 0.4],
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=volume_df.index,
                            y=volume_df["bid_qty"],
                            name="Bid Volume",
                            line={"color": COLORS["success"], "width": 1.5},
                            fill="tozeroy",
                            opacity=0.6,
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=volume_df.index,
                            y=volume_df["ask_qty"],
                            name="Ask Volume",
                            line={"color": COLORS["danger"], "width": 1.5},
                            fill="tozeroy",
                            opacity=0.6,
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=volume_df.index,
                            y=volume_df["volume_imbalance"],
                            name="Volume Imbalance",
                            line={"color": COLORS["primary"], "width": 2},
                            fill="tozeroy",
                            opacity=0.7,
                        ),
                        row=2,
                        col=1,
                    )

                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color=COLORS["text_muted"],
                        row=2,  # pyright: ignore[reportArgumentType]
                        col=1,  # pyright: ignore[reportArgumentType]
                    )

                    fig.update_layout(
                        height=700,
                        plot_bgcolor=COLORS["background"],
                        paper_bgcolor=COLORS["background"],
                        font={"color": COLORS["text"], "family": "Courier New"},
                        hovermode="x unified",
                        showlegend=True,
                        legend={
                            "bgcolor": COLORS["card_bg"],
                            "bordercolor": COLORS["border"],
                            "borderwidth": 1,
                        },
                    )

                    fig.update_xaxes(
                        gridcolor=COLORS["border"],
                        showgrid=True,
                        title_text="Time Index",
                        row=2,
                        col=1,
                    )

                    fig.update_yaxes(
                        gridcolor=COLORS["border"],
                        showgrid=True,
                        title_text="Quantity",
                        row=1,
                        col=1,
                    )

                    fig.update_yaxes(
                        gridcolor=COLORS["border"],
                        showgrid=True,
                        title_text="Imbalance",
                        row=2,
                        col=1,
                    )

                    st.plotly_chart(fig, width="stretch")

                    # Volume statistics
                    st.markdown("### 📊 Volume Statistics")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Mean Bid Volume", f"{volume_df['bid_qty'].mean():.2f}")
                        st.metric("Std Bid Volume", f"{volume_df['bid_qty'].std():.2f}")

                    with col2:
                        st.metric("Mean Ask Volume", f"{volume_df['ask_qty'].mean():.2f}")
                        st.metric("Std Ask Volume", f"{volume_df['ask_qty'].std():.2f}")

                    with col3:
                        st.metric("Max Bid Volume", f"{volume_df['bid_qty'].max():.2f}")
                        st.metric("Max Ask Volume", f"{volume_df['ask_qty'].max():.2f}")

                    with col4:
                        st.metric(
                            "Mean Imbalance",
                            f"{volume_df['volume_imbalance'].mean():.4f}",
                        )
                        st.metric(
                            "Std Imbalance",
                            f"{volume_df['volume_imbalance'].std():.4f}",
                        )

                else:
                    st.info("No volume data available")

            except Exception as e:
                st.error(f"Error creating volume charts: {str(e)}")

        with analysis_tab3:
            render_volume_analysis()

        # Spread Analysis
        def render_spread_analysis():
            """Render spread analysis as a fragment."""
            st.markdown("### 📉 Spread Analysis")

            try:
                l1_df = get_l1_data(result)

                if not l1_df.empty:
                    spread_df = compute_spread_data(l1_df)

                    fig = make_subplots(
                        rows=2,
                        cols=2,
                        subplot_titles=(
                            "Spread Over Time",
                            "Spread Distribution",
                            "Spread (Basis Points)",
                            "Spread vs Mid Price",
                        ),
                        specs=[
                            [{"type": "scatter"}, {"type": "histogram"}],
                            [{"type": "scatter"}, {"type": "scatter"}],
                        ],
                        vertical_spacing=0.12,
                        horizontal_spacing=0.1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=spread_df.index,
                            y=spread_df["spread"],
                            name="Spread",
                            line={"color": COLORS["primary"], "width": 1.5},
                            fill="tozeroy",
                            opacity=0.7,
                        ),
                        row=1,
                        col=1,
                    )

                    fig.add_trace(
                        go.Histogram(
                            x=spread_df["spread"],
                            name="Distribution",
                            marker={"color": COLORS["secondary"]},
                            opacity=0.7,
                            nbinsx=50,
                        ),
                        row=1,
                        col=2,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=spread_df.index,
                            y=spread_df["spread_bps"],
                            name="Spread (bps)",
                            line={"color": COLORS["success"], "width": 1.5},
                            fill="tozeroy",
                            opacity=0.7,
                        ),
                        row=2,
                        col=1,
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=spread_df["mid_price"],
                            y=spread_df["spread"],
                            mode="markers",
                            name="Spread vs Price",
                            marker={
                                "color": COLORS["danger"],
                                "size": 3,
                                "opacity": 0.5,
                            },
                        ),
                        row=2,
                        col=2,
                    )

                    fig.update_layout(
                        height=800,
                        plot_bgcolor=COLORS["background"],
                        paper_bgcolor=COLORS["background"],
                        font={"color": COLORS["text"], "family": "Courier New"},
                        showlegend=False,
                    )

                    fig.update_xaxes(gridcolor=COLORS["border"], showgrid=True)
                    fig.update_yaxes(gridcolor=COLORS["border"], showgrid=True)

                    st.plotly_chart(fig, width="stretch")

                    # Spread statistics
                    st.markdown("### 📊 Spread Statistics")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Mean Spread", f"{spread_df['spread'].mean():.4f}")
                        st.metric("Median Spread", f"{spread_df['spread'].median():.4f}")

                    with col2:
                        st.metric("Std Spread", f"{spread_df['spread'].std():.4f}")
                        st.metric("Min Spread", f"{spread_df['spread'].min():.4f}")

                    with col3:
                        st.metric("Max Spread", f"{spread_df['spread'].max():.4f}")
                        st.metric("Mean Spread (bps)", f"{spread_df['spread_bps'].mean():.2f}")

                    with col4:
                        p25 = spread_df["spread"].quantile(0.25)
                        p75 = spread_df["spread"].quantile(0.75)
                        st.metric("25th Percentile", f"{p25:.4f}")
                        st.metric("75th Percentile", f"{p75:.4f}")

                else:
                    st.info("No spread data available")

            except Exception as e:
                st.error(f"Error creating spread charts: {str(e)}")

        with analysis_tab4:
            render_spread_analysis()

        # Execution Logs
        def render_logs():
            """Render logs as a fragment."""
            st.markdown("### 📋 Execution Logs")

            try:
                logs_df = get_logs_data(result)

                if not logs_df.empty:
                    # Filter controls
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if "agent_id" in logs_df.columns:
                            agent_ids = ["All"] + sorted(logs_df["agent_id"].unique().tolist())
                            selected_agent = st.selectbox("Filter by Agent", agent_ids, key="analyze_agent_filter")
                        else:
                            selected_agent = "All"

                    with col2:
                        if "EventType" in logs_df.columns or "event_type" in logs_df.columns:
                            event_col = "EventType" if "EventType" in logs_df.columns else "event_type"
                            event_types = ["All"] + sorted(logs_df[event_col].unique().tolist())
                            selected_event = st.selectbox(
                                "Filter by Event Type",
                                event_types,
                                key="analyze_event_filter",
                            )
                        else:
                            selected_event = "All"

                    with col3:
                        max_rows = st.number_input(
                            "Max Rows to Display",
                            min_value=10,
                            max_value=10000,
                            value=100,
                            step=10,
                            key="analyze_max_rows_logs",
                        )

                    # Apply filters
                    filtered_df = logs_df.copy()

                    if selected_agent != "All" and "agent_id" in logs_df.columns:
                        filtered_df = filtered_df[filtered_df["agent_id"] == selected_agent]

                    if selected_event != "All":
                        event_col = "EventType" if "EventType" in logs_df.columns else "event_type"
                        if event_col in logs_df.columns:
                            filtered_df = filtered_df[filtered_df[event_col] == selected_event]

                    # Display logs - convert object columns to strings to avoid Arrow errors
                    display_df = filtered_df.head(max_rows).copy()

                    # Convert object columns to strings for display
                    for col in display_df.columns:
                        if display_df[col].dtype == "object":
                            display_df[col] = display_df[col].astype(str)

                    st.dataframe(display_df, width="stretch", height=500)

                    # Log statistics
                    st.markdown("### 📊 Log Statistics")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Log Entries", f"{len(logs_df):,}")

                    with col2:
                        st.metric("Filtered Entries", f"{len(filtered_df):,}")

                    with col3:
                        if "agent_id" in logs_df.columns:
                            st.metric("Unique Agents", f"{logs_df['agent_id'].nunique():,}")

                    # Download button
                    csv = filtered_df.to_csv(index=True)
                    st.download_button(
                        label="⬇️ Download Filtered Logs (CSV)",
                        data=csv,
                        file_name="execution_logs.csv",
                        mime="text/csv",
                    )

                else:
                    st.info("No execution logs available")

            except Exception as e:
                st.error(f"Error loading logs: {str(e)}")

        with analysis_tab5:
            render_logs()
