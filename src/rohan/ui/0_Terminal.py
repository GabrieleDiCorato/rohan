"""ABIDES Simulation Terminal - Rohan Edition.

Visual design follows the abides-ui project (glassmorphism cards, Carbon Dark
theme, 4-tab analytical layout).  Backend uses Rohan's config system,
SimulationService, AnalysisService, saved scenarios and baseline comparison.
"""

from __future__ import annotations

import html as _html
import logging
import time
import traceback
from collections.abc import Hashable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

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
from rohan.simulation.abides_impl.hasufel_output import HasufelOutput
from rohan.simulation.simulation_service import SimulationService
from rohan.ui import charts, metrics
from rohan.ui.utils.baseline_comparison import (
    build_baseline_context_table,
    ensure_baseline_comparable,
    get_baseline_compatibility_issues,
)
from rohan.ui.utils.components import agent_recipe_bar, execution_console, metric_row
from rohan.ui.utils.metric_display import build_comparison_table
from rohan.ui.utils.presets import get_preset_config, get_preset_names
from rohan.ui.utils.startup import ensure_db_initialized
from rohan.ui.utils.theme import CARBON_DARK_CSS

_logger = logging.getLogger(__name__)

# -- DB init & page config -----------------------------------------------------

ensure_db_initialized()
_scenario_repo = ScenarioRepository()

st.set_page_config(page_title="ABIDES Terminal", page_icon="\u2b21", layout="wide")
st.markdown(f"<style>{CARBON_DARK_CSS}</style>", unsafe_allow_html=True)

# -- Header (abides-ui style) --------------------------------------------------

st.markdown(
    '<div style="display:flex;align-items:center;gap:12px;margin-bottom:4px">'
    '<span style="font-size:1.6rem;opacity:0.25;line-height:1">\u2b21</span>'
    "<span style=\"font-family:'JetBrains Mono',monospace;font-size:1.3rem;"
    'font-weight:700;color:#E0E0E0;letter-spacing:0.08em">ABIDES TERMINAL</span>'
    "<span style=\"font-family:'JetBrains Mono',monospace;font-size:0.6rem;"
    "background:rgba(0,112,255,0.15);color:#0070FF;padding:2px 8px;"
    'border-radius:4px;border:1px solid rgba(0,112,255,0.25)">rohan</span>'
    "</div>"
    "<div style=\"font-family:'Inter',sans-serif;font-size:0.78rem;"
    'color:#6B7280;margin-bottom:12px">'
    "Agent-Based Interactive Discrete Event Simulation"
    "</div>",
    unsafe_allow_html=True,
)

# -- Session state --------------------------------------------------------------

if "simulation_config" not in st.session_state:
    st.session_state.simulation_config = None
if "draft_config" not in st.session_state:
    st.session_state.draft_config = SimulationSettings()
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

# -- Cached helpers -------------------------------------------------------------


@st.cache_data
def load_historical_csv_preview(
    csv_path: str,
    file_mtime: float,
    file_size: int,
) -> tuple[pd.DataFrame, str, int]:
    _ = (file_mtime, file_size)
    header = pd.read_csv(csv_path, nrows=0)
    columns = set(header.columns)
    price_col = "price_cents" if "price_cents" in columns else "price" if "price" in columns else ""
    if "timestamp" not in columns or not price_col:
        msg = "CSV must contain 'timestamp' and either 'price_cents' or 'price' columns"
        raise ValueError(msg)

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
    _ = (file_mtime, file_size)
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


# ==============================================================================
# SIDEBAR -- PREPARATION DESK
# ==============================================================================


def compact_input(label: str, widget_type: str, key: str, **kwargs: Any) -> Any:
    """Render a label+input pair in a compact two-column layout."""
    actual_key = f"{key}_{st.session_state.config_reset_counter}"
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"**{label}:**")
    with col2:
        if widget_type == "text":
            return st.text_input(
                label,
                key=actual_key,
                label_visibility="collapsed",
                **kwargs,
            )
        if widget_type == "number":
            return st.number_input(
                label,
                key=actual_key,
                label_visibility="collapsed",
                **kwargs,
            )
        if widget_type == "selectbox":
            return st.selectbox(
                label,
                key=actual_key,
                label_visibility="collapsed",
                **kwargs,
            )
        if widget_type == "checkbox":
            return st.checkbox(
                label,
                key=actual_key,
                label_visibility="collapsed",
                **kwargs,
            )
        msg = f"Unknown widget type: {widget_type}"
        raise ValueError(msg)


def _clear_config_widget_keys() -> None:
    st.session_state.config_reset_counter += 1


with st.sidebar:
    st.markdown(
        "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.75rem;font-weight:600;color:#8A919B;letter-spacing:0.1em;margin-bottom:12px\">PREPARATION DESK</div>",
        unsafe_allow_html=True,
    )

    # -- Presets ----------------------------------------------------------------
    st.subheader("Template")
    preset_name = st.selectbox(
        "Preset",
        options=["Custom"] + get_preset_names(),
        help="Load a preset configuration from the library.",
        key="preset_selector",
    )
    if st.button("Load Preset", type="primary", use_container_width=True) and preset_name != "Custom":
        new_config = get_preset_config(preset_name)
        st.session_state.draft_config = new_config.model_copy(deep=True)
        _clear_config_widget_keys()
        st.success("\u2705 Loaded: " + str(preset_name))
        st.rerun()

    # -- Saved scenarios --------------------------------------------------------
    st.divider()

    @st.dialog("Load Saved Scenario", width="large")
    def load_scenario_dialog() -> None:
        try:
            _saved_list = _scenario_repo.list_scenarios()
        except Exception:
            _saved_list = []
        if not _saved_list:
            st.info("No saved scenarios yet.")
            return
        st.markdown(f"**{len(_saved_list)} saved scenario(s)**")
        st.markdown("---")
        for _sc in _saved_list:
            _cfg = _sc.full_config or {}
            _sc_col1, _sc_col2, _sc_col3 = st.columns([5, 1, 1])
            with _sc_col1:
                st.markdown(f"**{_sc.name}**")
                if _sc.description:
                    st.caption(_sc.description)
                st.caption(
                    "\U0001f4c5 "
                    + str(_cfg.get("date", "?"))
                    + " \u00b7 "
                    + str(_cfg.get("start_time", "?"))
                    + "\u2013"
                    + str(_cfg.get("end_time", "?"))
                    + " \u00b7 \U0001f3f7\ufe0f "
                    + str(_cfg.get("ticker", "?"))
                    + " \u00b7 Created: "
                    + _sc.created_at.strftime("%Y-%m-%d %H:%M"),
                )
            with _sc_col2:
                if st.button(
                    "Load",
                    key=f"load_sc_{_sc.scenario_id}",
                    use_container_width=True,
                    type="primary",
                ):
                    st.session_state.draft_config = SimulationSettings.model_validate(_sc.full_config)
                    _clear_config_widget_keys()
                    st.toast("\u2705 Loaded: " + str(_sc.name))
                    st.rerun()
            with _sc_col3:
                if st.button(
                    "\U0001f5d1\ufe0f",
                    key=f"del_sc_{_sc.scenario_id}",
                    use_container_width=True,
                    help="Delete '" + str(_sc.name) + "'",
                ):
                    _scenario_repo.delete_scenario(_sc.scenario_id)
                    st.rerun()
            st.divider()

    if st.button("Load Saved Scenario", use_container_width=True):
        load_scenario_dialog()

    st.divider()

    # -- Market settings --------------------------------------------------------
    config = st.session_state.draft_config

    st.subheader("Market")
    ticker = compact_input("Ticker", "text", "cfg_ticker", value=config.ticker)
    date = compact_input("Date", "text", "cfg_date", value=config.date)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        start_time = st.text_input(
            "Open",
            value=config.start_time,
            key=f"cfg_start_time_{st.session_state.config_reset_counter}",
            label_visibility="collapsed",
        )
    with col_t2:
        end_time = st.text_input(
            "Close",
            value=config.end_time,
            key=f"cfg_end_time_{st.session_state.config_reset_counter}",
            label_visibility="collapsed",
        )

    seed = compact_input(
        "Seed",
        "number",
        "cfg_seed",
        value=config.seed,
        min_value=0,
    )
    starting_cash = compact_input(
        "Starting Cash",
        "number",
        "cfg_starting_cash",
        value=config.starting_cash,
        min_value=0,
        step=1000000,
    )

    _log_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
    stdout_log_level = compact_input(
        "Log Level",
        "selectbox",
        "cfg_log_level",
        options=_log_levels,
        index=_log_levels.index(config.stdout_log_level),
    )
    log_orders = compact_input(
        "Log Orders",
        "checkbox",
        "cfg_log_orders",
        value=config.log_orders,
    )
    computation_delay_ns = compact_input(
        "Comp Delay (ns)",
        "number",
        "cfg_computation_delay_ns",
        value=config.computation_delay_ns,
        min_value=0,
    )

    st.divider()

    # -- Exchange ---------------------------------------------------------------
    with st.expander("\U0001f3e6 Exchange"):
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

    # -- Noise ------------------------------------------------------------------
    with st.expander("\U0001f4e2 Noise Agents"):
        noise_num_agents = compact_input(
            "Count",
            "number",
            "cfg_noise_num_agents",
            value=config.agents.noise.num_agents,
            min_value=0,
        )

    # -- Value ------------------------------------------------------------------
    with st.expander("\U0001f48e Value Agents"):
        value_num_agents = compact_input(
            "Count",
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

    # -- Market Makers ----------------------------------------------------------
    with st.expander("\U0001f3af Market Makers"):
        amm_num_agents = compact_input(
            "Count",
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

    # -- Momentum ---------------------------------------------------------------
    with st.expander("\U0001f4c8 Momentum Agents"):
        momentum_num_agents = compact_input(
            "Count",
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

    # -- Oracle -----------------------------------------------------------------
    with st.expander("\U0001f52e Oracle"):
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
                index=[pt.value for pt in ProviderType].index(
                    config.agents.oracle.historical.provider_type.value,
                ),
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
                        help="Select a CSV dataset.",
                        key=f"cfg_oracle_historical_csv_{st.session_state.config_reset_counter}",
                    )
                    if historical_csv:
                        historical_csv_path = str(
                            datasets_dir / f"{historical_csv}.csv",
                        )
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
                    st.warning("No datasets found in database")
                else:
                    historical_db_dataset_id = st.selectbox(
                        "Database Dataset",
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
                    ),
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
                        "API Timezone",
                        "text",
                        "cfg_oracle_historical_api_timezone",
                        value=historical_source_timezone,
                    ),
                )

            historical_interpolation = compact_input(
                "Interpolation",
                "selectbox",
                "cfg_oracle_historical_interp",
                options=[m.value for m in InterpolationMode],
                index=[m.value for m in InterpolationMode].index(
                    config.agents.oracle.historical.interpolation.value,
                ),
            )
            historical_recenter = compact_input(
                "Recenter to R_Bar",
                "checkbox",
                "cfg_oracle_historical_recenter",
                value=config.agents.oracle.historical.recenter_r_bar,
            )

            if historical_provider_type == ProviderType.CSV and historical_csv_path:
                csv_path_obj = Path(historical_csv_path)
                try:
                    preview_df, price_col, total_rows = load_historical_csv_preview(
                        str(csv_path_obj),
                        csv_path_obj.stat().st_mtime,
                        csv_path_obj.stat().st_size,
                    )
                    preview_df = preview_df.set_index("timestamp")
                    st.markdown("**Preview**")
                    preview_fig = charts.csv_preview_chart(preview_df, price_col)
                    st.plotly_chart(
                        preview_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                    if total_rows > len(preview_df):
                        st.caption(
                            f"Showing {len(preview_df):,} of {total_rows:,} rows.",
                        )
                except Exception as e:
                    st.error(f"Cannot preview dataset: {e}")

    # -- Latency ----------------------------------------------------------------
    with st.expander("\u23f1\ufe0f Latency"):
        _latency_options = [lt.value for lt in LatencyType]
        latency_type = compact_input(
            "Type",
            "selectbox",
            "cfg_latency_type",
            options=_latency_options,
            index=_latency_options.index(str(config.latency.type)),
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

    st.divider()

    # -- Apply Configuration ----------------------------------------------------
    if st.button(
        "\U0001f680 Apply Configuration",
        type="primary",
        use_container_width=True,
    ):
        try:
            agents = AgentSettings(
                exchange=ExchangeAgentSettings(
                    book_logging=bool(exchange_book_logging),
                    book_log_depth=int(exchange_book_log_depth),
                    stream_history_length=int(exchange_stream_history),
                    exchange_log_orders=bool(exchange_log_orders),
                    pipeline_delay_ns=int(exchange_pipeline_delay),
                    computation_delay_ns=int(exchange_computation_delay),
                ),
                noise=NoiseAgentSettings(num_agents=int(noise_num_agents)),
                value=ValueAgentSettings(
                    num_agents=int(value_num_agents),
                    r_bar=int(value_r_bar),
                    kappa=float(value_kappa),
                    lambda_a=float(value_lambda_a),
                ),
                adaptive_market_maker=AdaptiveMarketMakerSettings(
                    num_agents=int(amm_num_agents),
                    window_size=(amm_window_size if amm_window_size == "adaptive" else int(amm_window_size)),
                    pov=float(amm_pov),
                    num_ticks=int(amm_num_ticks),
                    wake_up_freq=str(amm_wake_up_freq),
                    poisson_arrival=bool(amm_poisson),
                    min_order_size=int(amm_min_order_size),
                    skew_beta=int(amm_skew_beta),
                    price_skew=int(amm_price_skew),
                    level_spacing=int(amm_level_spacing),
                    spread_alpha=float(amm_spread_alpha),
                    backstop_quantity=int(amm_backstop_qty),
                ),
                momentum=MomentumAgentSettings(
                    num_agents=int(momentum_num_agents),
                    min_size=int(momentum_min_size),
                    max_size=int(momentum_max_size),
                    wake_up_freq=str(momentum_wake_up_freq),
                    poisson_arrival=bool(momentum_poisson),
                ),
                oracle=(
                    OracleSettings(
                        oracle_type=OracleType.SYNTHETIC,
                        kappa=float(oracle_kappa),
                        sigma_s=float(oracle_sigma_s),
                        fund_vol=float(oracle_fund_vol),
                        megashock_lambda_a=float(oracle_megashock_lambda),
                        megashock_mean=int(oracle_megashock_mean),
                        megashock_var=int(oracle_megashock_var),
                    )
                    if oracle_type == OracleType.SYNTHETIC
                    else OracleSettings(
                        oracle_type=OracleType.HISTORICAL,
                        historical=HistoricalOracleSettings(
                            provider_type=historical_provider_type,
                            interpolation=InterpolationMode(
                                str(historical_interpolation),
                            ),
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
                jitter=float(latency_jitter),
                jitter_clip=float(latency_jitter_clip),
                jitter_unit=float(latency_jitter_unit),
            )
            new_config = SimulationSettings(
                template=config.template,
                date=str(date),
                start_time=str(start_time),
                end_time=str(end_time),
                seed=int(seed),
                ticker=str(ticker),
                starting_cash=int(starting_cash),
                stdout_log_level=str(stdout_log_level),
                log_orders=bool(log_orders),
                computation_delay_ns=int(computation_delay_ns),
                agents=agents,
                latency=latency,
            )
            st.session_state.simulation_config = new_config
            st.session_state.draft_config = new_config.model_copy(deep=True)
            st.session_state.simulation_result = None
            st.session_state.simulation_metrics = None
            st.session_state.baseline_comparison = None
            st.session_state.pop("simulation_timestamp", None)
            st.session_state.pop("simulation_duration", None)
            st.session_state.pop("simulation_seed", None)
            st.success("\u2705 Configuration applied!")
            st.rerun()
        except Exception as e:
            st.error("\u274c Error: " + str(e))

    if st.button("\U0001f504 Reset to Default", use_container_width=True):
        st.session_state.draft_config = SimulationSettings()
        _clear_config_widget_keys()
        st.success("\u2705 Reset to defaults!")
        st.rerun()

    # -- Session status indicator -----------------------------------------------
    st.divider()
    _has_result = st.session_state.simulation_result is not None
    _result_ticker = st.session_state.simulation_config.ticker if st.session_state.simulation_config else ""
    if _has_result:
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.68rem;"
            "background:rgba(0,200,5,0.08);color:#00C805;padding:6px 10px;"
            "border-radius:4px;border:1px solid rgba(0,200,5,0.2);"
            'margin-bottom:8px">'
            "\u25cf Results loaded \u2014 " + _html.escape(str(_result_ticker)) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.68rem;"
            "color:#6B7280;padding:6px 10px;border-radius:4px;"
            "border:1px solid rgba(255,255,255,0.06);"
            'margin-bottom:8px">'
            "\u25cb No results yet</div>",
            unsafe_allow_html=True,
        )

    if "simulation_seed" in st.session_state:
        st.markdown("**\U0001f3b2 Last Run Seed**")
        st.code(str(st.session_state.simulation_seed), language=None)
        st.caption("Set this seed to reproduce results.")

    # -- About ------------------------------------------------------------------
    with st.expander("\u2139\ufe0f About", expanded=False):
        st.markdown(
            "**ABIDES Terminal** \u2014 Rohan Edition\n\n"
            "Interactive agent-based market simulation dashboard.\n\n"
            "**Quick start:** pick a template, customize parameters, "
            "hit **Run Simulation**.\n\n"
            "**Tips**\n"
            "- Press **R** to re-run the app\n"
            "- Use presets for fast experiments\n"
            "- Enable *Log Orders* for the Order Book tab",
        )


# ==============================================================================
# MAIN AREA -- Agent Summary + Run + Results
# ==============================================================================

if st.session_state.simulation_config is None:
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;color:#6B7280;'
        "font-family:'Inter',sans-serif\">"
        '<div style="font-size:2.5rem;margin-bottom:12px;opacity:0.2">'
        "\u2b21</div>"
        '<div style="font-size:1rem;color:#A0A8B4;margin-bottom:20px">'
        "No configuration applied yet</div>"
        '<div style="display:flex;justify-content:center;gap:32px;'
        'flex-wrap:wrap;margin-top:8px">'
        '<div style="text-align:center;max-width:160px">'
        '<div style="font-size:1.3rem;margin-bottom:4px">1</div>'
        '<div style="font-size:0.78rem">Pick a <b>template</b> in the '
        "Preparation Desk</div></div>"
        '<div style="text-align:center;max-width:160px">'
        '<div style="font-size:1.3rem;margin-bottom:4px">2</div>'
        '<div style="font-size:0.78rem">Customize <b>agents</b> and '
        "parameters</div></div>"
        '<div style="text-align:center;max-width:160px">'
        '<div style="font-size:1.3rem;margin-bottom:4px">3</div>'
        '<div style="font-size:0.78rem">Hit <b style="color:#0070FF">'
        "Apply Configuration</b></div></div>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

sim_config = st.session_state.simulation_config

# -- Agent composition summary + Recipe bar ------------------------------------

agent_counts = {
    "Noise": sim_config.agents.noise.num_agents,
    "Value": sim_config.agents.value.num_agents,
    "Market Maker": sim_config.agents.adaptive_market_maker.num_agents,
    "Momentum": sim_config.agents.momentum.num_agents,
}
total_agents = sum(agent_counts.values()) + 1  # +1 for exchange

st.markdown(
    metric_row(
        [
            {"label": "Ticker", "value": sim_config.ticker},
            {"label": "Date", "value": sim_config.date},
            {
                "label": "Window",
                "value": sim_config.start_time + " \u2013 " + sim_config.end_time,
            },
            {"label": "Seed", "value": str(sim_config.seed)},
            {"label": "Total Agents", "value": str(total_agents)},
        ]
    ),
    unsafe_allow_html=True,
)

_recipe_configs = {name: {"count": cnt} for name, cnt in agent_counts.items() if cnt > 0}
if _recipe_configs:
    st.markdown(agent_recipe_bar(_recipe_configs), unsafe_allow_html=True)

# -- Save scenario + Baseline + Run -------------------------------------------

st.divider()

# Save scenario
_save_col1, _save_col2 = st.columns([3, 1])
with _save_col1:
    _save_name = st.text_input(
        "Scenario name",
        placeholder="e.g. High-Vol Stress Test",
        key="save_scenario_name",
        label_visibility="collapsed",
    )
with _save_col2:
    _save_clicked = st.button(
        "\U0001f4be Save",
        use_container_width=True,
        disabled=not bool(_save_name and _save_name.strip()),
    )
if _save_clicked and _save_name and _save_name.strip():
    try:
        _scenario_repo.save_scenario(
            name=_save_name.strip(),
            full_config=sim_config.model_dump(),
        )
        st.toast("\u2705 Scenario '" + _save_name.strip() + "' saved!")
    except Exception as _save_err:
        st.error("Could not save: " + str(_save_err))

# Baseline comparison selector
try:
    _saved_scenarios = _scenario_repo.list_scenarios()
except Exception:
    _saved_scenarios = []

_baseline_options: list[str] = ["None"]
_baseline_id_map: dict[str, str] = {}
_baseline_scenario_map: dict[str, Any] = {}
for _sc in _saved_scenarios:
    _baseline_options.append(_sc.name)
    _baseline_id_map[_sc.name] = str(_sc.scenario_id)
    _baseline_scenario_map[_sc.name] = _sc

_baseline_choice = st.selectbox(
    "Baseline scenario",
    options=_baseline_options,
    index=0,
    help="Pick a saved scenario to run as baseline.",
    key="baseline_selector",
)
st.session_state.baseline_scenario_id = _baseline_id_map.get(_baseline_choice)

_selected_baseline = _baseline_scenario_map.get(_baseline_choice)
_baseline_preview_config: SimulationSettings | None = None
_baseline_preview_error: str | None = None
_baseline_issues: list[str] = []

if _selected_baseline is not None:
    if _selected_baseline.full_config is None:
        _baseline_preview_error = "Baseline has no saved configuration."
    else:
        try:
            _baseline_preview_config = SimulationSettings.model_validate(
                _selected_baseline.full_config,
            )
            _baseline_issues = get_baseline_compatibility_issues(
                sim_config,
                _baseline_preview_config,
            )
        except Exception as _preview_err:
            _baseline_preview_error = "Could not load baseline: " + str(_preview_err)

    if _baseline_preview_error is not None:
        st.error(_baseline_preview_error)
    elif _baseline_preview_config is not None:
        context_df = build_baseline_context_table(
            sim_config,
            _baseline_preview_config,
        )
        st.dataframe(context_df, use_container_width=True, hide_index=True)
        if _baseline_issues:
            st.warning(
                "Comparison skipped until checks match:\n" + "\n".join("- " + i for i in _baseline_issues),
            )
        else:
            st.success("Baseline eligible for comparison.")

st.divider()

# Run Simulation
run_clicked = st.button(
    "\U0001f680 Run Simulation",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.simulation_running,
)
st.caption("\U0001f4a1 Tip: press **R** to re-run the app at any time.")

if run_clicked:
    st.session_state.simulation_running = True
    _log_lines: list[str] = []

    with st.status("Executing simulation...", expanded=True) as _status:
        try:
            st.markdown(
                "<div style=\"font-family:'JetBrains Mono',monospace;"
                'font-size:0.72rem;color:#00C805;opacity:0.8">'
                "Agents: " + str(total_agents) + " | Seed: " + str(sim_config.seed) + "<br>Kernel boot sequence initiated..."
                "</div>",
                unsafe_allow_html=True,
            )

            _log_lines.append("Seed: " + str(sim_config.seed))
            _log_lines.append("Agents: " + str(total_agents))
            _log_lines.append("Kernel started")

            t0 = time.perf_counter()
            service = SimulationService()
            sim_result = service.run_simulation(sim_config)

            if sim_result.error is not None:
                raise sim_result.error

            result = sim_result.result
            assert result is not None  # noqa: S101

            wall_time = time.perf_counter() - t0
            _log_lines.append(f"Simulation finished in {wall_time:.2f}s")

            rohan_metrics = AnalysisService.compute_metrics(result)
            _log_lines.append("Metrics computed")
            _log_lines.append("Results ready.")

            st.session_state.previous_metrics = st.session_state.get(
                "last_run_metrics",
            )
            st.session_state.simulation_result = result
            st.session_state.simulation_metrics = rohan_metrics
            st.session_state.last_run_metrics = rohan_metrics
            st.session_state.simulation_duration = wall_time
            st.session_state.wall_time = wall_time
            st.session_state.simulation_timestamp = datetime.now()  # noqa: DTZ005
            st.session_state.simulation_seed = sim_config.seed

            _status.update(
                label=f"Simulation complete \u2014 {wall_time:.2f}s",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            _log_lines.append("[ERROR] " + str(e))
            _status.update(
                label="\u274c Simulation Failed",
                state="error",
                expanded=True,
            )
            st.error("Simulation failed: " + str(e))
            with st.expander("Error Details"):
                st.code(traceback.format_exc())

        finally:
            st.session_state.simulation_running = False

    st.markdown(
        execution_console(_log_lines, st.session_state.get("wall_time")),
        unsafe_allow_html=True,
    )

    # -- Baseline run -----------------------------------------------------------
    if _selected_baseline is not None and _baseline_preview_error is None and _baseline_preview_config is not None and not _baseline_issues and st.session_state.simulation_result is not None:
        bl_status = st.status(
            "Running baseline: " + str(_selected_baseline.name) + "\u2026",
            expanded=True,
        )
        try:
            with bl_status:
                ensure_baseline_comparable(sim_config, _baseline_preview_config)
                bl_service = SimulationService()
                bl_result = bl_service.run_simulation(_baseline_preview_config)
                if bl_result.error is not None:
                    raise bl_result.error
                bl_output = bl_result.result
                assert bl_output is not None  # noqa: S101
                bl_metrics = AnalysisService.compute_metrics(bl_output)
                st.session_state.baseline_comparison = {
                    "name": _selected_baseline.name,
                    "metrics": bl_metrics,
                    "timestamp": datetime.now(),  # noqa: DTZ005
                    "config_snapshot": {
                        "current": sim_config.model_dump(),
                        "baseline": _baseline_preview_config.model_dump(),
                    },
                }
            bl_status.update(
                label="\u2705 Baseline Complete!",
                state="complete",
                expanded=False,
            )
        except BaselineComparisonError as bl_err:
            bl_status.update(
                label="\u26a0\ufe0f Baseline Skipped",
                state="error",
                expanded=True,
            )
            st.warning("Baseline skipped: " + str(bl_err))
            st.session_state.baseline_comparison = None
        except Exception as bl_err:
            bl_status.update(
                label="\u274c Baseline Failed",
                state="error",
                expanded=True,
            )
            st.error("Baseline failed: " + str(bl_err))
    elif _selected_baseline is None:
        st.session_state.baseline_comparison = None


# ==============================================================================
# RESULTS -- Summary KPI + 4-Tab Analytics
# ==============================================================================

output = st.session_state.simulation_result
if output is None:
    st.stop()

rohan_metrics = st.session_state.simulation_metrics
wall_time_val: float = st.session_state.get("wall_time", 0.0)

# Access hasufel result for rich analytics
hasufel_result = None
hasufel_market = None
l1 = None
order_df = None

if isinstance(output, HasufelOutput):
    hasufel_result = output.hasufel_result
    hasufel_market = hasufel_result.markets.get(sim_config.ticker)
    if hasufel_market and hasufel_market.l1_series:
        l1 = metrics.derive_l1(hasufel_market.l1_series.as_dataframe())
    order_df = metrics.extract_order_log(hasufel_result)

# -- Summary KPI cards ---------------------------------------------------------

if l1 is not None and hasufel_market is not None:
    sm = metrics.compute_summary(hasufel_market, l1)
    _mid_str = f"${sm.mid_close:,.2f}" if sm.mid_close is not None else "N/A"
    _spread_str = f"${sm.spread_close:,.2f}" if sm.spread_close is not None else "N/A"
    _vwap_str = f"${sm.vwap:,.2f}" if sm.vwap is not None else "N/A"
    _rv_str = f"{sm.realized_vol:.6f}" if sm.realized_vol is not None else "N/A"
    _range_str = f"${sm.price_range:,.2f}" if sm.price_range is not None else "N/A"
    st.markdown(
        metric_row(
            [
                {"label": "Mid Price", "value": _mid_str},
                {"label": "Bid-Ask Spread", "value": _spread_str},
                {"label": "VWAP", "value": _vwap_str},
                {"label": "Volume", "value": f"{sm.volume:,}"},
                {"label": "Realized Vol (\u03c3)", "value": _rv_str},
                {"label": "Price Range", "value": _range_str},
                {"label": "Wall-Clock", "value": f"{wall_time_val:.1f}s"},
            ]
        ),
        unsafe_allow_html=True,
    )

# -- Tabbed analytics ----------------------------------------------------------

tab_micro, tab_alpha, tab_book, tab_config = st.tabs(
    [
        "Market Microstructure",
        "Agent Alpha",
        "Order Book Dynamics",
        "Configuration",
    ]
)
st.caption(
    "**Microstructure** \u2014 prices, spreads, volatility \u00b7 "
    "**Alpha** \u2014 agent P&L, equity curves \u00b7 "
    "**Order Book** \u2014 flow stats, trade attribution \u00b7 "
    "**Config** \u2014 JSON export",
)

# ==============================================================================
# TAB 1: MARKET MICROSTRUCTURE
# ==============================================================================

with tab_micro:
    if l1 is not None:
        st.plotly_chart(
            charts.price_series(l1.time, l1.bid, l1.ask, l1.mid),
            use_container_width=True,
        )

        _mean = l1.spread.mean()
        avg_spread = float(_mean) if pd.notna(_mean) else 0.0
        rv = metrics.compute_rolling_vol(l1.log_returns)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.plotly_chart(
                charts.spread_over_time(l1.time, l1.spread, avg_spread),
                use_container_width=True,
            )
        with mc2:
            if rv is not None:
                rolling_vol_series, window = rv
                ret_time = l1.time.iloc[l1.log_returns.index]
                st.plotly_chart(
                    charts.rolling_volatility(ret_time, rolling_vol_series, window),
                    use_container_width=True,
                )

        pressure = metrics.compute_book_pressure(l1.l1_df)
        rs = metrics.compute_return_stats(l1.log_returns)

        mc3, mc4 = st.columns(2)
        with mc3:
            st.plotly_chart(
                charts.book_pressure(l1.time, pressure),
                use_container_width=True,
            )
        with mc4:
            if rs is not None:
                st.plotly_chart(
                    charts.returns_histogram(l1.log_returns),
                    use_container_width=True,
                )

        # Spread statistics
        ss = metrics.compute_spread_stats(l1.spread, l1.mid)
        _ss_cards: list[dict[str, str]] = []
        if ss.mean is not None:
            _ss_cards.append({"label": "Mean Spread", "value": f"${ss.mean:.4f}"})
        if ss.median is not None:
            _ss_cards.append(
                {"label": "Median Spread", "value": f"${ss.median:.4f}"},
            )
        if ss.max is not None:
            _ss_cards.append({"label": "Max Spread", "value": f"${ss.max:.4f}"})
        if ss.std is not None:
            _ss_cards.append({"label": "Spread Std", "value": f"${ss.std:.4f}"})
        if ss.mean_pct is not None:
            _ss_cards.append(
                {"label": "Mean Spread %", "value": f"{ss.mean_pct:.4f}%"},
            )
        if ss.median_pct is not None:
            _ss_cards.append(
                {"label": "Median Spread %", "value": f"{ss.median_pct:.4f}%"},
            )
        if _ss_cards:
            st.markdown(metric_row(_ss_cards), unsafe_allow_html=True)
        if ss.n_one_sided > 0:
            pct_one_sided = ss.n_one_sided / ss.n_total * 100
            st.caption(
                f"\u26a0 {ss.n_one_sided} of {ss.n_total} ticks ({pct_one_sided:.1f}%) had a one-sided book.",
            )

        # Market quality
        if hasufel_market is not None:
            both_sides = 100 - max(
                hasufel_market.liquidity.pct_time_no_bid,
                hasufel_market.liquidity.pct_time_no_ask,
            )
            last_trade = hasufel_market.liquidity.last_trade_cents
            _lt_str = f"${last_trade / 100:.2f}" if last_trade is not None else "N/A"
            st.markdown(
                metric_row(
                    [
                        {
                            "label": "% Time No Bid",
                            "value": f"{hasufel_market.liquidity.pct_time_no_bid:.1f}%",
                        },
                        {
                            "label": "% Time No Ask",
                            "value": f"{hasufel_market.liquidity.pct_time_no_ask:.1f}%",
                        },
                        {"label": "% Time Two-Sided", "value": f"{both_sides:.1f}%"},
                        {"label": "Last Trade", "value": _lt_str},
                    ]
                ),
                unsafe_allow_html=True,
            )

        # Returns distribution
        if rs is not None:
            st.markdown(
                metric_row(
                    [
                        {"label": "Mean Return", "value": f"{rs.mean:.8f}"},
                        {"label": "Std Dev", "value": f"{rs.std:.6f}"},
                        {"label": "Skewness", "value": f"{rs.skewness:.4f}"},
                        {"label": "Excess Kurtosis", "value": f"{rs.kurtosis:.4f}"},
                    ]
                ),
                unsafe_allow_html=True,
            )

        # Advanced microstructure
        if hasufel_result is not None:
            micro = metrics.compute_microstructure_metrics(
                hasufel_result,
                sim_config.ticker,
            )
            if micro is not None:
                _micro_cards: list[dict[str, str]] = []
                if micro.mean_spread_cents is not None:
                    _micro_cards.append(
                        {
                            "label": "Mean Spread (\u00a2)",
                            "value": f"{micro.mean_spread_cents:.2f}",
                        }
                    )
                if micro.volatility_ann is not None:
                    _micro_cards.append(
                        {
                            "label": "Ann. Volatility",
                            "value": f"{micro.volatility_ann:.4f}",
                        }
                    )
                if micro.sharpe_ratio is not None:
                    _micro_cards.append(
                        {
                            "label": "Sharpe Ratio",
                            "value": f"{micro.sharpe_ratio:.2f}",
                        }
                    )
                if micro.avg_bid_liquidity is not None:
                    _micro_cards.append(
                        {
                            "label": "Avg Bid Depth",
                            "value": f"{micro.avg_bid_liquidity:,.0f}",
                        }
                    )
                if micro.avg_ask_liquidity is not None:
                    _micro_cards.append(
                        {
                            "label": "Avg Ask Depth",
                            "value": f"{micro.avg_ask_liquidity:,.0f}",
                        }
                    )
                if micro.lob_imbalance_mean is not None:
                    _micro_cards.append(
                        {
                            "label": "LOB Imbalance \u03bc",
                            "value": f"{micro.lob_imbalance_mean:+.4f}",
                        }
                    )
                if micro.lob_imbalance_std is not None:
                    _micro_cards.append(
                        {
                            "label": "LOB Imbalance \u03c3",
                            "value": f"{micro.lob_imbalance_std:.4f}",
                        }
                    )
                if micro.vpin is not None:
                    _micro_cards.append(
                        {
                            "label": "VPIN",
                            "value": f"{micro.vpin:.4f}",
                        }
                    )
                if micro.resilience_ns is not None:
                    _micro_cards.append(
                        {
                            "label": "Resilience (ms)",
                            "value": f"{micro.resilience_ns / 1e6:,.1f}",
                        }
                    )
                if micro.effective_spread_cents is not None:
                    _micro_cards.append(
                        {
                            "label": "Effective Spread (\u00a2)",
                            "value": f"{micro.effective_spread_cents:.2f}",
                        }
                    )
                if micro.market_ott_ratio is not None:
                    _micro_cards.append(
                        {
                            "label": "Market OTT",
                            "value": f"{micro.market_ott_ratio:.2f}",
                        }
                    )
                if micro.pct_time_two_sided is not None:
                    _micro_cards.append(
                        {
                            "label": "% Two-Sided",
                            "value": f"{micro.pct_time_two_sided:.1f}%",
                        }
                    )
                if _micro_cards:
                    st.markdown(
                        "<div style=\"font-family:'Inter',sans-serif;"
                        "font-size:0.72rem;color:#6B7280;margin:12px 0 4px 0;"
                        'text-transform:uppercase;letter-spacing:0.06em">'
                        "Advanced Microstructure</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        metric_row(_micro_cards),
                        unsafe_allow_html=True,
                    )

        # Baseline comparison in microstructure tab
        bl_data = st.session_state.get("baseline_comparison")
        if bl_data is not None and rohan_metrics is not None:
            st.markdown("---")
            st.markdown(
                "### \u2696\ufe0f Current vs Baseline: *" + str(bl_data.get("name", "baseline")) + "*",
            )
            snapshots = bl_data.get("config_snapshot")
            if snapshots is not None:
                cur_cfg = SimulationSettings.model_validate(snapshots["current"])
                bl_cfg = SimulationSettings.model_validate(snapshots["baseline"])
                ctx_df = build_baseline_context_table(cur_cfg, bl_cfg)
                st.dataframe(
                    ctx_df,
                    use_container_width=True,
                    hide_index=True,
                )
            cmp_df = build_comparison_table(rohan_metrics, bl_data["metrics"])
            st.dataframe(
                cmp_df.set_index("Metric"),
                use_container_width=True,
            )

        with st.expander("Raw L1 data"):
            st.dataframe(l1.l1_df, use_container_width=True)
    else:
        st.warning("L1 price series not available.")


# ==============================================================================
# TAB 2: AGENT ALPHA
# ==============================================================================

with tab_alpha:
    if hasufel_result is not None and hasufel_result.agents:
        agent_df = metrics.build_agent_dataframe(hasufel_result)
        exec_agents = metrics.get_execution_agents(hasufel_result)

        _cat_counts = agent_df["Category"].value_counts()
        if len(_cat_counts) > 0:
            _cat_cards = [{"label": cat.title(), "value": str(count)} for cat, count in _cat_counts.items() if cat]
            if _cat_cards:
                st.markdown(metric_row(_cat_cards), unsafe_allow_html=True)

        if exec_agents:
            exec_summary = metrics.compute_execution_summary(exec_agents)
            if exec_summary is not None:
                _dd = f"${exec_summary.max_drawdown_cents / 100:,.2f}" if exec_summary.max_drawdown_cents is not None else "N/A"
                st.markdown(
                    metric_row(
                        [
                            {
                                "label": "Exec Agents",
                                "value": str(len(exec_agents)),
                            },
                            {
                                "label": "Total Filled",
                                "value": (f"{exec_summary.total_filled:,} / {exec_summary.total_target:,}"),
                            },
                            {
                                "label": "Avg Fill Rate",
                                "value": f"{exec_summary.avg_fill_rate:.1f}%",
                            },
                            {
                                "label": "Avg VWAP Slippage",
                                "value": (f"{exec_summary.avg_vwap_slippage_bps:.2f} bps"),
                            },
                            {"label": "Max Drawdown", "value": _dd},
                        ]
                    ),
                    unsafe_allow_html=True,
                )

        agg = metrics.compute_agent_performance(agent_df)
        st.dataframe(agg, use_container_width=True, hide_index=True)

        # Rich agent metrics
        if isinstance(output, HasufelOutput):
            try:
                rich = output.rich_metrics
                rs_rich = metrics.compute_rich_summary(rich)
                _rich_cards: list[dict[str, str]] = []
                if rs_rich.avg_sharpe is not None:
                    _rich_cards.append(
                        {
                            "label": "Avg Sharpe",
                            "value": f"{rs_rich.avg_sharpe:.4f}",
                        }
                    )
                if rs_rich.avg_ott_ratio is not None:
                    _rich_cards.append(
                        {
                            "label": "Avg OTT Ratio",
                            "value": f"{rs_rich.avg_ott_ratio:.2f}",
                        }
                    )
                if rs_rich.avg_inventory_std is not None:
                    _rich_cards.append(
                        {
                            "label": "Avg Inventory \u03c3",
                            "value": f"{rs_rich.avg_inventory_std:.2f}",
                        }
                    )
                _rich_cards.append(
                    {
                        "label": "Total Trades (Rich)",
                        "value": f"{rs_rich.total_trade_count:,}",
                    }
                )
                if rs_rich.avg_fill_slippage_bps is not None:
                    _rich_cards.append(
                        {
                            "label": "Avg Fill Slippage",
                            "value": f"{rs_rich.avg_fill_slippage_bps:.1f} bps",
                        }
                    )
                if _rich_cards:
                    st.markdown(
                        "<div style=\"font-family:'Inter',sans-serif;font-size:0.72rem;color:#6B7280;margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em\">Rich Agent Analytics</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        metric_row(_rich_cards),
                        unsafe_allow_html=True,
                    )

                rich_df = metrics.build_rich_agent_dataframe(rich)
                if not rich_df.empty:
                    _ra1, _ra2 = st.columns(2)
                    with _ra1:
                        st.plotly_chart(
                            charts.rich_agent_comparison(rich_df),
                            use_container_width=True,
                        )
                    with _ra2:
                        if rich.fills:
                            fill_df = metrics.build_fill_records_df(rich.fills)
                            st.plotly_chart(
                                charts.fill_slippage_histogram(fill_df),
                                use_container_width=True,
                            )
                    with st.expander("Rich agent detail table"):
                        st.dataframe(
                            rich_df,
                            use_container_width=True,
                            hide_index=True,
                        )
            except Exception:
                pass  # rich metrics may not be available

        # P&L box plot + equity curves
        if exec_agents:
            aa1, aa2 = st.columns(2)
            with aa1:
                st.plotly_chart(
                    charts.pnl_box_plot(agent_df),
                    use_container_width=True,
                )
            with aa2:
                for agent in exec_agents[:1]:
                    ec_df = metrics.build_equity_curve_df(agent)
                    if ec_df is not None:
                        st.plotly_chart(
                            charts.equity_curve(ec_df, agent.agent_name),
                            use_container_width=True,
                        )
                    else:
                        st.caption("No equity curve data.")
        else:
            st.plotly_chart(
                charts.pnl_box_plot(agent_df),
                use_container_width=True,
            )

        hold_agg = metrics.build_holdings_table(hasufel_result)
        if hold_agg is not None:
            st.dataframe(hold_agg, use_container_width=True, hide_index=True)

        if len(exec_agents) > 1:
            slip_data = [
                {
                    "name": a.agent_name,
                    "vwap_slippage_bps": (a.execution_metrics.vwap_slippage_bps or 0.0),
                }
                for a in exec_agents
            ]
            st.plotly_chart(
                charts.slippage_comparison(slip_data),
                use_container_width=True,
            )

        if exec_agents:
            with st.expander(
                "Execution agent details (" + str(len(exec_agents)) + ")",
            ):
                for agent in exec_agents:
                    detail_df = metrics.build_execution_detail_df(agent)
                    if len(detail_df) > 0:
                        st.caption(
                            "**" + str(agent.agent_name) + "** (" + str(agent.agent_type) + ")",
                        )
                        st.dataframe(
                            detail_df,
                            use_container_width=True,
                            hide_index=True,
                        )
                    ec_df = metrics.build_equity_curve_df(agent)
                    if ec_df is not None:
                        st.plotly_chart(
                            charts.equity_curve(ec_df, agent.agent_name),
                            use_container_width=True,
                        )

        with st.expander("Agent Leaderboard"):
            st.dataframe(
                metrics.build_leaderboard(agent_df),
                use_container_width=True,
            )
    else:
        st.info("No agent data available.")


# ==============================================================================
# TAB 3: ORDER BOOK DYNAMICS
# ==============================================================================

with tab_book:
    _has_orders = order_df is not None and len(order_df) > 0
    _has_trades = hasufel_market is not None and hasufel_market.trades is not None and len(hasufel_market.trades) > 0

    if _has_orders:
        ofs = metrics.compute_order_flow_stats(order_df)
        st.markdown(
            metric_row(
                [
                    {"label": "Orders Submitted", "value": f"{ofs.total_submitted:,}"},
                    {"label": "Executions", "value": f"{ofs.executed:,}"},
                    {"label": "Cancellations", "value": f"{ofs.cancelled:,}"},
                    {"label": "Fill Rate", "value": f"{ofs.fill_rate:.1f}%"},
                    {"label": "Cancel Rate", "value": f"{ofs.cancel_rate:.1f}%"},
                ]
            ),
            unsafe_allow_html=True,
        )

        if "EventType" in order_df.columns:
            event_counts = order_df["EventType"].value_counts()
            ob1, ob2 = st.columns(2)
            with ob1:
                st.plotly_chart(
                    charts.event_type_pie(event_counts),
                    use_container_width=True,
                )
            with ob2:
                if "side" in order_df.columns:
                    submitted = order_df[order_df["EventType"] == "ORDER_SUBMITTED"]
                    side_counts = submitted["side"].value_counts()
                    st.plotly_chart(
                        charts.side_balance(side_counts),
                        use_container_width=True,
                    )

        imb_df = metrics.compute_cumulative_imbalance(order_df)
        if imb_df is not None:
            flow_time = pd.to_datetime(imb_df["EventTime"], unit="ns")
            st.plotly_chart(
                charts.cumulative_imbalance(flow_time, imb_df["cum_imbalance"]),
                use_container_width=True,
            )

        if "agent_type" in order_df.columns:
            exec_df = order_df[order_df["EventType"] == "ORDER_EXECUTED"]
            if len(exec_df) > 0 and "quantity" in exec_df.columns:
                vol_by_type = exec_df.groupby("agent_type")["quantity"].sum().sort_values(ascending=True)
                st.plotly_chart(
                    charts.volume_by_agent_type(vol_by_type),
                    use_container_width=True,
                )

        with st.expander("Raw order logs"):
            st.dataframe(order_df, use_container_width=True)

    # Trade attribution
    if _has_trades and hasufel_result is not None:
        attr_df = metrics.build_trade_attribution_df(
            hasufel_market.trades,
            hasufel_result.agents,
        )
        mts = metrics.compute_maker_taker_summary(attr_df)

        if not _has_orders:
            st.markdown(
                metric_row(
                    [
                        {"label": "Total Trades", "value": f"{mts.total_trades:,}"},
                        {
                            "label": "Maker Types",
                            "value": str(len(mts.maker_volume_by_type)),
                        },
                        {
                            "label": "Taker Types",
                            "value": str(len(mts.taker_volume_by_type)),
                        },
                    ]
                ),
                unsafe_allow_html=True,
            )

        ob3, ob4 = st.columns(2)
        with ob3:
            st.plotly_chart(
                charts.maker_taker_volume(
                    mts.maker_volume_by_type,
                    mts.taker_volume_by_type,
                ),
                use_container_width=True,
            )
        with ob4:
            st.plotly_chart(
                charts.trade_price_scatter(attr_df),
                use_container_width=True,
            )

        if len(attr_df) > 0 and "maker_type" in attr_df.columns and "taker_type" in attr_df.columns:
            st.plotly_chart(
                charts.trade_attribution_heatmap(attr_df),
                use_container_width=True,
            )

        with st.expander("Raw trade attribution data"):
            st.dataframe(attr_df, use_container_width=True)

    # Fill-level analysis
    if isinstance(output, HasufelOutput):
        try:
            rich_ob = output.rich_metrics
            if rich_ob.fills:
                st.markdown(
                    "<div style=\"font-family:'Inter',sans-serif;font-size:0.72rem;color:#6B7280;margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em\">Fill-Level Analysis</div>",
                    unsafe_allow_html=True,
                )
                fill_df = metrics.build_fill_records_df(rich_ob.fills)
                if "slippage (bps)" in fill_df.columns:
                    _slip = fill_df["slippage (bps)"].dropna()
                    if len(_slip) > 0:
                        st.markdown(
                            metric_row(
                                [
                                    {
                                        "label": "Fills Analysed",
                                        "value": f"{len(fill_df):,}",
                                    },
                                    {
                                        "label": "Avg Fill Slippage",
                                        "value": f"{_slip.mean():.1f} bps",
                                    },
                                    {
                                        "label": "Median Slippage",
                                        "value": f"{_slip.median():.1f} bps",
                                    },
                                    {
                                        "label": "Max Slippage",
                                        "value": f"{_slip.max():.0f} bps",
                                    },
                                    {
                                        "label": "% Adverse",
                                        "value": (f"{(_slip > 0).mean() * 100:.1f}%"),
                                    },
                                ]
                            ),
                            unsafe_allow_html=True,
                        )
                st.plotly_chart(
                    charts.fill_slippage_histogram(fill_df),
                    use_container_width=True,
                )
                _as_cols = [c for c in fill_df.columns if c.startswith("AS ") and c.endswith("(bps)")]
                if _as_cols:
                    st.plotly_chart(
                        charts.adverse_selection_by_window(fill_df),
                        use_container_width=True,
                    )
                with st.expander("Raw fill records"):
                    st.dataframe(fill_df, use_container_width=True)
        except Exception:
            pass

    # L2 Order Book Depth
    if hasufel_market is not None and hasufel_market.l2_series is not None:
        st.markdown(
            "<div style=\"font-family:'Inter',sans-serif;font-size:0.72rem;color:#6B7280;margin:12px 0 4px 0;text-transform:uppercase;letter-spacing:0.06em\">L2 Order Book Depth</div>",
            unsafe_allow_html=True,
        )
        l2_df = hasufel_market.l2_series.as_dataframe()
        if len(l2_df) > 0:
            _l2_mid = l1.mid if l1 is not None else None
            st.plotly_chart(
                charts.l2_depth_heatmap(l2_df, mid=_l2_mid),
                use_container_width=True,
            )
            _l2_c1, _l2_c2 = st.columns(2)
            with _l2_c1:
                st.plotly_chart(
                    charts.l2_depth_profile(l2_df),
                    use_container_width=True,
                )
            with _l2_c2:
                _bid_levels = l2_df[l2_df["side"] == "bid"]
                _ask_levels = l2_df[l2_df["side"] == "ask"]
                _n_snapshots = l2_df["time_ns"].nunique()
                _max_depth = int(l2_df["level"].max()) + 1 if len(l2_df) > 0 else 0
                _avg_bid_depth = float(_bid_levels["qty"].mean()) if len(_bid_levels) > 0 else 0
                _avg_ask_depth = float(_ask_levels["qty"].mean()) if len(_ask_levels) > 0 else 0
                st.markdown(
                    metric_row(
                        [
                            {"label": "L2 Snapshots", "value": f"{_n_snapshots:,}"},
                            {
                                "label": "Max Depth",
                                "value": str(_max_depth) + " levels",
                            },
                            {
                                "label": "Avg Bid Qty/Level",
                                "value": f"{_avg_bid_depth:,.0f}",
                            },
                            {
                                "label": "Avg Ask Qty/Level",
                                "value": f"{_avg_ask_depth:,.0f}",
                            },
                        ]
                    ),
                    unsafe_allow_html=True,
                )
            with st.expander("Raw L2 data"):
                st.dataframe(l2_df, use_container_width=True)
        else:
            st.info("L2 data is empty.")

    if not _has_orders and not _has_trades:
        st.warning(
            "Order log data not available. Enable **Log Orders** in the sidebar to populate this tab.",
        )


# ==============================================================================
# TAB 4: CONFIGURATION
# ==============================================================================

with tab_config:
    config_json = sim_config.model_dump_json(indent=2)
    st.download_button(
        "\U0001f4e5 Download config.json",
        data=config_json,
        file_name="abides_config.json",
        mime="application/json",
    )
    st.code(config_json, language="json")
