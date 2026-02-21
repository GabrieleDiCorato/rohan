"""ABIDES Simulation Terminal - Optimized Single Page Application.

Performance optimizations:
- @st.cache_data for expensive data operations
- @st.fragment for isolated component updates
- Lazy loading of analysis components
- Conditional rendering
"""

import contextlib
import traceback
from datetime import datetime

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from rohan.config import SimulationSettings
from rohan.config.agent_settings import (
    AdaptiveMarketMakerSettings,
    AgentSettings,
    ExchangeAgentSettings,
    MomentumAgentSettings,
    NoiseAgentSettings,
    OracleSettings,
    ValueAgentSettings,
)
from rohan.config.latency_settings import LatencyModelSettings, LatencyType
from rohan.framework.analysis_service import AnalysisService
from rohan.framework.database import initialize_database
from rohan.framework.scenario_repository import ScenarioRepository
from rohan.simulation.simulation_service import SimulationService
from rohan.ui.utils.presets import get_preset_config, get_preset_names
from rohan.ui.utils.theme import COLORS, apply_theme

# Ensure DB tables exist
with contextlib.suppress(Exception):  # DB may not be configured; features degrade gracefully
    initialize_database()

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
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["returns"] = df["mid_price"].pct_change().fillna(0)
    return df


@st.cache_data
def compute_volume_data(_l1_df):
    """Cache volume calculations."""
    df = _l1_df.copy()
    df["volume_imbalance"] = (df["bid_qty"] - df["ask_qty"]) / (df["bid_qty"] + df["ask_qty"])
    return df


@st.cache_data
def compute_spread_data(_l1_df):
    """Cache spread calculations."""
    df = _l1_df.copy()
    df["spread"] = df["ask_price"] - df["bid_price"]
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["spread_bps"] = (df["spread"] / df["mid_price"]) * 10000
    return df


# ============================================================================
# SIDEBAR: CONFIGURATION
# ============================================================================


def compact_input(label, widget_type, key, **kwargs):
    """Create a compact key-value input with label and widget on the same line."""
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown(f"**{label}:**")
    with col2:
        if widget_type == "text":
            return st.text_input("", key=key, label_visibility="collapsed", **kwargs)
        if widget_type == "number":
            return st.number_input("", key=key, label_visibility="collapsed", **kwargs)
        if widget_type == "selectbox":
            return st.selectbox("", key=key, label_visibility="collapsed", **kwargs)
        if widget_type == "checkbox":
            return st.checkbox("", key=key, label_visibility="collapsed", **kwargs)
        raise ValueError(f"Unknown widget type: {widget_type}")


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

    if st.button("Load Preset", type="primary", use_container_width=True) and preset_name != "Custom":
        st.session_state.draft_config = get_preset_config(preset_name)
        st.success(f"✅ Loaded preset: {preset_name} (click 'Apply Configuration' to use it)")
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
                if st.button("Load", key=f"load_sc_{_sc.scenario_id}", use_container_width=True, type="primary"):
                    from rohan.config import SimulationSettings as _SimSettings

                    st.session_state.draft_config = _SimSettings.model_validate(_sc.full_config)
                    st.session_state.simulation_config = st.session_state.draft_config
                    st.toast(f"✅ Loaded: {_sc.name}")
                    st.rerun()
            with _sc_col3:
                if st.button("🗑️", key=f"del_sc_{_sc.scenario_id}", use_container_width=True, help=f"Delete '{_sc.name}'"):
                    _scenario_repo.delete_scenario(_sc.scenario_id)
                    st.rerun()
            st.divider()

    if st.button("Load Configuration", use_container_width=True):
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
    if st.button("🚀 Apply Configuration", type="primary", use_container_width=True):
        try:
            # Build configuration
            agents = AgentSettings(
                exchange=ExchangeAgentSettings(
                    book_logging=exchange_book_logging,
                    book_log_depth=exchange_book_log_depth,
                    stream_history_length=exchange_stream_history,
                    exchange_log_orders=exchange_log_orders,
                    pipeline_delay_ns=exchange_pipeline_delay,
                    computation_delay_ns=exchange_computation_delay,
                ),
                noise=NoiseAgentSettings(num_agents=noise_num_agents),
                value=ValueAgentSettings(
                    num_agents=value_num_agents,
                    r_bar=value_r_bar,
                    kappa=value_kappa,
                    lambda_a=value_lambda_a,
                ),
                adaptive_market_maker=AdaptiveMarketMakerSettings(
                    num_agents=amm_num_agents,
                    window_size=(amm_window_size if amm_window_size == "adaptive" else int(amm_window_size)),
                    pov=amm_pov,
                    num_ticks=amm_num_ticks,
                    wake_up_freq=amm_wake_up_freq,
                    poisson_arrival=amm_poisson,
                    min_order_size=amm_min_order_size,
                    skew_beta=amm_skew_beta,
                    price_skew=amm_price_skew,
                    level_spacing=amm_level_spacing,
                    spread_alpha=amm_spread_alpha,
                    backstop_quantity=amm_backstop_qty,
                ),
                momentum=MomentumAgentSettings(
                    num_agents=momentum_num_agents,
                    min_size=momentum_min_size,
                    max_size=momentum_max_size,
                    wake_up_freq=momentum_wake_up_freq,
                    poisson_arrival=momentum_poisson,
                ),
                oracle=OracleSettings(
                    kappa=oracle_kappa,
                    sigma_s=oracle_sigma_s,
                    fund_vol=oracle_fund_vol,
                    megashock_lambda_a=oracle_megashock_lambda,
                    megashock_mean=oracle_megashock_mean,
                    megashock_var=oracle_megashock_var,
                ),
            )

            latency = LatencyModelSettings(
                type=LatencyType(latency_type),
                jitter=latency_jitter,
                jitter_clip=latency_jitter_clip,
                jitter_unit=latency_jitter_unit,
            )

            new_config = SimulationSettings(
                date=date,
                start_time=start_time,
                end_time=end_time,
                seed=seed,
                ticker=ticker,
                starting_cash=starting_cash,
                stdout_log_level=stdout_log_level,
                log_orders=log_orders,
                computation_delay_ns=computation_delay_ns,
                agents=agents,
                latency=latency,
            )

            st.session_state.simulation_config = new_config
            st.success("✅ Configuration applied!")
            st.rerun()  # Force full page refresh to update Execute tab

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

    # Reset button
    if st.button("🔄 Reset to Default", use_container_width=True):
        st.session_state.simulation_config = SimulationSettings()
        st.success("✅ Reset to defaults!")
        st.rerun()


# Render sidebar
with st.sidebar:
    render_sidebar_config()

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
        _save_clicked = st.button("💾 Save", use_container_width=True, disabled=not bool(_save_name and _save_name.strip()))
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
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.date}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["secondary"]};'>
                <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>TIME RANGE</p>
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.start_time} - {config.end_time}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["success"]};'>
                <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>TICKER</p>
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.ticker}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style='background-color: {COLORS["card_bg"]}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS["danger"]};'>
                <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;'>SEED</p>
                <p style='color: {COLORS["text"]}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.seed}</p>
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
            st.markdown(f"- **Kappa:** {config.agents.oracle.kappa:.2e}")
            st.markdown(f"- **Sigma S:** {config.agents.oracle.sigma_s:.2e}")
            st.markdown(f"- **Fund Vol:** {config.agents.oracle.fund_vol:.2e}")
            st.markdown(f"- **Megashock Lambda:** {config.agents.oracle.megashock_lambda_a:.2e}")
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

    # Run Simulation
    st.markdown("### 🚀 Run Simulation")

    run_button = st.button(
        "▶️ RUN SIMULATION",
        type="primary",
        use_container_width=True,
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

                # Save to session state
                st.session_state.simulation_result = result
                st.session_state.simulation_metrics = metrics
                st.session_state.simulation_duration = duration
                st.session_state.simulation_timestamp = datetime.now()

                # Clear caches to force fresh data
                get_l1_data.clear()
                get_logs_data.clear()
                compute_price_data.clear()
                compute_volume_data.clear()
                compute_spread_data.clear()

            status_container.update(label="✅ Simulation Complete!", state="complete", expanded=False)

            st.success(f"🎉 Simulation completed successfully in {duration:.2f} seconds!")

            # Show quick metrics
            st.markdown("### 📊 Quick Metrics")

            col1, col2, col3, col4 = st.columns(4)

            def _m(v: float | None, fmt: str = ".6f") -> str:
                return f"{v:{fmt}}" if v is not None else "N/A"

            def _m_dollar(v: float | None) -> str:
                """Format a cents value as dollars."""
                return f"${v / 100:,.4f}" if v is not None else "N/A"

            with col1:
                st.metric("Volatility", _m(metrics.volatility))

            with col2:
                st.metric("Mean Spread", _m_dollar(metrics.mean_spread))

            with col3:
                st.metric("Avg Bid Liquidity", _m(metrics.avg_bid_liquidity, ".2f"))

            with col4:
                st.metric("Avg Ask Liquidity", _m(metrics.avg_ask_liquidity, ".2f"))

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

            def _mv(v: float | None, fmt: str = ".6f") -> str:
                return f"{v:{fmt}}" if v is not None else "N/A"

            def _mv_dollar(v: float | None) -> str:
                """Format a cents value as dollars."""
                return f"${v / 100:,.4f}" if v is not None else "N/A"

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Volatility", _mv(metrics.volatility))
                st.metric("Mean Spread", _mv_dollar(metrics.mean_spread))

            with col2:
                st.metric("Effective Spread", _mv_dollar(metrics.effective_spread))
                st.metric("Avg Bid Liquidity", _mv(metrics.avg_bid_liquidity, ".2f"))

            with col3:
                st.metric("Avg Ask Liquidity", _mv(metrics.avg_ask_liquidity, ".2f"))
                vol_str = f"{metrics.traded_volume:,}" if metrics.traded_volume is not None else "N/A"
                st.metric("Traded Volume", vol_str)

            st.markdown("---")

            # Order book L1 summary
            st.markdown("### 📖 Order Book Summary (L1)")

            try:
                l1_df = get_l1_data(result)

                if not l1_df.empty:
                    st.dataframe(l1_df.head(20), use_container_width=True)

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

                if not l1_df.empty:
                    price_df = compute_price_data(l1_df)

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

                    st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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

                    st.plotly_chart(fig, use_container_width=True)

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
