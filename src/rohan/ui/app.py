"""ABIDES Simulation Terminal - Optimized Single Page Application.

Performance optimizations:
- @st.cache_data for expensive data operations
- @st.fragment for isolated component updates
- Lazy loading of analysis components
- Conditional rendering
"""

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
from rohan.simulation.simulation_service import SimulationService
from rohan.ui.utils.presets import get_preset_config, get_preset_names
from rohan.ui.utils.theme import COLORS, apply_theme

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
    st.session_state.simulation_config = None

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


@st.fragment
def render_sidebar_config():
    """Render configuration sidebar as a fragment for isolated updates."""
    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: {COLORS['primary']}; font-size: 1.8rem; margin: 0;'>
                ⚡ ABIDES
            </h1>
            <p style='color: {COLORS['secondary']}; font-size: 1.2rem; margin: 5px 0 0 0; letter-spacing: 2px;'>
                TERMINAL
            </p>
            <hr style='border-color: {COLORS['border']}; margin: 15px 0;'>
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
        st.session_state.simulation_config = get_preset_config(preset_name)
        st.success(f"✅ Loaded: {preset_name}")
        st.rerun()

    st.markdown("---")

    # Initialize config
    config = SimulationSettings() if st.session_state.simulation_config is None else st.session_state.simulation_config

    # Simulation Parameters
    with st.expander("🎯 SIMULATION", expanded=True):
        date = st.text_input("Date (YYYYMMDD)", value=config.date, key="cfg_date")
        start_time = st.text_input("Start Time", value=config.start_time, key="cfg_start_time")
        end_time = st.text_input("End Time", value=config.end_time, key="cfg_end_time")
        seed = st.number_input("Random Seed", value=config.seed, min_value=0, key="cfg_seed")
        ticker = st.text_input("Ticker", value=config.ticker, key="cfg_ticker")
        starting_cash = st.number_input(
            "Starting Cash (cents)",
            value=config.starting_cash,
            min_value=0,
            step=1000000,
            key="cfg_starting_cash",
        )
        stdout_log_level = st.selectbox(
            "Log Level",
            options=["DEBUG", "INFO", "WARNING", "ERROR"],
            index=1,
            key="cfg_log_level",
        )
        log_orders = st.checkbox("Log Orders", value=config.log_orders, key="cfg_log_orders")
        computation_delay_ns = st.number_input(
            "Computation Delay (ns)",
            value=config.computation_delay_ns,
            min_value=0,
            key="cfg_computation_delay_ns",
        )

    # Exchange Agent
    with st.expander("🏦 EXCHANGE"):
        exchange_book_logging = st.checkbox(
            "Book Logging",
            value=config.agents.exchange.book_logging,
            key="cfg_exchange_book_logging",
        )
        exchange_book_log_depth = st.number_input(
            "Book Log Depth",
            value=config.agents.exchange.book_log_depth,
            min_value=1,
            key="cfg_exchange_book_log_depth",
        )
        exchange_stream_history = st.number_input(
            "Stream History",
            value=config.agents.exchange.stream_history_length,
            min_value=1,
            key="cfg_exchange_stream_history",
        )
        exchange_log_orders = st.checkbox(
            "Exchange Log Orders",
            value=config.agents.exchange.exchange_log_orders,
            key="cfg_exchange_log_orders",
        )
        exchange_pipeline_delay = st.number_input(
            "Pipeline Delay (ns)",
            value=config.agents.exchange.pipeline_delay_ns,
            min_value=0,
            key="cfg_exchange_pipeline_delay",
        )
        exchange_computation_delay = st.number_input(
            "Computation Delay (ns)",
            value=config.agents.exchange.computation_delay_ns,
            min_value=0,
            key="cfg_exchange_computation_delay",
        )

    # Noise Agents
    with st.expander("📢 NOISE AGENTS"):
        noise_num_agents = st.number_input(
            "Number of Agents",
            value=config.agents.noise.num_agents,
            min_value=0,
            key="cfg_noise_num_agents",
        )

    # Value Agents
    with st.expander("💎 VALUE AGENTS"):
        value_num_agents = st.number_input(
            "Number of Agents",
            value=config.agents.value.num_agents,
            min_value=0,
            key="cfg_value_num_agents",
        )
        value_r_bar = st.number_input(
            "R Bar",
            value=config.agents.value.r_bar,
            min_value=0,
            key="cfg_value_r_bar",
        )
        value_kappa = st.number_input(
            "Kappa",
            value=config.agents.value.kappa,
            format="%.2e",
            key="cfg_value_kappa",
        )
        value_lambda_a = st.number_input(
            "Lambda A",
            value=config.agents.value.lambda_a,
            format="%.2e",
            key="cfg_value_lambda_a",
        )

    # Adaptive Market Makers
    with st.expander("🎯 MARKET MAKERS"):
        amm_num_agents = st.number_input(
            "Number of Agents",
            value=config.agents.adaptive_market_maker.num_agents,
            min_value=0,
            key="cfg_amm_num_agents",
        )
        amm_window_size = st.text_input(
            "Window Size",
            value=str(config.agents.adaptive_market_maker.window_size),
            key="cfg_amm_window_size",
        )
        amm_pov = st.number_input(
            "POV",
            value=config.agents.adaptive_market_maker.pov,
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
            key="cfg_amm_pov",
        )
        amm_num_ticks = st.number_input(
            "Num Ticks",
            value=config.agents.adaptive_market_maker.num_ticks,
            min_value=1,
            key="cfg_amm_num_ticks",
        )
        amm_wake_up_freq = st.text_input(
            "Wake Up Freq",
            value=config.agents.adaptive_market_maker.wake_up_freq,
            key="cfg_amm_wake_up_freq",
        )
        amm_poisson = st.checkbox(
            "Poisson Arrival",
            value=config.agents.adaptive_market_maker.poisson_arrival,
            key="cfg_amm_poisson",
        )
        amm_min_order_size = st.number_input(
            "Min Order Size",
            value=config.agents.adaptive_market_maker.min_order_size,
            min_value=1,
            key="cfg_amm_min_order_size",
        )
        amm_level_spacing = st.number_input(
            "Level Spacing",
            value=config.agents.adaptive_market_maker.level_spacing,
            min_value=1,
            key="cfg_amm_level_spacing",
        )
        amm_skew_beta = st.number_input(
            "Skew Beta",
            value=config.agents.adaptive_market_maker.skew_beta,
            key="cfg_amm_skew_beta",
        )
        amm_price_skew = st.number_input(
            "Price Skew",
            value=config.agents.adaptive_market_maker.price_skew,
            key="cfg_amm_price_skew",
        )
        amm_spread_alpha = st.number_input(
            "Spread Alpha",
            value=config.agents.adaptive_market_maker.spread_alpha,
            min_value=0.0,
            max_value=1.0,
            format="%.2f",
            key="cfg_amm_spread_alpha",
        )
        amm_backstop_qty = st.number_input(
            "Backstop Qty",
            value=config.agents.adaptive_market_maker.backstop_quantity,
            min_value=0,
            key="cfg_amm_backstop_qty",
        )

    # Momentum Agents
    with st.expander("📈 MOMENTUM AGENTS"):
        momentum_num_agents = st.number_input(
            "Number of Agents",
            value=config.agents.momentum.num_agents,
            min_value=0,
            key="cfg_momentum_num_agents",
        )
        momentum_min_size = st.number_input(
            "Min Size",
            value=config.agents.momentum.min_size,
            min_value=1,
            key="cfg_momentum_min_size",
        )
        momentum_max_size = st.number_input(
            "Max Size",
            value=config.agents.momentum.max_size,
            min_value=1,
            key="cfg_momentum_max_size",
        )
        momentum_wake_up_freq = st.text_input(
            "Wake Up Freq",
            value=config.agents.momentum.wake_up_freq,
            key="cfg_momentum_wake_up_freq",
        )
        momentum_poisson = st.checkbox(
            "Poisson Arrival",
            value=config.agents.momentum.poisson_arrival,
            key="cfg_momentum_poisson",
        )

    # Oracle Settings
    with st.expander("🔮 ORACLE"):
        oracle_kappa = st.number_input(
            "Kappa",
            value=config.agents.oracle.kappa,
            format="%.2e",
            key="cfg_oracle_kappa",
        )
        oracle_sigma_s = st.number_input(
            "Sigma S",
            value=config.agents.oracle.sigma_s,
            format="%.2e",
            key="cfg_oracle_sigma_s",
        )
        oracle_fund_vol = st.number_input(
            "Fund Vol",
            value=config.agents.oracle.fund_vol,
            format="%.2e",
            key="cfg_oracle_fund_vol",
        )
        oracle_megashock_lambda = st.number_input(
            "Megashock Lambda",
            value=config.agents.oracle.megashock_lambda_a,
            format="%.2e",
            key="cfg_oracle_megashock_lambda",
        )
        oracle_megashock_mean = st.number_input(
            "Megashock Mean",
            value=config.agents.oracle.megashock_mean,
            min_value=0,
            key="cfg_oracle_megashock_mean",
        )
        oracle_megashock_var = st.number_input(
            "Megashock Var",
            value=config.agents.oracle.megashock_var,
            min_value=0,
            key="cfg_oracle_megashock_var",
        )

    # Latency Model
    with st.expander("⏱️ LATENCY"):
        latency_type = st.selectbox(
            "Type",
            options=[lt.value for lt in LatencyType],
            index=1,
            key="cfg_latency_type",
        )
        if latency_type == LatencyType.CUBIC.value:
            latency_jitter = st.number_input(
                "Jitter",
                value=config.latency.jitter,
                min_value=0.0,
                key="cfg_latency_jitter",
            )
            latency_jitter_clip = st.number_input(
                "Jitter Clip",
                value=config.latency.jitter_clip,
                min_value=0.0,
                key="cfg_latency_jitter_clip",
            )
            latency_jitter_unit = st.number_input(
                "Jitter Unit",
                value=config.latency.jitter_unit,
                min_value=0.0,
                key="cfg_latency_jitter_unit",
            )
        else:
            latency_jitter = config.latency.jitter
            latency_jitter_clip = config.latency.jitter_clip
            latency_jitter_unit = config.latency.jitter_unit

    st.markdown("---")

    # Save configuration button
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
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
            st.success("✅ Configuration saved!")

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
        <h1 style='color: {COLORS['primary']}; font-size: 2.5rem; margin: 0;'>
            ABIDES Simulation Terminal
        </h1>
        <p style='color: {COLORS['text_muted']}; font-size: 1.1rem; margin: 10px 0 0 0;'>
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


@st.fragment
def render_execute_tab():
    """Render execute tab as a fragment for isolated updates."""
    # Check if configuration exists
    if st.session_state.simulation_config is None:
        st.warning("⚠️ No configuration saved. Please configure and save in the sidebar first.")
        return

    config = st.session_state.simulation_config

    # Configuration Summary
    st.markdown("### 📋 Configuration Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
            <div style='background-color: {COLORS['card_bg']}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS['primary']};'>
                <p style='color: {COLORS['text_muted']}; margin: 0; font-size: 0.8rem;'>DATE</p>
                <p style='color: {COLORS['text']}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.date}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div style='background-color: {COLORS['card_bg']}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS['secondary']};'>
                <p style='color: {COLORS['text_muted']}; margin: 0; font-size: 0.8rem;'>TIME RANGE</p>
                <p style='color: {COLORS['text']}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.start_time} - {config.end_time}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
            <div style='background-color: {COLORS['card_bg']}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS['success']};'>
                <p style='color: {COLORS['text_muted']}; margin: 0; font-size: 0.8rem;'>TICKER</p>
                <p style='color: {COLORS['text']}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.ticker}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
            <div style='background-color: {COLORS['card_bg']}; padding: 15px; border-radius: 8px; border-left: 4px solid {COLORS['danger']};'>
                <p style='color: {COLORS['text_muted']}; margin: 0; font-size: 0.8rem;'>SEED</p>
                <p style='color: {COLORS['text']}; margin: 5px 0 0 0; font-size: 1.2rem; font-weight: bold;'>{config.seed}</p>
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
                service = SimulationService(config)
                st.write("✓ Simulation engine initialized")

                # Step 3: Run simulation
                st.write("⏳ Running simulation (this may take a few minutes)...")
                start_time = datetime.now()

                result = service.run_simulation()

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

            with col1:
                st.metric("Volatility", f"{metrics.volatility:.6f}")

            with col2:
                st.metric(
                    "Mean Spread",
                    f"{metrics.custom_metrics.get('mean_spread', 0):.2f}",
                )

            with col3:
                st.metric(
                    "Avg Bid Liquidity",
                    f"{metrics.custom_metrics.get('avg_bid_liquidity', 0):.2f}",
                )

            with col4:
                st.metric(
                    "Avg Ask Liquidity",
                    f"{metrics.custom_metrics.get('avg_ask_liquidity', 0):.2f}",
                )

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
                <div style='background-color: {COLORS['card_bg']}; padding: 15px; border-radius: 8px;'>
                    <p style='color: {COLORS['text_muted']}; margin: 0; font-size: 0.8rem;'>TIMESTAMP</p>
                    <p style='color: {COLORS['text']}; margin: 5px 0 0 0;'>{st.session_state.simulation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div style='background-color: {COLORS['card_bg']}; padding: 15px; border-radius: 8px;'>
                    <p style='color: {COLORS['text_muted']}; margin: 0; font-size: 0.8rem;'>DURATION</p>
                    <p style='color: {COLORS['text']}; margin: 5px 0 0 0;'>{st.session_state.simulation_duration:.2f} seconds</p>
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
        @st.fragment
        def render_metrics_tab():
            """Render metrics tab as a fragment."""
            st.markdown("### 📊 Key Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Volatility", f"{metrics.volatility:.6f}")
                st.metric("Total PnL", f"${metrics.total_pnl:,.2f}")

            with col2:
                st.metric("Sharpe Ratio", f"{metrics.sharpe_ratio:.4f}")
                st.metric("Max Drawdown", f"{metrics.max_drawdown:.4f}")

            with col3:
                st.metric("Win Rate", f"{metrics.win_rate:.2%}")
                st.metric("Traded Volume", f"{metrics.traded_volume:,}")

            st.markdown("---")

            # Custom metrics
            if metrics.custom_metrics:
                st.markdown("### 🔧 Custom Metrics")

                custom_cols = st.columns(len(metrics.custom_metrics))

                for i, (key, value) in enumerate(metrics.custom_metrics.items()):
                    with custom_cols[i]:
                        display_key = key.replace("_", " ").title()
                        display_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                        st.metric(display_key, display_value)

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
        @st.fragment
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
        @st.fragment
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
        @st.fragment
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
                            marker={"color": COLORS["danger"], "size": 3, "opacity": 0.5},
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
        @st.fragment
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

                    # Display logs
                    st.dataframe(filtered_df.head(max_rows), use_container_width=True, height=500)

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
