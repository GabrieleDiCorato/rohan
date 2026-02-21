"""ROHAN — Refinement Lab

Autonomous LLM-driven strategy optimization.  This page lets you:

1. Define a trading strategy goal in natural language
2. Launch the refinement loop (Writer → Validator → Executor → Explainer → Judge)
3. Monitor progress in real-time as each agent completes
4. Inspect iteration history, metrics, and the final strategy code
5. Save / load complete runs to the database
"""

from __future__ import annotations

import contextlib
import logging
import time
import traceback
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from rohan.config import SimulationSettings
from rohan.framework.database import initialize_database
from rohan.framework.refinement_repository import (
    IterationData,
    RefinementRepository,
    ScenarioResultData,
)
from rohan.framework.scenario_repository import ScenarioRepository
from rohan.llm.graph import build_refinement_graph
from rohan.llm.state import RefinementState, ScenarioConfig
from rohan.ui.utils.presets import get_preset_config, get_preset_names
from rohan.ui.utils.theme import COLORS, apply_theme

logger = logging.getLogger(__name__)

# ============================================================================
# PAGE CONFIG & THEME
# ============================================================================

st.set_page_config(
    page_title="ROHAN — Refinement Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# Ensure DB tables exist
with contextlib.suppress(Exception):
    initialize_database()

_scenario_repo = ScenarioRepository()
_refinement_repo = RefinementRepository()

# ============================================================================
# CONSTANTS
# ============================================================================

EXAMPLE_GOALS: dict[str, str] = {
    "📊 Market Making": (
        "Create a market-making strategy that captures the bid-ask spread "
        "while keeping inventory risk low and maintaining a near-neutral "
        "position. Cancel stale quotes every tick and skew pricing based "
        "on current inventory."
    ),
    "📈 Momentum": ("Design a momentum-following strategy that detects short-term price trends using recent trade data and places directional limit orders slightly ahead of the move."),
    "🔄 Mean Reversion": (
        "Build a mean-reversion strategy that identifies when the price deviates significantly from its recent moving average and places contrarian limit orders to profit from the reversion."
    ),
    "⚡ Aggressive PnL": (
        "Create an aggressive strategy that maximises PnL by detecting order-flow imbalance from the bid/ask volume ratio and taking directional positions ahead of anticipated price moves."
    ),
}


# ============================================================================
# SESSION STATE INITIALISATION
# ============================================================================

_DEFAULTS: dict[str, Any] = {
    "refine_goal": "",
    "refine_max_iterations": 3,
    "refine_running": False,
    "refine_final_state": None,
    "refine_duration": None,
    "refine_progress": [],
    "refine_timestamp": None,
    "refine_error": None,
    # ── Dirty tracking ──
    "refine_is_dirty": False,
    "refine_saved": False,
    "refine_saved_id": None,
    "refine_prev_goal": "",
    "refine_prev_scenarios": [],
}

for _key, _val in _DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _val


# ============================================================================
# HELPERS
# ============================================================================


def _card(label: str, value: str, accent: str) -> str:
    """Return HTML for a Bloomberg-style metric card."""
    return f"""
    <div style='background-color: {COLORS["card_bg"]}; padding: 15px;
                border-radius: 8px; border-left: 4px solid {accent};'>
        <p style='color: {COLORS["text_muted"]}; margin: 0; font-size: 0.8rem;
                  text-transform: uppercase; letter-spacing: 1px;'>{label}</p>
        <p style='color: {COLORS["text"]}; margin: 5px 0 0 0;
                  font-size: 1.4rem; font-weight: bold;'>{value}</p>
    </div>
    """


def _build_scenarios_from_selection(selected: list[str]) -> list[ScenarioConfig]:
    """Convert the multi-select result into a list of ScenarioConfig.

    Preset names use a diff vs. default SimulationSettings.
    Saved scenarios carry the full config as override.
    """
    default_dump = SimulationSettings().model_dump()
    configs: list[ScenarioConfig] = []

    for name in selected:
        if name.startswith("💾 "):
            # Saved scenario — load full config from DB
            raw_name = name[len("💾 ") :]
            sc = _scenario_repo.get_scenario_by_name(raw_name)
            if sc:
                configs.append(ScenarioConfig(name=raw_name, config_override=sc.full_config))
            else:
                configs.append(ScenarioConfig(name=raw_name))
        elif name.startswith("📦 "):
            # Preset — compute diff from defaults
            raw_name = name[len("📦 ") :]
            preset_settings = get_preset_config(raw_name)
            preset_dump = preset_settings.model_dump()
            override = _dict_diff(default_dump, preset_dump)
            configs.append(ScenarioConfig(name=raw_name, config_override=override))
        else:
            # Plain default
            configs.append(ScenarioConfig(name=name))

    return configs if configs else [ScenarioConfig(name="default")]


def _dict_diff(base: dict, other: dict) -> dict:
    """Return only the keys in *other* that differ from *base* (shallow top-level)."""
    diff: dict[str, Any] = {}
    for k, v in other.items():
        if k not in base or base[k] != v:
            diff[k] = v
    return diff


def _pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:+.1f}%"


def _dollar(v: float | None) -> str:
    """Format cents → dollars."""
    if v is None:
        return "N/A"
    return f"${v / 100:,.2f}"


def _reset_run_state() -> None:
    """Clear all run-related session state (called before new run / load)."""
    st.session_state.refine_final_state = None
    st.session_state.refine_duration = None
    st.session_state.refine_progress = []
    st.session_state.refine_timestamp = None
    st.session_state.refine_error = None
    st.session_state.refine_saved = False
    st.session_state.refine_saved_id = None
    st.session_state.refine_is_dirty = False


def _save_current_run(run_name: str | None = None) -> bool:
    """Persist the current refine_final_state to the DB. Returns success."""
    fs = st.session_state.refine_final_state
    if fs is None:
        return False

    iterations_data: list[IterationData] = []
    for it in fs.get("iterations", []):
        sc_results = [
            ScenarioResultData(
                scenario_name=sm.scenario_name,
                total_pnl=sm.total_pnl,
                sharpe_ratio=sm.sharpe_ratio,
                max_drawdown=sm.max_drawdown,
                trade_count=sm.trade_count,
                volatility_delta_pct=sm.volatility_delta_pct,
                spread_delta_pct=sm.spread_delta_pct,
                price_chart_b64=sm.price_chart_b64,
                spread_chart_b64=sm.spread_chart_b64,
                volume_chart_b64=sm.volume_chart_b64,
            )
            for sm in it.scenario_metrics.values()
        ]
        iterations_data.append(
            IterationData(
                iteration_number=it.iteration_number,
                strategy_code=it.strategy_code,
                class_name=fs.get("current_class_name"),
                reasoning=fs.get("current_reasoning"),
                judge_score=it.judge_score,
                judge_reasoning=it.judge_reasoning,
                aggregated_explanation=it.aggregated_explanation,
                scenario_results=sc_results,
            )
        )

    goal = fs.get("goal", "")
    if not run_name:
        ts = st.session_state.refine_timestamp or time.strftime("%Y-%m-%d %H:%M")
        run_name = f"{goal[:50]} — {ts}"

    # Serialize scenario configs
    scenarios_raw = fs.get("scenarios", [])
    scenario_dicts = [s.model_dump() if hasattr(s, "model_dump") else s for s in scenarios_raw]

    iterations = fs.get("iterations", [])
    final_score = iterations[-1].judge_score if iterations else None

    try:
        saved = _refinement_repo.save_session(
            name=run_name,
            goal=goal,
            max_iterations=fs.get("max_iterations", 3),
            scenario_configs=scenario_dicts,
            status=fs.get("status", "done"),
            final_score=final_score,
            total_duration=st.session_state.refine_duration,
            progress_log=st.session_state.refine_progress or [],
            final_code=fs.get("current_code"),
            final_class_name=fs.get("current_class_name"),
            final_reasoning=fs.get("current_reasoning"),
            iterations=iterations_data,
        )
        st.session_state.refine_saved = True
        st.session_state.refine_saved_id = saved.session_id
        st.session_state.refine_is_dirty = False
        return True
    except Exception as exc:
        logger.error("Failed to save refinement run: %s", exc)
        return False


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown(
        f"""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: {COLORS["primary"]}; font-size: 1.8rem; margin: 0;'>
                ROHAN
            </h1>
            <p style='color: {COLORS["secondary"]}; font-size: 1.2rem;
                      margin: 5px 0 0 0; letter-spacing: 2px;'>
                REFINEMENT LAB
            </p>
            <hr style='border-color: {COLORS["border"]}; margin: 15px 0;'>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Model info ────────────────────────────────────────────────────
    st.markdown("### 🤖 Models")
    try:
        from rohan.config.llm_settings import LLMSettings

        _settings = LLMSettings()
        st.markdown(f"**Provider:** `{_settings.provider.value}`")
        st.markdown(f"**Codegen:** `{_settings.codegen_model}`")
        st.markdown(f"**Analysis:** `{_settings.analysis_model}`")
        st.markdown(f"**Judge:** `{_settings.judge_model}`")

        if _settings.provider.value == "openrouter" and not _settings.openrouter_api_key:
            st.error("⚠️ `OPENROUTER_API_KEY` not set! Add it to your `.env` file.")
        elif _settings.provider.value == "openai" and not _settings.openai_api_key:
            st.error("⚠️ `OPENAI_API_KEY` not set!")
    except Exception as exc:
        st.warning(f"Could not load LLM settings: {exc}")

    st.markdown("---")

    # ── Saved Refinement Runs ─────────────────────────────────────────
    st.markdown("### 💾 Saved Runs")

    @st.dialog("Load Past Run", width="large")
    def _sidebar_load_run_dialog():
        try:
            _sessions = _refinement_repo.list_sessions()
        except Exception as _exc:
            st.error(f"Could not load past runs: {_exc}")
            return

        if not _sessions:
            st.info("No saved runs found. Run a refinement and save it first.")
            return

        st.markdown(f"**{len(_sessions)} saved run(s)**")
        st.markdown("---")

        for _s in _sessions:
            _s_col1, _s_col2, _s_col3 = st.columns([5, 1, 1])
            with _s_col1:
                _score_str = f"{_s.final_score:.1f}/10" if _s.final_score is not None else "N/A"
                st.markdown(f"**{_s.name}**")
                st.caption(_s.goal)
                st.caption(f"🏆 Score: {_score_str} · 🔄 {_s.iteration_count} iter · 📌 {_s.status} · 📅 {_s.created_at:%Y-%m-%d %H:%M}")
            with _s_col2:
                if st.button("Load", key=f"sb_load_{_s.session_id}", use_container_width=True, type="primary"):
                    _loaded = _refinement_repo.load_session(_s.session_id)
                    if _loaded:
                        for _k, _v in _loaded.items():
                            st.session_state[_k] = _v
                        st.session_state.refine_prev_goal = _loaded.get("refine_goal", "")
                        st.session_state.refine_prev_scenarios = []
                        st.session_state.refine_running = False
                        st.toast("✅ Run loaded!")
                        st.rerun()
                    else:
                        st.error("Failed to load run.")
            with _s_col3:
                if st.button("🗑️", key=f"sb_del_{_s.session_id}", use_container_width=True, help="Delete this run"):
                    _refinement_repo.delete_session(_s.session_id)
                    st.rerun()
            st.divider()

    if st.button("Load Past Run", use_container_width=True):
        _sidebar_load_run_dialog()

    st.markdown("---")

    # ── Example goals ─────────────────────────────────────────────────
    st.markdown("### 💡 Example Goals")
    for _name, _goal_text in EXAMPLE_GOALS.items():
        if st.button(_name, use_container_width=True, key=f"eg_{_name}"):
            st.session_state.refine_goal = _goal_text
            st.session_state.goal_input = _goal_text
            st.rerun()

    st.markdown("---")

    # ── Tips ──────────────────────────────────────────────────────────
    with st.expander("ℹ️ Quick Tips"):
        st.markdown(
            """
- Be **specific** about the strategy's risk constraints
- Mention inventory limits, spread targets, position sizing
- Start with **2–3 iterations** for quick experiments
- Pipeline: Writer → Validator → Simulator → Explainer → Judge
- Convergence: score ≥ 8/10 or plateau → auto-stop
- Each iteration takes ~60–120 s (simulation is the bottleneck)
            """
        )


# ============================================================================
# MAIN: HEADER
# ============================================================================

st.markdown(
    f"""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: {COLORS["primary"]}; font-size: 2.5rem; margin: 0;'>
            🧪 Refinement Lab
        </h1>
        <p style='color: {COLORS["text_muted"]}; font-size: 1.1rem;
                  margin: 10px 0 0 0;'>
            Autonomous LLM-Driven Strategy Optimization
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# ============================================================================
# MAIN: GOAL INPUT + CONTROLS
# ============================================================================

st.markdown("### 🎯 Strategy Goal")

goal = st.text_area(
    "Describe your trading strategy",
    value=st.session_state.refine_goal,
    height=120,
    placeholder=("e.g. Create a market-making strategy that captures the bid-ask spread while keeping inventory risk low…"),
    key="goal_input",
    label_visibility="collapsed",
)
st.session_state.refine_goal = goal

# Controls row
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1, 2, 1, 1])

with ctrl_col1:
    max_iterations = st.number_input(
        "Max Iterations",
        min_value=1,
        max_value=10,
        value=st.session_state.refine_max_iterations,
        step=1,
        key="max_iter_input",
    )
    st.session_state.refine_max_iterations = max_iterations

with ctrl_col2:
    # Build scenario options: presets + saved
    _preset_names = get_preset_names()
    _preset_options = [f"📦 {n}" for n in _preset_names]
    try:
        _saved_scenarios = _scenario_repo.list_scenarios()
        _saved_options = [f"💾 {s.name}" for s in _saved_scenarios]
    except Exception:
        _saved_options = []
    _all_scenario_options = _preset_options + _saved_options

    selected_scenarios = st.multiselect(
        "Scenarios",
        options=_all_scenario_options,
        default=[_all_scenario_options[0]] if _all_scenario_options else [],
        key="scenario_multiselect",
        help="Select one or more scenarios to run the strategy against",
    )

with ctrl_col3:
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    can_launch = bool(goal and goal.strip()) and not st.session_state.refine_running and len(selected_scenarios) > 0
    launch_pressed = st.button(
        "🚀 LAUNCH",
        type="primary",
        use_container_width=True,
        disabled=not can_launch,
    )

with ctrl_col4:
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
    load_pressed = st.button(
        "📂 Load Run",
        use_container_width=True,
    )

if not (goal and goal.strip()) and st.session_state.refine_final_state is None:
    st.caption("👆 Enter a strategy goal above to get started. Try one of the **example goals** in the sidebar.")


# ============================================================================
# DIRTY TRACKING
# ============================================================================

# Detect if inputs changed after a run completed
if st.session_state.refine_final_state is not None and not st.session_state.refine_saved:
    _goal_changed = goal != st.session_state.refine_prev_goal
    _scenarios_changed = sorted(selected_scenarios) != sorted(st.session_state.refine_prev_scenarios)
    if _goal_changed or _scenarios_changed:
        st.session_state.refine_is_dirty = True


# ============================================================================
# DIALOGS
# ============================================================================


@st.dialog("⚠️ Unsaved Changes")
def _unsaved_changes_dialog(action: str) -> None:
    """Show save-before-reset prompt. *action* = 'launch' or 'load'."""
    st.markdown("You have **unsaved results** from the current run. What would you like to do?")

    d_col1, d_col2, d_col3 = st.columns(3)
    with d_col1:
        if st.button("💾 Save & Continue", use_container_width=True, type="primary"):
            if _save_current_run():
                st.toast("✅ Run saved!")
            _reset_run_state()
            if action == "launch":
                st.session_state._pending_launch = True
            elif action == "load":
                st.session_state._pending_load = True
            st.rerun()
    with d_col2:
        if st.button("🗑️ Discard", use_container_width=True):
            _reset_run_state()
            if action == "launch":
                st.session_state._pending_launch = True
            elif action == "load":
                st.session_state._pending_load = True
            st.rerun()
    with d_col3:
        if st.button("Cancel", use_container_width=True):
            st.rerun()


@st.dialog("📂 Load Past Run", width="large")
def _load_run_dialog() -> None:
    """Browse and load a past refinement run from the database."""
    try:
        sessions = _refinement_repo.list_sessions()
    except Exception as exc:
        st.error(f"Could not load past runs: {exc}")
        return

    if not sessions:
        st.info("No saved runs found. Run a refinement and save it first.")
        return

    st.markdown(f"**{len(sessions)} saved run(s)**")
    st.markdown("---")

    for s in sessions:
        s_col1, s_col2, s_col3 = st.columns([4, 1, 1])
        with s_col1:
            score_str = f"{s.final_score:.1f}/10" if s.final_score is not None else "N/A"
            st.markdown(f"**{s.name}**")
            st.caption(f"Score: {score_str} · {s.iteration_count} iter · {s.status} · {s.created_at:%Y-%m-%d %H:%M}")
        with s_col2:
            if st.button("Load", key=f"dlg_load_{s.session_id}", use_container_width=True, type="primary"):
                loaded = _refinement_repo.load_session(s.session_id)
                if loaded:
                    for k, v in loaded.items():
                        st.session_state[k] = v
                    # Sync prev-tracking to loaded values
                    st.session_state.refine_prev_goal = loaded.get("refine_goal", "")
                    st.session_state.refine_prev_scenarios = []
                    st.session_state.refine_running = False
                    st.toast("✅ Run loaded!")
                    st.rerun()
                else:
                    st.error("Failed to load run.")
        with s_col3:
            if st.button("🗑️", key=f"dlg_del_{s.session_id}", use_container_width=True, help="Delete this run"):
                _refinement_repo.delete_session(s.session_id)
                st.rerun()


# ============================================================================
# DIALOG TRIGGERS
# ============================================================================

_has_unsaved_results = st.session_state.refine_final_state is not None and not st.session_state.refine_saved

# Load button logic
if load_pressed:
    if _has_unsaved_results:
        _unsaved_changes_dialog("load")
    else:
        st.session_state._pending_load = True
        st.rerun()

if st.session_state.pop("_pending_load", False):
    _load_run_dialog()

# Launch button logic — guard for unsaved results
if launch_pressed:
    if _has_unsaved_results:
        _unsaved_changes_dialog("launch")
    else:
        st.session_state._pending_launch = True
        st.rerun()


st.markdown("---")


# ============================================================================
# EXECUTION ENGINE
# ============================================================================


def _run_refinement(
    _goal: str,
    _max_iters: int,
    _scenarios: list[ScenarioConfig],
) -> None:
    """Execute the refinement graph with live streaming progress."""

    initial_state: RefinementState = {
        "goal": _goal,
        "max_iterations": _max_iters,
        "scenarios": _scenarios,
        "current_code": None,
        "current_class_name": None,
        "current_reasoning": None,
        "validation_errors": [],
        "validation_attempts": 0,
        "scenario_results": [],
        "explanations": [],
        "aggregated_feedback": None,
        "iterations": [],
        "iteration_number": 1,
        "status": "writing",
        "messages": [],
    }

    graph = build_refinement_graph()
    t0 = time.time()
    progress: list[str] = []
    accumulated: dict[str, Any] = {**initial_state}  # manual state merge

    status_container = st.status("🔄 Launching refinement loop…", expanded=True)

    try:
        with status_container:
            st.write(f"**Goal:** {_goal[:120]}{'…' if len(_goal) > 120 else ''}")
            st.write(f"**Config:** {_max_iters} iteration(s), {len(_scenarios)} scenario(s)")
            st.write("")

            iter_display = 1
            st.write(f"### Iteration {iter_display}")

            for event in graph.stream(initial_state, stream_mode="updates"):
                if not event:
                    continue
                node_name = next(iter(event))
                if node_name.startswith("__"):
                    continue
                node_output: dict[str, Any] = event[node_name]
                accumulated.update(node_output)

                elapsed = time.time() - t0

                # ── Writer ────────────────────────────────────────
                if node_name == "writer":
                    cls = node_output.get("current_class_name", "?")
                    msg = f"✓ **Writer** — generated `{cls}` ({elapsed:.1f}s)"
                    st.write(msg)
                    progress.append(msg)

                # ── Validator ─────────────────────────────────────
                elif node_name == "validator":
                    errors = node_output.get("validation_errors", [])
                    attempts = node_output.get("validation_attempts", 0)
                    msg = (  # noqa: SIM108
                        f"⚠️ **Validator** — attempt {attempts} failed: _{errors[0][:80]}_" if errors else f"✓ **Validator** — passed on attempt {attempts} ({elapsed:.1f}s)"
                    )
                    st.write(msg)
                    progress.append(msg)

                # ── Executor ──────────────────────────────────────
                elif node_name == "executor":
                    results = node_output.get("scenario_results", [])
                    n_ok = sum(1 for r in results if not r.error)
                    msg = f"✓ **Executor** — {n_ok}/{len(results)} scenario(s) succeeded ({elapsed:.1f}s)"
                    st.write(msg)
                    progress.append(msg)

                # ── Explainer ─────────────────────────────────────
                elif node_name == "explainer":
                    n_exp = len(node_output.get("explanations", []))
                    msg = f"✓ **Explainer** — analysed {n_exp} scenario(s) ({elapsed:.1f}s)"
                    st.write(msg)
                    progress.append(msg)

                # ── Aggregator ────────────────────────────────────
                elif node_name == "aggregator":
                    fb = node_output.get("aggregated_feedback")
                    new_status = node_output.get("status", "done")

                    if fb:
                        sc = fb.verdict.score
                        rec = fb.verdict.recommendation
                        msg = f"✓ **Judge** — score **{sc:.1f}/10** · _{rec}_ ({elapsed:.1f}s)"
                        st.write(msg)
                        progress.append(msg)

                    if new_status == "writing":
                        new_iter = node_output.get("iteration_number", 2)
                        iter_display = new_iter
                        st.write("")
                        st.write(f"### Iteration {iter_display}")
                        progress.append(f"─── Iteration {iter_display} ───")
                    elif new_status == "done":
                        if fb and fb.verdict.recommendation == "stop_converged":
                            st.write("🎯 **Converged** — strategy meets quality threshold")
                        elif fb and fb.verdict.recommendation == "stop_plateau":
                            st.write("📊 **Plateau** — no further improvement detected")
                        else:
                            st.write("🏁 **Max iterations reached**")

            total = time.time() - t0
            st.write("")
            st.write(f"⏱️ **Total time:** {total:.1f}s")

        # ── Update status bar ──
        iters = accumulated.get("iterations", [])
        if iters:
            last = iters[-1]
            last_score = f"{last.judge_score:.1f}" if last.judge_score is not None else "?"
            status_container.update(
                label=(f"✅ Refinement complete — {len(iters)} iteration(s), score {last_score}/10 ({total:.1f}s)"),
                state="complete",
                expanded=False,
            )
        else:
            status_container.update(
                label=f"⚠️ No iterations completed ({total:.1f}s)",
                state="error",
                expanded=True,
            )

        # ── Persist to session state ──
        st.session_state.refine_final_state = accumulated
        st.session_state.refine_duration = total
        st.session_state.refine_progress = progress
        st.session_state.refine_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.refine_error = None
        st.session_state.refine_saved = False
        st.session_state.refine_is_dirty = False
        st.session_state.refine_prev_goal = goal
        st.session_state.refine_prev_scenarios = sorted(selected_scenarios)

    except Exception as exc:
        total = time.time() - t0
        status_container.update(
            label=f"❌ Refinement failed ({total:.1f}s)",
            state="error",
            expanded=True,
        )
        st.error(f"**Error:** {exc}")
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())

        st.session_state.refine_error = str(exc)

    finally:
        st.session_state.refine_running = False


# ── Launch trigger ────────────────────────────────────────────────────────
if st.session_state.pop("_pending_launch", False) or (launch_pressed and not _has_unsaved_results):
    _reset_run_state()
    st.session_state.refine_running = True
    _run_refinement(str(goal), max_iterations, _build_scenarios_from_selection(selected_scenarios))


# ============================================================================
# RESULTS — shown when a run has completed
# ============================================================================

final_state: dict[str, Any] | None = st.session_state.refine_final_state

if final_state is not None:
    iterations = final_state.get("iterations", [])

    # ── Save bar ──────────────────────────────────────────────────────
    save_col1, save_col2, save_col3 = st.columns([3, 1, 1])
    with save_col1:
        _run_name = st.text_input(
            "Run name",
            value="" if not st.session_state.refine_saved else "✅ Saved",
            placeholder="e.g. MM strategy v3",
            key="save_run_name",
            label_visibility="collapsed",
            disabled=st.session_state.refine_saved,
        )
    with save_col2:
        _save_btn = st.button(
            "✅ Saved" if st.session_state.refine_saved else "💾 Save Run",
            use_container_width=True,
            disabled=st.session_state.refine_saved,
            type="primary" if not st.session_state.refine_saved else "secondary",
        )
    with save_col3:
        if st.session_state.refine_is_dirty:
            st.markdown(
                f"<div style='padding:8px; color:{COLORS['secondary']};'>⚠️ Unsaved changes</div>",
                unsafe_allow_html=True,
            )

    if _save_btn and not st.session_state.refine_saved:
        if _save_current_run(_run_name if _run_name and _run_name.strip() else None):
            st.toast("✅ Run saved successfully!")
            st.rerun()
        else:
            st.error("Failed to save run.")

    tab_monitor, tab_results, tab_strategy = st.tabs(["📡 Monitor", "📊 Results", "💻 Strategy"])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — Monitor
    # ══════════════════════════════════════════════════════════════════
    with tab_monitor:
        # Summary cards
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.markdown(
                _card("Iterations", str(len(iterations)), COLORS["primary"]),
                unsafe_allow_html=True,
            )
        with c2:
            if iterations and iterations[-1].judge_score is not None:
                sc = iterations[-1].judge_score
                sc_color = COLORS["success"] if sc >= 7 else COLORS["secondary"]
                st.markdown(
                    _card("Final Score", f"{sc:.1f}/10", sc_color),
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    _card("Final Score", "N/A", COLORS["text_muted"]),
                    unsafe_allow_html=True,
                )
        with c3:
            raw_status = final_state.get("status", "unknown")
            status_map = {
                "done": ("✅ Complete", COLORS["success"]),
                "failed": ("❌ Failed", COLORS["danger"]),
                "writing": ("🔄 Running", COLORS["secondary"]),
            }
            s_text, s_color = status_map.get(raw_status, (raw_status, COLORS["text_muted"]))
            st.markdown(_card("Status", s_text, s_color), unsafe_allow_html=True)
        with c4:
            dur = st.session_state.refine_duration or 0
            st.markdown(
                _card("Duration", f"{dur:.1f}s", COLORS["secondary"]),
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Timestamp & goal
        ts = st.session_state.refine_timestamp
        if ts:
            st.caption(f"Run completed at **{ts}**")

        st.markdown("---")

        # Execution log
        st.markdown("### 📋 Execution Log")
        progress_log: list[str] = st.session_state.refine_progress or []
        if progress_log:
            for msg in progress_log:
                if msg.startswith("───"):
                    st.markdown(f"**{msg}**")
                else:
                    st.write(msg)
        else:
            st.info("No progress data available.")

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — Results
    # ══════════════════════════════════════════════════════════════════
    with tab_results:
        if not iterations:
            st.warning("No completed iterations. Check the Monitor tab for errors.")
        else:
            import pandas as pd

            # ── Iteration history table ───────────────────────────
            st.markdown("### 📊 Iteration History")

            history_rows: list[dict[str, Any]] = []
            for it in iterations:
                sc_names = list(it.scenario_metrics.keys())
                first = it.scenario_metrics[sc_names[0]] if sc_names else None
                history_rows.append(
                    {
                        "Iteration": it.iteration_number,
                        "Score": (f"{it.judge_score:.1f}/10" if it.judge_score is not None else "N/A"),
                        "PnL ($)": (_dollar(first.total_pnl) if first else "N/A"),
                        "Trades": first.trade_count if first else 0,
                        "Vol Δ%": (_pct(first.volatility_delta_pct) if first else "N/A"),
                        "Spread Δ%": (_pct(first.spread_delta_pct) if first else "N/A"),
                    }
                )

            df = pd.DataFrame(history_rows)
            st.dataframe(df.set_index("Iteration"), use_container_width=True)

            st.markdown("---")

            # ── Score progression chart ───────────────────────────
            st.markdown("### 📈 Score Progression")

            scores = [it.judge_score for it in iterations if it.judge_score is not None]
            if scores:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=list(range(1, len(scores) + 1)),
                        y=scores,
                        mode="lines+markers",
                        name="Score",
                        line={"color": COLORS["primary"], "width": 3},
                        marker={"size": 12, "color": COLORS["primary"]},
                    )
                )

                # Convergence threshold line
                fig.add_hline(
                    y=8,
                    line_dash="dash",
                    line_color=COLORS["success"],
                    annotation_text="Convergence (8/10)",
                    annotation_position="top right",
                    annotation_font_color=COLORS["success"],
                )

                fig.update_layout(
                    height=400,
                    plot_bgcolor=COLORS["background"],
                    paper_bgcolor=COLORS["background"],
                    font={
                        "color": COLORS["text"],
                        "family": "Courier New",
                    },
                    xaxis={
                        "title": "Iteration",
                        "gridcolor": COLORS["border"],
                        "dtick": 1,
                    },
                    yaxis={
                        "title": "Score",
                        "gridcolor": COLORS["border"],
                        "range": [0, 10.5],
                    },
                    showlegend=False,
                    margin={"l": 60, "r": 30, "t": 30, "b": 50},
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No scores recorded.")

            st.markdown("---")

            # ── Judge reasoning ───────────────────────────────────
            st.markdown("### 🧑‍⚖️ Judge Reasoning")

            for it in iterations:
                label = f"Iteration {it.iteration_number} — Score: {it.judge_score:.1f}/10" if it.judge_score is not None else f"Iteration {it.iteration_number}"
                with st.expander(label, expanded=(it == iterations[-1])):
                    st.markdown(it.judge_reasoning or "_(no reasoning recorded)_")

                    # Scenario metrics summary
                    for sc_name, sm in it.scenario_metrics.items():
                        st.markdown(f"**Scenario: {sc_name}**")
                        mc1, mc2, mc3 = st.columns(3)
                        with mc1:
                            st.metric(
                                "PnL",
                                _dollar(sm.total_pnl),
                            )
                        with mc2:
                            st.metric("Trades", sm.trade_count)
                        with mc3:
                            st.metric(
                                "Vol Δ%",
                                _pct(sm.volatility_delta_pct),
                            )

                        # Simulation charts
                        import base64

                        chart_cols = st.columns(3)
                        chart_data = [
                            ("📈 Price Series", sm.price_chart_b64),
                            ("📊 Bid-Ask Spread", sm.spread_chart_b64),
                            ("📉 Volume at BBO", sm.volume_chart_b64),
                        ]
                        for col, (chart_label, chart_b64) in zip(chart_cols, chart_data, strict=False):
                            with col:
                                if chart_b64:
                                    st.caption(chart_label)
                                    st.image(base64.b64decode(chart_b64))

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — Strategy
    # ══════════════════════════════════════════════════════════════════
    with tab_strategy:
        final_code: str | None = final_state.get("current_code")

        if final_code:
            st.markdown("### 💻 Final Strategy")
            st.code(final_code, language="python")

            # Download button
            dl_col1, dl_col2, _ = st.columns([1, 1, 2])
            with dl_col1:
                st.download_button(
                    "⬇️ Download .py",
                    data=final_code,
                    file_name="strategy.py",
                    mime="text/x-python",
                    use_container_width=True,
                )
            with dl_col2:
                class_name = final_state.get("current_class_name", "?")
                st.markdown(
                    f"<div style='padding: 8px 0; color: {COLORS['text_muted']};'>Class: <code>{class_name}</code></div>",
                    unsafe_allow_html=True,
                )

            # Writer reasoning
            reasoning = final_state.get("current_reasoning")
            if reasoning:
                st.markdown("---")
                st.markdown("### 💭 Writer Reasoning")
                st.markdown(reasoning)

            # Per-iteration code history
            if iterations:
                st.markdown("---")
                st.markdown("### 📜 Code History")
                st.caption("Expand an iteration to see the strategy code and feedback at that point.")

                for it in reversed(iterations):
                    label = f"Iteration {it.iteration_number} — {it.judge_score:.1f}/10" if it.judge_score is not None else f"Iteration {it.iteration_number}"
                    with st.expander(label, expanded=False):
                        st.code(it.strategy_code, language="python")

                        if it.aggregated_explanation:
                            st.markdown("**Feedback:**")
                            st.markdown(it.aggregated_explanation[:800])
                            if len(it.aggregated_explanation) > 800:
                                st.caption(f"…({len(it.aggregated_explanation) - 800} more chars)")
        else:
            st.warning("No strategy code produced. Check the Monitor tab for errors.")
            errors = final_state.get("validation_errors", [])
            if errors:
                st.error("**Validation errors:**")
                for err in errors:
                    st.markdown(f"- `{err}`")


# ============================================================================
# EMPTY STATE — no run yet
# ============================================================================

elif not st.session_state.refine_running:
    st.markdown(
        f"""
        <div style='text-align: center; padding: 60px 20px;
                    background-color: {COLORS["card_bg"]};
                    border-radius: 12px;
                    border: 1px solid {COLORS["border"]};'>
            <p style='font-size: 3rem; margin: 0;'>🧪</p>
            <h3 style='color: {COLORS["primary"]};
                       margin: 15px 0 10px 0;'>
                Ready to Refine
            </h3>
            <p style='color: {COLORS["text_muted"]};
                      max-width: 600px; margin: 0 auto; line-height: 1.6;'>
                Enter a strategy goal above and launch the refinement loop.
                The LLM agents will iteratively <strong>generate</strong>,
                <strong>test</strong>, <strong>analyse</strong>, and
                <strong>improve</strong> your trading strategy against the
                ABIDES market simulator.
            </p>
            <div style='margin-top: 25px; padding: 15px;
                        background-color: {COLORS["secondary_bg"]};
                        border-radius: 8px; display: inline-block;
                        text-align: left;'>
                <p style='color: {COLORS["primary"]}; margin: 0 0 8px 0;
                          font-size: 0.85rem; letter-spacing: 1px;'>
                    PIPELINE
                </p>
                <p style='color: {COLORS["text"]}; margin: 0;
                          font-family: Courier New, monospace;
                          font-size: 0.9rem;'>
                    Writer → Validator → Simulator → Explainer → Judge
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
