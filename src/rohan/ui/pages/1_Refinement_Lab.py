"""ROHAN — Refinement Lab

Autonomous LLM-driven strategy optimization.  This page lets you:

1. Define a trading strategy goal in natural language
2. Launch the refinement loop (Writer → Validator → Executor → Explainer → Judge)
3. Monitor progress in real-time as each agent completes
4. Inspect iteration history, metrics, and the final strategy code
"""

from __future__ import annotations

import logging
import time
import traceback
from typing import Any

import plotly.graph_objects as go
import streamlit as st

from rohan.llm.graph import build_refinement_graph
from rohan.llm.state import RefinementState, ScenarioConfig
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


def _build_scenarios(preset: str) -> list[ScenarioConfig]:
    """Convert a UI preset choice into a scenario list."""
    if preset == "Multi-Scenario":
        return [
            ScenarioConfig(name="default"),
            ScenarioConfig(name="volatile", config_override={"seed": 99}),
        ]
    return [ScenarioConfig(name="default")]


def _pct(v: float | None) -> str:
    if v is None:
        return "N/A"
    return f"{v * 100:+.1f}%"


def _dollar(v: float | None) -> str:
    """Format cents → dollars."""
    if v is None:
        return "N/A"
    return f"${v / 100:,.2f}"


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

    # ── Example goals ─────────────────────────────────────────────────
    st.markdown("### 💡 Example Goals")
    for _name, _goal_text in EXAMPLE_GOALS.items():
        if st.button(_name, use_container_width=True, key=f"eg_{_name}"):
            st.session_state.refine_goal = _goal_text
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
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])

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
    scenario_preset = st.selectbox(
        "Scenarios",
        options=["Default (single)", "Multi-Scenario"],
        key="scenario_select",
    )

with ctrl_col3:
    # Vertical alignment spacer
    st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)

    can_launch = bool(goal.strip()) and not st.session_state.refine_running
    launch_pressed = st.button(
        "🚀 LAUNCH REFINEMENT",
        type="primary",
        use_container_width=True,
        disabled=not can_launch,
    )

if not goal.strip() and st.session_state.refine_final_state is None:
    st.caption("👆 Enter a strategy goal above to get started. Try one of the **example goals** in the sidebar.")

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
    accumulated: dict[str, Any] = dict(initial_state)  # manual state merge

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
if launch_pressed:
    st.session_state.refine_running = True
    _run_refinement(goal, max_iterations, _build_scenarios(scenario_preset))


# ============================================================================
# RESULTS — shown when a run has completed
# ============================================================================

final_state: dict[str, Any] | None = st.session_state.refine_final_state

if final_state is not None:
    iterations = final_state.get("iterations", [])

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
