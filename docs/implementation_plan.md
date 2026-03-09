# ROHAN — Refinement Loop Fix & Architecture Evolution Plan

## Product Vision & Direction

ROHAN is a **strategy stress-testing and analysis platform**. The long-term product is NOT an automated strategy generator — it is a system where:

1. **The user writes (or provides) the strategy** — the LLM codegen loop is scaffolding for the PoC, not the product.
2. **The system generates adversarial scenarios** — controlled, diverse market conditions designed to expose strategy weaknesses.
3. **The system deeply interprets what happens** — an AI investigator with full simulation observability explains *why* the strategy behaved the way it did, with code-level precision.

The refinement loop serves the PoC today, but every architectural investment must be evaluated against this future: **scenario control quality** and **interpretation depth** are the product differentiators. Quantitative scoring provides grounding, data visualization, and comparison baselines — it supports interpretation but does not replace it.

---

## Current State: What's Broken

### Critical Issue #1: Scoring architecture is structurally broken

The current system asks an LLM to score strategy performance against a **hardcoded rubric** with fixed dollar thresholds (e.g., "score 5-6: PnL $200–$1,500"). This fails because:

1. **Thresholds are environment-dependent.** Achievable PnL varies with simulation duration, competing market makers, agent population, starting capital, and market volatility. No single set of static thresholds can be correct across configurations.

2. **LLM scoring is inherently noisy.** Even with ground-truth metrics injected, the LLM judge produces ±1-2 point variance across identical inputs. When scores are already low (1-3 range), noise dominates signal.

**Solution:** Replace LLM scoring with deterministic, baseline-relative formulas. Keep the LLM for qualitative analysis only.

### Critical Issue #2: 90% of simulation observability is discarded

ABIDES provides rich, multi-dimensional data that is **computed/available but never surfaced** to the analysis agents:

| Data source | Status |
|---|---|
| L2 order book depth (multi-level, time-series) | Implemented via `get_order_book_l2()`, never called |
| Per-fill execution data (fill price, quantity, timestamp, side) | Available in agent logs, only used for aggregate effective spread |
| Intra-simulation PnL curve | Reconstructed in `_agent_risk_metrics`, then discarded |
| Inventory trajectory over time | Available via holdings, never captured |
| Per-agent interaction logs (who your strategy traded against) | Available via `get_logs_by_agent()`, never used |
| Post-fill adverse selection (price movement after each fill) | Computable from L1 + fill logs, never computed |
| Order lifecycle (submission → acceptance → fill/cancel + timing) | Available in agent logs, never analysed |
| Volume chart | `plot_volume()` implemented, never called in pipeline |

The explainer receives a 22-line text summary of what was a rich simulation. This is the biggest architectural bottleneck for interpretation quality.

### Critical Issue #3: Non-deterministic seeds across iterations

Each iteration creates fresh `SimulationSettings()` in the executor, generating a new random seed each time. Within one iteration, strategy and baseline share the same seed (good), but across iterations different seeds produce different market conditions, making score comparisons unreliable.

**Solution:** Fix seeds per scenario at session start.

### Secondary Issues

- **OTT ratio explosion**: cancel-all + 1s wakeup + dual on_tick/on_market_data → OTT 10-20+. Rubric punishes harshly.
- **Sharpe = None masking**: `or 0` in nodes.py displays None as `$0.00`, judge scores 1-2 on risk.
- **Only 3 iterations**: iteration 1 often consumed by validation fixes, leaving 1-2 productive cycles.
- **Feedback bottleneck**: explainer produces structured weaknesses/recommendations, aggregator flattens them into a string, writer loses code-level specificity.
- **Explainer is not agentic**: the codebase says *"For MVP, this uses a simple prompt-based analysis without full ReAct tool calling"* — the explainer should be the primary agentic component, not the writer.

---

## Architecture Principle: Interpretation > Scoring > Generation

The value chain flows:

```
Simulation (ABIDES)  →  Observation (data extraction)  →  Interpretation (explainer)  →  Scoring (deterministic)  →  Feedback (to user or writer)
```

Every step must be excellent. Current state:
- Simulation: **good** (ABIDES is solid, L1/L2/agent logs all available)
- Observation: **broken** (90% of data discarded before analysis)
- Interpretation: **weak** (prompt-only, no tool access, no investigation)
- Scoring: **broken** (LLM-based, noisy, miscalibrated)
- Feedback: **lossy** (structured data compressed to flat strings)

The plan addresses these in priority order.

---

## Phase 1 — Unblock the Loop (P0: makes the PoC functional)

These changes fix the acute symptoms that keep the system from producing any useful output. They are prerequisites for everything else.

### ~~Step 1: Deterministic scoring formulas~~ ✅ DONE
**File: scoring.py** — Implemented and tested (103 tests).

<details>
<summary>Completed implementation details (click to expand)</summary>

Replace `build_scoring_rubric()` with pure Python functions. Keep `classify_goal_weights()`.

**Why:** Eliminates scoring noise, automatically adapts to any simulation configuration. Provides grounded, reproducible numbers for comparison dashboards and data visualization. These scores become the quantitative baseline that interpretation explains.

Update `ScoringWeights` to 6 fields: `profitability`, `risk_adjusted`, `volatility_impact`, `spread_impact`, `liquidity_impact`, `execution_quality`.

Update `WEIGHT_PROFILES` to 6-axis tables.

Add `AxisScores` dataclass (6 fields) and two functions:
- `compute_axis_scores(scenario_result, starting_capital_cents, sim_duration_hours, baseline_mean_spread, baseline_traded_volume) → AxisScores`
- `compute_final_score(axis_scores, weights) → float`

#### Scoring Formulas

**Axis 1 — Profitability** (normalized as *opportunity capture rate*):

```
opportunity = baseline_mean_spread × baseline_traded_volume / 2
capture_rate = strategy_pnl / opportunity
```

Represents what fraction of the total spread revenue pool the strategy captured. Automatically adapts to duration, volatility, agent density, and capital.

| capture_rate | Score range | Meaning |
|---|---|---|
| < 0 | 1 → 3 | Losing money |
| 0 → 0.1% | 3 → 5 | Minimal capture |
| 0.1% → 0.5% | 5 → 7 | Solid capture |
| 0.5% → 2% | 7 → 9 | Strong capture |
| > 2% | 9 → 10 (capped) | Exceptional |

Zero trades → score **1.0** (hard floor).

**Axis 2 — Risk-Adjusted Performance** (Sharpe is already annualized/scale-invariant):

| Sharpe range | Score range |
|---|---|
| None (< 5 trades) | 3.0 (insufficient data) |
| None (≥ 5 trades) | 2.0 (suspicious) |
| < −1 | 1.0 |
| −1 → 0 | 1 → 3 |
| 0 → 0.5 | 3 → 4 |
| 0.5 → 1.5 | 4 → 6 |
| 1.5 → 3.0 | 6 → 8 |
| > 3.0 | 8 → 10 |

Drawdown penalty (% of starting capital):
- `max_drawdown > 5% of capital` → −2 points
- `max_drawdown > 2% of capital` → −1 point

**Axis 3 — Volatility Impact** (baseline-relative):

| volatility_delta_pct | Score range | Meaning |
|---|---|---|
| > +10% | 1.0 | Strongly destabilizing |
| +5% → +10% | 1 → 4 | Destabilizing |
| −5% → +5% | 4 → 7 | Neutral |
| −15% → −5% | 7 → 9 | Stabilizing |
| < −15% | 9.0 (capped) | Strongly stabilizing |

**Axis 4 — Spread Impact** (baseline-relative, same shape as volatility).

**Axis 5 — Liquidity Impact** (baseline-relative, **inverted** — positive delta = good):

| liquidity_delta_pct | Score range | Meaning |
|---|---|---|
| < −15% | 1.0 | Strongly draining |
| −15% → −5% | 1 → 4 | Draining |
| −5% → +5% | 4 → 7 | Neutral |
| +5% → +15% | 7 → 9 | Improving |
| > +15% | 9.0 (capped) | Strongly improving |

**Axis 6 — Execution Quality** (fill rate primary, OTT as penalty modifier):

| Fill rate | Base score |
|---|---|
| < 5% | 1 → 2 |
| 5−15% | 2 → 4 |
| 15−30% | 4 → 6 |
| 30−50% | 6 → 8 |
| > 50% | 8 → 10 |

OTT penalty: > 200 → −2 points; > 100 → −1 point.

**Weight Profiles:**

| Profile | Profit | Risk | Vol | Spread | Liquidity | Execution |
|---------|--------|------|-----|--------|-----------|-----------|
| **default** | 35% | 20% | 7% | 8% | 5% | 25% |
| impact_focused | 15% | 10% | 20% | 20% | 15% | 20% |
| risk_focused | 15% | 40% | 7% | 8% | 5% | 25% |
| execution_focused | 15% | 15% | 7% | 8% | 5% | 50% |

**Deterministic convergence/comparison:**
- `comparison`: `"better"` if score > best_score, `"worse"` if score < best_score − 0.5, else `"similar"`.
- `recommendation`: `"stop_converged"` if score ≥ 7.0 and plateau; `"stop_plateau"` if 3+ consecutive within ±0.5; else `"continue"`.

</details>

### ~~Step 2: Propagate simulation context through state~~ ✅ DONE
**Files: state.py, nodes.py (executor)** — Implemented and tested (11 tests).

<details>
<summary>Completed implementation details (click to expand)</summary>

**state.py**: Added 5 fields to `ScenarioResult`: `starting_capital_cents` (default 10M), `baseline_mean_spread`, `baseline_traded_volume`, `bid_liquidity_delta_pct`, `ask_liquidity_delta_pct`. `sim_duration_hours` dropped — derivable from settings and not needed by scoring formulas.

**nodes.py (executor)**: Populated the new fields from `settings.starting_cash`, `base_market.mean_spread`, `base_market.traded_volume`, and `impact.bid_liquidity_delta_pct`/`impact.ask_liquidity_delta_pct`.

</details>

### Step 3: Rewire aggregator for deterministic scoring ✅ DONE

<details>
<summary>Completed implementation details (click to expand)</summary>

**nodes.py (aggregator)**: Complete rewrite (~240 lines). Calls `compute_axis_scores()` per scenario → averages across scenarios → `compute_final_score()`. Deterministic comparison (better/worse/similar thresholds), deterministic recommendation (plateau detection at ±0.5, convergence at ≥7.0 + plateau). LLM now returns `QualitativeAnalysis` (reasoning only). Builds `JudgeVerdict` programmatically from deterministic scores + LLM reasoning. Simplified rollback: single >1.0 threshold, no borderline re-score. Regression forces `recommendation = "continue"`.

**models.py**: `JudgeVerdict` updated from 4 to 6 sub-score fields (`volatility_impact_score`, `spread_impact_score`, `liquidity_impact_score` replace `impact_score`). Added `QualitativeAnalysis` model. `IterationSummary` updated to 6 sub-score fields.

**prompts.py**: `AGGREGATOR_SYSTEM` rewritten — removed `{scoring_rubric}` placeholder, removed convergence rules, now instructs LLM to produce `QualitativeAnalysis` structured output (qualitative analysis only).

**refinement_repository.py**: `IterationData` dataclass updated from `impact_score` → 3 separate impact fields. Save/load mappings updated.

**database/models.py**: `RefinementIteration` ORM: added `volatility_impact_score`, `spread_impact_score`, `liquidity_impact_score` columns. `impact_score` kept for backward compat with existing DBs.

**UI (1_Refinement_Lab.py, metric_display.py)**: Updated from 4-axis to 6-axis scoring display — history table, score progression chart, radar chart, metric cards, per-iteration breakdown all show 6 axes. `SCORING_AXIS_CONFIG` help text updated for new axes.

**Tests**: `test_llm_nodes.py` aggregator tests rewritten (9 tests: basic flow, convergence, max iterations, LLM failure fallback, regression/rollback, best-tracking, plateau, sub-score propagation, error scenario floor). `test_llm_prompts.py` updated (removed `build_scoring_rubric` import, rewrote aggregator prompt tests for qualitative-only). `test_refinement_repository.py` updated for 6-axis fields.

**Pyright error resolved**: The `build_scoring_rubric` import in `nodes.py` (broken since Step 1) is now replaced with `compute_axis_scores, compute_final_score`.

</details>

### Step 4: Fix `or 0` masking and metrics formatting ✅ DONE
**File: nodes.py**
- ~~Replace `(sr.strategy_pnl or 0)` with explicit `None` checks → "N/A" instead of "$0.00".~~
- ~~Fix negative PnL formatting: `$-7.41` → `-$7.41`.~~

**Completed:** Added 3 private formatting helpers (`_fmt_dollar`, `_fmt_pct`, `_fmt_float`) to `nodes.py`.
Replaced all 18 `or 0` occurrences across 4 formatting blocks (writer feedback, history table,
explanations, aggregator metrics). None values now display as "N/A"; negative PnL renders as
`-$7.41` not `$-7.41`. 20 new tests added (13 for helpers, 4 for integration edge cases,
3 for history table formatting).

### Step 5: Fix seed consistency — per-scenario fixed seeds ✅ DONE
**File: graph.py, state.py, nodes.py**
- ~~In `run_refinement()`, assign deterministic seed per scenario (hash of name + session timestamp) if not already set.~~
- ~~Same seed reused across all iterations for that scenario.~~

**Completed:** Added `seed: int | None` field to `ScenarioConfig`.  `run_refinement()` now assigns
deterministic seeds via `_deterministic_seed(name, session_ts)` (SHA-256 hash, uint32 range).
`scenario_executor_node` injects scenario seed into `SimulationSettings`.  4 tests added for
seed determinism and `ScenarioConfig.seed`.

### Step 6: Increase default max iterations ✅ DONE
**File: graph.py**
- ~~`max_iterations`: 3 → 5.~~
- ~~`_DEFAULT_RECURSION_LIMIT`: 50 → 80.~~

**Completed:** `max_iterations` default changed to 5, `_DEFAULT_RECURSION_LIMIT` changed to 80.
Test added to verify constant value.

### Step 7: Parametrized test suite for scoring ✅ DONE
**File: tests/test_deterministic_scoring.py (new)**
- ~~15-20+ test vectors covering every piecewise boundary, edge case, and guard condition in all 6 axes.~~

**Completed:** New test file with 89 parametrized test vectors across 8 test classes covering
all 6 axis boundary conditions, guard conditions (None, zero trades, missing baselines),
OTT penalties, drawdown penalties, liquidity averaging, weighted final score, and rounding.

---

## Phase 2 — Interpretation Quality (P1: the core product investment)

These changes build toward the future product: deep, investigative analysis of strategy behaviour. They have lasting architectural value regardless of whether the refinement loop stays or goes.

### Step 8: Expose rich simulation data — build the analysis toolbox ✅ DONE

**Why this matters most:** The explainer (and in the future, the interpretation system) currently receives a 22-line text summary of a simulation that produces thousands of data points. You have a full MRI and you're handing your AI analyst a Post-it note. Every improvement to interpretation quality starts here.

**File: analysis_service.py** — add new computation methods:

| Tool function | Input | Output | What it reveals |
|---|---|---|---|
| `get_fill_analysis(output, agent_id)` | Sim output + agent ID | DataFrame: timestamp, side, price, qty, mid_at_fill, slippage_bps | Per-fill execution quality. Shows adverse selection, aggressive vs passive fills |
| `get_pnl_curve(output, agent_id, initial_cash)` | Sim output + agent ID | DataFrame: timestamp, mark_to_market_pnl | Where the strategy made/lost money over time. Currently reconstructed then discarded — persist it |
| `get_inventory_trajectory(output, agent_id)` | Sim output + agent ID | DataFrame: timestamp, position | Position buildup/unwinding patterns. Exposes inventory risk |
| `get_adverse_selection(output, agent_id, window_ns)` | Sim output + agent ID + look-ahead window | float: avg mid-price move against the fill direction | Is the strategy consistently trading on the wrong side of impending moves? |
| `get_order_lifecycle(output, agent_id)` | Sim output + agent ID | DataFrame: order_id, submitted_at, status, resting_time_ns, fill_rate | How long orders rest, what fraction fill, cancel rate |
| `get_counterparty_breakdown(output, agent_id)` | Sim output (full logs) | dict: {agent_type: {count, avg_size, pnl_contribution}} | Who is the strategy trading against? Noise (easy) vs informed (hard)? |
| `query_book_depth(output, timestamp, n_levels)` | Sim output + timestamp | dict: {bids: [(p,q)...], asks: [(p,q)...]} | L2 snapshot at any moment — enables "what did the book look like when X happened?" |

**File: analysis_service.py** — add new chart methods:

| Chart | Method | Visualization |
|---|---|---|
| PnL over time | `plot_pnl_curve()` | Line chart — where money was made/lost, drawdown periods visible |
| Inventory over time | `plot_inventory()` | Line chart — position buildup, risk accumulation, unwinding |
| Fill scatter | `plot_fills_vs_mid()` | Scatter: fills plotted against mid-price progression with buy/sell color coding |

### Step 9: Tool-equipped explainer agent ✔️ DONE

**Why:** This is the architectural change that transforms ROHAN from "system that computes metrics" into "system that investigates strategy behaviour." The explainer should be a ReAct agent that can query simulation data on demand, follow chains of reasoning, and produce deep, causally-grounded analysis.

#### Architecture Decision: Enriched Serializable Bundle (Option A)

Three architectural options were evaluated for how the explainer agent accesses simulation data:

| Option | Description | Verdict |
|--------|-------------|--------|
| **A — Enriched serializable bundle** | Extend `RichAnalysisBundle` with raw queryable data (mid-price series, L2 snapshots, multi-window adverse selection). Tools parse the serialised JSON; NO live `SimulationOutput` needed. | **✅ Chosen** |
| B — Co-located explainer in container | Run the explainer inside the simulation container while `SimulationOutput` is still alive. | ❌ Rejected: couples explainer to container lifecycle, breaks re-explainability |
| C — Full serializable SimulationOutput proxy | Serialize the entire `SimulationOutput` (DataFrames → Parquet → base64). | ❌ Rejected: massive state bloat (~10–50 MB per scenario), quadratic graph checkpoint growth |

**Rationale:** Option A keeps all data in a single JSON blob on `ScenarioResult.rich_analysis_json` (already stored since Step 8). The explainer is container-independent, re-runnable, and checkpoint-safe. Can evolve into B or C later if needed.

**Critical constraint:** `SimulationOutput` is NOT stored in `RefinementState`. It depends on live ABIDES objects, is not JSON-serialisable, and would break LangGraph checkpointing, replay, and container scaling.

#### Sub-step 9.1 — Extend `RichAnalysisBundle` ✅ DONE

**File: analysis_models.py** — Added two new models and three new fields:

| Addition | Purpose |
|----------|---------|
| `MidPricePoint(timestamp_ns, mid_price)` | Single L1 mid-price observation |
| `L2Snapshot(timestamp_ns, bids, asks)` | L2 order-book snapshot (list of (price, qty) tuples) |
| `RichAnalysisBundle.adverse_selection_by_window` | `dict[str, float]` — adverse selection at 100ms, 500ms, 1s, 5s windows |
| `RichAnalysisBundle.mid_price_series` | `list[MidPricePoint]` — full L1 mid-price time-series for tool recomputation |
| `RichAnalysisBundle.l2_snapshots` | `list[L2Snapshot]` — sampled L2 snapshots at fills, PnL turning points, and every ~5s |

#### Sub-step 9.2 — Enrich `compute_rich_analysis` ✅ DONE

**File: analysis_service.py** — `compute_rich_analysis()` now:
1. Computes multi-window adverse selection (100ms, 500ms, 1s, 5s)
2. Extracts the full L1 mid-price series from `_build_mid_lookup()`
3. Samples L2 snapshots at fill timestamps + PnL turning points + every ~5s interval, capped at 200 snapshots

#### Sub-step 9.3 — Rewrite investigation tools ✅ DONE

**File: tools.py** — Replace `make_explainer_tools(output: SimulationOutput)` (8 tools that were NEVER USED in the graph) with `make_investigation_tools(rich_json: str) -> list[BaseTool]`.

The new factory parses `rich_json` into a `RichAnalysisBundle` once at creation time, then creates closure-bound tools. When `rich_json` is `None`, tools return "No analysis data available."

**Parameterised tools** (8 total):

```python
def query_fills(start_ns: int | None = None, end_ns: int | None = None,
                side: str | None = None, limit: int = 50) -> str:
    """Get per-fill execution data with optional time-range and side filters."""

def query_pnl_curve(start_ns: int | None = None, end_ns: int | None = None,
                    limit: int = 100) -> str:
    """Get mark-to-market PnL over time with optional time-range filter."""

def query_inventory(start_ns: int | None = None, end_ns: int | None = None,
                    limit: int = 100) -> str:
    """Get position trajectory with optional time-range filter."""

def query_adverse_selection(window_label: str | None = None) -> str:
    """Get adverse selection metrics. If window_label is given, return that
    specific window; otherwise return all pre-computed windows."""

def query_book_at_time(timestamp_ns: int, n_levels: int = 5) -> str:
    """Get nearest L2 snapshot to a specific timestamp."""

def query_counterparties() -> str:
    """Get breakdown of which agent types the strategy traded against."""

def query_order_lifecycle(status: str | None = None, limit: int = 50) -> str:
    """Get order lifecycle records with optional status filter."""

def get_simulation_summary() -> str:
    """Get high-level summary: fill count, PnL range, inventory range,
    adverse selection, time span."""
```

All tools return human-readable strings with truncation guards. Each tool's return is ≤ 4 KB to keep LLM context manageable.

#### Sub-step 9.4 — Rewrite explainer prompts ✅ DONE

**File: prompts.py** — Replace `EXPLAINER_SYSTEM` / `EXPLAINER_HUMAN` with `EXPLAINER_SYSTEM_REACT` / `EXPLAINER_HUMAN_REACT`.

- System prompt: investigation methodology (query data → form hypothesis → verify with tools → conclude). Instructs the agent to act as a quantitative analyst.
- Human prompt: adds `{regime_context}` placeholder for forward-compatibility with adversarial scenarios. When no regime is active, the slot is empty.

#### Sub-step 9.5 — Rewrite `explainer_node` with `create_react_agent` ✅ DONE

**File: nodes.py** — Replace the single structured-output LLM call with a ReAct agent:

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(
    model=llm,
    tools=make_investigation_tools(rich_json),
    prompt=react_system_prompt,
    response_format=ScenarioExplanation,  # structured final output
)
```

- `recursion_limit=25` to cap tool-calling loops.
- Fallback: on agent failure, fall back to a single structured-output call (same as pre-Step 9) with a warning log. The pipeline never breaks.
- Helper functions: `_error_explanation()`, `_extract_explanation()`, `_fallback_structured_explanation()`.

- The explainer prompt instructs the agent to investigate, not just summarize. Example flow:
  1. Look at the PnL curve → notice a sharp drop at minute 12
  2. Query fills around that timestamp → find a cluster of sells at below-mid prices
  3. Query adverse selection → confirm fills were systematically adversely selected
  4. Query book depth at minute 12 → see bid-side depth collapsed
  5. Conclude: "The momentum agents drove price down at minute 12. Your strategy was holding long inventory and couldn't unwind fast enough. Recommendation: add a VPIN-based spread widening when `vpin > 0.6` to reduce fill probability during informed trading periods."

- The cost of this is more LLM calls per iteration, but each call is small (tool responses are tabular, not prose). The payoff is dramatically better feedback quality.

#### Sub-step 9.6 — Add `regime_context` to `ScenarioResult` ✅ DONE

**File: state.py** — Add `regime_context: str = ""` to `ScenarioResult`. This field carries the regime description for the explainer prompt's `{regime_context}` slot. Empty for now; populated by the adversary node in Phase A of the adversarial plan. Forward-compatible.

#### Data flow (end-to-end)

```
Simulation → SimulationOutput (live, in-process)
    ↓
compute_rich_analysis() → RichAnalysisBundle (Pydantic)
    ↓
.model_dump_json() → ScenarioResult.rich_analysis_json (str, serialisable)
    ↓
make_investigation_tools(rich_json) → 8 closure-bound tools
    ↓
create_react_agent(model, tools, prompt, response_format=ScenarioExplanation)
    ↓
ScenarioExplanation (structured output)
```

`SimulationOutput` is consumed and discarded in the executor. Only the serialised bundle crosses node boundaries.

### Step 10: Persist and surface richer data to both UI and writer ✅ DONE

**File: state.py / ScenarioResult** — Already has `pnl_chart_b64`, `inventory_chart_b64`, `fill_scatter_b64` (added in Step 9).

**File: nodes.py (executor)** — Already generates the 3 rich charts (added in Step 9).

**File: llm/models.py (`ScenarioMetrics`)** — Added `pnl_chart_b64`, `inventory_chart_b64`, `fill_scatter_b64` to carry rich charts through aggregation.

**File: nodes.py (aggregator)** — Copies the 3 rich charts from `ScenarioResult` into `ScenarioMetrics`.

**File: framework/database/models.py (`RefinementScenarioResult`)** — Added 3 rich chart `Text` columns + `rich_analysis_json` column for persistence.

**File: framework/refinement_repository.py** — `ScenarioResultData` DTO, `save_session()`, and `load_session()` all thread the 6 chart fields + `rich_analysis_json`.

**File: ui/pages/1_Refinement_Lab.py** — `_save_current_run` maps all 6 charts. Chart display upgraded to 2×3 grid: Market row (Price/Spread/Volume) + Strategy Performance row (PnL/Inventory/Fills).

**Writer multimodal injection** — Deferred. Current focus was persist + surface; writer receives text-only feedback.

---

## Phase 3 — Prompt & Model Cleanup (P2: polish and consistency)

These changes clean up the prompt architecture and data models to align with the deterministic scoring + qualitative analysis split.

### Step 11: Simplify aggregator prompt ✅ DONE (completed in Step 3)

### Step 12: Update models for 6-axis scoring + qualitative output ✅ DONE (completed in Step 3)

### Step 13: Restructure feedback routing ✅ DONE

Implemented structured per-scenario feedback routing:
- Added `scenario_weaknesses` and `scenario_recommendations` fields to `AggregatedFeedback` (models.py).
- Aggregator populates them from explainer `ScenarioExplanation` objects (nodes.py).
- Added `_render_per_scenario_feedback()` helper to format per-scenario bullet lists for the writer prompt (nodes.py).
- Writer feedback template now includes structured weaknesses/recommendations per scenario instead of flattened verdict reasoning.

### Step 14: Audit UI for 6-axis scoring ✅ DONE (completed in Step 3)

UI updated: 6-axis display (history table, score progression, radar chart, metric cards), `SCORING_AXIS_CONFIG` help text, `IterationData`/repository mappings. PnL curve/inventory/fill scatter charts deferred to Step 10.

### Step 15: Production hardening pass ✅ DONE

Post-Phase 3 review & hardening based on full codebase audit:
- **Deleted `SecretSettings`**: Removed `src/rohan/config/secrets_settings.py` entirely (dead class, hardcoded Unix `/run/secrets` path, redundant with `LLMSettings`).
- **Extended validator smoke test**: `smoke_test()` now exercises all 5 protocol methods — added `on_order_update()` with a mock `Order(FILLED)` and `on_simulation_end()` with a `MarketState`. Catches runtime crashes before expensive ABIDES simulation.
- **All-failed-scenarios guard**: Aggregator now detects when every scenario errored. Forces `recommendation="continue"` with explicit logging, preventing misleading 1.0 scores from being treated as legitimate results.
- **Graph-level timeout**: `run_refinement()` now accepts `max_wall_clock_seconds` (default 3600s) and logs a warning if the full loop exceeds the timeout.
- **Seed collision fix**: `_deterministic_seed()` now includes scenario index in the hash to disambiguate same-name scenarios: `hash(f"{index}:{name}:{timestamp}")`.
- **Error context on scenario failure**: Failed scenarios now set `interpreter_prompt` to a truncated error summary instead of `None`, so the explainer receives meaningful context.

---

## Phase 4 — Future Architecture (P3: post-PoC, product evolution)

These are not part of the current implementation sprint but define the architectural direction.

### F1: Adversarial scenario generation

When the user provides their own strategy, the system's job is to **find conditions that break it**. This means:

- **Scenario parameter space**: agent population mix (more informed traders, fewer noise), volatility regimes, liquidity shocks (sudden book depth collapse), trend vs mean-reversion markets, spread regimes.
- **LLM-driven scenario design**: an agent that proposes scenarios designed to stress-test specific strategy weaknesses identified by the explainer. "Your strategy accumulates inventory during trends → let's test with 50 momentum agents and 500 noise agents."
- **Outcome**: the system produces a stress-test report: "Your strategy survives normal conditions but loses $X in regime Y because of Z. Here's the inventory trajectory and PnL curve."

### F2: Session memory and cross-session learning

A vector store of `(strategy_code, scenario_config, scores, qualitative_analysis)` from past sessions enables:
- Starting from the best-known strategy instead of from scratch
- Identifying recurring failure patterns across strategy variants
- Building a library of "known-bad patterns" that the writer/user is warned about

### F3: Parametric tuning separation

Once a structurally sound strategy exists, separate code logic from parameters. Let the system run grid/random search on parameter space while holding code fixed. LLMs are bad at numerical optimization; systematic search is better by orders of magnitude.

---

## Verification

Phase 1:
- Run a full refinement loop with goal "Create a basic market-making strategy"
- Scores are deterministic (identical inputs → identical scores)
- First-iteration scores reach 3-6 range (not stuck at 1-2)
- Scores improve across iterations (no false rollbacks from seed noise)
- Seeds remain consistent across iterations for each scenario
- `pytest tests/` passes (including test_deterministic_scoring.py)

Phase 2: ✅
- Explainer agent uses tools to investigate simulation data
- Explainer output references specific timestamps, fill data, inventory trajectories — not just aggregate metrics
- PnL curve, inventory trajectory, and fill scatter charts appear in the UI (2×3 grid)
- Rich analysis JSON + all 6 charts persisted to DB and survive save/load round-trip
- Writer receives richer textual feedback (multimodal deferred)

Phase 3:
- UI displays 6 axes correctly
- Qualitative analysis model produces structured strengths/weaknesses/recommendations
- Feedback routing preserves per-scenario structure
- `SecretSettings` class deleted — no imports remain
- Validator smoke test covers all 5 protocol methods
- All-failed-scenarios guard prevents misleading scores
- Graph timeout parameter available via `run_refinement()`
- Same-name scenarios get distinct seeds

---

## Decisions Log

| Decision | Chosen | Over | Rationale |
|----------|--------|------|-----------|
| Scoring method | Deterministic formulas | LLM scoring | Eliminates noise, auto-adapts to config, debuggable. LLM kept for qualitative analysis |
| Profitability normalizer | Opportunity capture rate (spread × volume) | Fixed dollar thresholds, bps/hr | Adapts to any duration, capital, volatility, competitive landscape |
| Scoring axes | 6 (vol/spread/liquidity split) | 4 (composite market impact) | Prevents mixed signals from being hidden by averaging |
| Drawdown normalizer | % of starting capital | Drawdown/PnL ratio | Avoids division-by-zero near zero PnL |
| Seed strategy | Per-scenario fixed, assigned at session start | Random per iteration, session-level single seed | Cross-iteration comparability + multi-condition robustness |
| Feedback routing | Structured per-scenario weaknesses/recommendations | Flattened string | Preserves code-level specificity |
| Explainer architecture | Tool-equipped ReAct agent | Prompt-only summarizer | Enables investigation, not just summarization — core product differentiator |
| Product direction | Scenario control + interpretation depth | Automated codegen loop | Loop is PoC scaffolding; interpretation quality is the product |
| DB backward compat | Drop old sessions | Migration | Still in development, no production data |
| Explainer data source | Enriched serializable bundle (Option A) | Co-located explainer in container (B), full SimulationOutput proxy (C) | Container-independent, re-explainable, checkpoint-safe. SimulationOutput is not JSON-serialisable and would break LangGraph checkpointing, replay, and container scaling. Can evolve into B or C later. |
| SimulationOutput in state | NOT stored in RefinementState | Store as `simulation_outputs: dict` | SimulationOutput depends on live ABIDES objects, is not serialisable, breaks checkpointing. All data flows through `rich_analysis_json` on ScenarioResult. |
| Old explainer tools | Replace entirely | Extend alongside new tools | `make_explainer_tools` contained 8 tools that were NEVER USED in the graph (dead code). Clean replacement with `make_investigation_tools` working from serialized data. |
| Tool parameterization | Parameterized with time-range, filters, limit | Fixed-parameter tools | Enables targeted investigation ("show fills between minute 10-12") instead of dumping all data. Keeps tool responses under ~4 KB. |
| Regime context | `regime_context: str` slot on ScenarioResult + explainer prompt | No regime awareness | Forward-compatible with adversarial scenario plan. Empty for now; populated by adversary node when implemented. |
| Rich chart persistence | Thread through all layers (DTO → ORM → UI) | Only in LangGraph state | Charts were generated but lost at the aggregator boundary. Consistent with "persist everything for replay". |
| `rich_analysis_json` storage | Inline in `RefinementScenarioResult` (Text column) | Separate artifacts table | Acceptable for dev/PoC (50–200 KB per scenario). Production should migrate to `artifacts` table. |
| Writer multimodal | Deferred | Inline chart injection | Focus on persist + surface first. Writer multimodal requires model capability gating. |
| UI chart layout | 2×3 grid (Market + Strategy) | Single row of 3 | Strategy-performance charts (PnL, Inventory, Fills) deserve equal visibility with market microstructure charts. |
| Defaults centralization | Constants in graph.py | Per-module hardcoded | Single source of truth for max_iterations (5), convergence threshold (7.0), recursion_limit (80). Eliminates UI/CLI/nodes.py drift. |
| Convergence threshold | 7.0 | 8.0 | Matches deterministic scoring where 7+ is "good". UI chart line + sidebar tips now consistent with code. |
| Writer prompt OTT fix | modify()/replace() example pattern | cancel_all()+re-place | cancel_all inflates OTT 10–20×, penalizing OTT score. Rules 5-6 now warn against it. |
| Simulation duration | 2 hours (09:30–11:30) | 30 minutes (09:30–10:00) | Enough for multiple mean-reversion cycles, meaningful PnL curves, realistic market microstructure dynamics. |
