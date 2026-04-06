---
marp: true
theme: default
math: mathjax
paginate: true
size: 16:9
style: |
  section {
    background-color: #ffffff;
    font-family: "Inter", "Roboto", sans-serif;
    font-size: 20px;
    padding: 40px 60px;
  }
  section.lead {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.lead h1 {
    font-size: 38px;
    border-bottom: 3px solid #2b6cb0;
    padding-bottom: 10px;
    color: #1a365d;
  }
  section.lead .subtitle {
    font-size: 20px;
    color: #4a5568;
    margin-top: 16px;
  }
  section.lead .meta {
    font-size: 16px;
    color: #718096;
    margin-top: 30px;
    line-height: 1.8;
  }
  h1 {
    color: #1a365d;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 5px;
    font-size: 34px;
    margin-top: 0;
    margin-bottom: 18px;
  }
  h2 { color: #2b6cb0; font-size: 24px; margin-top: 14px; margin-bottom: 6px; }
  h3 { color: #2d3748; font-size: 20px; margin-top: 10px; margin-bottom: 4px; }
  code {
    color: #38a169;
    background: #f0fff4;
    font-family: "Fira Code", monospace;
    font-size: 0.88em;
    padding: 2px 5px;
    border-radius: 4px;
  }
  pre { background: #f7fafc; border: 1px solid #cbd5e0; border-radius: 6px; padding: 12px; font-size: 0.82em; }
  table { border-collapse: collapse; width: 100%; font-size: 0.82em; margin: 12px 0; }
  th, td { border: 1px solid #cbd5e0; padding: 6px 10px; text-align: left; }
  th { background-color: #f7fafc; color: #2d3748; font-weight: 600; }
  blockquote {
    border-left: 4px solid #2b6cb0;
    background: #ebf8ff;
    padding: 10px 16px;
    margin: 12px 0;
    color: #1a365d;
    font-style: normal;
  }
  ul li, ol li { margin-bottom: 4px; line-height: 1.5; }
  .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 30px; }
  .col-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; }
  .box {
    border: 1px solid #cbd5e0;
    border-radius: 6px;
    padding: 14px;
    background: #f7fafc;
  }
  .box-green { border-color: #38a169; background: #f0fff4; }
  .box-orange { border-color: #dd6b20; background: #fffaf0; }
  .box-blue { border-color: #2b6cb0; background: #ebf8ff; }
  .tag { display: inline-block; background: #e2e8f0; border-radius: 3px; padding: 1px 6px; font-size: 0.8em; }
  .diagram-placeholder {
    border: 2px dashed #a0aec0;
    border-radius: 8px;
    background: #f7fafc;
    color: #718096;
    text-align: center;
    padding: 20px;
    font-size: 0.85em;
    margin-top: 10px;
  }
---

---


<!-- _class: lead -->

# Evolutionary Market Microstructure
## An Agentic Framework for Automated Strategy Design and Stress-Testing in Simulated Financial Environments

<div class="subtitle">Combining Agent-Based Market Simulation with LLM-Driven Agentic Orchestration</div>

<div class="meta">
Gabriele Di Corato &nbsp;·&nbsp; Master in AI & Data Analytics for Business &nbsp;·&nbsp; April 2026<br>
<strong>R.O.H.A.N.</strong> — Risk Optimizing Holistic Agentic Network &nbsp;|&nbsp; <strong>ABIDES-Hasufel</strong>
</div>




---


# Agenda

<div class="columns">
<div>

**Foundation**
1. Motivation & Problem Statement
2. Market Microstructure & ABM Background
3. ABIDES-Hasufel: What I Built

**ROHAN Core**

4. System Architecture Overview
5. The Agentic Refinement Loop
6. Strategy Protocol & Sandbox
7. Metrics Framework

</div>
<div>

**ROHAN Core (cont.)**

8. Deterministic 6-Axis Scoring
9. ReAct Explainer Agent
10. Adversarial Scenario Planning
11. Simulation Charts & Output

**Production Path**

12. UI & User Workflow
13. Quality Engineering & CI
14. Target Architecture
15. Cost/Benefit Analysis

**Wrap-Up**

16. Risks & Trade-offs
17. Conclusions & Roadmap

</div>
</div>



<!--
Group the agenda visually with thin horizontal rules between Foundation / ROHAN Core / Production Path / Wrap-Up sections.
-->




---


# Why Automate Strategy Validation?

<div class="col-3">
<div class="box">

**Backtesting is Static**
- Cannot simulate *market impact* or emergent feedback loops
- Flash crashes, liquidity droughts invisible until live deployment
- A strategy passing backtests may still destabilize markets

</div>
<div class="box-orange">

**Manual Validation Costs Scale Poorly**
- 2–5 days / strategy · €150/hr senior quant analyst
- 50-strategy portfolio: **€240K–€480K/year**
- SR 11-7, MiFID II Art. 17 mandate systematic stress testing

</div>
<div class="box-blue">

**ROHAN Closes the Loop**
- LLMs can write code but lack a *closed-loop environment* to observe failure
- ROHAN: goal → generate → simulate → analyze → refine → converge
- Minutes, not days. Auditable and reproducible.

</div>
</div>



<!--
Diagram: Before/After workflow.
BEFORE (grey boxes): Historical Data → Backtest → Human Reviews → Manual Fix → Repeat (days)
AFTER (blue boxes): Goal → AI Generates → Simulates → AI Analyzes → AI Refines → Converged Strategy (minutes)
White background, white boxes, directional arrows, color-coded paths.
-->




---


# Agent-Based Simulation of Financial Markets

## Limit Order Book (LOB)

Two-sided queue: bids (buy) and asks (sell) ordered by price.
Key quantities: spread $s = P_{ask} - P_{bid}$, &nbsp; mid-price $P_{mid} = (P_{bid}+P_{ask})/2$, &nbsp; LOB imbalance $I = (Q^{bid}-Q^{ask})/(Q^{bid}+Q^{ask})$ &nbsp;*(Cont et al., 2014)*

## Agent-Based Modeling (ABM)

Market as an **emergent phenomenon** from heterogeneous agents — each with own strategy, latency, information set.
Captures market impact, feedback loops, flash crashes that reduced-form stochastic models miss.

## ABIDES *(Byrd & Balch, 2019)*

Discrete-event Kernel · priority-queue message bus · configurable latency.
Agent types: `NoiseAgent` · `ValueAgent` · `MomentumAgent` · `MarketMakerAgent` · `POVExecutionAgent` · `ExchangeAgent`



<!--
Diagram 1: LOB visualization — horizontal price axis, green bid bars left, red ask bars right, spread labeled.
Diagram 2: Agent ecosystem — Noise, Value, Momentum, Market Maker, Strategic Agent all connected to central Exchange node with latency arrows.
-->




---


# Extending ABIDES: The Hasufel Fork
## What I Found (ABIDES Legacy)

<div class="box-orange">

- **Archived** by J.P. Morgan — no longer maintained, Python 3.8, NumPy 1.x
- **Broken tests** — regression suite compared a hardcoded commit ID with itself
- **Missing `FinancialAgent`** base class — simulator would not run at all
- **No declarative config** — each simulation required a procedural `build_config()` function
- **No structured output** — results extracted by manually iterating raw `end_state` dicts

</div>

> Starting point: a simulator that could not simulate.



<!--
Column layout: 'What I Found' and 'What I Built' side-by-side. Left column grey/neutral, right column green/positive.
-->




---


# abides-hasufel v2.5.8 — What I Built

<div class="columns">
<div>

1. **Dependency modernization** — UV, current deps, semantic versioning
2. **Restored `FinancialAgent`** — simulation pipeline functional
3. **Declarative Config System** — 4-layer architecture:
   - *Models* — Pydantic `SimulationConfig` (typed sections)
   - *Registry* — `@register_agent` pluggable agent types
   - *Builder* — `.from_template("rmsc04").market(ticker="AAPL").seed(42).build()`
   - *Compiler* — immutable config → Kernel runtime dict
   - *Templates* — `rmsc04`, `liquid_market`, `thin_market` + overlays

</div>
<div>

4. **Typed `SimulationResult`** — Pydantic, `ResultProfile` depth control (SUMMARY/QUANT/FULL)
5. **Rich Metrics API** — 30+ metrics: Sharpe, VWAP, fill rate, VPIN, LOB imbalance, resilience, OTT ratio
6. **SHA-256 deterministic seeding** — identity-based; adding/removing agents never shifts seeds → fair A/B comparison
7. **`run_batch()` API** — process-based parallelism, `spawn` context, picklable results

</div>
</div>



<!--
Layered architecture diagram of the Hasufel config system — 4 stacked horizontal bands:
Top: Templates (rmsc04, liquid_market, thin_market, overlays)
Upper-mid: Builder (fluent API: .from_template().market().seed().build())
Lower-mid: Models (SimulationConfig: market / agents / infrastructure / simulation)
Bottom: Compiler (→ Kernel runtime dict → run_simulation() → SimulationResult)
Blue data-flow arrows, white background, distinct borders.
-->




---


# ROHAN — System Architecture

| Layer | Technology | Purpose |
|---|---|---|
| Orchestration | **LangGraph** (Python) | Directed state-machine graph |
| Simulation | **abides-hasufel v2.5.8** | Discrete-event LOB simulator |
| LLM Integration | **LangChain + OpenRouter** | Model-agnostic endpoint |
| Persistence | **SQLAlchemy ORM** | SQLite (dev) / PostgreSQL (prod) |
| UI | **Streamlit** multipage | Terminal + Refinement Lab |
| Strategy Sandbox | **AST + ThreadPoolExecutor** | Unsafe code rejection + timeout |
| Quality | **ruff · pyright · pytest · hypothesis** | CI via GitHub Actions |

**Key modules:** `src/rohan/simulation/` · `src/rohan/llm/` · `src/rohan/framework/` · `src/rohan/ui/`



<!--
System context diagram (C4 Context level):
Center: ROHAN Platform with 4 sub-blocks (Simulation Engine, Agentic Graph, Analysis & Persistence, UI)
Left: User → UI
Top: LLM Provider (OpenRouter) → Agentic Graph
Right: ABIDES-Hasufel Engine → Simulation Engine
Bottom: Database (SQLite/PostgreSQL) → Persistence
Arrows labeled with data types: SimulationSettings, HasufelOutput, RichAnalysisBundle
White background, distinct borders for each component box.
SOURCE: docs/agentic_framework.drawio — export to SVG and embed here.
-->




---


# LangGraph Orchestration — The Core Loop

Each node has a **single responsibility**. Communication via shared `RefinementState` TypedDict — no direct calls.

| Step | Node | Role |
|---|---|---|
| 1 | **User Goal** | Natural language strategy objective |
| 2 | **Planner** | ReAct agent proposes 3–5 adversarial scenarios (3-tier fallback) |
| 3 | **Writer** | Generates `StrategicAgent` Python code from goal + prior feedback |
| 4 | **Validator** | AST safety check + sandboxed execution (up to 3 retries) |
| 5 | **Scenario Executor** | Runs strategy across all scenarios → `RichAnalysisBundle` + 6 charts |
| 6 | **Explainer** | ReAct agent with 8 tools investigates each scenario result |
| 7 | **Aggregator** | Deterministic 6-axis scoring · convergence · rollback detection |
| 8 | **Loop** | Converged → DONE · Drop > 1.0 → rollback · Else → next iteration |

> **Critical:** Scoring is fully deterministic — no LLM ever assigns a number.



<!--
THIS IS THE KEY DIAGRAM. Produce in Draw.io or Mermaid.
Nodes (left to right, top path):
  [User Goal] → [Planner] → [Writer] → [Validator] (retry loop back to Writer on ❌)
  Validator ✅ → [Scenario Executor] fans out to [Scenario 1..N]
  Each scenario → [Explainer] → merge back → [Aggregator]
  Aggregator: converged → [DONE], regression → [Rollback → Writer], plateau → [DONE]
Color coding: green=generation, blue=execution, purple=analysis, grey=aggregation.
White background, black borders, solid arrows for main flow, dashed for feedback/rollback.
SOURCE: docs/agentic_framework_presentation.drawio
-->




---


# Data Contracts Across Boundaries

> Design principle: every boundary has a **typed data contract**. No implicit behavior.

| Boundary | Format | Rationale |
|---|---|---|
| Simulator → Executor | `HasufelOutput` (Python objects) | Access to nanosecond-precision typed `SimulationResult` |
| **Executor → Explainer** | `RichAnalysisBundle` (JSON, `.model_dump_json()`) | **Checkpoint-safe** · container-independent · re-explainable |
| Explainer → Aggregator | `ScenarioExplanation` (Pydantic structured output) | Structured qualitative analysis |
| Aggregator → DB | `ScenarioMetrics` → `RefinementScenarioResult` (ORM) | Full round-trip persistence |

**Why JSON at the Executor→Explainer boundary?**
`SimulationOutput` depends on live ABIDES objects — not serializable. JSON enables: checkpointing, container scaling, and **re-explaining any past scenario** without re-running the simulation.



<!--
Horizontal pipeline diagram:
[Simulation] --HasufelOutput--> [Executor] --RichAnalysisBundle (JSON)--> [Explainer] --ScenarioExplanation--> [Aggregator] --ORM--> [Database]
Add a dashed vertical line between Executor and Explainer labeled "Checkpoint boundary"
Each stage is a bordered box. Arrows labeled with the transformation method.
-->




---


# StrategicAgent Protocol & Safe Execution

<div class="columns">
<div>

## The Protocol (`strategy_api.py`)
The **only** interface LLM-generated code implements.

- Units: prices (¢) · quantities (shares) · time (ns)
- `MarketState`: L1/L2 book · portfolio · open orders · `mid_price` / `spread`
- `OrderAction`: discriminated union of 6 types (PLACE, CANCEL, CANCEL_ALL, MODIFY, PARTIAL_CANCEL, REPLACE)
- **5 callbacks:** `initialize()` · `on_tick()` · `on_market_data()` · `on_order_update()` · `on_simulation_end()`

</div>
<div>

## Adapter + AST Sandbox

`StrategicAgentAdapter` wraps strategy inside `TradingAgent` subclass.
Translates: `wakeup()` → `on_tick(MarketState)` · `OrderAction` → ABIDES calls.

**Forbidden:** `eval` · `exec` · `__import__` · `ctypes` · `subprocess`

**Allowed imports:** `numpy` · `math` · `collections` · `dataclasses` · `enum` · `typing`

Execution **timeout** via `ThreadPoolExecutor.shutdown(cancel_futures=True)`

</div>
</div>



<!--
3-layer sandwich diagram:
  TOP (green border): LLM-Generated Strategy (StrategicAgent Protocol)
  MIDDLE (orange border): StrategicAgentAdapter (ABIDES ↔ Protocol bridge)
  BOTTOM (blue border): ABIDES/Hasufel Discrete Event Engine
Inter-layer arrows labeled: "wakeup() → on_tick(MarketState)", "OrderAction.PLACE → place_limit_order()"
Side panel: AST Sandbox checklist listing FORBIDDEN_CALLS and allowed imports.
-->




---


# Performance Metrics — Agent Analytics

## Agent Performance Metrics

| Metric | Formula |
|---|---|
| Total PnL | $PnL_{total} = PnL_{realized} + Inventory_{end} \times P_{mid,end}$ |
| Sharpe (annualized) | $\text{Sharpe} = \dfrac{\bar{r}}{\sigma_r} \times \sqrt{\dfrac{T_{ns/year}}{\Delta t_{median}}}$ &nbsp; (requires ≥ 30 fills) |
| Max Drawdown | $\text{MaxDD} = \max_t\!\left(\max_{s \le t} PnL_s - PnL_t\right)$ |
| Fill Rate | $N_{executed} / N_{submitted}$ |
| Avg Fill Slippage | $\dfrac{1}{N}\sum sign_i \times \dfrac{P_{fill,i} - P_{mid,i}}{P_{mid,i}} \times 10{,}000$ &nbsp; (basis points) |
| Multi-window Adverse Selection | $AS_w = \dfrac{1}{N}\sum sign_i \times (P_{mid}(t_i+w) - P_{mid}(t_i))$ &nbsp; for $w \in \{100\text{ms}, 500\text{ms}, 1\text{s}, 5\text{s}\}$ |



<!--
Three-panel layout: Panel 1 for agent metrics (formulas), Panel 2 for microstructure metrics, Panel 3 for market impact delta. Use LaTeX-rendered formulas in each panel. White background, thin borders.
-->




---


# Performance Metrics — Market Microstructure & Impact

## Market Microstructure Metrics

| Metric | Formula / Definition |
|---|---|
| LOB Imbalance | $I_t = (Q_t^{bid} - Q_t^{ask}) / (Q_t^{bid} + Q_t^{ask})$ &nbsp; *(Cont et al., 2014)* |
| VPIN | Volume-Synchronized P(Informed Trading) · Lee-Ready tick test · 50 equal-volume buckets &nbsp; *(Easley et al., 2012)* |
| Market Resilience | Mean spread recovery time after $s_t > \bar{s} + 2\sigma_s$ shocks &nbsp; *(Foucault et al., 2013)* |
| OTT Ratio | $N_{submissions} / N_{fills}$ across all agents &nbsp; *(MiFID II RTS 9)* |

## Market Impact (Strategy vs. Baseline)

$$\Delta X(\%) = \frac{X_{strategy} - X_{baseline}}{X_{baseline}} \times 100$$

Applied to: volatility · spread · liquidity · LOB imbalance · VPIN · resilience · OTT ratio.
Enables **attributing market quality changes** directly to the strategy.



<!--
Continue the three-panel layout from slide 10a. Panel 2: VPIN, LOB Imbalance, Resilience formulas. Panel 3: Delta formula with arrows from Baseline Run and Strategy Run converging.
-->




---


# Deterministic Scoring — No LLM in the Loop

> LLM-based scoring introduced noise in early prototypes: same results → different scores → false rollbacks. **The LLM contributes qualitative analysis only.**

**6 Axes — each clamped $ [1.0, 10.0]$, weighted average → final score:**

| Axis | Signal | Scoring Logic |
|---|---|---|
| Profitability | $PnL / (\bar{s}_{baseline} \times V_{baseline}/2)$ | loss→1–3 · marginal→3–5 · strong→7–9 · exceptional→9–10 |
| Risk-Adjusted | Sharpe piecewise | + Drawdown penalty: >5% capital → −2, >2% → −1 |
| Volatility Impact | Δ% vs. baseline | Stabilizing (negative Δ) → high score |
| Spread Impact | Δ% vs. baseline | Tighter spreads = better |
| Liquidity Impact | Δ% vs. baseline | More liquidity added = better; >10% drop → −2 |
| Execution Quality | Fill rate | < 5% → 1–2 · > 50% → 8–10; OTT > 200 → −2; slippage > 10 bps → −2 |

**Convergence:** improvement < threshold over 2 iterations → plateau → stop.
**Rollback:** score drop > 1.0 → revert to best iteration + structured feedback.



<!--
Radar/spider chart: 6 axes = Profitability, Risk-Adjusted, Volatility Impact, Spread Impact, Liquidity Impact, Execution Quality.
Show two overlapping polygons: Iteration 1 (smaller, red/orange) and Iteration 3 (larger, blue/green).
White background, distinct borders. Generate with Python Matplotlib from a real refinement session.
-->




---


# ReAct Explainer — AI That Investigates, Not Just Summarizes

Built with `create_react_agent` (LangGraph). Works from serialized `RichAnalysisBundle` — **never** raw simulation objects.

| Tool | Returns |
|---|---|
| `get_simulation_summary` | High-level stats: PnL, fill rate, Sharpe |
| `query_fills(side, start_ns, end_ns)` | Fill records with per-fill slippage |
| `query_adverse_selection(window_label)` | AS scores for 100ms / 500ms / 1s / 5s windows |
| `query_book_at_time(timestamp_ns)` | Nearest L2 snapshot |
| `query_pnl_curve` · `query_inventory` · `query_counterparties` · `query_order_lifecycle` | Full forensic data |

**Example chain:** `get_simulation_summary` → negative PnL, low fill rate → `query_fills(side="BUY")` → high slippage → `query_adverse_selection("100ms")` → toxic flow confirmed → `query_book_at_time(<worst fill>)` → thin ask side → **Conclusion:** *"Strategy buys into one-sided books and gets adversely selected."*

**Fallback:** ReAct failure → structured-output LLM call. Pipeline never breaks.



<!--
Sequence diagram (2 columns):
Left: Explainer Agent (LLM) with thought-bubble steps.
Right: Tool responses (bordered boxes).
Show 4 iterations: Think → Tool Call → Observation → Think → Conclusion.
Colors: blue=agent reasoning, green=tool calls, orange=tool responses.
-->




---


# AI-Designed Stress Tests

**Planner Agent** — pre-graph ReAct agent. Proposes adversarial scenarios targeting *your strategy's specific weaknesses.*

**3-Tier Fallback:**
1. ReAct agent with `build_scenario` / `validate_scenario` / `explain_metrics` tools
2. Structured LLM output (single call)
3. Keyword heuristic (e.g., "market maker" → volatile + thin market)

| Scenario | Template | Market Condition |
|---|---|---|
| Baseline | `rmsc04` | 1,000 Noise · 102 Value · 12 Momentum · 2 MM |
| Volatile | `rmsc04` + high `fund_vol` | 3× normal volatility, wider spreads |
| Thin Market | `thin_market` | 100 Noise · 20 Value · no MM — illiquid |
| Toxic Flow | `rmsc04` + high informed ratio | Elevated adverse selection pressure |

**Determinism:** `SHA-256(scenario_name + session_timestamp)` → same seed across all iterations → fair comparison.



<!--
Fan-out diagram:
Left: Single 'Strategy Code' box
Center: arrows fan out to 4 scenario boxes (Baseline=green, Volatile=yellow, Thin Market=orange, Toxic Flow=red)
Right: each scenario produces 'Scores + Charts' output box
Below: seed derivation — SHA-256(scenario_name + session_timestamp) → deterministic seed → reproducible simulation
-->




---


# 6 Charts Per Scenario — Forensic Observability

For each scenario execution, the system generates a **2 × 3 chart grid** — persisted as base64 PNG in ORM Text columns.

<div class="columns">
<div>

**Row 1 — Market Microstructure**

1. **Price Series** — mid-price over time; volatility regime, trend
2. **Bid-Ask Spread** — spread widening events; strategy's liquidity impact
3. **Volume at BBO** — depth dynamics; thinning events

</div>
<div>

**Row 2 — Strategy Performance**

4. **PnL Curve** — mark-to-market trajectory; drawdown patterns
5. **Inventory Trajectory** — position buildup, unwinding, exposure risk
6. **Fill Scatter** — each fill (time × price) colored buy/sell, overlaid on mid-price

</div>
</div>

All charts survive **save/load round-trip** — no re-simulation needed to revisit.



<!--
2×3 grid of schematic chart placeholders (or real screenshots from Streamlit UI):
Row 1 (Market Microstructure): Price Series | Bid-Ask Spread | Volume at BBO
Row 2 (Strategy Performance): PnL Curve | Inventory Trajectory | Fill Scatter (buy=green dots, sell=red dots, mid-price line)
Each chart is a bordered rectangle with title and axis labels. White background.
-->




---


# Streamlit UI — Terminal & Refinement Lab

<div class="columns">
<div class="box-blue">

## Terminal Page
*Simulation Explorer*

- Select preset or configure custom scenario parameters
- Run single simulation: baseline vs. strategy comparison
- View 6-chart grid, rich metric tables, LOB snapshots
- Compare across scenarios interactively

</div>
<div class="box-green">

## Refinement Lab
*LLM Orchestration UI*

- Enter a natural-language goal
- Configure: model, max iterations, convergence threshold, scenario count
- Watch **real-time progress**: scores + charts per iteration
- **Iteration history table:** PnL · Trades · Fill Rate · Slippage · Vol Δ · Spread Δ · Score
- Drill into any scenario's explanation and charts
- **Session persistence:** save/load full runs from database

</div>
</div>



<!--
Two annotated screenshots side-by-side:
Left: Terminal page showing simulation result with charts and metric tables.
Right: Refinement Lab showing iteration history table, current scores radar, and 2×3 chart grid.
If mockups: bordered wireframes on white background with labeled sections.
-->




---


# Testing, Linting & CI Pipeline

<div class="columns">
<div>

## Test Suite
- **749+ unit tests** across all modules
- **89 parametrized boundary tests** for all 6 scoring axes — every piecewise breakpoint verified
- **Property-based testing** (Hypothesis) — `MarketState` computed fields, `OrderAction` factory methods
- Integration + failure-path tests

## Security Hardening (v0.2.1)
- Sandbox: `__import__` removed from builtins; `_make_safe_import` whitelist added
- ORM: fixed mutable defaults, eager loading, secured connection strings
- Mitigated HTML injection in Streamlit templates

</div>
<div>

## Toolchain

| Tool | Role |
|---|---|
| `ruff` | Linting + formatting |
| `pyright` | Static type checking (strict) |
| `pre-commit` | ruff + pyright + nbstripout |
| **pytest** | 749+ tests |
| **Hypothesis** | Property-based testing |
| **CodeQL** | Security scanning |
| **Codecov** | Coverage tracking |

CI runs on every push/PR. All gates must pass.

</div>
</div>



<!--
CI pipeline diagram: Git Push → Pre-commit (ruff, pyright, nbstripout) → GitHub Actions (pytest, codecov, CodeQL) → Pass/Fail badge.
Add a summary stats block below the pipeline showing test counts.
-->




---


# Target Architecture — From PoC to Platform

**Key Evolution:** Strategy *generation* loop → Strategy *validation* platform. Accept user-supplied code; produce forensic-quality audit reports.

**Target Users:** Risk management offices, model validation teams in quantitative investment firms.

## The Linear Validation DAG

```
[Ingest] → [Analyze] → [Plan] → [Execute] → [Explain] → [Report]
```

| Stage | Input | Output |
|---|---|---|
| **Ingest** | Strategy code + metadata | AST-validated, adapter-translated code |
| **Analyze** | Strategy code | `StrategyProfile` (inferred style, risk hypotheses) |
| **Plan** | Profile + mandatory registry | `ScenarioPlan` (regulatory + AI adversarial scenarios) |
| **Execute** | ScenarioPlan | `ScenarioResult[]` with rich data |
| **Explain** | ScenarioResult[] | `ExplanationBundle` ranked by severity |
| **Report** | ExplanationBundle | PDF / JSON / dashboard — immutable, versioned, audit-ready |



<!--
Linear 6-node DAG diagram (horizontal pipeline):
[Ingest] → [Analyze] → [Plan] → [Execute] → [Explain] → [Report]
Each box labeled with stage name + key output artifact below it:
  Ingest: StrategyCode | Analyze: StrategyProfile | Plan: ScenarioPlan | Execute: ScenarioResult[] | Explain: ExplanationBundle | Report: ValidationReport
Arrow style: solid, left-to-right. White background, colored borders by stage type (blue=AI, green=execution, purple=analysis).
-->




---


# Target Architecture — Cloud Infrastructure

<div class="columns">
<div>

## Service Architecture
- **API Gateway** (FastAPI) — authentication, RBAC, rate limiting
- **Validation Orchestrator** (LangGraph) — DAG execution, checkpointing
- **Simulation Worker Pool** (hasufel) — auto-scaled, isolated processes
- **PostgreSQL** — results, audit log, session history
- **S3** — chart artifacts, strategy code snapshots
- **Redis** — scenario cache, job queue

</div>
<div>

## Key Simplifications vs. PoC

| PoC | Target |
|---|---|
| Cyclic loop (generation) | Linear DAG (validation) |
| LLM writes strategy code | User supplies strategy |
| Single-process | Container-scaled worker pool |
| SQLite | PostgreSQL + S3 + Redis |

**Trigger modes:** on-demand · scheduled (cron) · CI webhook

**~80% of core algorithm logic already built and tested.**

</div>
</div>



<!--
Cloud component diagram:
API Gateway (FastAPI) → Validation Orchestrator (LangGraph) → Simulation Worker Pool (hasufel, auto-scaled)
→ PostgreSQL (results) + S3 (charts/artifacts) + Redis (cache/queue)
Trigger modes: on-demand, scheduled (cron), CI webhook (shown as 3 input arrows to API Gateway)
RBAC roles shown as a small legend: analyst, risk_manager, compliance, admin
-->




---


# Cost/Benefit Analysis — TCO & LLM Costs

## 3-Year TCO

| Profile | Build (Year 1) | Annual Ops | 3-Year TCO |
|---|---|---|---|
| Lean (1 eng, cost-optimized) | €80K–€170K | €42K–€127K | **~€250K** |
| Mid-scale (2 eng, prod infra) | same | same | **~€380K** |
| Enterprise (3 eng, high-throughput) | same | same | **~€550K** |

## LLM Cost Per Run

| Architecture | 3 scenarios | 12 scenarios |
|---|---|---|
| Current (cyclic, avg 3 iterations) | ~$2.38 | ~$8–$10 |
| Target (linear DAG) | ~$0.35–$1.00 | ~$1.00 |

> **Key insight: LLM costs are < 5% of TCO. Infrastructure dominates.**



<!--
Bar chart: Annual Savings by Portfolio Size (10, 50, 200 strategies). Stacked bars: ROHAN Cost (bottom, blue) + Savings (top, green). Manual Cost as dotted line above each bar. White background, distinct borders.
-->




---


# Cost/Benefit Analysis — Annual Savings & Efficiency

## Annual Savings (Base Case: 30% human oversight)

| Portfolio | Manual Cost | ROHAN Cost | **Annual Savings** | Payback |
|---|---|---|---|---|
| 10 strategies | €60K–€120K | €24K–€46K | €14K–€96K | 2–5 years |
| 50 strategies | €240K–€480K | €83K–€170K | **€70K–€397K** | 5–18 months |
| 200 strategies | €800K–€1.5M | €269K–€504K | **€296K–€1M** | 2–7 months |

## Efficiency Multiplier

| Metric | Manual | ROHAN | **Improvement** |
|---|---|---|---|
| Time to validate | 2–5 days | 15–45 min | **10–50×** |
| Scenarios per validation | 3–5 | 8–15 | **2–5× coverage** |
| Audit response time | 2–5 days | Instant | **> 100×** |



<!--
Line chart: Payback Period (months) vs. Portfolio Size. Three lines for 20%, 30%, 40% human oversight rates. Show breakeven point. White background, distinct border around chart.
-->




---


# Risks & Trade-offs

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Single-ticker limitation** — ABIDES supports one ticker | High | High | Document; plan multi-ticker hasufel support; many strategies decomposable per-ticker |
| **Adapter fidelity** — real-time strategies may not translate faithfully | Medium | High | Protocol-native path: zero fidelity loss; adapter built incrementally |
| **Simulation fidelity** — ABIDES is stylized, not a real exchange | Medium | Medium | Scoring is relative (strategy vs. baseline), not absolute claim of real-world P&L |
| **LLM reliability** — Explainer/Planner may hallucinate | Medium | Medium | 3-tier fallbacks; deterministic scoring is LLM-free; human review expected |



<!--
Risk matrix (2×2 grid, Probability vs Impact). Plot 4 risks as labeled circles: Single-ticker (High/High, red), Adapter fidelity (Med/High, orange), Sim fidelity (Med/Med, yellow), LLM reliability (Med/Med, yellow). White background.
-->




---


# Build vs. Buy vs. Manual

| Dimension | Manual | Build (ROHAN) | Buy (Vendor) |
|---|---|---|---|
| Annual cost (50 strategies) | €240K–€480K | **€83K–€170K** | €150K–€400K |
| Forensic depth | Analyst skill | **8-tool investigation** | Aggregate metrics |
| Scenario design | Human judgment | **AI + mandatory registry** | Vendor-defined |
| IP / strategy privacy | Full | **Full (open-source stack)** | Strategy shared with vendor |
| Time-to-validate | Days | **Minutes** | Hours |
| Audit-ready reports | Manual | **Immutable, versioned** | Varies |

> **Key differentiator:** ROHAN combines AI adversarial scenario design + full LOB simulation + forensic explainability. No commercial vendor offers this combination. Strategy code never leaves the firm.



<!--
Comparison table color-coded: green cells = ROHAN advantages, grey = parity, red = ROHAN disadvantages. Add a callout box at the bottom for the key differentiator statement.
-->




---


# Conclusions & Future Steps

<div class="columns">
<div>

## What Was Built

**ABIDES-Hasufel v2.5.8**
Modernized declarative simulator · 4-layer config · 30+ metrics · deterministic seeding · parallel `run_batch()` · 7 documentation guides

**ROHAN v0.3.1**
LangGraph orchestration · 5 specialized agents · deterministic 6-axis scoring · 8-tool ReAct Explainer · adversarial scenario planner · Streamlit UI · SQLAlchemy persistence · AST sandbox · **749+ tests**

</div>
<div>

## Phased Roadmap

| Phase | Scope | Timeline |
|---|---|---|
| **1 — MVP** | Linear DAG · on-demand trigger · PostgreSQL · single-server | 3–4 months |
| **2 — Production** | FastAPI · containers · cloud · PDF reports · RBAC | 2–3 months |
| **3 — Enterprise** | Strategy adapter · CI webhooks · scenario governance | 2–3 months |
| **4 — Evolution** | Multi-ticker · historical data integration · RL environments | Ongoing |

</div>
</div>

> ~80% of core algorithm and data pipeline logic is already built and tested.



<!--
Horizontal timeline: 4 colored phase segments on a bar. Each phase labeled with name and key deliverable below. Add a 'What Was Built' summary block above the timeline with two sub-blocks: ABIDES-Hasufel and ROHAN.
-->
