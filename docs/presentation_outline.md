# ROHAN Presentation — Detailed 20-Slide Outline

> **Target audience:** Professors specializing in statistics and software architecture, financial engineers.
> **Master program:** AI & Data Analytics for Business.
> **For input into NotebookLM.**
> **Visual style:** White/clean background, distinct borders on all diagrams/charts, re-usable standalone images.

---

## Slide 1 — Title Slide

**Title:** *Evolutionary Market Microstructure: An Agentic Framework for Automated Strategy Design and Stress-Testing in Simulated Financial Environments*

**Content:**
- Author: Gabriele Di Corato
- Program: Master in AI & Data Analytics for Business
- Date: April 2026
- Two logos/icons: ROHAN (R.O.H.A.N. — Risk Optimizing Holistic Agentic Network) and ABIDES-Hasufel

**Visual:** Clean title layout. Below the title, a single-line subtitle: "Combining Agent-Based Market Simulation with LLM-Driven Agentic Orchestration."

---

## Slide 2 — Agenda

**Content:** Numbered visual agenda with the following sections:

1. Motivation & Problem Statement
2. Background: Market Microstructure & Agent-Based Simulation
3. ABIDES & The Hasufel Fork — What I Did
4. ROHAN Architecture Overview
5. The Agentic Loop — LangGraph Orchestration
6. Strategy Protocol & Sandbox Execution
7. Simulation Data Pipeline & Metrics
8. Deterministic Scoring System
9. Tool-Equipped Explainer Agent
10. Adversarial Scenario Planning
11. UI & User Workflow
12. Quality Engineering & Testing
13. Target Architecture — From PoC to Platform
14. Cost/Benefit Analysis
15. Demo / Results Walkthrough
16. Future Work & Conclusions

**Visual:** A clean numbered list with subtle section-grouping (e.g., "Foundation" for 1–3, "ROHAN Core" for 4–10, "Production Path" for 11–14, "Wrap-Up" for 15–16). Use thin horizontal dividers between groups.

---

## Slide 3 — Motivation & Problem Statement

**Title:** *Why Automate Strategy Validation?*

**Content — present these as three columns or stacked blocks with headers:**

**1. The Limitation of Backtesting**
- Traditional strategy validation uses backtesting against historical data.
- Backtesting is *static*: it cannot simulate market impact (how the market reacts to the strategy itself) or emergent behavior (unpredictable feedback loops between agents).
- A strategy that performs well in backtesting may trigger unintended consequences when deployed — cascading order cancellations, liquidity droughts, flash crashes.

**2. The Cost of Manual Validation**
- Industry benchmarks (McKinsey 2023, EY surveys): initial strategy validation takes 2–5 days per strategy by a senior quant analyst (~€150/hr).
- For a portfolio of 50 strategies with quarterly re-validation: **€240K–€480K/year** in analyst labor.
- Regulatory frameworks (SR 11-7, MiFID II Art. 17) mandate systematic stress testing and independent model validation — manual processes cannot scale.

**3. The Regulatory Mandate — MiFID II and Algorithmic Trading Testing**

The EU regulatory framework imposes explicit, binding obligations on investment firms engaged in algorithmic trading to test, validate, and stress-test their trading systems and strategies *before and after* deployment:

- **MiFID II Art. 17(1)** (Directive 2014/65/EU) — Investment firms engaged in algorithmic trading must have "*effective systems and risk controls suitable to the business it operates*" ensuring that trading systems are resilient, have sufficient capacity, are subject to appropriate thresholds and limits, and prevent erroneous orders or contributions to disorderly markets. Systems must be "*fully tested and properly monitored*."

- **RTS 6 Art. 5 — General Testing Methodology** (Commission Delegated Regulation (EU) 2017/589) — Prior to deployment or substantial update of any algorithmic trading system, trading algorithm, or strategy, firms must establish *clearly delineated methodologies* to develop and test them. Specifically, the methodology must ensure that the algorithm: (a) does not behave in an unintended manner; (b) complies with regulatory obligations; (c) complies with rules of connected trading venues; (d) does not contribute to disorderly trading conditions, **continues to work effectively in stressed market conditions**, and allows for shutdown under those conditions.

- **RTS 6 Art. 7 — Separated Testing Environments** — Testing must be conducted in an environment *separated from production*, used specifically for development and testing of algorithmic trading systems.

- **RTS 6 Art. 8 — Controlled Deployment** — Before deployment, firms must set predefined limits on: number of instruments traded, price/value/number of orders, strategy positions, and number of trading venues.

- **RTS 6 Art. 9 — Annual Self-Assessment and Validation** — Firms must annually perform a self-assessment and produce a validation report reviewing: (a) algorithmic trading systems, algorithms, and strategies; (b) governance, accountability, and approval framework; (c) business continuity; (d) overall compliance with Art. 17. The validation report must be drawn up by the **risk management function**, audited by internal audit, and approved by senior management.

- **RTS 6 Art. 10 — Mandatory Stress Testing** — As part of the annual self-assessment, firms must test that their systems can withstand increased order flows or market stresses. Prescribed tests include: high messaging volume (2× the highest in the previous 6 months) and high trade volume (2× the highest in the previous 6 months). Tests must not affect the production environment.

- **RTS 6 Art. 12 — Kill Functionality** — Firms must be able to immediately cancel all unexecuted orders across all connected trading venues as an emergency measure.

- **MiFID II Art. 48(6) — Trading Venue Requirements** — Regulated markets must require members/participants to carry out *appropriate testing of algorithms* and must *provide environments to facilitate such testing*, ensuring algorithms cannot create or contribute to disorderly trading conditions.

**U.S. Parallel — SR 11-7 (Federal Reserve / OCC, 2011):**
- While MiFID II governs the EU, U.S. banking organizations are subject to SR 11-7 — Guidance on Model Risk Management — which requires: rigorous model development and testing; independent model validation (*evaluation of conceptual soundness*, *ongoing monitoring*, *outcomes analysis/back-testing*); annual review at minimum; and governance with board-level oversight.
- The guidance mandates that validation involves "effective challenge" by objective, informed parties independent from model development.

**Why this matters for ROHAN:**
- These regulations collectively establish that algorithmic strategy testing is *not optional* — it is a legal obligation with specific procedural requirements (separated test environments, stress scenarios, annual validation, documented methodologies).
- Manual, ad-hoc validation processes cannot satisfy the rigor, reproducibility, and auditability demanded by RTS 6 Arts. 5–10 and SR 11-7 at the scale required by modern algorithmic trading desks.
- ROHAN directly addresses several of these requirements: sandboxed execution (Art. 7 — separated environment), adversarial stress scenarios (Art. 10 — stress testing), deterministic reproducible scoring (Art. 9 — auditable validation), and automated kill/timeout mechanisms (Art. 12 — kill functionality).

**4. The Gap We Address**
- LLMs can write and analyze code, but they lack a *closed-loop environment* to test that code, observe failure, and self-correct based on quantitative feedback.
- ROHAN closes this loop: an agentic system that autonomously generates, simulates, analyzes, and iteratively refines trading strategies against a high-fidelity market simulator.

**Visual/Diagram:**
- A simple before/after diagram:
  - **Before:** "Historical Data → Backtest → Human Reviews → Manual Fix → Repeat (days)"
  - **After:** "Goal → AI Generates → Simulates → AI Analyzes → AI Refines → Converged Strategy (minutes)"
- Use arrows and clearly bordered boxes on white background. Color-code the "before" path in grey (manual/slow) and the "after" path in blue (automated/fast).

---

## Slide 4 — Background: Market Microstructure & ABM

**Title:** *Agent-Based Simulation of Financial Markets*

**Content — explain these concepts for the audience (professors of statistics and financial engineering will appreciate rigorous definitions):**

**Limit Order Book (LOB)**
- The central mechanism of modern electronic exchanges. Two sides: bids (buy orders) and asks (sell orders), each an ordered queue of (price, quantity) tuples.
- Key microstructure quantities: best bid, best ask, spread ($s = P_{ask} - P_{bid}$), mid-price ($P_{mid} = (P_{bid} + P_{ask}) / 2$), LOB imbalance ($I = (Q_{bid} - Q_{ask}) / (Q_{bid} + Q_{ask})$, Cont, Kukanov & Stoikov, 2014).

**Agent-Based Modeling (ABM)**
- Instead of modeling the market as a single stochastic process, ABM models the market as an emergent phenomenon arising from the interaction of heterogeneous agents, each with their own strategy, latency, and information set.
- Key advantage: captures market impact, feedback loops, and emergent phenomena (flash crashes, liquidity spirals) that reduced-form models miss.

**ABIDES (Byrd & Balch, 2019)**
- Agent-Based Interactive Discrete Event Simulation — originally developed at Georgia Tech, maintained by J.P. Morgan Chase.
- Discrete-event architecture: a single Kernel processes a priority queue of timestamped messages. Agents communicate exclusively through messages with configurable latency.
- Agent types: NoiseAgent, ValueAgent (mean-reverts to fundamental), MomentumAgent, MarketMakerAgent, POV ExecutionAgent — plus a full NASDAQ-style matching engine (ExchangeAgent).

**Visual/Diagram:**
- A diagram of the Limit Order Book with bid and ask sides, showing price levels and quantities. Use a horizontal price axis with green (bids, left) and red (asks, right) bars showing depth at each level. Label the spread.
- Below it, a small "agent ecosystem" diagram: multiple agents (Noise, Value, Momentum, Market Maker, *Strategic Agent*) all connected to a central Exchange node via message arrows (labeled "orders, market data"). Add a latency icon on the arrows.

---

## Slide 5 — ABIDES-Hasufel: What I Inherited and What I Built

**Title:** *Extending ABIDES: The Hasufel Fork*

**Content — this slide documents YOUR contribution to the simulation layer. Two columns: "What I found" vs. "What I built."**

**What I Found (ABIDES Legacy)**
- Archived by J.P. Morgan, no longer maintained. Outdated dependencies (Python 3.8, NumPy 1.x).
- Test suite was broken — non-regression tests compared a hardcoded GitHub commit ID with itself (not meaningful).
- A critical `FinancialAgent` base class was missing; the simulator would not run out of the box.
- No declarative configuration — each simulation required a procedural `build_config()` function, tightly coupling agent instantiation with simulation parameters.
- No structured output — results extracted by manually iterating over raw `end_state` dicts.

**What I Built (abides-hasufel v2.5.8)**
1. **Dependency modernization** — Migrated to UV package manager, upgraded all dependencies to current versions, proper versioning with semantic releases.
2. **Re-implemented missing infrastructure** — Restored the `FinancialAgent` class, fixed the simulation pipeline.
3. **Declarative Configuration System** — Four-layer architecture:
   - *Models* — Pydantic `SimulationConfig` with typed sections (market, agents, infrastructure, simulation).
   - *Registry* — `@register_agent` decorator for pluggable agent types with `BaseAgentConfig`.
   - *Builder* — Fluent `SimulationBuilder` API: `.from_template("rmsc04").market(ticker="AAPL").seed(42).build()`.
   - *Compiler* — Converts immutable `SimulationConfig` → Kernel runtime dict.
   - *Templates* — Composable presets (base: `rmsc04`, `liquid_market`, `thin_market`; overlays: `with_momentum`, `with_execution`).
4. **Typed `SimulationResult`** — Immutable Pydantic model with `ResultProfile` depth control (SUMMARY/QUANT/FULL). Replaces manual `end_state` dict parsing.
5. **Rich Metrics API** — `compute_rich_metrics()` returns `RichSimulationMetrics` with per-agent analytics (Sharpe, drawdown, VWAP, fill rate, order lifecycle, multi-window adverse selection) and market microstructure metrics (LOB imbalance, resilience, VPIN, OTT ratio).
6. **Identity-based deterministic seeding** — SHA-256 derivation: each component's seed depends on master seed + component name. Adding/removing agent groups never shifts other agents' seeds. Enables fair baseline-vs-strategy A/B comparison.
7. **Parallel simulation** — `run_batch()` API: process-based parallelism with `spawn` context, unique log directories, picklable results.

**Visual/Diagram:**
- A layered architecture diagram of the Hasufel config system. Four stacked horizontal bands:
  - Top: **Templates** (rmsc04, liquid_market, thin_market, overlays)
  - Middle-top: **Builder** (fluent API)
  - Middle-bottom: **Models** (SimulationConfig → market, agents, infrastructure, simulation sections)
  - Bottom: **Compiler** (→ Kernel runtime dict → `run_simulation()` → `SimulationResult`)
- Use distinct borders, white background, minimal colors (blue for data flow arrows).

---

## Slide 6 — ROHAN Architecture Overview

**Title:** *ROHAN — System Architecture*

**Content — present the technology stack and module map:**

**Technology Stack Table:**

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Orchestration | LangGraph (Python) | Directed state-machine graph for the refinement loop |
| Simulation | abides-hasufel v2.5.8 | Discrete-event LOB simulator (fork of JP Morgan ABIDES) |
| LLM Integration | LangChain + OpenRouter | Model-agnostic; any OpenAI-compatible endpoint |
| Persistence | SQLAlchemy ORM | SQLite (dev) / PostgreSQL (prod) |
| UI | Streamlit (multipage) | Terminal (sim explorer) + Refinement Lab |
| Strategy Sandbox | AST validation + ThreadPoolExecutor | Rejects unsafe imports/constructs; timeout-bounded |
| Quality | ruff, pyright, pytest, hypothesis, pre-commit | CI via GitHub Actions |

**Module Map** (select the key 4 modules):
- `src/rohan/simulation/` — ABIDES integration: `config_builder.py`, `HasufelOutput`, `StrategicAgentAdapter`, `StrategicAgentConfig`
- `src/rohan/llm/` — Agentic graph: `graph.py`, `nodes.py`, `state.py`, `scoring.py`, `tools.py`, `planner.py`
- `src/rohan/framework/` — Analysis, persistence: `analysis_service.py`, `database/`, `repository.py`
- `src/rohan/ui/` — Streamlit: `0_Terminal.py`, `1_Refinement_Lab.py`

**Visual/Diagram:**
- A high-level system context diagram. Center block: "ROHAN Platform" containing four sub-blocks (Simulation Engine, Agentic Graph, Analysis & Persistence, UI). External actors on left/right: "User" connects to UI; "LLM Provider (OpenRouter)" connects to Agentic Graph; "ABIDES-Hasufel Engine" connects to Simulation Engine; "Database (SQLite/PostgreSQL)" connects to Persistence.
- Use clear bordered boxes, white background, directional arrows labeled with data types (e.g., "SimulationSettings", "HasufelOutput", "RichAnalysisBundle").

---

## Slide 7 — The Agentic Refinement Loop

**Title:** *LangGraph Orchestration — The Core Loop*

**Content — this is the central contribution. Walk through the data flow step by step:**

The refinement loop is a LangGraph directed state-machine graph. Each agent is a separate node with a single responsibility. Agents communicate through shared state (`RefinementState` TypedDict), not direct calls.

**Step-by-step flow:**

1. **User Goal** (natural language) → e.g., "Create a market maker that provides liquidity on ABM with tight spreads"
2. **Planner** (optional) — Pre-graph ReAct agent that proposes 3–5 adversarial scenarios using hasufel template-discovery tools. 3-tier fallback: ReAct agent → structured LLM output → keyword heuristic.
3. **Writer Agent** — Generates `StrategicAgent` Python code from goal + feedback from previous iterations. Receives `scenario_context` block describing market conditions across all scenarios.
4. **Validator Agent** — AST safety check (forbidden calls, dangerous dunders, import whitelist) + sandboxed execution with timeout. Up to 3 retries on failure.
5. **Scenario Executor** — Runs validated strategy in abides-hasufel across multiple scenarios (volatile, stable, custom). Per scenario: `HasufelOutput` → `compute_rich_analysis()` → `RichAnalysisBundle` → 6 charts (Price, Spread, Volume, PnL, Inventory, Fills).
6. **Explainer Agent** — ReAct agent with 8 investigation tools drills into per-fill data, PnL trajectory, order lifecycle, L2 snapshots. Produces `ScenarioExplanation` with findings.
7. **Aggregator** — Deterministic 6-axis scoring (no LLM in scoring loop!) + LLM qualitative analysis. Convergence/plateau/rollback detection.
8. **Loop decision:** Converged → DONE. Regression (>1.0 score drop) → rollback. Otherwise → next iteration (Writer receives structured per-scenario feedback).

**Visual/Diagram — THIS IS THE KEY DIAGRAM:**
- A flowchart following this layout (use clear, bordered boxes on white background):

```
[User Goal] ──→ [Planner] ──→ ┌─→ [Writer] ──→ [Validator] ──┐
                                │                    │ ❌ retry │
                                │                    ▼ ✅       │
                                │              [Scenario Executor]
                                │         ┌──────────┼──────────┐
                                │    [Scenario 1] [Scenario 2] [Scenario N]
                                │         │          │          │
                                │    [Explainer]  [Explainer]  [Explainer]
                                │         └──────────┼──────────┘
                                │              [Aggregator]
                                │              /    │    \
                                │       converged   │   regression
                                │          ↓        │      ↓
                                │        [DONE]  iterate  [rollback]
                                └──────────────────←┘
```

- Color code: green for generation nodes, blue for execution, purple for analysis, grey for aggregation. White background, black borders, clear arrows.

---

## Slide 8 — Data Contracts & State Management

**Title:** *Data Contracts Across Boundaries*

**Content — this slide is for the software architecture professors. Explain the explicit data boundaries:**

**Design principle:** Each boundary has a typed data contract. No implicit behavior.

| Boundary | Format | Rationale |
|----------|--------|-----------|
| Simulator → Executor node | `HasufelOutput` (live Python objects) | Access to typed `SimulationResult` with nanosecond-precision data |
| Executor → Explainer | `RichAnalysisBundle` (JSON string via `.model_dump_json()`) | Checkpoint-safe, container-independent, re-explainable without re-simulation |
| Explainer → Aggregator | `ScenarioExplanation` (Pydantic structured output) | Structured qualitative analysis |
| Aggregator → DB | `ScenarioMetrics` → `RefinementScenarioResult` (ORM) | Full round-trip persistence |

**Why JSON serialization at the Executor → Explainer boundary?**
- `SimulationOutput` depends on live ABIDES objects — not JSON-serializable. Storing it in LangGraph state would break checkpointing, replay, and container scaling.
- The `RichAnalysisBundle` captures everything the Explainer needs: fills, PnL curve, inventory trajectory, adverse selection windows, counterparty breakdown, order lifecycle, L2 snapshots.
- Any `ScenarioResult` with `rich_analysis_json` can be re-analyzed without re-running the simulation (re-explainability).

**Persistence Pipeline:**
```
ScenarioResult (LangGraph state)
  ↓ [aggregator_node]
ScenarioMetrics (6 chart fields + scores)
  ↓ [UI _save_current_run]
ScenarioResultData (DTO: 6 charts + rich_analysis_json)
  ↓ [save_session()]
RefinementScenarioResult (ORM: 6 Text columns + rich_analysis_json)
  ↓ [load_session()]
ScenarioMetrics (round-trip restored)
```

**Visual/Diagram:**
- A horizontal pipeline diagram showing the data transformations from left (Simulation) to right (Database). Each stage is a bordered box with the model name inside. Arrows labeled with the transformation (e.g., "`.model_dump_json()`", "ORM mapping"). Highlight the JSON serialization boundary with a dashed vertical line labeled "Checkpoint boundary."

---

## Slide 9 — Strategy Protocol & Sandbox

**Title:** *StrategicAgent Protocol & Safe Execution*

**Content:**

**The StrategicAgent Protocol** (defined in `strategy_api.py`) — the ONLY interface the LLM-generated code implements:
- **Units:** prices in integer cents, quantities in shares, cash in cents, timestamps in nanoseconds.
- **`MarketState`:** L1/L2 book data, portfolio, liquidity metrics, time remaining, open orders. `mid_price` and `spread` are `@computed_field`s.
- **`OrderAction`:** Discriminated union of 6 types (PLACE, CANCEL, CANCEL_ALL, MODIFY, PARTIAL_CANCEL, REPLACE). Validated by `@model_validator`.
- **5 callbacks:** `initialize()`, `on_tick()`, `on_market_data()`, `on_order_update()`, `on_simulation_end()`.

**The Adapter Pattern** (`StrategicAgentAdapter`):
- Thin wrapper: the generated `StrategicAgent` is wrapped inside an ABIDES `TradingAgent` subclass.
- Translates ABIDES discrete-event callbacks (`wakeup`, `receive_message`) → protocol callbacks (`on_tick`, `on_market_data`, `on_order_update`).
- Builds `MarketState` snapshots including portfolio valuation, real-time L1/L2 data, open orders.
- Dispatches `OrderAction`s via `match` on `OrderActionType` (handlers for place, modify, partial-cancel, replace).

**AST Sandbox:**
- `FORBIDDEN_CALLS`: `eval`, `exec`, `compile`, `getattr`, `__import__`, `ctypes`, `subprocess`.
- Import whitelist: `numpy`, `math`, `collections`, `dataclasses`, `enum`, `typing`.
- `DANGEROUS_DUNDERS` allowlist (rejects overly broad `__` usage but permits `super().__init__()`).
- Execution timeout via `ThreadPoolExecutor` with manual `shutdown(wait=False, cancel_futures=True)`.

**Visual/Diagram:**
- A three-layer "sandwich" diagram:
  - Top layer: "LLM-Generated Strategy (StrategicAgent Protocol)" — green border
  - Middle layer: "StrategicAgentAdapter (ABIDES ↔ Protocol bridge)" — orange border
  - Bottom layer: "ABIDES/Hasufel Discrete Event Engine" — blue border
  - Arrows between layers labeled with the translations (e.g., "wakeup() → on_tick(MarketState)", "OrderAction.PLACE → place_limit_order()").
- Side panel: "AST Sandbox" box with a checklist icon listing the forbidden/allowed constructs.

---

## Slide 10 — Metrics Framework

**Title:** *Performance Metrics & Evaluation Framework*

**Content — organize metrics in three groups. Emphasize the mathematical rigor for the statistics professors:**

**1. Agent Performance Metrics**
- **PnL:** $PnL_{total} = PnL_{realized} + PnL_{unrealized}$, where $PnL_{unrealized} = Inventory_{end} \times P_{mid,end}$
- **Sharpe Ratio:** $\text{Sharpe} = \frac{\bar{r}}{\sigma_r} \times \sqrt{\frac{T_{ns/year}}{\Delta t_{median}}}$ (annualized from fill-by-fill PnL curve; requires ≥30 observations)
- **Max Drawdown:** $\text{MaxDD} = \max_t(\max_{s \le t} PnL_s - PnL_t)$
- **Fill Rate:** $N_{executed} / N_{submitted}$
- **VWAP:** $\sum P_{fill,i} \times Q_i / \sum Q_i$ (integer cents)
- **Effective Spread:** $\frac{1}{N} \sum 2|P_{fill,i} - P_{mid,i}|$
- **Average Fill Slippage:** $\frac{1}{N} \sum sign_i \times \frac{P_{fill,i} - P_{mid,i}}{P_{mid,i}} \times 10000$ (basis points)
- **Multi-window Adverse Selection** (100ms, 500ms, 1s, 5s): $AS_w = \frac{1}{N} \sum sign_i \times (P_{mid}(t_i + w) - P_{mid}(t_i))$

**2. Market Microstructure Metrics**
- **Volatility:** Annualized σ of mid-price returns from two-sided L1 snapshots.
- **Mean Spread, Avg Bid/Ask Liquidity** — from L1 data.
- **LOB Imbalance** (Cont et al., 2014): $I_t = (Q_t^{bid} - Q_t^{ask}) / (Q_t^{bid} + Q_t^{ask})$
- **VPIN** (Easley, López de Prado & O'Hara, 2012): Volume-Synchronized Probability of Informed Trading using Lee-Ready tick test, 50 equal-volume buckets.
- **Market Resilience** (Foucault, Kadan & Kandel, 2013): Mean spread recovery time after shock events ($s_t > \bar{s} + 2\sigma_s$).
- **Market OTT Ratio** (MiFID II RTS 9): $N_{submissions} / N_{fills}$ across all agents.

**3. Market Impact (Strategy vs. Baseline Δ%)**
- $\Delta X(\%) = (X_{strategy} - X_{baseline}) / X_{baseline} \times 100$ for: volatility, spread, liquidity, LOB imbalance, VPIN, resilience, OTT ratio.
- Enables attributing market quality changes to the strategy.

**Visual/Diagram:**
- A structured table or three-panel layout. Each panel is a bordered box.
- Panel 1 (Agent Performance): key formulas for PnL, Sharpe, Slippage.
- Panel 2 (Market Microstructure): key formulas for VPIN, LOB Imbalance, Resilience.
- Panel 3 (Market Impact): the delta formula with arrows from "Baseline Run" and "Strategy Run" converging into the Δ calculation.
- Use LaTeX-rendered formulas. White background, thin borders.

---

## Slide 11 — Deterministic 6-Axis Scoring

**Title:** *Deterministic Scoring — No LLM in the Loop*

**Content — this is a critical design decision. Explain why and how:**

**Why deterministic scoring?**
- LLM-based scoring introduced *noise* in early prototypes: the same simulation results would receive different scores depending on LLM temperature, phrasing, and context window. This caused false rollbacks and prevented reliable convergence detection.
- All quantitative evaluation must be reproducible and auditable. The LLM contributes *qualitative analysis only* (reasoning, strengths, weaknesses, recommendations).

**The 6 Axes — each clamped to [1.0, 10.0], weighted average → final score:**

1. **Profitability** — Opportunity capture rate: $PnL / (\bar{s}_{baseline} \times V_{baseline} / 2)$. Maps loss → 1–3, marginal → 3–5, moderate → 5–7, strong → 7–9, exceptional → 9–10.
2. **Risk-Adjusted Performance** — Sharpe ratio piecewise mapping + drawdown penalty (>5% starting capital → −2, >2% → −1).
3. **Volatility Impact** — Baseline-relative Δ volatility. Stabilizing (negative Δ) → high score; destabilizing (positive Δ) → low score.
4. **Spread Impact** — Same piecewise as volatility. Tighter spreads = better.
5. **Liquidity Impact** — Inverted: more liquidity added = better. Availability penalty: >10% liquidity drop → −2, >5% → −1.
6. **Execution Quality** — Fill rate primary signal (< 5% → 1–2, > 50% → 8–10). OTT penalty (>200 → −2, >100 → −1). Slippage adjustment (negative slippage → +0.5; >10bps → −2).

**Convergence & rollback:**
- Score improvement < threshold over 2 consecutive iterations → plateau → stop.
- Score drop > 1.0 → regression → rollback to best iteration's strategy + feedback.

**Visual/Diagram:**
- A radar/spider chart showing the 6 axes (Profitability, Risk, Volatility Impact, Spread Impact, Liquidity Impact, Execution Quality). Show two overlaid polygons: Iteration 1 (smaller, red-ish) and Iteration 3 (larger, blue/green), demonstrating convergence.
- Alternatively: a stacked horizontal bar chart showing each axis score with the piecewise thresholds marked.
- White background, clear axis labels, distinct border around the chart.

---

## Slide 12 — The Tool-Equipped Explainer

**Title:** *ReAct Explainer — AI That Investigates, Not Just Summarizes*

**Content — this is the core analytical investment. Explain the ReAct pattern and tools:**

**Architecture:**
- Built with `create_react_agent` from `langgraph.prebuilt`.
- The agent follows the ReAct (Reasoning + Acting) pattern: think → call a tool → observe result → think → call another tool → ... → produce structured output.
- Does NOT receive raw simulation output. It works from serialized `RichAnalysisBundle` JSON via 8 parameterized closure-bound tools.

**The 8 Investigation Tools:**

| Tool | Parameters | What It Returns |
|------|-----------|----------------|
| `query_fills` | start_ns, end_ns, side, limit | Filtered fill records with slippage |
| `query_pnl_curve` | start_ns, end_ns, limit | PnL points in time range |
| `query_inventory` | start_ns, end_ns, limit | Position trajectory |
| `query_adverse_selection` | window_label | Per-window or all-window AS |
| `query_book_at_time` | timestamp_ns, n_levels | Nearest L2 snapshot |
| `query_counterparties` | — | Agent-type breakdown |
| `query_order_lifecycle` | status, limit | Order records (filled/cancelled/resting) |
| `get_simulation_summary` | — | High-level statistics |

**Output:** `ScenarioExplanation` (Pydantic structured output) with:
- Findings, root causes, recommendations.
- Interpretation guidance for multi-window adverse selection, VWAP comparison, fill slippage analysis.

**Fallback:** On ReAct agent failure → falls back to single structured-output LLM call. Pipeline never breaks.

**Example investigation chain:**
1. Agent calls `get_simulation_summary` → sees negative PnL and low fill rate.
2. Agent calls `query_fills(side="BUY")` → notices buy fills have high positive slippage (cost).
3. Agent calls `query_adverse_selection(window_label="100ms")` → confirms short-term toxic flow.
4. Agent calls `query_book_at_time(timestamp_ns=<worst_fill_time>)` → sees thin ask side at that moment.
5. Agent concludes: "Strategy is buying into one-sided books and getting adversely selected."

**Visual/Diagram:**
- A sequence diagram or step-by-step flow. Left column: "Explainer Agent (LLM)" with thought bubbles. Right column: "Tools" with response boxes. Show 3–4 iterations of Think → Tool Call → Observation → Think → Conclusion.
- Use bordered boxes with distinct colors: blue for agent reasoning, green for tool calls, orange for tool responses.
- White background, clear arrows.

---

## Slide 13 — Adversarial Scenario Planning

**Title:** *AI-Designed Stress Tests*

**Content:**

**The Planner Agent** — A pre-graph ReAct agent that proposes adversarial scenarios targeting strategy weaknesses.

**3-Tier Fallback Design:**
1. **ReAct agent** with 3 scenario tools (`build_scenario`, `validate_scenario`, `explain_metrics`) — discovers hasufel templates and composes adversarial scenario configurations.
2. **Structured LLM output** — If the ReAct agent fails, falls back to a single structured-output call that proposes scenario names + rationale.
3. **Keyword heuristic** — If LLM completely fails, selects from a hardcoded set based on goal keywords (e.g., "market maker" → volatile + thin market scenarios).

**Scenario Execution:**
- Each scenario receives a deterministic seed: SHA-256(scenario_name + session_timestamp). Same seed across all iterations → fair comparison.
- Baseline run (no strategy) establishes microstructure reference values. Cached when config + seed match a prior baseline.
- Strategy run with identical config → `HasufelOutput` → `RichAnalysisBundle` → scoring → charts.

**Example Scenarios:**
| Scenario | Template | Market Condition |
|----------|----------|-----------------|
| Baseline | rmsc04 | Standard: 1000 Noise, 102 Value, 12 Momentum, 2 MM |
| Volatile | rmsc04 + high fund_vol | 3× normal volatility, wider spreads |
| Thin Market | thin_market | 100 Noise, 20 Value, no MM — low liquidity |
| Toxic Flow | rmsc04 + high informed ratio | Elevated adverse selection pressure |

**Visual/Diagram:**
- A fan-out diagram: single "Strategy Code" box at left, arrow fans out to 3–4 scenario boxes (Baseline, Volatile, Thin, Toxic), each producing a "Scores + Charts" output box on the right. Below, show the seed derivation: "SHA-256(name + timestamp) → deterministic seed → reproducible simulation."
- White background, bordered boxes, scenario boxes color-coded by severity (green = baseline, yellow = moderate, red = extreme).

---

## Slide 14 — Simulation Charts & Analysis Output

**Title:** *6 Charts Per Scenario — Forensic Observability*

**Content:**

For each scenario execution, the system generates 6 charts arranged in a 2×3 grid:

**Market Microstructure Row:**
1. **Price Series** — Mid-price over simulation time (ns from midnight). Shows price dynamics, trend, and volatility regime.
2. **Bid-Ask Spread** — Spread over time. Reveals liquidity conditions, spread widening events, and strategy impact on market quality.
3. **Volume at BBO** — Best bid and ask quantities over time. Shows depth available to the strategy.

**Strategy Performance Row:**
4. **PnL Curve** — Mark-to-market PnL trajectory (dense, L1-sampled via hasufel `compute_equity_curve`). Shows drawdowns, recovery patterns, and cumulative performance.
5. **Inventory Trajectory** — Position (shares) over time. Reveals inventory buildup, unwinding, and exposure patterns.
6. **Fill Scatter** — Each fill plotted as a point (time × price), colored by side (buy/sell), overlaid on mid-price. Shows execution timing relative to market moves.

All charts are base64-encoded PNG, persisted as ORM Text columns, and survive save/load round-trip.

**Visual/Diagram:**
- A 2×3 grid of EXAMPLE charts (schematic/placeholder). Each chart is a bordered rectangle with axis labels and a title. Market row on top, Strategy row below. Use simple line/scatter representations — the actual charts will come from a real run.
- Alternatively, annotate a real screenshot from the Streamlit UI showing the chart grid.
- White background, distinct borders on each chart panel.

---

## Slide 15 — UI & User Workflow

**Title:** *Streamlit UI — Terminal & Refinement Lab*

**Content:**

**Two-Page Application:**

1. **Terminal Page (0_Terminal.py)** — Simulation explorer.
   - Configure market scenarios from presets or custom parameters.
   - Run single simulations, view results, compare baseline vs. strategy.
   - Interactive charts and metric tables.

2. **Refinement Lab Page (1_Refinement_Lab.py)** — LLM refinement UI.
   - Enter a natural language goal (e.g., "Create a market maker that provides liquidity with tight spreads").
   - Configure: model selection, max iterations, convergence threshold, scenario count.
   - Watch real-time progress: iteration-by-iteration scores, charts, explanations.
   - Iteration history table: Iter, PnL, Trades, Fill Rate, Slippage, Vol Δ, Spread Δ, Score, Summary.
   - Drill down into any scenario's charts, explanation, and analysis.
   - Session persistence: save/load entire refinement sessions from database.

**Visual/Diagram:**
- Two annotated screenshots (or schematic mockups) side by side:
  - Left: Terminal page showing a simulation result with charts and metrics.
  - Right: Refinement Lab showing iteration history, current scores, and the 2×3 chart grid.
- If using mockups: bordered wireframes on white background with labeled sections (sidebar, main content, chart grid, history table).

---

## Slide 16 — Quality Engineering

**Title:** *Testing, Linting & CI Pipeline*

**Content:**

**Test Suite:**
- **749+ unit tests** covering all modules — pytest with structured test organization.
- **89 parametrized piecewise-boundary tests** — full coverage of all 6 scoring axes with edge cases, ensuring deterministic scoring is correct at every breakpoint.
- **Property-based testing** (Hypothesis) — for `MarketState` computed fields and `OrderAction` factory methods. Tests invariants across random inputs.
- **Integration tests** — minimal end-to-end tests through the simulation pipeline.
- **Failure-path tests** — explicit tests for error handling, timeouts, and edge cases.

**Linting & Type Checking:**
- `ruff` — fast Python linter/formatter. Rules enforced on every commit.
- `pyright` — static type checker. Strict mode on critical modules.
- `pre-commit` hooks — ruff lint/format + pyright + nbstripout. CI fails on any violation.

**CI/CD (GitHub Actions):**
- Full test suite on every push/PR.
- Codecov integration for coverage tracking.
- CodeQL for security scanning.

**Security Hardening (v0.2.1):**
- Sandbox escape chain closed: removed `__import__` from safe builtins, added `_make_safe_import` whitelist.
- Database: fixed mutable ORM defaults, added eager loading, secured connection strings.
- HTML injection mitigated in Streamlit templates.
- Matplotlib figure leak fixed.

**Visual/Diagram:**
- A CI pipeline diagram: "Git Push" → "Pre-commit (ruff, pyright, nbstripout)" → "GitHub Actions (pytest, codecov, CodeQL)" → "Pass/Fail badge".
- Below it, a small table: "749 unit tests | 89 scoring boundary tests | Hypothesis property tests | ruff + pyright strict | CodeQL security scan"
- White background, clean flow arrows, green checkmark icons.

---

## Slide 17 — Target Architecture: From PoC to Platform

**Title:** *Target Architecture — Agentic Strategy Validation Platform*

**Content — the vision for production:**

**Key Evolution:** ROHAN evolves from an autonomous strategy-*generation* loop into an agentic strategy-*validation* platform. The system accepts user-supplied strategies (not LLM-generated), applies AI-driven adversarial scenario design, and produces forensic-quality validation reports.

**Target User:** Risk management offices and model validation teams within quantitative investment firms.

**The Linear DAG (replaces the cyclic loop):**

```
[Ingest] → [Analyze] → [Plan] → [Execute] → [Explain] → [Report]
```

1. **Ingest** — Strategy code + metadata. AST validation. Strategy adapter translates real-time idioms → discrete-event callbacks.
2. **Analyze** — LLM reads code, produces `StrategyProfile` (inferred style, risk dimensions, weakness hypotheses). Cross-validates declared vs. inferred intent.
3. **Plan** — Merges mandatory scenarios (3-tier registry: regulatory, strategy-type, organization) + AI adversarial scenarios. Deterministic seeds.
4. **Execute** — Parallel simulation across all scenarios. Rich data capture.
5. **Explain** — ReAct Explainer per scenario + cross-scenario aggregation. Ranks findings by severity and frequency.
6. **Report** — Multi-format output: interactive dashboard, PDF, JSON, email digest. Immutable, versioned, audit-ready.

**Key Simplifications:**
- No Writer node (strategies are user-supplied).
- No loop edges (one pass through the DAG). Predictable cost, simpler checkpointing.
- Deterministic scoring preserved as-is.
- Tool-equipped Explainer preserved as-is.

**Cloud Infrastructure:**
- API Gateway (FastAPI) → Validation Orchestrator (LangGraph) → Simulation Worker Pool (hasufel) → PostgreSQL + S3 + Redis.
- RBAC: analyst, risk_manager, compliance, admin roles.
- Three trigger modes: on-demand, scheduled (cron), CI webhook.

**Visual/Diagram:**
- The linear DAG as a clean horizontal pipeline: 6 bordered boxes connected by arrows, labeled with the stage name and key output artifact (StrategyProfile, ScenarioPlan, ScenarioResult[], ValidationReport).
- Below it, the cloud component diagram: API Gateway, Orchestrator, Worker Pool, PostgreSQL, S3 — as bordered boxes with connecting arrows.
- White background, minimal colors, distinct borders.

---

## Slide 18 — Cost/Benefit Analysis

**Title:** *Cost/Benefit Analysis — Is It Worth It?*

**Content — present the key numbers. The financial engineers in the audience will care about TCO, payback period, and sensitivity analysis:**

**Total Cost of Ownership (3-Year Horizon):**

| Profile | Build (Year 1) | Annual Ops | 3-Year TCO |
|---------|----------------|------------|------------|
| Lean startup (1 eng, cost-optimized) | €80K–€170K | €42K–€127K | ~€250K |
| Mid-scale (2 eng, prod infra) | same | same | ~€380K |
| Enterprise (3 eng, high-throughput) | same | same | ~€550K |

**LLM Costs — Current vs. Target Architecture:**

| Architecture | Per-Run Cost (3 scenarios) | Per-Run Cost (12 scenarios) |
|---|---|---|
| Current (cyclic, 3 iter avg) | ~$2.38 | ~$8–$10 |
| Target (linear DAG) | ~$0.35–$1.00 | ~$1.00 |

Key insight: **LLM costs are <5% of TCO. Infrastructure dominates.**

**Annual Savings (Base Case: 30% human oversight):**

| Portfolio Size | Manual Cost | ROHAN Cost | Annual Savings | Payback |
|---|---|---|---|---|
| 10 strategies | €60K–€120K | €24K–€46K | €14K–€96K | 2–5 years |
| 50 strategies | €240K–€480K | €83K–€170K | €70K–€397K | 5–18 months |
| 200 strategies | €800K–€1.5M | €269K–€504K | €296K–€1M | 2–7 months |

**Efficiency Multiplier:**

| Metric | Manual | ROHAN | Improvement |
|---|---|---|---|
| Time to validate (initial) | 2–5 days | 15–45 min | 10–50× |
| Scenarios per validation | 3–5 | 8–15 | 2–5× coverage |
| Audit response time | 2–5 days | Instant | >100× |

**PoC Maturity:** ~80% of core algorithm and data pipeline logic is already built and tested. The target architecture build phase is service decomposition + deployment — not core R&D.

**Visual/Diagram:**
- Two charts:
  1. A bar chart showing "Annual Savings by Portfolio Size" with three grouped bars (10, 50, 200 strategies). Each bar has two segments: "ROHAN Cost" (bottom, blue) and "Savings" (top, green). The "Manual Cost" is shown as a dotted line.
  2. A line chart showing "Payback Period vs. Portfolio Size" — three lines for 20%, 30%, 40% oversight rates, showing the breakeven in months vs. number of strategies.
- White background, distinct borders, clear axis labels.

---

## Slide 19 — Key Risks & Build vs. Buy

**Title:** *Risks, Trade-Offs, and Alternatives*

**Content:**

**Top Risks:**

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Single-ticker limitation** — ABIDES supports one ticker; production strategies often trade multiple assets | High | High | Document limitation; plan multi-ticker hasufel support; some strategies decomposable per-ticker |
| **Adapter fidelity** — many real-time strategies can't be faithfully translated to discrete-time | Medium | High | Protocol-native path has zero fidelity loss; adapter builds incrementally |
| **Simulation fidelity** — ABIDES is stylized, not a real exchange | Medium | Medium | Reports include model limitations; scoring is relative (strategy vs. baseline), not absolute |
| **LLM reliability** — Explainer/Planner may produce incorrect analysis | Medium | Medium | 3-tier fallbacks; deterministic scoring is LLM-free; human review expected |

**Build vs. Buy vs. Manual:**

| Dimension | Manual | Build (ROHAN) | Buy (Vendor) |
|---|---|---|---|
| Annual cost (50 strats) | €240K–€480K | €83K–€170K | €150K–€400K |
| Forensic depth | Analyst skill | 8-tool investigation | Aggregate metrics |
| Scenario design | Human judgment | AI + mandatory registry | Vendor-defined |
| IP control | Full | Full (open-source stack) | Strategy shared with vendor |
| Time-to-validate | Days | Minutes | Hours |

**Key differentiator:** ROHAN combines AI adversarial scenario design + full LOB simulation + forensic explainability. No commercial vendor offers this combination, and strategy code never leaves the firm.

**Visual/Diagram:**
- A risk matrix (2×2 grid: Probability vs. Impact) with the top risks plotted as labeled circles. Use red/orange/green coloring.
- Or: a comparison table with the Build vs. Buy vs. Manual content above, color-coded (green cells for ROHAN advantages, neutral for parity, red for disadvantages).

---

## Slide 20 — Future Work & Conclusions

**Title:** *Conclusions & Future Steps*

**Content — summarize accomplishments and forward-looking work:**

**What Was Built:**
1. **ABIDES-Hasufel** (v2.5.8) — A modernized, declarative, AI-friendly market simulator forked from J.P. Morgan's ABIDES. Includes: config system (4-layer architecture), typed `SimulationResult`, rich metrics API, identity-based deterministic seeding, parallel `run_batch()`, 7 documentation guides.
2. **ROHAN** (v0.3.1) — An agentic framework for autonomous strategy refinement. Includes: LangGraph orchestration with 5 specialized agents, deterministic 6-axis scoring, tool-equipped ReAct Explainer with 8 investigation tools, adversarial scenario planner with 3-tier fallback, Streamlit UI (Terminal + Refinement Lab), SQLAlchemy persistence, AST strategy sandbox, 749+ tests.
3. **Full Documentation Suite** — Technical architecture, functional metrics specification, target architecture, cost/benefit analysis — all grounded in financial literature.

**Key Contributions:**
- Demonstrated that an LLM agentic system *can* autonomously generate, simulate, analyze, and iteratively refine trading strategies with convergent improvement.
- Scoring is fully deterministic — no LLM in the quantitative evaluation loop. This is essential for reproducibility and auditability.
- The ReAct Explainer with 8 tools provides forensic-depth analysis that exceeds what a human analyst would typically produce per strategy per scenario.
- Bridged the gap between ABIDES's outdated, procedural codebase and modern AI-friendly interfaces (Pydantic models, fluent builders, typed results).

**Phased Roadmap:**

| Phase | Scope | Duration |
|---|---|---|
| Phase 1: MVP | Linear DAG, on-demand trigger, PostgreSQL, single-server | 3–4 months |
| Phase 2: Production | FastAPI, containerization, cloud, PDF reports, RBAC | 2–3 months |
| Phase 3: Enterprise | Strategy adapter, CI webhooks, scenario governance, multi-format | 2–3 months |
| Phase 4: Evolution | Multi-ticker support, historical data integration, RL environments | Ongoing |

**Open Research Questions:**
- Multi-ticker simulation for portfolio-level strategy validation.
- Historical market data integration for more realistic oracle calibration.
- Reinforcement learning agent training within the ABIDES-Gym environment.

**Visual/Diagram:**
- A horizontal timeline showing the 4 phases as colored segments on a bar, with key deliverables labeled below each phase.
- Or: a summary "achievements" graphic showing the two projects side-by-side (ABIDES-Hasufel and ROHAN) with key stats (lines of code, test count, document count, agent count, metric count).

---

## Appendix — Notes on Visual Production

**General rules for all diagrams and charts:**
- **Background:** Pure white (#FFFFFF). No gradients.
- **Borders:** All boxes/panels have distinct, visible borders (1–2px, dark grey or black).
- **Colors:** Use a limited palette — primary blue, green for positive/success, red/orange for negative/risk, grey for neutral. No more than 5 colors per diagram.
- **Fonts:** Clean sans-serif (Inter, Helvetica, or similar). Title: 24–28pt bold. Body: 16–18pt. Formulas: LaTeX-rendered where possible.
- **Arrows:** Solid lines for data flow, dashed for optional/fallback paths. Arrow heads clearly visible.
- **Spacing:** Generous whitespace. Do not crowd elements. Better to spread across two slides than pack too tightly.
- **Standalone images:** Each diagram should work as a standalone image with its own title/caption. This allows re-use in the university template or a thesis document.

**Charts that should be real data (generate from actual simulation runs):**
- Slide 11: Radar chart with actual scores from a real refinement session (iterations 1 and 3).
- Slide 14: The 2×3 chart grid from a real simulation run.
- Slide 15: Screenshots from the actual Streamlit UI.
- Slide 18: Bar chart and line chart generated from the CBA tables above.

**Charts that should be schematic diagrams (produce in draw.io, Mermaid, or similar):**
- Slide 3: Before/after workflow.
- Slide 4: LOB diagram + agent ecosystem.
- Slide 5: Hasufel config system layers.
- Slide 6: System context diagram.
- Slide 7: Agentic loop flowchart (the KEY diagram — invest time here).
- Slide 8: Data contract pipeline.
- Slide 9: Strategy adapter sandwich + sandbox.
- Slide 12: Explainer ReAct sequence diagram.
- Slide 13: Scenario fan-out diagram.
- Slide 17: Target architecture DAG + cloud components.
- Slide 19: Risk matrix or comparison table.
- Slide 20: Timeline / roadmap.

---

## Speaker Notes Summary (per slide)

| Slide | Key Talking Points | Time (approx.) |
|---|---|---|
| 1 | Introduce yourself and the project name (ROHAN = Risk Optimizing Holistic Agentic Network). | 30s |
| 2 | Walk through the agenda. Mention the two sub-projects (ABIDES-Hasufel + ROHAN). | 30s |
| 3 | "The traditional approach is static backtesting. Real markets are dynamic ecosystems." Cite McKinsey/EY benchmarks for manual validation cost. | 2 min |
| 4 | "The LOB is the central mechanism. ABM captures emergent phenomena that stochastic models miss." Define ABIDES. | 1.5 min |
| 5 | "The simulator was archived and broken. I had to modernize it before I could build on it." Walk through the 7 contributions. | 2 min |
| 6 | "ROHAN sits on top of ABIDES-Hasufel." Show stack table + module map. | 1 min |
| 7 | "This is the core loop." Walk through each node. Emphasize: 'scoring is deterministic, the LLM never assigns a number.' | 2.5 min |
| 8 | "Every boundary has an explicit, typed data contract." For software architecture audience. | 1.5 min |
| 9 | "The LLM generates Python code against a clean protocol — it never sees ABIDES internals." Explain the adapter + sandbox. | 1.5 min |
| 10 | "We compute 30+ metrics grounded in market microstructure literature." Highlight Sharpe, VPIN, adverse selection. | 2 min |
| 11 | "Every metric is computed by formula, not by LLM. This eliminates scoring noise." Show radar chart. | 1.5 min |
| 12 | "The Explainer doesn't just summarize — it investigates. Like an analyst with programmatic access to the data." Walk through the example chain. | 2 min |
| 13 | "The system designs stress tests targeting YOUR strategy's specific weaknesses." Explain 3-tier fallback. | 1.5 min |
| 14 | "Every scenario produces 6 charts covering both market quality and strategy performance." Show the grid. | 1 min |
| 15 | "The UI lets users run simulations interactively or launch full refinement sessions." Show screenshots. | 1 min |
| 16 | "749 tests, property-based testing, static analysis, CI/CD. The quality bar is production-grade." | 1 min |
| 17 | "The target architecture moves from generation to validation — no more code generation loop." Explain the DAG + regulatory framing. | 2 min |
| 18 | "At 50+ strategies, the platform pays for itself within 18 months. LLM costs are under 5% of TCO." | 2 min |
| 19 | "The biggest risk is the single-ticker limitation. The biggest differentiator is forensic explainability." | 1 min |
| 20 | "I built a modernized simulator AND an agentic system on top of it. The PoC covers ~80% of the target architecture's core logic." | 1.5 min |
| **Total** | | **~30 min** (20 slides, ~1.5 min avg, with flexibility for Q&A) |
