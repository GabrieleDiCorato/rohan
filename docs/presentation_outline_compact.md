# ROHAN — 20-Slide Presentation (Compact Outline for NotebookLM)

Audience: professors of statistics, software architecture, and financial engineering. Master in AI & Data Analytics for Business. White/clean background, distinct-bordered diagrams reusable as standalone images. All content grounded in project documentation.

## Slide 1 — Title
"Evolutionary Market Microstructure: An Agentic Framework for Automated Strategy Design and Stress-Testing in Simulated Financial Environments." Author: Gabriele Di Corato. Two projects: ROHAN (Risk Optimizing Holistic Agentic Network) — the agentic framework; ABIDES-Hasufel — the modernized market simulator.

## Slide 2 — Agenda
Six sections: (1) Motivation, (2) Background on ABM & market microstructure, (3) ABIDES-Hasufel contributions, (4-10) ROHAN core architecture, (11-14) Production path & CBA, (15-16) Results & conclusions. Visual: numbered list grouped by section.

## Slide 3 — Motivation
Three problems. (A) Static backtesting cannot capture market impact or emergent feedback loops. (B) Manual validation costs €240K–€480K/yr for 50 strategies (McKinsey 2023, SR 11-7, MiFID II Art. 17 mandate systematic stress testing). (C) LLMs can write code but lack a closed-loop environment to test, observe failure, and self-correct. ROHAN closes this loop. Diagram: before/after workflow — "days of manual iteration" vs. "minutes of autonomous refinement."

## Slide 4 — Background: LOB, ABM, ABIDES
Limit Order Book: bids/asks as (price, qty) queues; spread, mid-price, LOB imbalance (Cont et al. 2014). Agent-Based Modeling: heterogeneous agents → emergent market dynamics (flash crashes, liquidity spirals). ABIDES (Byrd & Balch 2019): discrete-event simulator from Georgia Tech / J.P. Morgan — Kernel processes priority queue of timestamped messages; agents: Noise, Value, Momentum, MarketMaker, Exchange. Diagram: LOB depth chart + agent ecosystem around a central Exchange node.

## Slide 5 — ABIDES-Hasufel: My Contributions
Inherited: archived repo, broken tests, missing FinancialAgent class, procedural configs, no typed output. Built (v2.5.8): (1) UV migration + dependency modernization, (2) 4-layer declarative config system (Pydantic Models → Registry with @register_agent → fluent SimulationBuilder → Compiler), composable templates (rmsc04, liquid_market, thin_market + overlays), (3) typed immutable SimulationResult with ResultProfile depth control (SUMMARY/QUANT/FULL), (4) Rich Metrics API: compute_rich_metrics() returning RichSimulationMetrics with per-agent Sharpe/drawdown/VWAP/fill-rate/order-lifecycle/multi-window-adverse-selection + market microstructure (LOB imbalance, resilience, VPIN, OTT), (5) identity-based SHA-256 deterministic seeding — composition-invariant, (6) parallel run_batch() with spawn-context process pool, (7) 7 documentation guides. Diagram: 4-layer config architecture stack.

## Slide 6 — ROHAN Architecture Overview
Stack: LangGraph orchestration, abides-hasufel simulation, LangChain+OpenRouter LLM integration, SQLAlchemy persistence (SQLite/PostgreSQL), Streamlit UI, AST strategy sandbox, ruff+pyright+pytest+hypothesis quality. Four modules: simulation/ (config_builder, HasufelOutput, StrategicAgentAdapter), llm/ (graph, nodes, state, scoring, tools, planner), framework/ (analysis_service, database, repository), ui/ (Terminal, Refinement Lab). Diagram: system context — ROHAN platform with 4 internal blocks connected to external actors (User, LLM Provider, ABIDES Engine, Database).

## Slide 7 — The Agentic Refinement Loop (KEY SLIDE)
LangGraph directed state-machine. Flow: User Goal → Planner (optional, proposes adversarial scenarios) → Writer (generates StrategicAgent Python code from goal+feedback) → Validator (AST safety + sandbox, ≤3 retries) → Scenario Executor (runs strategy in hasufel across multiple scenarios; produces HasufelOutput → RichAnalysisBundle → 6 charts per scenario) → Explainer (ReAct agent with 8 investigation tools → ScenarioExplanation) → Aggregator (deterministic 6-axis scoring + LLM qualitative analysis; convergence/rollback detection) → loop or DONE. Critical: scoring is fully deterministic — NO LLM assigns numbers. Diagram: flowchart with color-coded nodes (green=generation, blue=execution, purple=analysis, grey=aggregation), loop arrow from Aggregator back to Writer.

## Slide 8 — Data Contracts & State Management
Explicit typed boundaries: Simulator→Executor = HasufelOutput (live Python); Executor→Explainer = RichAnalysisBundle JSON (checkpoint-safe, container-independent, re-explainable); Explainer→Aggregator = ScenarioExplanation (Pydantic); Aggregator→DB = ORM models. SimulationOutput is NOT stored in LangGraph state — it depends on live ABIDES objects and would break checkpointing. The RichAnalysisBundle captures: fills, PnL curve, inventory trajectory, multi-window adverse selection, counterparty breakdown, order lifecycle, L2 snapshots. Diagram: horizontal pipeline with JSON serialization boundary highlighted.

## Slide 9 — Strategy Protocol & Sandbox
StrategicAgent Protocol: units in integer cents/shares/nanoseconds. MarketState (L1/L2 book, portfolio, liquidity, open orders), OrderAction (discriminated union: PLACE/CANCEL/MODIFY/REPLACE/etc.), 5 callbacks (initialize, on_tick, on_market_data, on_order_update, on_simulation_end). StrategicAgentAdapter wraps generated code inside ABIDES TradingAgent — translates discrete-event callbacks → protocol callbacks. AST Sandbox: FORBIDDEN_CALLS (eval, exec, compile, getattr, __import__), import whitelist, ThreadPoolExecutor timeout. Diagram: 3-layer sandwich (Strategy Protocol → Adapter → ABIDES Engine) + sidebar Sandbox checklist.

## Slide 10 — Metrics Framework
30+ metrics in 3 groups. Agent Performance: PnL (realized+unrealized), Sharpe ratio (annualized from fill-by-fill curve, ≥30 obs), max drawdown, fill rate, VWAP, effective spread, avg slippage (bps), multi-window adverse selection (100ms/500ms/1s/5s). Market Microstructure: annualized volatility from two-sided L1, mean spread, LOB imbalance (Cont et al. 2014), VPIN (Easley, López de Prado & O'Hara 2012, Lee-Ready tick test, 50 buckets), market resilience (Foucault et al. 2013, recovery after spread shocks), market OTT ratio (MiFID II RTS 9). Market Impact: Δ% = (strategy − baseline)/baseline × 100 for each microstructure metric. Diagram: 3-panel layout with key formulas.

## Slide 11 — Deterministic 6-Axis Scoring
Why: LLM-based scoring caused noise → false rollbacks. All quant evaluation is deterministic. 6 axes [1.0–10.0]: (1) Profitability = capture rate vs theoretical spread revenue, (2) Risk = Sharpe piecewise + drawdown penalty, (3) Volatility Impact = baseline Δ, (4) Spread Impact = baseline Δ, (5) Liquidity Impact = inverted Δ + availability penalty, (6) Execution Quality = fill rate primary + OTT penalty + slippage adjustment. Convergence: plateau after 2 stale iterations → stop; regression >1.0 → rollback. Diagram: radar chart with Iteration 1 vs Iteration 3 overlay showing convergence.

## Slide 12 — Tool-Equipped Explainer (ReAct Agent)
Built with create_react_agent (langgraph.prebuilt). Think→Act→Observe loop over serialized RichAnalysisBundle. 8 tools: query_fills (time/side filter), query_pnl_curve, query_inventory, query_adverse_selection (per window), query_book_at_time (L2 snapshot), query_counterparties, query_order_lifecycle, get_simulation_summary. Output: ScenarioExplanation with findings/root-causes/recommendations. Fallback: single structured-output call if ReAct fails. Example chain: summary→sees low fill rate→queries buy fills→high slippage→checks adverse selection 100ms→confirms toxic flow→inspects L2 at worst fill→thin ask side→concludes: "buying into one-sided books." Diagram: sequence diagram showing 4 think→tool→observe steps.

## Slide 13 — Adversarial Scenario Planning
Planner: pre-graph ReAct agent with 3-tier fallback: (1) ReAct with scenario tools (build_scenario, validate_scenario, explain_metrics), (2) structured LLM output, (3) keyword heuristic. Per-scenario deterministic seed: SHA-256(name + session_timestamp). Baseline runs (no strategy) cached. Example scenarios: Baseline (rmsc04), Volatile (3× fund_vol), Thin Market (100 Noise, no MM), Toxic Flow (high informed ratio). Diagram: fan-out from single strategy to N scenarios, each producing scores+charts.

## Slide 14 — Charts & Analysis Output
6 charts per scenario in 2×3 grid. Market row: Price Series (mid-price over time), Bid-Ask Spread (liquidity conditions), Volume at BBO (depth). Strategy row: PnL Curve (dense L1-sampled via compute_equity_curve), Inventory Trajectory (position over time), Fill Scatter (fills vs mid-price, colored by side). All base64 PNG, persisted in ORM, survive save/load round-trip. Diagram: 2×3 grid of labeled chart panels.

## Slide 15 — UI & User Workflow
Streamlit multipage app. Terminal page: configure scenarios from presets/custom, run single simulations, compare baseline vs strategy. Refinement Lab: natural language goal input, model/iteration/threshold config, real-time progress with iteration history table (Iter, PnL, Trades, Fill Rate, Slippage, Vol Δ, Spread Δ, Score, Summary), drill-down into any scenario's charts/explanation, session save/load. Diagram: two annotated UI screenshots or wireframes.

## Slide 16 — Quality Engineering
749+ unit tests (pytest), 89 parametrized scoring boundary tests, Hypothesis property-based tests, integration tests, failure-path tests. Linting: ruff + pyright strict. Pre-commit: ruff lint/format + pyright + nbstripout. CI: GitHub Actions (pytest + Codecov + CodeQL). Security: sandbox escape chain closed (v0.2.1), DB hardening, HTML injection mitigated. Diagram: CI pipeline flow with badge summary.

## Slide 17 — Target Architecture
Evolution: strategy-generation loop → strategy-validation platform. Linear DAG (no loop): Ingest (AST + adapter translation) → Analyze (LLM produces StrategyProfile, cross-validates intent) → Plan (3-tier mandatory registry: regulatory/strategy-type/organization + AI adversarial) → Execute (parallel simulation) → Explain (ReAct per scenario + cross-scenario synthesis) → Report (PDF/JSON/email, immutable, audit-ready). Cloud: FastAPI gateway → LangGraph orchestrator → hasufel worker pool → PostgreSQL + S3 + Redis. RBAC, 3 triggers (on-demand/scheduled/CI webhook). Regulatory compliance: SR 11-7, MiFID II Art. 17, EBA GL 2017/11. Diagram: linear DAG pipeline + cloud component diagram.

## Slide 18 — Cost/Benefit Analysis
TCO 3-year: €250K (lean) to €550K (enterprise). LLM costs <5% of TCO; infrastructure dominates. Target DAG: ~$1/run (12 scenarios) vs current cyclic ~$8–10/run. Savings at 30% human oversight: 50 strategies → €70K–€397K/yr savings, 5–18 month payback. Efficiency: 10–50× faster initial validation, 2–5× scenario coverage, >100× audit response time. PoC maturity: ~80% of core logic already built. Diagram: (1) bar chart — annual savings by portfolio size, (2) payback period line chart at 20/30/40% oversight rates.

## Slide 19 — Risks & Build vs Buy
Top risks: single-ticker limitation (High/High — plan multi-ticker hasufel), adapter fidelity (Medium/High — protocol-native path = zero loss), simulation fidelity (Medium/Medium — relative scoring mitigates), LLM reliability (Medium/Medium — 3-tier fallbacks + deterministic scoring). Build vs Buy vs Manual: ROHAN = €83K–€170K/yr (50 strats), 8-tool forensic depth, AI+mandatory scenario design, full IP control, minutes to validate. Vendor = €150K–€400K, aggregate metrics only, strategy shared with vendor. Key differentiator: no commercial product combines AI adversarial scenario design + full LOB simulation + forensic explainability. Diagram: comparison table or risk matrix.

## Slide 20 — Conclusions & Future Steps
Built: ABIDES-Hasufel v2.5.8 (modernized simulator with declarative config, typed results, rich metrics, parallel execution, 7 docs) + ROHAN v0.3.1 (agentic framework with 5 specialized agents, deterministic scoring, 8 explainer tools, adversarial planner, Streamlit UI, 749+ tests). Key contributions: demonstrated LLM agentic systems CAN autonomously refine trading strategies with convergent improvement; scoring is fully deterministic; ReAct Explainer exceeds typical human analyst depth; bridged legacy ABIDES to modern AI-friendly interfaces. Roadmap: Phase 1 MVP (linear DAG, 3–4 mo) → Phase 2 Production (FastAPI, cloud, 2–3 mo) → Phase 3 Enterprise (adapter layer, governance, 2–3 mo) → Phase 4 Evolution (multi-ticker, RL, ongoing). Diagram: horizontal timeline with phase labels.
