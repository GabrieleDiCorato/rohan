20-slide academic presentation. Audience: professors of statistics, software architecture, and financial engineering (MSc AI & Data Analytics for Business). White background, bordered diagrams, standalone reusable images. Two projects: ABIDES-Hasufel v2.5.8 (modernized LOB simulator) + ROHAN v0.3.1 (agentic trading framework).

1 TITLE: "Evolutionary Market Microstructure: An Agentic Framework for Automated Strategy Design and Stress-Testing in Simulated Financial Environments." Author Gabriele Di Corato.

2 AGENDA: Motivation → ABM background → ABIDES-Hasufel → ROHAN → CBA → Conclusions.

3 MOTIVATION: Static backtesting misses market impact. Manual validation EUR 240K-480K/yr (50 strategies); SR 11-7+MiFID II Art.17 mandate stress tests. LLMs write code but lack closed test-observe-fix loop — ROHAN closes it. Diagram: manual-days vs agentic-minutes.

4 LOB & ABM: LOB: bids/asks queues, spread, mid-price, imbalance (Cont 2014). ABM: emergent dynamics (flash crashes). ABIDES (Byrd & Balch 2019): Kernel, Noise/Value/Momentum/MarketMaker/Exchange agents. Diagram: depth chart + agent ecosystem.

5 ABIDES-HASUFEL v2.5.8: Modernized broken J.P.Morgan archive. 4-layer config: Pydantic models → @register_agent registry → SimulationBuilder → Compiler; composable templates rmsc04/liquid/thin. Typed SimulationResult+ResultProfile. Rich Metrics: Sharpe, drawdown, VWAP, slippage, adverse selection, VPIN, OTT. SHA-256 seeding. Parallel run_batch(). 7 docs. Diagram: 4-layer stack.

6 ROHAN STACK: LangGraph + hasufel + LangChain/OpenRouter + SQLAlchemy + Streamlit + AST sandbox. Modules: simulation/, llm/, framework/, ui/. Diagram: system context with external actors.

7 AGENTIC LOOP: Goal → Planner → Writer (StrategicAgent code) → Validator (AST+sandbox, 3 retries) → Executor (hasufel → HasufelOutput → RichAnalysisBundle → 6 charts) → Explainer (ReAct+8 tools → ScenarioExplanation) → Aggregator (6-axis scores + LLM qualitative; convergence/rollback) → loop or DONE. Diagram: LangGraph flowchart with loop arrow.

8 DATA CONTRACTS: HasufelOutput (live) → RichAnalysisBundle JSON (checkpoint-safe: fills, PnL, inventory, adverse selection, counterparties, order lifecycle, L2 snapshots) → ScenarioExplanation (Pydantic) → ORM. Diagram: serialization boundary.

9 STRATEGY PROTOCOL & SANDBOX: StrategicAgent (integer prices/shares, 5 callbacks, OrderAction union), Adapter bridges to ABIDES TradingAgent. AST sandbox: forbids eval/exec/__import__, import whitelist, timeout. Diagram: 3-layer sandwich.

10 METRICS (30+): Agent: PnL, Sharpe, drawdown, fill rate, VWAP, slippage bps, adverse selection 100ms–5s. Market: volatility, spread, LOB imbalance (Cont 2014), VPIN (Easley 2012), resilience (Foucault 2013), OTT (MiFID II). Impact: delta% vs baseline. Diagram: 3-panel metric cards.

11 SCORING: LLM scoring → noise/false rollbacks → replaced with formulas [1–10]. Axes: Profitability, Risk (Sharpe+drawdown), Volatility/Spread/Liquidity/Execution Impact. Convergence: plateau 2 iters → stop; regression >1.0 → rollback. Diagram: radar chart.

12 REACT EXPLAINER: create_react_agent over serialized bundle. 8 tools: fills, PnL curve, inventory, adverse selection, L2 book, counterparties, order lifecycle, sim summary. Example: low fills → 100ms toxic flow → “buying into one-sided books.” Diagram: think→tool→observe.

13 ADVERSARIAL PLANNER: Pre-graph ReAct + 3-tier fallback (tools → LLM → keyword). SHA-256(name+timestamp) seeding. Scenarios: Baseline (rmsc04), Volatile (3× vol), Thin Market, Toxic Flow. Diagram: strategy fan-out.

14 CHARTS: 6 per scenario (2×3). Market row: Price Series, Spread, Volume@BBO. Strategy row: PnL Curve (L1-sampled), Inventory, Fill Scatter (colored by side). Base64 PNG, ORM-persisted. Diagram: labeled grid.

15 UI: Streamlit multipage. Terminal: presets, baseline comparison. Refinement Lab: goal input, iteration history (PnL/Trades/Fill/Slippage/Deltas/Score), drill-down, session save/load. Diagram: annotated screenshots.

16 QUALITY: 749+ tests (89 scoring boundaries, Hypothesis). ruff+pyright strict, pre-commit. CI: GitHub Actions+Codecov+CodeQL. Security: hardened sandbox+DB. Diagram: CI pipeline.

17 TARGET PLATFORM: Linear DAG — Ingest → Analyze (StrategyProfile) → Plan (3-tier: regulatory/type/org + AI adversarial) → Execute → Explain → immutable Report (PDF/JSON). Cloud: FastAPI → LangGraph → hasufel → PostgreSQL+S3+Redis. RBAC, 3 triggers. SR 11-7, MiFID II Art.17, EBA GL 2017/11. Diagram: DAG + cloud stack.

18 CBA: 3Y TCO EUR 250K–550K; LLM <5%. DAG ~$1/run vs cyclic ~$8–10. At 30% oversight, 50 strats: EUR 70K–397K/yr savings, 5–18mo payback. 10–50× faster, >100× audit speed. PoC ≈80% of target. Diagram: savings bars + payback curves.

19 BUILD VS BUY: Risks: single-ticker (High/High), adapter (Med/High), sim fidelity (Med/Med), LLM (Med/Med). ROHAN EUR 83K–170K/yr vs vendor EUR 150K–400K vs manual EUR 240K–480K (50 strats). Unique: AI adversarial + LOB sim + forensic explainability + IP ownership. Diagram: comparison table.

20 CONCLUSIONS: abides-hasufel v2.5.8 (4-layer config, typed results, 30+ metrics, parallel) + ROHAN v0.3.1 (5 agents, deterministic 6-axis scoring, ReAct×8 tools, planner, 749+ tests). Convergent refinement, noise-free scoring, forensic explainability. Roadmap: MVP 3–4mo → cloud 2–3mo → enterprise 2–3mo → multi-ticker+RL. Diagram: timeline.
