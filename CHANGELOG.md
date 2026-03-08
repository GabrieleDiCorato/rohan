# Changelog

All notable changes to ROHAN are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] — 2026-03-08

### Added

- **Deterministic 6-axis scoring system** — Replaced LLM-based scoring with formulaic computation across Profitability, Risk, Volatility Impact, Spread Impact, Liquidity Impact, and Execution Quality. Auto-normalizes to scenario config (duration, capital, spread, competition).
- **Rich analysis pipeline** — `compute_rich_analysis()` produces a `RichAnalysisBundle` (fills, PnL curve, inventory trajectory, adverse-selection windows, counterparty breakdown, order lifecycle, L2 snapshots). Serialized via `.model_dump_json()` and stored on `ScenarioResult.rich_analysis_json`.
- **Tool-equipped Explainer Agent** — ReAct agent with 8 investigation tools (`query_fills`, `query_pnl_curve`, `query_inventory`, `query_adverse_selection`, `query_book_at_time`, `query_counterparties`, `query_order_lifecycle`, `get_simulation_summary`). Falls back to single structured-output call on failure.
- **3 strategy-performance charts** — PnL Curve, Inventory Trajectory, Fill Scatter generated per scenario in the executor node.
- **Full chart persistence pipeline** — All 6 charts (Price, Spread, Volume, PnL, Inventory, Fills) + `rich_analysis_json` threaded through `ScenarioMetrics` → `ScenarioResultData` DTO → `RefinementScenarioResult` ORM → DB. Survive save/load round-trip.
- **2×3 chart grid in UI** — Market row (Price Series, Bid-Ask Spread, Volume at BBO) + Strategy Performance row (PnL Curve, Inventory, Fills vs Mid) in the Refinement Lab.
- **Deterministic per-scenario seeds** — SHA-256 hash of scenario name + session timestamp, assigned once per `run_refinement()` call.
- **89 parametrized piecewise-boundary scoring tests** — Full coverage of all 6 scoring axes with edge cases.
- **Simulation context on ScenarioResult** — `sim_duration_ns`, `sim_start_capital`, `sim_num_background_agents`, `sim_spread_oracle_widths` for score normalization.
- **`regime_context` field** — Forward-compatible slot on `ScenarioResult` for adversarial scenario descriptions.
- **Formatting utilities** — `_fmt_dollar`, `_fmt_pct`, `_fmt_float` for consistent `None→"N/A"` handling in LLM prompts.
- **Features section in README**.
- **CHANGELOG.md**.

### Changed

- **Aggregator prompt** — Simplified to focus on qualitative analysis; scoring moved to deterministic computation.
- **Aggregator models** — `IterationSummary` and `ScenarioMetrics` carry 6 sub-scores + 6 chart fields.
- **Writer feedback** — Receives per-scenario structured weaknesses/recommendations from the explainer instead of flat text.
- **Default `max_iterations`** — Changed from 10 to 5.
- **Market metrics computation** — Refactored into shared utility functions used by both `IterationPipeline` and `scenario_executor_node`.

### Fixed

- **Scoring noise** — Eliminated false rollbacks caused by LLM scoring variance across iterations.
- **Seed consistency** — Fixed non-deterministic simulation results that made cross-iteration comparison unreliable.
- **Dead explainer tools** — Removed 8 unused tools from `make_explainer_tools` (never wired into the graph).

## [0.1.2] — 2026-02-25

### Added

- Streamlit multipage UI (Terminal + Refinement Lab).
- Session save/load with SQLAlchemy ORM (SQLite / PostgreSQL).
- CI pipeline (ruff, pyright, pytest + Codecov, CodeQL).
- Pre-commit hooks (ruff lint/format, pyright, nbstripout).

### Changed

- Upgraded Strategy API and ABIDES adapter.

### Fixed

- UI state management in Terminal page.
- Writer LLM retry loop and `include_raw` structured output guard.
- Feedback loop quality: code-to-explainer routing, richer metrics, deduplication.

## [0.1.0] — 2026-01-31

### Added

- Initial ABIDES market simulation wrapper.
- `StrategicAgent` protocol and AST-based strategy validator.
- `SimulationService` with configurable scenarios.
- Database schema (sessions, iterations, scenarios, runs, market data, artifacts).
- LangGraph refinement loop (Writer → Executor → Explainer → Judge → Writer).
- Streamlit Terminal UI for running simulations.
- Reproducibility tests.

[0.2.0]: https://github.com/GabrieleDiCorato/rohan/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/GabrieleDiCorato/rohan/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/GabrieleDiCorato/rohan/releases/tag/v0.1.0
