# Changelog

All notable changes to ROHAN are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] — 2026-03-09

### Added

- **Centralized refinement defaults** — `DEFAULT_MAX_ITERATIONS`, `DEFAULT_CONVERGENCE_THRESHOLD`, `_DEFAULT_RECURSION_LIMIT` constants in `graph.py`, imported by UI, CLI, and nodes.py. Eliminates scattered hardcoded values.
- **Structured per-scenario feedback routing (Step 13)** — `AggregatedFeedback` now carries `scenario_weaknesses` and `scenario_recommendations` populated from explainer `ScenarioExplanation` objects. `_render_per_scenario_feedback()` helper formats them for the writer prompt.

### Changed

- **Simulation duration** — Default session length extended from 30 minutes (09:30–10:00) to 2 hours (09:30–11:30) for meaningful market dynamics and PnL curves. All 5 presets updated.
- **Writer prompt example pattern** — Replaced `cancel_all()` + re-place with `OrderAction.modify()` to avoid inflating OTT 10–20×.
- **Prompt Rules 5-6** — Rule 5 now warns against OTT inflation from cancel_all. Rule 6 recommends `modify()`/`replace()` for stale order management; reserves `cancel_all()` for emergency flatten.
- **Example goals** — All 4 UI example prompts rewritten to use `modify()`/`replace()` instead of `cancel_all()` every tick.
- **Sidebar tips** — Updated to "3–5 iterations" (was 2-3) and "score ≥ 7/10" (was 8/10) to match actual convergence threshold.

### Fixed

- **Convergence threshold mismatch** — UI chart line and sidebar displayed threshold as 8 while code used 7. Now uses `DEFAULT_CONVERGENCE_THRESHOLD` (7.0) everywhere.
- **`max_iterations` default drift** — UI and CLI defaulted to 3 while docs said 5. Now all reference `DEFAULT_MAX_ITERATIONS` (5).
- **`recursion_limit` too low** — Was hardcoded at 50, causing premature graph termination. Increased to 80 via `_DEFAULT_RECURSION_LIMIT`.
- **`_save_current_run` fallback** — Used hardcoded 3 instead of centralized default.

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

[0.2.1]: https://github.com/GabrieleDiCorato/rohan/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/GabrieleDiCorato/rohan/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/GabrieleDiCorato/rohan/compare/v0.1.0...v0.1.2
[0.1.0]: https://github.com/GabrieleDiCorato/rohan/releases/tag/v0.1.0
