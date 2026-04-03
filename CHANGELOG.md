# Changelog

All notable changes to ROHAN are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] — 2026-04-03

### Changed

- **Upgraded abides-hasufel to v2.5.6** — Unlocks `run_simulation(oracle_instance=...)` for historical oracle mode; all simulation paths now use the high-level API.
- **Migrated `SimulationRunnerAbides` to `run_simulation()`** — Replaced the low-level `build_and_compile()` → `abides_run()` pipeline with a single `run_simulation(config, profile=ResultProfile.FULL, oracle_instance=..., runtime_agents=...)` call. Returns typed `SimulationResult`.
- **Added `strategy_spec` support to `run_batch()`** — Parallel batch path now accepts a `StrategySpec`, registers the `rohan_strategy` agent type in worker processes via `worker_initializer`, and discovers strategic agent IDs from results.
- **Unified output architecture** — All simulation paths (single-run, batch, historical) now return `HasufelOutput`. The dual `AbidesOutput` / `HasufelOutput` architecture is eliminated.

### Removed

- **`AbidesOutput`** (308 lines) — Replaced by `HasufelOutput` for all execution paths. Raw `end_state` dict parsing eliminated.
- **`test_abides_output.py`** (350 lines) and **`test_abides_integration.py`** (205 lines) — Tests for deleted `AbidesOutput` class.
- **`ABIDES_INTERFACE_REVIEW.md`** — All Rohan-side recommendations implemented; review completed and archived.

## [0.3.0] — 2026-03-31

### Changed

- **Migrated from legacy ABIDES to abides-hasufel v2.5.1** — Replaced the 504-line `AbidesConfigMapper` and manual agent instantiation with a thin `config_builder.py` (~145 lines) that maps `SimulationSettings` onto hasufel's declarative `SimulationBuilder` API. Agent instantiation, seed derivation, and latency model construction are now handled by hasufel's `compile()` step.
- **`SimulationRunnerAbides` rewritten** — Uses `build_simulation_config()` → `compile()` → `config_add_agents()` → `abides_run()` pipeline. Strategic agent injected post-compilation via `config_add_agents()`.
- **Dependency source** — `abides-hasufel` now sourced from GitHub (`GabrieleDiCorato/abides-hasufel@v2.5.1`) instead of local editable install.
- **Value agent sigma_s auto-inheritance** — For synthetic oracle, value agents auto-inherit `sigma_s = fund_vol²` from hasufel's oracle context. For historical oracle (ExternalDataOracle), `sigma_s` is computed explicitly using the same `fund_vol²` convention.
- **Noise agent wakeup window** — `noise_mkt_close_time` set to `settings.end_time` instead of default `"16:00:00"` to ensure agents wake during short simulations.
- **Log level mapping** — `"OFF"` mapped to `"CRITICAL"` (hasufel only accepts `DEBUG`/`INFO`/`WARNING`/`ERROR`/`CRITICAL`).
- **Test assertion relaxed** — `market_ott_ratio` assertion changed from `>= 1.0` to `> 0.0` since partial fills can produce ratio < 1.
- **Historical integration test** — Updated assertions to use `AbidesOutput` API (`get_order_book_l1()`, `get_logs_df()`) instead of removed `exchange_messages`/`agent_logs` attributes.

### Removed

- **`AbidesConfigMapper`** — Replaced by `config_builder.build_simulation_config()`.
- **`RandomStateHandler`** — Replaced by direct `np.random.RandomState` usage and hasufel's identity-based SHA-256 seed derivation.
- **`test_random_state_handler.py`** — Tests for deleted module.

## [0.2.1] — 2026-03-09

### Added

- **56 new unit tests** — `AbidesConfigMapper` (12), `RandomStateHandler` (17), `StrategicAgentAdapter` callbacks (16), property-based tests for `MarketState` computed fields and `OrderAction` factory methods (14). Total suite: 749 passing tests.
- **Pandera schema constraints** — `ge=0` on price, quantity, and time fields in L1/L2 schemas.
- **Shared DB init helper** — `ensure_db_initialized()` in `rohan.ui.utils.startup` replaces duplicated init logic across UI pages.
- **Centralized refinement defaults** — `DEFAULT_MAX_ITERATIONS`, `DEFAULT_CONVERGENCE_THRESHOLD`, `_DEFAULT_RECURSION_LIMIT` constants in `LLMSettings`, imported by UI, CLI, and nodes.py. Eliminates scattered hardcoded values.
- **Structured per-scenario feedback routing (Step 13)** — `AggregatedFeedback` now carries `scenario_weaknesses` and `scenario_recommendations` populated from explainer `ScenarioExplanation` objects. `_render_per_scenario_feedback()` helper formats them for the writer prompt.
- **Composite DB index** on `strategy_iterations(session_id, generation_number)`.

### Changed

- **Simulation duration** — Default session length extended from 30 minutes (09:30–10:00) to 2 hours (09:30–11:30) for meaningful market dynamics and PnL curves. All 5 presets updated.
- **Writer prompt example pattern** — Replaced `cancel_all()` + re-place with `OrderAction.modify()` to avoid inflating OTT 10–20×.
- **Prompt Rules 5-6** — Rule 5 now warns against OTT inflation from cancel_all. Rule 6 recommends `modify()`/`replace()` for stale order management; reserves `cancel_all()` for emergency flatten.
- **Example goals** — All 4 UI example prompts rewritten to use `modify()`/`replace()` instead of `cancel_all()` every tick.
- **Sidebar tips** — Updated to "3–5 iterations" (was 2-3) and "score ≥ 7/10" (was 8/10) to match actual convergence threshold.

### Fixed

#### Security (WS1)
- **Sandbox escape chain closed** — Removed `__import__` from `SAFE_BUILTINS`; injected `_make_safe_import` whitelist wrapper. Removed `rohan.config` from `SAFE_IMPORTS` (prevented API key exfiltration).
- **ThreadPoolExecutor timeout** — Replaced `with` context manager with manual `shutdown(wait=False, cancel_futures=True)` to prevent blocking on stuck threads.
- **AST hardening** — Added `FORBIDDEN_CALLS` set (`eval`, `exec`, `compile`, `getattr`, etc.) and `DANGEROUS_DUNDERS` allowlist (replaces overly broad `__` block that rejected `super().__init__()`).

#### Database (WS2)
- **Mutable ORM defaults** — Replaced `default={}` / `default=[]` with `default=dict` / `default=list` callables.
- **Detached ORM objects & N+1 queries** — Added `selectinload` eager loading in `list_sessions()` and `load_session()`.
- **Price forward-fill removed** — `save_market_data` no longer applies `ffill().bfill()` on prices, preserving NaN illiquidity signals per `AnalysisService` contract.
- **`SecretSettings` removed** — Eliminated redundant/broken secrets class; `LLMSettings` is the single source of truth for API keys.
- **ORM type annotations fixed** — `session_id` → `Mapped[uuid.UUID | None]`, `scenario_configs` → `Mapped[list[...]]`, `progress_log` → `Mapped[list[str]]`.
- **Connection string secured** — `DatabaseSettings.connection_string` changed to `SecretStr`; log output masked.

#### Reliability (WS3)
- **Matplotlib figure leak** — All 6 chart blocks now close figures in `finally` clauses.
- **LLM parse error logging** — `get_structured_model` logs `parsing_error` at WARNING instead of silently discarding.
- **`_pct_change` capped** — Returns `None` instead of `float("inf")` when baseline is zero.
- **Agent ID fix** — `run_with_baseline` uses `res.result.strategic_agent_id` instead of hardcoded `1`.
- **Missing `explanations` key** — Error path in `process_scenario_node` now includes `explanations` via `_error_explanation()`.
- **`LatencyType` enum values** — Fixed accidental tuple values (trailing commas) → plain strings.
- **UI delta display** — `prev` now reads from `st.session_state.get("previous_metrics")` instead of always being `None`.
- **DB init logging** — Replaced `contextlib.suppress(Exception)` with `try/except` + `logger.warning`.
- **Chart log level** — Generation failures logged at WARNING (was DEBUG).
- **UTC timestamps** — `IterationSummary.timestamp` uses `datetime.now(timezone.utc)`.
- **Timeout centralized** — `SimulationSettings.timeout_seconds` field replaces three hardcoded `300` values.

#### Architecture (WS4)
- **Circular import resolved** — Refinement loop constants moved to `LLMSettings`; `graph.py` re-exports for backward compatibility. Deleted `constants.py`.
- **Environment variable mutation** — `LANGCHAIN_TRACING_V2` assignment moved from module level into `build_refinement_graph()`.
- **`lru_cache` clarified** — `maxsize=8` → `maxsize=1` (zero-arg function produces one entry).
- **Ticker parameterized** — `AbidesOutput` accepts `ticker` constructor parameter (was hardcoded `"ABM"`).
- **`save_logs` vectorized** — Replaced slow `iterrows()` with bulk `to_dict(orient="records")`.
- **HTML injection mitigated** — Dynamic values in `unsafe_allow_html` templates wrapped with `html.escape()`.
- **Dead code removed** — `_save_rich_analysis_artifacts` (50 lines) deleted from `SimulationEngine`.
- **`pytestmark` fix** — Triple assignment in `test_property_based.py` → single module-level declaration.

### Removed

- `src/rohan/llm/constants.py` — contents migrated to `LLMSettings`.
- `SecretSettings` class and its export from `rohan.config`.
- `_save_rich_analysis_artifacts` dead method from `SimulationEngine`.

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
