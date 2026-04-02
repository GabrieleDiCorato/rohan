# abides-hasufel v2.5 Interface Review

> **Reviewer:** AI Framework Architect (Financial Engineering / Risk Management)
> **Scope:** Usability of abides-hasufel v2.5.x within Rohan's agentic strategy refinement loop (Rev 2 — updated to v2.5.5, expanded scope)
> **Date:** 2026-04-02

---

## Executive Summary

The migration from the legacy ABIDES fork (`abides-rohan v1.2.7`) to `abides-hasufel v2.5` is **complete**. The old `AbidesConfigMapper` (504 lines) has been replaced by a 293-line `config_builder.py` backed by hasufel's declarative `SimulationBuilder` API. All original hasufel-side requests are resolved (v2.5.3–v2.5.5). Phase 2 Rohan implementation is complete: `StrategySpec` model, `@register_agent("rohan_strategy")`, parallel `run_batch()`, 11 LangChain tools (3 scenario + 8 investigation), and 6-axis deterministic scoring.

**Remaining work** falls into six areas: (1) migrating the runner from `build_and_compile()` + `abides_run()` to `run_simulation()` (blocked by oracle-instance dependency), (2) unifying the dual-output architecture (`AbidesOutput` vs `HasufelOutput`), (3) historical oracle data-provider assessment, (4) LLM tooling gap analysis, (5) strategy compilation security hardening, and (6) operational integration concerns.

This document contains **11 findings** with prioritised recommendations in [§13](#13-prioritised-recommendations).

---

## Table of Contents

1. [What Works Well](#1-what-works-well)
2. [Remaining Adoption Opportunities](#2-remaining-adoption-opportunities)
3. [Remaining Hasufel Requests](#3-remaining-hasufel-requests)
4. [Dual-Output Architecture](#4-dual-output-architecture)
5. [Historical Oracle & Data Providers](#5-historical-oracle--data-providers)
6. [Template System Assessment](#6-template-system-assessment)
7. [Metrics & Output Pipeline Assessment](#7-metrics--output-pipeline-assessment)
8. [LLM Tooling & Refinement Loop](#8-llm-tooling--refinement-loop)
9. [Strategy Compilation Security](#9-strategy-compilation-security)
10. [Operational Integration Concerns](#10-operational-integration-concerns)
11. [Testing Strategy & Coverage](#11-testing-strategy--coverage)
12. [Risk & Migration Notes](#12-risk--migration-notes)
13. [Prioritised Recommendations](#13-prioritised-recommendations)
- [Appendix A: File-Level Impact Map](#appendix-a-file-level-impact-map)
- [Appendix B: Documentation Quality Score](#appendix-b-documentation-quality-score)

---

## 1. What Works Well

### 1.1 SimulationBuilder Fluent API

The builder pattern is clean and intuitive. `create_simulation_builder()` reads top-to-bottom: market → exchange → oracle → agents → infrastructure → metadata. Each section maps 1:1 to a `SimulationSettings` sub-model with minimal transformation. This is a dramatic improvement over the old mapper's procedural instantiation with manual ID assignment and numpy `RandomState` plumbing.

**Code evidence:** `config_builder.py` lines 57–111 — the full builder setup is 55 lines of declarative calls with no conditional agent instantiation logic.

### 1.2 Identity-Based Seed Derivation

The SHA-256 identity-based RNG hierarchy (`sha256(f"{seed}:{component}:{index}")`) solves a real composition-invariance problem. Adding the `StrategicAgentAdapter` post-compilation no longer shifts every other agent's random state. This was a persistent source of non-reproducibility in the old system that required the now-deleted `RandomStateHandler`.

### 1.3 Build-Time Parameter Validation

`SimulationBuilder.build()` validates agent parameters eagerly via Pydantic. Invalid configurations fail immediately with structured error messages rather than producing cryptic runtime crashes deep inside the kernel. This is critical for the agentic loop where an LLM proposes configurations — fast validation feedback enables faster iteration.

### 1.4 `config_add_agents()` Post-Compilation Injection

The ability to inject agents into a compiled runtime dict cleanly separates "market environment setup" (hasufel's domain) from "strategy injection" (Rohan's domain). `SimulationRunnerAbides.run()` is only 30 lines of glue as a result.

### 1.5 Config System Discoverability API

`list_agent_types()`, `list_templates()`, `get_config_schema()`, and `get_full_manifest()` are purpose-built for AI consumers. An LLM agent can query available agent types, their parameters, valid ranges, and recommended compositions without parsing source code. This is a rare and commendable design decision.

### 1.6 Comprehensive Documentation Suite

The six documentation files (`ABIDES_REFERENCE.md`, `ABIDES_CONFIG_SYSTEM.md`, `ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md`, `ABIDES_LLM_INTEGRATION_GOTCHAS.md`, `ABIDES_DATA_EXTRACTION.md`, `PARALLEL_SIMULATION_GUIDE.md`) form a thorough and internally consistent reference. The gotchas document is particularly valuable — it catalogs every `None`/`KeyError`/`NaN` trap with safe-access patterns, directly targeting the failure modes of LLM-generated code.

---

## 2. Remaining Adoption Opportunities

Features that hasufel already provides but Rohan does not yet use.

### 2.1 Migrate to `run_simulation()`

**Current state:** `SimulationRunnerAbides.run()` calls `builder.build_and_compile()` → `abides_run()`, manually managing the runtime dict, log directory, and kernel random state. This path is required because the historical oracle mode uses `builder.oracle_instance()`, which `run_simulation()` does not accept.

**Hasufel offers:** `run_simulation(config)` handles compilation, log directory, kernel RNG, and returns a typed `SimulationResult` with structured per-agent PnL, market summaries, and optional L1/L2 series. `runtime_agents` parameter (v2.5.4) supports post-compile agent injection. `StrategicAgentConfig` is registered and the StrategySpec path goes through hasufel's compile pipeline.

**Remaining blocker:** Oracle-instance dependency. Migrate once Rohan switches to template-based oracle configuration or hasufel adds oracle-instance support to `run_simulation()`.

**Impact:** Typed `SimulationResult` extraction, elimination of `end_state` dict coupling (see [§12.2](#122-end_state-dict-stability)).

### 2.2 Use `ResultProfile.QUANT` for Metrics Extraction

**Current state:** `AbidesOutput` parses `end_state` → `ExchangeAgent` → `OrderBook` → `get_L1_snapshots()` → manual DataFrame construction → Pandera validation.

**Hasufel offers:** `ResultProfile.QUANT` includes per-symbol L1/L2 time-series as numpy arrays, plus a `MarketSummary` with `l1_close`, `vwap_cents`, `liquidity_pct`, `volume`, and `spread_close`. The standalone `compute_metrics()` function (`abides_markets.simulation.metrics`) also provides canonical metric computation.

**Recommendation:** Audit the overlap between `MarketSummary` fields and Rohan's `SimulationMetrics` to determine which metrics can be read directly.

### 2.3 Use `summary_dict()` for LLM-Friendly Output

**Current state:** `RunSummary` is constructed manually from `ComparisonResult`, which is assembled from separate `AgentMetrics`, `MarketMetrics`, and `MarketImpact` computations.

**Hasufel offers:** `SimulationResult.summary()` produces a human-readable narrative; `summary_dict()` produces a structured dict with `metadata`, `markets`, `agent_leaderboard`, and `warnings` — designed for dashboards and JSON APIs.

**Why adopt:** The explainer agent and writer agent in the refinement loop currently receive `RunSummary` serialized as JSON. Hasufel's `summary_dict()` could serve as the basis, enriched with Rohan-specific fields (market impact, scoring).

---

## 3. Remaining Hasufel Requests

Only one hasufel-side request remains open.

### 3.1 Risk Guards Documentation in Template Metadata

**Problem:** Templates do not advertise what risk guard defaults they set. A user choosing `rmsc04` vs. `thin_market` cannot know whether position limits or max drawdown guards are pre-configured.

**Request:** Include `default_risk_guards` in template metadata (or at least document it in the template description string). This is important for the agentic loop, where the LLM writer needs to know whether the simulation environment already has guardrails.

---

## 4. Dual-Output Architecture

### 4.1 Current State

- `AbidesOutput` wraps raw `end_state` dict (single runs via `build_and_compile()` + `abides_run()`)
- `HasufelOutput` wraps hasufel's typed `SimulationResult` (batch runs via `run_batch()`)
- Both implement the `SimulationOutput` ABC; downstream consumers are output-path-agnostic

### 4.2 Feature Asymmetry

- `HasufelOutput.get_logs_by_agent()` → `NotImplementedError`
- `HasufelOutput` has no `end_state` — callers degrade via `hasattr()` guards
- L1/L2 conversion paths differ: `AbidesOutput` reads `OrderBook.get_L1_snapshots()` arrays; `HasufelOutput` reads hasufel numpy arrays. Both produce Pandera-validated DataFrames, but conversion logic is independent

### 4.3 Risks

- **Metric drift:** two extraction paths could produce numerically different results for identical simulations
- **API surface:** `get_logs_by_agent()` callers silently fail on batch path
- **Maintenance:** two adapters to maintain for every schema change

### 4.4 Checklist

- [ ] Add integration test comparing `AbidesOutput` and `HasufelOutput` metrics for same simulation seed
- [ ] Audit all `get_logs_by_agent()` call sites — confirm none are on batch path
- [ ] Audit all `end_state` accesses — confirm `hasattr()` guards exist everywhere
- [ ] Plan migration to retire `AbidesOutput` once `run_simulation()` path is adopted (depends on §2.1)
- [ ] Consider making `HasufelOutput` the single output adapter once historical oracle supports `run_simulation()`

**References:** `abides_output.py` (308 lines), `hasufel_output.py` (149 lines), `simulation_service.py` (280 lines)

---

## 5. Historical Oracle & Data Providers

### 5.1 Current State

- `src/rohan/simulation/data/` contains 7 modules: `csv_provider.py`, `database_provider.py`, `api_provider.py`, `provider_protocol.py`, `normalization.py`, `generator.py`, `models.py`
- `OracleSettings` supports `OracleType.HISTORICAL` with sub-configs per provider (CSV path + price unit + timezone; database dataset_id; API provider + symbol + key)
- `config_builder.py` has dedicated `_build_historical_oracle()` path with `_resolve_historical_provider()`

### 5.2 Key Constraints

- Historical oracle mode **disables parallelization**: `SimulationService.run_batch()` falls back to sequential when any setting uses historical oracle
- `sigma_s` gotcha: for historical oracle, `fund_vol**2` must be passed explicitly (synthetic oracle auto-inherits from oracle context)
- Oracle instance needs `builder.oracle_instance()`, which blocks `run_simulation()` adoption (see §2.1)

### 5.3 Checklist

- [ ] Test coverage for each provider type (CSV, Database, API) — confirm existence and adequacy
- [ ] Data validation at ingestion boundary: price unit normalization (DOLLARS → CENTS), timezone handling, gap/NaN handling
- [ ] Caching/staleness strategy: are API provider results cached? Are database queries re-run per simulation?
- [ ] Consider enabling parallelization for historical oracle (pre-fetch data, pass as config field instead of oracle instance)
- [ ] Document provider selection criteria and error modes for each provider type

**References:** `config_builder.py` (`_build_historical_oracle`, `_resolve_historical_provider`), `src/rohan/simulation/data/` (7 modules, 662 lines total)

---

## 6. Template System Assessment

### 6.1 Template Coverage

As of v2.5.5, hasufel provides eight base templates covering a broad range of market regimes:

| Template | Regime | Key Characteristics | Use Case |
|---|---|---|---|
| `rmsc04` | Reference | 1000 noise, 102 value, 12 momentum, 2 MM | Default benchmark, balanced market |
| `liquid_market` | High liquidity | 5000 noise, 200 value, 25 momentum, 4 MM | Stress-test market impact in deep markets |
| `thin_market` | Low liquidity | 100 noise, 20 value, no momentum/MM | Illiquid conditions, large spreads |
| `stable_day` | Low volatility | Low fund_vol, no megashocks | Control scenario |
| `volatile_day` | High volatility | High fund_vol, megashocks every ~6h | Strategy robustness testing |
| `low_liquidity` | Illiquid | No MM, 25 slow noise, 10 slow value | Extreme illiquidity |
| `trending_day` | Trending | Weak mean-reversion (365d half-life), momentum | Directional markets |
| `stress_test` | Extreme stress | fund_vol=3e-4, 3+ megashocks/session | Tail-risk scenarios |

All templates include `scenario_description` and `regime_tags` for programmatic selection.

**Remaining gap:** A **short-duration `quick_test` template** (10-minute window, reduced agent counts) would accelerate the inner loop of strategy refinement.

**Overlay templates** (`with_momentum`, `with_execution`) remain a strong composition primitive. They enable adding agent groups without replacing the base market environment.

### 6.2 Template Composability for Agentic Use

The `SimulationBuilder.from_template()` composability is well-designed. Stacking `from_template("rmsc04").from_template("with_execution")` is intuitive. However, the interaction between overlays and existing agent groups needs documentation:

- Does `from_template("with_momentum")` on top of `rmsc04` (which already has momentum agents) add 12 more, or replace the existing 12?
- What happens when overlays conflict (e.g., two overlays both set `noise.count`)?

**Recommendation:** Document overlay semantics clearly. Ideally, last-wins for same-name agent groups, additive for new groups.

### 6.3 Template Recommendations for Agentic Scenario Generation

Templates are a strong foundation for agentic scenario composition. With structured metadata (`scenario_description`, `regime_tags`), an LLM agent selecting market conditions can:

1. Query `list_templates()` for available regimes with tags
2. Filter by `regime_tags` to match desired market conditions programmatically
3. Stack overlays for additional agent groups
4. Override specific parameters via `enable_agent(..., param=value)`

---

## 7. Metrics & Output Pipeline Assessment

### 7.1 Metrics Inventory

Rohan computes a comprehensive set of 22 metrics across three categories:

| Category | Metrics | Source |
|---|---|---|
| **Agent Performance** (7) | total_pnl, sharpe_ratio, max_drawdown, inventory_std, fill_rate, trade_count, traded_volume | Agent logs, L1 |
| **Market Microstructure** (9) | volatility, mean_spread, effective_spread, avg_bid_liquidity, avg_ask_liquidity, lob_imbalance_mean, lob_imbalance_std, vpin, resilience_mean_ns, market_ott_ratio, pct_time_two_sided | L1, agent logs |
| **Market Impact** (9) | Percentage deltas of all microstructure metrics between strategy and baseline runs | Computed from above |

This is a strong metric set for an agentic strategy refinement system. The inclusion of VPIN (adverse selection), resilience (market recovery), and LOB imbalance (directional pressure) goes well beyond the basic PnL/Sharpe metrics typical of retail backtesting.

### 7.2 Metric Computation Architecture

**Current flow:**
```
AbidesOutput (raw end_state)
  → .get_order_book_l1() → L1 DataFrame
  → .get_logs_df() → Agent logs DataFrame
  → AnalysisService.compute_*() → SimulationMetrics / AgentMetrics
```

**Hasufel alternative (available since v2.5.3):**
```
run_simulation(config, profile=ResultProfile.QUANT)
  → SimulationResult.markets[symbol] → MarketSummary (L1 close, VWAP, volume)
  → SimulationResult.agents[i] → AgentData (PnL, holdings, agent_category)
  → SimulationResult.markets[symbol].l1_series → numpy array

# Or standalone:
from abides_markets.simulation.metrics import compute_metrics
metrics = compute_metrics(book_log2=snapshots, exec_trades=fills,
                          agent_holdings=holdings, symbol="AAPL")
```

**Assessment:** The hasufel path now provides a superset of basic metrics plus a standalone `compute_metrics()` function. Rohan's advanced microstructure metrics (VPIN, resilience, LOB imbalance) should be audited against hasufel's canonical implementation to determine overlap. Metrics not covered by hasufel should remain in Rohan's `AnalysisService`.

### 7.3 `AbidesOutput` Extraction Quality

The extraction code in `AbidesOutput` is well-written and defensively coded:
- Handles one-sided books (NaN fill instead of crash)
- Validates parallel-array invariant for L1 snapshots
- Uses `cached_property` correctly (instance-level cache, garbage-collected with instance)
- Warns on monotonicity violations (cross-midnight detection)

The main concern is **coupling to ABIDES internals**: direct access to `end_state["agents"][0]` (exchange agent by positional convention) and `order_book.get_L1_snapshots()` return format. If hasufel changes these internals, Rohan breaks. Using `SimulationResult` would insulate Rohan from these changes.

---

## 8. LLM Tooling & Refinement Loop

### 8.1 Implemented Tools (11 Total)

**Scenario tools** (`scenario_tools.py` — `make_scenario_tools()`):
- `build_scenario`: template selection via `regime_tags` matching, supports overrides
- `validate_scenario`: wraps hasufel `validate_config()`, returns structured errors/warnings
- `explain_metrics`: rule-based diagnostic patterns from `metrics_definition.md` §4

**Investigation tools** (`tools.py` — `make_investigation_tools(rich_json)`):
- `query_fills`, `query_pnl_curve`, `query_inventory`, `query_adverse_selection`
- `query_book_at_time`, `query_counterparties`, `query_order_lifecycle`, `get_simulation_summary`
- All closure-bound to a `RichAnalysisBundle` JSON string, enabling container-independent invocation

### 8.2 Refinement Loop Architecture

- **RichAnalysisBundle:** fully JSON-serializable analysis output (fill records, PnL curve, inventory trajectory, adverse selection, counterparty breakdown, order lifecycle, mid-price series, L2 snapshots)
- **3-tier explainer fallback:** ReAct agent with investigation tools → structured LLM call → deterministic template
- **6-axis deterministic scoring** (`scoring.py`): Profitability, Risk-Adjusted, Volatility Impact, Spread Impact, Liquidity Impact, Execution Quality. Goal-adaptive weights via keyword classification
- **Regression detection + automatic rollback:** aggregator detects score drops > 1.0, preserves best-known code

### 8.3 Remaining Opportunity — Market Regime Classifier

A market regime classifier tool would enable dynamic scenario selection based on observed L1 conditions:

```python
def classify_regime(l1_df: DataFrame) -> RegimeClassification:
    """Classify the market regime from L1 order book data.
    Returns volatility bucket, liquidity bucket, and directional bias.
    """
```

The refinement loop currently does not adapt scenarios based on observed market conditions. A `query_regime()` tool could feed this into the scenario selection logic.

### 8.4 Checklist

- [ ] Assess investigation tool coverage: are there scenarios where the explainer needs data not available through the 8 tools?
- [ ] Evaluate scoring weight profiles: are the 4 profiles (default, impact_focused, risk_focused, execution_focused) sufficient?
- [ ] Consider adding a `query_regime()` tool for market condition classification (§8.3)

**References:** `scenario_tools.py` (273 lines, 3 tools), `tools.py` (380 lines, 8 tools), `scoring.py` (439 lines, 6-axis scoring), `nodes.py` (5 graph nodes), `state.py` (RefinementState)

---

## 9. Strategy Compilation Security

### 9.1 Current Sandbox Model

- `StrategySpec.compile()` calls `StrategyValidator.execute_strategy(source_code, class_name)`
- Source code executed via `exec()` in a restricted namespace
- Restricted builtins: only safe builtins allowed (no `eval`, `exec`, `compile`, `__import__`)
- Import whitelist: only approved modules (math, numpy, collections, etc.)
- Forbidden-call checks in AST analysis (pre-execution)

### 9.2 Attack Surface

- `exec()` on LLM-generated code is inherently risky — the sandbox is the only barrier
- Potential escape vectors: `__subclasses__()` traversal, `object.__init__.__globals__`, `ctypes` if imported, `sys.modules` manipulation
- `run_batch()` workers execute compiled strategies in separate processes — partial isolation, but the `worker_initializer` imports strategy modules

### 9.3 Checklist

- [ ] Audit restricted builtins list against known Python sandbox escapes
- [ ] Verify import whitelist cannot be bypassed via `importlib` or `__builtins__` manipulation
- [ ] Verify AST forbidden-call checks cover `getattr`-based dynamic dispatch
- [ ] Assess timeout enforcement: is `timeout_seconds` enforced at the process level or only at the thread level?
- [ ] Consider containerized execution (Docker/subprocess with seccomp) for production deployments
- [ ] Verify that `StrategySpec.compile()` cannot access the host filesystem or network

**References:** `strategy_spec.py` (61 lines, `compile()`), `strategy_validator.py` (400 lines, `execute_strategy()`, AST checks), `simulation_service.py` (timeout handling)

---

## 10. Operational Integration Concerns

### 10.1 Known Integration Gotchas

- `noise_mkt_close_time` must be set to `settings.end_time` for short simulations (default `"16:00:00"` spreads wakeups across a full trading day)
- Log level `"OFF"` rejected by hasufel — config builder maps to `"CRITICAL"`
- Kernel starts at midnight (`date_ns`), not at `mkt_open` — time calculations must account for this
- `market_ott_ratio` can be < 1.0 due to partial fills making `submitted/executed < 1`
- `SparseMeanRevertingOracle.sigma_s` is legacy/unused — actual volatility parameter is `fund_vol`

### 10.2 Error Propagation

- `SimulationResult` pattern: `result` XOR `error` (never both) — callers must check `result.error` before accessing `result.result`
- `timeout_seconds` (10–3600s) enforces wall-clock limit on simulation runs
- Hasufel kernel exceptions propagate through `abides_run()` — caught in `SimulationService` and wrapped into `SimulationResult(error=...)`

### 10.3 Dependency Versioning

- Hasufel pinned at `git+https://github.com/GabrieleDiCorato/abides-hasufel.git@v2.5.5`
- No automated version-compatibility testing between Rohan and hasufel releases
- Upgrade procedure: update pin → run `tests/test_abides_integration.py` + `tests/test_reproducibility.py` → verify seed stability

### 10.4 Baseline Cache

- Cache key: SHA-256 of JSON-serialized `SimulationSettings` (sorted keys)
- LRU eviction: `OrderedDict` with configurable `baseline_cache_max_entries` (1–10000)
- In-memory only: no cross-process sharing, cache lost on restart
- Feature-flagged: `baseline_cache_v1` via `FEATURE_BASELINE_CACHE_V1` env var

### 10.5 Checklist

- [ ] Document gotchas in a centralized location (or verify they're in `ABIDES_LLM_INTEGRATION_GOTCHAS.md`)
- [ ] Add hasufel version-compatibility CI step (run integration tests on upgrade)
- [ ] Evaluate persistent baseline cache (e.g., disk or Redis) for long-running refinement loops
- [ ] Verify timeout enforcement for parallel `run_batch()` workers

**References:** `config_builder.py` (293 lines), `simulation_service.py` (280 lines), `feature_flags.py` (4 feature toggles)

---

## 11. Testing Strategy & Coverage

### 11.1 Current Test Inventory

- 51 test files across LLM, simulation, framework, UI, and infrastructure
- Notable coverage:
  - `test_deterministic_scoring.py`: 89 parametrized boundary tests for 6-axis scoring
  - `test_property_based.py`: Hypothesis property-based testing for edge cases
  - `test_reproducibility.py`: seed-stability verification
  - `test_abides_integration.py`: ABIDES simulation execution end-to-end
  - `test_strategy_api_validation.py`, `test_strategy_validator.py`, `test_strategy_spec.py`: protocol compliance

### 11.2 Checklist

- [ ] Output path consistency: add test comparing `AbidesOutput` and `HasufelOutput` metrics for identical seeds (see §4)
- [ ] Seed reproducibility across hasufel versions: pin a reference seed + expected metrics, verify on upgrade
- [ ] Historical oracle provider tests: confirm each provider (CSV, DB, API) has integration tests
- [ ] Full refinement loop integration test: verify Writer → Validator → Executor → Explainer → Aggregator pipeline end-to-end
- [ ] Security sandbox tests: verify escape-vector protection in `StrategyValidator` (see §9)

**References:** `tests/` directory (51 files)

---

## 12. Risk & Migration Notes

### 12.1 Seed Reproducibility

The v2.4.0+ identity-based seed derivation breaks backward compatibility with stored seeds from v1.x. Any cached baseline results computed under the old system are invalidated. This is documented in hasufel's changelog and is the correct trade-off (composition invariance is more important than backward seed compatibility).

**Rohan impact:** The baseline cache in `SimulationService` keys on `SimulationSettings` hash, which includes the seed. Old cache entries will simply miss and be recomputed. No action needed.

### 12.2 `end_state` Dict Stability

Rohan's `AbidesOutput` accesses `end_state["agents"]`, `end_state["seed"]`, and `end_state["random_state_kernel"]`. These keys are part of hasufel's `compile()` output contract but are not formally typed or versioned.

**Risk:** If hasufel restructures the runtime dict, Rohan's extraction code breaks silently (wrong data) or loudly (KeyError). The `SimulationResult` API is the stable alternative.

### 12.3 Strategic Agent Registration Visibility

`StrategicAgentConfig` is registered as `"rohan_strategy"` with `category="strategy"` via `@register_agent`. This makes it visible to `list_agent_types()`, meaning the LLM discoverability API returns it as an available agent type. The description marks it as framework-internal, but it is not hidden from queries.

**Risk:** The scenario builder could attempt to use it as a user-selectable agent type. The agent is discovered via `agent.category == "strategy"`, not string matching.

**Mitigation:** Register with `hidden=True` to exclude from discovery queries, or add explicit filtering in the scenario builder's agent-type selection logic.

---

## 13. Prioritised Recommendations

### Tier 1 — High Impact, Low-Medium Effort

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| R1 | Migrate runner to `run_simulation()` — resolve oracle-instance dependency | Rohan | Medium | Typed `SimulationResult`, eliminates `end_state` coupling |
| R2 | Unify output paths: retire `AbidesOutput`, use `HasufelOutput` everywhere | Rohan | Medium | Single extraction path, no metric drift risk |
| R3 | Add output-path consistency test (`AbidesOutput` vs `HasufelOutput`) | Rohan | Low | Catch metric drift before it matters |
| R4 | Security sandbox audit for `StrategyValidator` escape vectors | Rohan | Low | Harden code execution against injection |

### Tier 2 — Medium Impact

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| R5 | Audit Rohan metrics against hasufel `compute_metrics()` for dedup | Rohan | Medium | Delete overlapping metric implementations |
| R6 | Enable parallelization for historical oracle mode | Rohan | Medium | Remove sequential bottleneck for historical scenarios |
| R7 | Add `default_risk_guards` to template metadata | Hasufel | Low | LLM writer can reason about existing guardrails |
| R8 | Add short-duration `quick_test` template | Hasufel | Low | Faster inner refinement loop |

### Tier 3 — Strategic

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| R9 | Build market regime classifier tool (§8.3) | Rohan | Medium | Dynamic scenario selection based on observed conditions |
| R10 | Persistent baseline cache (disk/Redis) | Rohan | Medium | Survives process restarts, cross-worker sharing |
| R11 | Containerized strategy execution (Docker/seccomp) | Rohan | High | Production-grade security for LLM code execution |

---

## Appendix A: File-Level Impact Map

| Rohan File | Lines | Status / Remaining Work |
|---|---|---|
| `config_builder.py` | 293 | Stable. Oracle-instance dependency blocks `run_simulation()` adoption (R1) |
| `simulation_runner_abides.py` | 108 | Dual-path runner. R1 would simplify to single `run_simulation()` call |
| `abides_output.py` | 308 | R2 retires this file entirely once `HasufelOutput` covers all paths |
| `hasufel_output.py` | 149 | Batch output adapter. Would become sole output adapter after R2 |
| `strategic_agent_adapter.py` | 459 | Stable. Protocol bridge between `StrategicAgent` and ABIDES `TradingAgent` |
| `strategic_agent_config.py` | 84 | Stable. `@register_agent("rohan_strategy")` |
| `strategy_spec.py` | 61 | Stable. Frozen serializable strategy representation |
| `simulation_metrics.py` | 117 | Stable. `MarketMetrics` alias, `AgentMetrics.vwap_cents` |
| `simulation_service.py` | 280 | Parallel `run_batch()` implemented. R6 would add historical oracle parallelization |
| `strategy_validator.py` | 400 | R4 security audit target |
| `scenario_tools.py` | 273 | 3 scenario tools (`build_scenario`, `validate_scenario`, `explain_metrics`) |
| `tools.py` | 380 | 8 investigation tools for explainer agent |
| `scoring.py` | 439 | 6-axis deterministic scoring. Stable |

---

## Appendix B: Documentation Quality Score

| Document | Completeness | Accuracy | Agentic Utility | Score |
|---|---|---|---|---|
| `ABIDES_REFERENCE.md` | High — covers all critical traps | Verified against code | High — crash table is directly usable | 9/10 |
| `ABIDES_CONFIG_SYSTEM.md` | High — full builder API + templates | Verified against code | High — discoverability API section | 9/10 |
| `ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md` | High — two patterns + scaffold | Verified against code | High — copy-paste ready | 8/10 |
| `ABIDES_LLM_INTEGRATION_GOTCHAS.md` | High — exhaustive None/NaN catalog | Verified against code | Very high — directly targets LLM failure modes | 10/10 |
| `ABIDES_DATA_EXTRACTION.md` | Medium — covers L1/L2 but not all `ResultProfile` fields | Partially verified | Medium — should be updated with `summary_dict()` examples (available since v2.5.3) | 7/10 |
| `PARALLEL_SIMULATION_GUIDE.md` | High — covers `run_batch()` + manual parallelism | Verified against code | Medium — mostly operational, less agentic | 8/10 |

**Overall documentation score: 8.5/10** — Exceptional for a simulation library. The LLM gotchas document alone would prevent most common integration failures.
