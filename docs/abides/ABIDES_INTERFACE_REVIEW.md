# abides-hasufel v2.5 Interface Review

> **Reviewer:** AI Framework Architect (Financial Engineering / Risk Management)
> **Scope:** Usability of abides-hasufel v2.5.x within Rohan's agentic strategy refinement loop
> **Date:** 2026-04-01
> **Last updated:** 2026-04-01 — refreshed after hasufel v2.5.4 upgrade and Phase 2 implementation

---

## Executive Summary

The migration from the legacy ABIDES fork (`abides-rohan v1.2.7`) to `abides-hasufel v2.5` represents a **major architectural improvement**. The old `AbidesConfigMapper` (504 lines of brittle manual agent instantiation) has been replaced by a 145-line `config_builder.py` translation layer backed by hasufel's declarative `SimulationBuilder` API. The new system delivers real benefits: build-time parameter validation, composable templates, identity-based seed derivation, and a typed `SimulationResult` extraction API that eliminates ad-hoc `end_state` dict mining.

The v2.5.4 upgrade resolved all original hasufel-side requests: template metadata with regime tags (§3.3), raw physical parameter acceptance (§3.4), standalone `compute_metrics()` (§3.2), new scenario templates (§5.1), `agent.category` with `get_agents_by_category()` (§3.5), `runtime_agents` on `run_simulation()` (§3.1), `worker_initializer` on `run_batch()`, and public `derive_seed()`. No hasufel-side blockers remain.

**Phase 2 implementation completed:** Rohan now has a `StrategySpec` model (serialisable strategy representation) and a `StrategicAgentConfig` registered as `"rohan_strategy"` via `@register_agent`. The config builder integrates strategy agents through hasufel's config pipeline, while the runner supports both the new StrategySpec path (config-driven, serialisable) and the legacy strategy-instance path (for test observability). The remaining integration gaps are Rohan-side adoption tasks. With the changes proposed here, the Rohan↔hasufel boundary could shrink by an estimated 30–40% in lines-of-code while gaining new agentic capabilities.

---

## Table of Contents

1. [What Works Well](#1-what-works-well)
2. [Adopt from Hasufel (Rohan-Side Changes)](#2-adopt-from-hasufel-rohan-side-changes)
3. [Remaining Requests for Hasufel](#3-remaining-requests-for-hasufel)
4. [Rohan Internal Simplifications](#4-rohan-internal-simplifications)
5. [Template System Assessment](#5-template-system-assessment)
6. [Metrics & Output Pipeline Assessment](#6-metrics--output-pipeline-assessment)
7. [Agentic Tooling Opportunities](#7-agentic-tooling-opportunities)
8. [Risk & Migration Notes](#8-risk--migration-notes)
9. [Prioritised Recommendations](#9-prioritised-recommendations)
10. [Resolved Items (v2.5.3)](#10-resolved-items-v253)
11. [Resolved Items (v2.5.4) and Phase 2 Implementation](#11-resolved-items-v254-and-phase-2-implementation)

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

## 2. Adopt from Hasufel (Rohan-Side Changes)

These are features that hasufel already provides but Rohan does not yet use. Each one would reduce code, improve capability, or both.

### 2.1 Use `run_simulation()` Instead of Low-Level `compile()` → `abides.run()`

**Current state:** `SimulationRunnerAbides.run()` calls `builder.build_and_compile()` → `abides_run()`, manually managing the runtime dict, log directory (UUID-based), and kernel random state.

**Hasufel offers:** `run_simulation(config)` handles compilation, log directory, kernel RNG, and returns an immutable typed `SimulationResult` with structured per-agent PnL, market summaries, and optional L1/L2 series.

**Why adopt:** Rohan's `AbidesOutput` class (309 lines) manually reconstructs L1/L2 DataFrames from raw `OrderBook` objects with ns-to-midnight time conversion, NaN handling, and Pandera validation. If `SimulationResult` with `ResultProfile.QUANT` already provides validated L1/L2 numpy arrays, `AbidesOutput._compute_order_book_l1()` (80 lines) and `_compute_order_book_l2()` (75 lines) become redundant.

**Previous blocker (resolved):** Rohan injected `StrategicAgentAdapter` post-compilation via `config_add_agents()`. This has been resolved: `StrategicAgentConfig` is now registered as `"rohan_strategy"` via `@register_agent`, and the config builder injects strategy agents through hasufel's compile pipeline when a `StrategySpec` is provided.

**Current status:** The runner still uses `build_and_compile()` + `abides_run()` rather than `run_simulation()` because the historical oracle mode requires `builder.oracle_instance()`, which is not exposed through `run_simulation()`. The dual-path runner supports both StrategySpec (config pipeline) and legacy strategy instances (post-compile injection via `config_add_agents()`).

**Remaining effort:** Low — migrate to `run_simulation()` once oracle instance support is added, or when Rohan switches to template-based oracle configuration exclusively.

**Impact:** Already achieved: strategy agents participate in hasufel's compile pipeline (seed derivation, latency model, agent group management). Remaining: typed `SimulationResult` extraction (blocked by oracle mode).

### 2.2 Use `ResultProfile.QUANT` for Metrics Extraction

**Current state:** `AbidesOutput` parses `end_state` → `ExchangeAgent` → `OrderBook` → `get_L1_snapshots()` → manual DataFrame construction → Pandera validation.

**Hasufel offers:** `ResultProfile.QUANT` includes per-symbol L1/L2 time-series as numpy arrays, plus a `MarketSummary` with `l1_close`, `vwap_cents`, `liquidity_pct`, `volume`, and `spread_close`.

**Why adopt:** If the `MarketSummary` fields overlap with Rohan's `SimulationMetrics` / `MarketMetrics`, some or all of `AnalysisService.compute_market_metrics()` could be replaced by reading from `SimulationResult.markets`.

**Recommendation:** Audit the overlap between `MarketSummary` fields and Rohan's `SimulationMetrics` to determine which metrics can be read directly.

### 2.3 Use `run_batch()` for Parallel Baseline Execution

**Current state:** `SimulationService.run_batch()` runs simulations sequentially via a for-loop.

**Hasufel offers:** `run_batch(configs, n_workers=N)` runs simulations in parallel using `ProcessPoolExecutor`, returns results in input order, handles per-process log isolation automatically.

**Why adopt:** The refinement loop runs at least 2 simulations per iteration (strategy + baseline) across potentially multiple scenarios. Sequential execution is a latency bottleneck.

**Caveat:** Rohan's current `StrategicAgent` is a protocol with non-picklable strategy instances (they hold closures and dynamically-generated code). The baseline runs (strategy=None) could use `run_batch()` immediately; strategy runs would need the `@register_agent` registration from §2.1 to make strategies compilable rather than injected.

**Effort:** Low for baseline parallelization; medium if combined with §2.1.

### 2.4 Use `summary()` / `summary_dict()` for LLM-Friendly Output

**Current state:** `RunSummary` is constructed manually from `ComparisonResult`, which is assembled from separate `AgentMetrics`, `MarketMetrics`, and `MarketImpact` computations.

**Hasufel offers:** `SimulationResult.summary()` produces a human-readable narrative; `summary_dict()` produces a structured dict with `metadata`, `markets`, `agent_leaderboard`, and `warnings` — designed for dashboards and JSON APIs.

**Why adopt:** The explainer agent and writer agent in the refinement loop currently receive `RunSummary` serialized as JSON. Hasufel's `summary_dict()` could serve as the basis, enriched with Rohan-specific fields (market impact, scoring).

### 2.5 Leverage VWAP from `LiquidityMetrics`

**Current state:** Rohan does not compute VWAP.

**Hasufel offers:** `LiquidityMetrics.vwap_cents` is computed automatically from execution history.

**Why adopt:** VWAP is a standard benchmark for execution quality assessment. Adding it to `AgentMetrics` or using it in the scoring system (execution quality axis) would improve the refinement loop's feedback quality.

---

## 3. Remaining Requests for Hasufel

All original hasufel-side requests have been addressed in v2.5.3–v2.5.4 (see [§10](#10-resolved-items-v253) and [§11](#11-resolved-items-v254)). The items below are lower-priority enhancements.

### 3.1 ~~First-Class Post-Compilation Agent Injection in `run_simulation()`~~ ✅ Resolved in v2.5.4

**Resolved:** Hasufel v2.5.4 added `runtime_agents` parameter to `run_simulation()` and `worker_initializer` parameter to `run_batch()`. This, combined with Rohan's `@register_agent("rohan_strategy")` registration, fully resolves the blocker. See [§11](#11-resolved-items-v254).

### 3.2 Risk Guards Documentation in Template Metadata

**Problem:** Templates do not advertise what risk guard defaults they set. A user choosing `rmsc04` vs. `thin_market` cannot know whether position limits or max drawdown guards are pre-configured.

**Request:** Include `default_risk_guards` in template metadata (or at least document it in the template description string). This is important for the agentic loop, where the LLM writer needs to know whether the simulation environment already has guardrails.

---

## 4. Rohan Internal Simplifications

These are changes entirely within Rohan's codebase that do not require hasufel changes.

### 4.1 Eliminate Dual Parameter Vocabularies

**Problem:** Rohan's `AgentSettings` stores oracle and value agent parameters using raw physical names (`kappa`, `lambda_a`, `sigma_s`, `fund_vol`), while hasufel's builder historically only accepted human-friendly names (`mean_reversion_half_life`, `mean_wakeup_gap`, `fund_vol`). This dual vocabulary required 35 lines of unit-conversion helpers (`_kappa_to_half_life`, `_lambda_to_interval`).

**v2.5.3 update:** Hasufel now accepts raw physical parameters (`kappa`, `lambda_a`) directly alongside human-readable strings. The conversion helpers in `config_builder.py` can be deleted, and Rohan can pass its raw parameters straight through.

**Recommendation:** Remove `_kappa_to_half_life()` and `_lambda_to_interval()` from `config_builder.py` and pass `kappa`/`lambda_a` directly to the builder. This eliminates ~35 lines of boilerplate and a class of unit-conversion bugs.

### 4.2 ~~Register `StrategicAgentAdapter` via `@register_agent`~~ ✅ Implemented

**Completed.** `StrategicAgentConfig(BaseAgentConfig)` is registered as `"rohan_strategy"` with `category="strategy"` and `typical_count_range=(1, 1)`. The config holds a serialisable `StrategySpec` (source_code + class_name + params). Compilation is deferred to `create_agents()`, which calls `spec.compile()` → `StrategyValidator.execute_strategy()` to produce a `StrategicAgent` class, then wraps it in `StrategicAgentAdapter`.

This unlocks: config-pipeline seed derivation, latency model integration, and agent group management for strategy agents. `run_simulation()` usage and typed `SimulationResult` extraction remain blocked by oracle-instance compatibility (see §2.1).

### 4.3 Consolidate `SimulationMetrics` / `MarketMetrics` Duplication

**Problem:** `SimulationMetrics` and `MarketMetrics` have 11 identical fields. Both models exist because `SimulationMetrics` was the original aggregate model and `MarketMetrics` was added for baseline comparison. But they're structurally identical.

**Recommendation:** Use a single `MarketMetrics` model for both. `SimulationMetrics` becomes an alias or is removed.

### 4.4 Replace `build_simulation_config()` Backward-Compat Wrapper

**Problem:** `build_simulation_config()` exists as a "backward-compatible wrapper" around `create_simulation_builder()` but returns a tuple `(SimulationConfig, oracle_instance)` that is no longer used anywhere in the codebase — `SimulationRunnerAbides` calls `create_simulation_builder()` directly.

**Recommendation:** If no external code depends on this function, remove it to reduce the public API surface.

---

## 5. Template System Assessment

### 5.1 Template Coverage

As of v2.5.3, hasufel provides eight base templates covering a broad range of market regimes:

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

All templates now include `scenario_description` and `regime_tags` for programmatic selection.

**Remaining gap:** A **short-duration `quick_test` template** (10-minute window, reduced agent counts) would accelerate the inner loop of strategy refinement. This would be a useful addition for rapid iteration.

**Overlay templates** (`with_momentum`, `with_execution`) remain a strong composition primitive. They enable adding agent groups without replacing the base market environment.

### 5.2 Template Composability for Agentic Use

The `SimulationBuilder.from_template()` composability is well-designed. Stacking `from_template("rmsc04").from_template("with_execution")` is intuitive. However, the interaction between overlays and existing agent groups needs documentation:

- Does `from_template("with_momentum")` on top of `rmsc04` (which already has momentum agents) add 12 more, or replace the existing 12?
- What happens when overlays conflict (e.g., two overlays both set `noise.count`)?

**Recommendation:** Document overlay semantics clearly. Ideally, last-wins for same-name agent groups, additive for new groups.

### 5.3 Template Recommendations for Agentic Scenario Generation

Templates are a strong foundation for agentic scenario composition. With v2.5.3's structured metadata (`scenario_description`, `regime_tags`), an LLM agent selecting market conditions can now:

1. Query `list_templates()` for available regimes with tags
2. Filter by `regime_tags` to match desired market conditions programmatically
3. Stack overlays for additional agent groups
4. Override specific parameters via `enable_agent(..., param=value)`

**Rohan action:** Update the scenario executor to use `regime_tags` for template selection instead of hardcoded string matching.

---

## 6. Metrics & Output Pipeline Assessment

### 6.1 Metrics Inventory

Rohan computes a comprehensive set of 22 metrics across three categories:

| Category | Metrics | Source |
|---|---|---|
| **Agent Performance** (7) | total_pnl, sharpe_ratio, max_drawdown, inventory_std, fill_rate, trade_count, traded_volume | Agent logs, L1 |
| **Market Microstructure** (9) | volatility, mean_spread, effective_spread, avg_bid_liquidity, avg_ask_liquidity, lob_imbalance_mean, lob_imbalance_std, vpin, resilience_mean_ns, market_ott_ratio, pct_time_two_sided | L1, agent logs |
| **Market Impact** (9) | Percentage deltas of all microstructure metrics between strategy and baseline runs | Computed from above |

This is a strong metric set for an agentic strategy refinement system. The inclusion of VPIN (adverse selection), resilience (market recovery), and LOB imbalance (directional pressure) goes well beyond the basic PnL/Sharpe metrics typical of retail backtesting.

### 6.2 Metric Computation Architecture

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

### 6.3 `AbidesOutput` Extraction Quality

The extraction code in `AbidesOutput` is well-written and defensively coded:
- Handles one-sided books (NaN fill instead of crash)
- Validates parallel-array invariant for L1 snapshots
- Uses `cached_property` correctly (instance-level cache, garbage-collected with instance)
- Warns on monotonicity violations (cross-midnight detection)

The main concern is **coupling to ABIDES internals**: direct access to `end_state["agents"][0]` (exchange agent by positional convention) and `order_book.get_L1_snapshots()` return format. If hasufel changes these internals, Rohan breaks. Using `SimulationResult` would insulate Rohan from these changes.

---

## 7. Agentic Tooling Opportunities

The hasufel upgrade enables new capabilities that could become tools in Rohan's agentic framework.

### 7.1 Template-Based Scenario Generation Tool

**Concept:** An LLM tool that takes a natural-language market condition description and returns a `SimulationConfig` assembled from templates and parameter overrides.

```python
# Agentic tool signature
def build_scenario(description: str, ticker: str = "ABM", seed: int = 42) -> SimulationConfig:
    """Build a simulation scenario from a natural-language description.

    Uses list_templates() and list_agent_types() to select appropriate
    templates and parameter overrides.
    """
```

**Prerequisites:** Structured parameter descriptions from `get_full_manifest()`. Template metadata (`scenario_description`, `regime_tags`) is now available in v2.5.3.

### 7.2 Config Validation Tool

**Concept:** Wrap `validate_config()` as an agentic tool that checks a proposed configuration before simulation.

```python
def validate_scenario(config_dict: dict) -> ValidationResult:
    """Validate a proposed simulation configuration.

    Returns structured errors and warnings with field paths and agent names.
    """
```

**Already available:** `validate_config()` returns structured `ValidationIssue` objects. This just needs a tool wrapper.

### 7.3 Metric Explanation Tool

**Concept:** Given a `ComparisonResult`, generate a structured interpretation of what each metric means for the strategy.

The diagnostic feedback patterns in `docs/functional/metrics_definition.md` §4 already define the logic:
- High PnL + high `volatility_delta_pct` → "profitable but destabilising"
- Low fill_rate + low PnL → "too passive"
- High VPIN → "reduce order aggressiveness"
- High inventory_std + high max_drawdown → "large unhedged positions"

This could be codified as a rule-based explainer tool that supplements the LLM explainer agent. With hasufel v2.5.3's standalone `compute_metrics()`, the canonical metric values can be fed directly into such a tool.

### 7.4 Market Regime Classifier Tool

**Concept:** Given L1 data from a simulation, classify the prevailing market regime (liquid/illiquid, trending/mean-reverting, calm/volatile).

```python
def classify_regime(l1_df: DataFrame) -> RegimeClassification:
    """Classify the market regime from L1 order book data.

    Returns volatility bucket, liquidity bucket, and directional bias.
    """
```

**Use case:** The refinement loop currently does not adapt scenarios based on observed market conditions. A regime classifier would enable dynamic scenario selection.

---

## 8. Risk & Migration Notes

### 8.1 Seed Reproducibility

The v2.4.0+ identity-based seed derivation breaks backward compatibility with stored seeds from v1.x. Any cached baseline results computed under the old system are invalidated. This is documented in hasufel's changelog and is the correct trade-off (composition invariance is more important than backward seed compatibility).

**Rohan impact:** The baseline cache in `SimulationService` keys on `SimulationSettings` hash, which includes the seed. Old cache entries will simply miss and be recomputed. No action needed.

### 8.2 `end_state` Dict Stability

Rohan's `AbidesOutput` accesses `end_state["agents"]`, `end_state["seed"]`, and `end_state["random_state_kernel"]`. These keys are part of hasufel's `compile()` output contract but are not formally typed or versioned.

**Risk:** If hasufel restructures the runtime dict, Rohan's extraction code breaks silently (wrong data) or loudly (KeyError). The `SimulationResult` API is the stable alternative.

### 8.3 Strategic Agent Registration

If `StrategicAgentAdapter` is registered via `@register_agent`, it becomes visible to `list_agent_types()`. This means the LLM discoverability API will return it as an available agent type, which could confuse the scenario builder (it's not a user-selectable agent; it's injected by the refinement loop).

**Mitigation:** Hasufel v2.5.3 now supports `agent_category` on `AgentData`. Register the adapter with `category="internal"` or `hidden=True` to exclude it from discovery queries. The registration description should clearly mark it as "framework-internal, not user-selectable."

---

## 9. Prioritised Recommendations

### Tier 1 — High Impact, Low Effort

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| ~~R1~~ | ~~Add `extra_agents` parameter to `run_simulation()` or `compile()`~~ | ~~**Hasufel**~~ | ~~Low~~ | ✅ Resolved in v2.5.4 via `runtime_agents` parameter |
| R2 | Remove `build_simulation_config()` backward-compat wrapper | **Rohan** | Trivial | Reduces API surface, dead code |
| R3 | Delete `_kappa_to_half_life()` / `_lambda_to_interval()`, pass raw params to builder | **Rohan** | Trivial | −35 lines, leverages v2.5.3 raw parameter support |

### Tier 2 — High Impact, Medium Effort

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| ~~R4~~ | ~~Register `StrategicAgentAdapter` via `@register_agent`~~ | ~~**Rohan**~~ | ~~Medium~~ | ✅ Implemented — `StrategicAgentConfig` registered as `"rohan_strategy"` |
| R5 | Adopt `run_batch()` for parallel baseline runs | **Rohan** | Low | Reduces iteration latency |
| R6 | Consolidate `SimulationMetrics` / `MarketMetrics` duplication | **Rohan** | Low | Cleaner models |
| R7 | Use `get_agents_by_category("strategy")` instead of type string matching | **Rohan** | Trivial | Leverages v2.5.3; replaces fragile identification |

### Tier 3 — Strategic, Higher Effort

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| R8 | Migrate `AbidesOutput` to use `SimulationResult` (after R1 or R4) | **Rohan** | Medium | Eliminates ~155 lines, insulates from internal dict changes |
| R9 | Audit Rohan metrics against hasufel `compute_metrics()` for deduplication | **Rohan** | Medium | Canonical metrics; delete overlapping Rohan implementations |
| R10 | Add short-duration `quick_test` template | **Hasufel** | Low | Accelerates inner refinement loop |
| R11 | Build template-based scenario generation tool (§7.1) | **Rohan** | Medium | New agentic capability (prerequisites met in v2.5.3) |
| R12 | Build rule-based metric explanation tool (§7.3) | **Rohan** | Medium | Augments LLM explainer with deterministic interpretation |
| R13 | Add `default_risk_guards` to template metadata | **Hasufel** | Low | Enables LLM writer to reason about existing guardrails |

---

## Appendix A: File-Level Impact Map

| Rohan File | Lines | Change Impact |
|---|---|---|
| `config_builder.py` | 325 | R2 removes dead wrapper; R3 deletes unit-conversion helpers (−35 lines) |
| `simulation_runner_abides.py` | 82 | R4 completed — dual-path runner supports StrategySpec (config pipeline) and legacy instance injection |
| `abides_output.py` | 309 | R8 replaces L1/L2 extraction with `SimulationResult` accessors (−155 lines); R7 replaces type string matching |
| `strategic_agent_adapter.py` | 460 | R4 completed — `strategic_agent_config.py` (+75 lines) registered as `"rohan_strategy"` |
| `strategy_spec.py` *(new)* | 55 | Frozen Pydantic model carrying serialisable strategy code for the config pipeline |
| `strategic_agent_config.py` *(new)* | 75 | `@register_agent("rohan_strategy")` + `StrategicAgentConfig(BaseAgentConfig)` |
| `simulation_metrics.py` | 128 | R6 merges `SimulationMetrics`/`MarketMetrics` (−30 lines) |
| `simulation_service.py` | 200 | R5 adds `run_batch()` call path |

**Net estimated change:** −185 lines removed, +50 lines added = **−135 lines** total reduction in integration code.

## Appendix B: Hasufel Documentation Quality Score

| Document | Completeness | Accuracy | Agentic Utility | Score |
|---|---|---|---|---|
| `ABIDES_REFERENCE.md` | High — covers all critical traps | Verified against code | High — crash table is directly usable | 9/10 |
| `ABIDES_CONFIG_SYSTEM.md` | High — full builder API + templates | Verified against code | High — discoverability API section | 9/10 |
| `ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md` | High — two patterns + scaffold | Verified against code | High — copy-paste ready | 8/10 |
| `ABIDES_LLM_INTEGRATION_GOTCHAS.md` | High — exhaustive None/NaN catalog | Verified against code | Very high — directly targets LLM failure modes | 10/10 |
| `ABIDES_DATA_EXTRACTION.md` | Medium — covers L1/L2 but not all `ResultProfile` fields | Partially verified | Medium — `summary_dict()` now available in v2.5.3 but examples may need updating | 7/10 |
| `PARALLEL_SIMULATION_GUIDE.md` | High — covers `run_batch()` + manual parallelism | Verified against code | Medium — mostly operational, less agentic | 8/10 |

**Overall documentation score: 8.5/10** — Exceptional for a simulation library. The LLM gotchas document alone would prevent most common integration failures.

---

## 10. Resolved Items (v2.5.3)

The following items from the original review were addressed in hasufel v2.5.3 and no longer require action on the hasufel side.

| Original # | Item | Resolution |
|---|---|---|
| §3.2 | Expose `compute_metrics()` as standalone function | ✅ `abides_markets.simulation.metrics.compute_metrics()` — accepts book_log2, exec_trades, agent_holdings |
| §3.3 | Add `scenario_description` and `regime_tags` to template metadata | ✅ `TemplateInfo` extended; `list_templates()` returns structured metadata with tags |
| §3.4 | Accept raw physical parameters (`kappa`, `lambda_a`) | ✅ Both oracle and value agent configs accept raw per-ns rates as alternatives to human-readable strings |
| §3.5 | `agent.category` on `AgentData` + `get_agents_by_category()` | ✅ Categories: "background", "strategy", "execution", "market_maker" |
| §5.1 gaps | Missing high-volatility and stress templates | ✅ Added: `stable_day`, `volatile_day`, `low_liquidity`, `trending_day`, `stress_test` |
| §2.4 | `summary_dict()` for LLM-friendly output | ✅ `SimulationResult.summary_dict()` returns structured dict (metadata, markets, leaderboard, warnings) |

---

## 11. Resolved Items (v2.5.4) and Phase 2 Implementation

### Hasufel v2.5.4 Changes

| Original # | Item | Resolution |
|---|---|---|
| §3.1 | `runtime_agents` on `run_simulation()` | ✅ `run_simulation(config, runtime_agents=[...])` — accepts pre-built agents for post-compile injection |
| §3.1 | `worker_initializer` on `run_batch()` | ✅ `run_batch(configs, worker_initializer=fn)` — per-process initialization hook for parallel runs |
| §1.2 | Public `derive_seed()` | ✅ `derive_seed(master_seed, component, index)` — moved from internal to public API |

### Rohan Phase 2 Implementation

| Component | File | Description |
|---|---|---|
| `StrategySpec` | `simulation/models/strategy_spec.py` | Frozen Pydantic model carrying `(source_code, class_name, params)`. Serialisable, hashable, picklable. `compile()` defers to `StrategyValidator.execute_strategy()`. |
| `StrategicAgentConfig` | `simulation/abides_impl/strategic_agent_config.py` | `@register_agent("rohan_strategy", category="strategy")`. Creates `StrategicAgentAdapter` instances in `create_agents()` by compiling the spec once and instantiating per agent. |
| Config builder integration | `simulation/abides_impl/config_builder.py` | `create_simulation_builder(settings, strategy_spec=...)` calls `builder.enable_agent("rohan_strategy", ...)` when a spec is provided. |
| Dual-path runner | `simulation/abides_impl/simulation_runner_abides.py` | Supports both StrategySpec (config pipeline) and legacy strategy instances (post-compile `config_add_agents()`). Discovers strategic agent by `isinstance(a, StrategicAgentAdapter)`. |
| Service layer | `simulation/simulation_service.py` | Accepts both `strategy_spec` and `strategy` parameters; routes to appropriate runner path. |
| Validation pipeline | `simulation/strategy_validator.py` | Creates `StrategySpec` after validation; adds smoke-test step between validation and simulation. |

### Architecture Decision Record

**Decision:** Keep `build_and_compile()` + `abides_run()` flow instead of migrating to `run_simulation()`.

**Rationale:** The historical oracle mode uses `builder.oracle_instance(oracle)` to inject a pre-built oracle. `run_simulation()` does not accept an external oracle instance — it creates one from the config. Until Rohan migrates to template-based oracle configuration or hasufel adds oracle-instance support to `run_simulation()`, the low-level path is required.

**Decision:** Support both `strategy_spec` (StrategySpec) and `strategy` (StrategicAgent instance) parameters.

**Rationale:** The StrategySpec path is used by the LLM pipeline (serialisable code through the config system). The legacy instance path is used by integration tests that need to inspect strategy state after simulation (e.g., `strategy.initialize_called`, `strategy.market_data_count`). Both paths are tested and supported.
