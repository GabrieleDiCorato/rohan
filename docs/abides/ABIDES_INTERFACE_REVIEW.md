# abides-hasufel v2.5 Interface Review

> **Reviewer:** AI Framework Architect (Financial Engineering / Risk Management)
> **Scope:** Usability of abides-hasufel v2.5.x within Rohan's agentic strategy refinement loop
> **Date:** 2026-04-01

---

## Executive Summary

The migration from the legacy ABIDES fork (`abides-rohan v1.2.7`) to `abides-hasufel v2.5` represents a **major architectural improvement**. The old `AbidesConfigMapper` (504 lines of brittle manual agent instantiation) has been replaced by a 145-line `config_builder.py` translation layer backed by hasufel's declarative `SimulationBuilder` API. The new system delivers real benefits: build-time parameter validation, composable templates, identity-based seed derivation, and a typed `SimulationResult` extraction API that eliminates ad-hoc `end_state` dict mining.

That said, the current integration does not yet leverage several high-value features of the new API, and there are areas where the boundary between Rohan and hasufel creates unnecessary friction. This review identifies **13 concrete findings** across four categories: things that work well, things that should be adopted, things that should be simplified on the abides side, and things that should be simplified on the Rohan side.

**Overall assessment:** The hasufel upgrade is a clear net positive. The remaining integration gaps are tactical, not architectural. With the changes proposed here, the Rohan↔hasufel boundary could shrink by an estimated 30–40% in lines-of-code while gaining new agentic capabilities.

---

## Table of Contents

1. [What Works Well](#1-what-works-well)
2. [Adopt from Hasufel (Rohan-Side Changes)](#2-adopt-from-hasufel-rohan-side-changes)
3. [Requests for Hasufel (Abides-Side Changes)](#3-requests-for-hasufel-abides-side-changes)
4. [Rohan Internal Simplifications](#4-rohan-internal-simplifications)
5. [Template System Assessment](#5-template-system-assessment)
6. [Metrics & Output Pipeline Assessment](#6-metrics--output-pipeline-assessment)
7. [Agentic Tooling Opportunities](#7-agentic-tooling-opportunities)
8. [Risk & Migration Notes](#8-risk--migration-notes)
9. [Prioritised Recommendations](#9-prioritised-recommendations)

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

**Blocker:** Rohan injects `StrategicAgentAdapter` post-compilation via `config_add_agents()`. This requires the low-level path. The fix is to register `StrategicAgentAdapter` via `@register_agent` so it participates in the normal `compile()` pipeline. This would let Rohan use `run_simulation()` directly.

**Effort:** Medium — requires adding a `StrategicAgentConfig` and registering the adapter.

**Impact:** Eliminates ~155 lines of manual L1/L2 extraction code, gains typed `SimulationResult`, and removes the raw `end_state` dict dependency.

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

## 3. Requests for Hasufel (Abides-Side Changes)

These are areas where changes to hasufel itself would simplify the Rohan integration or improve the agentic developer experience. Since the maintainer is the same, these are concrete feature requests.

### 3.1 First-Class Post-Compilation Agent Injection in `run_simulation()`

**Problem:** `run_simulation()` accepts a `SimulationConfig` but not a pre-compiled agent to inject. Rohan's `StrategicAgentAdapter` is constructed with a non-serializable `StrategicAgent` protocol instance (LLM-generated code), so it cannot be described as a `BaseAgentConfig` field.

**Request:** Add a `run_simulation(config, extra_agents=[...])` parameter or a `compile(config, extra_agents=[...])` variant that appends agents after compilation but before execution. This would let Rohan use the high-level path without needing to register ephemeral strategy adapters.

**Alternative:** A `post_compile_hook: Callable[[dict], dict]` parameter on `run_simulation()` would be maximally flexible.

**Rationale:** This is the single blocker preventing Rohan from using `run_simulation()` and `ResultProfile.QUANT`. It would eliminate ~155 lines of manual extraction code on the Rohan side.

### 3.2 Expose `compute_metrics()` as a Standalone Function

**Problem:** Rohan has its own `AnalysisService` that computes volatility, mean spread, LOB imbalance, VPIN, resilience, OTT ratio, effective spread, Sharpe, max drawdown, and inventory std from raw L1 DataFrames and agent logs. Many of these are standard microstructure metrics.

**Request:** Factor the metrics computation in hasufel's `SimulationResult` pipeline into a standalone `compute_metrics(l1_df, logs_df, ...)` function. This would let Rohan (and any other consumer) compute the same metric set from any data source without running through the full `run_simulation()` → `SimulationResult` pipeline.

**Rationale:** Rohan already computes these metrics; if hasufel provides a canonical, tested implementation, Rohan can delete its own and use the canonical one. This also positions hasufel's metrics as agentic tools (see §7).

### 3.3 Template Metadata: Add `scenario_description` and `regime_tags`

**Problem:** Templates provide composable presets (e.g., `rmsc04`, `liquid_market`, `thin_market`) but the metadata returned by `list_templates()` only includes `name`, `description`, and `agent_types`.

**Request:** Add:
- `scenario_description: str` — a 1–2 sentence human-readable description of the market regime the template models (e.g., "Normal liquidity conditions with balanced order flow and moderate volatility").
- `regime_tags: list[str]` — searchable tags like `["liquid", "balanced", "low_vol"]` or `["thin", "illiquid", "high_vol"]`.

**Rationale:** In the agentic loop, the scenario executor selects templates based on a textual goal. Without structured metadata, template selection requires hardcoded string matching or LLM-based description parsing. Tags enable programmatic matching.

### 3.4 Oracle Parameterisation: Accept `fund_vol` Directly for Value Agent

**Problem:** Rohan's `config_builder.py` must manually convert `kappa` (per-nanosecond mean-reversion rate) to `mean_reversion_half_life` (human-readable duration string) and `lambda_a` (per-nanosecond Poisson rate) to `mean_wakeup_gap` (duration string). These conversions are in `_kappa_to_half_life()` and `_lambda_to_interval()` — 35 lines of unit-conversion boilerplate.

**Request:** Accept the raw physical parameters (`kappa`, `lambda_a`, `fund_vol`, `sigma_s`) as alternatives to the human-friendly strings. The builder could accept either form and convert internally.

**Rationale:** The human-readable duration strings (`"48d"`, `"175s"`) are good for YAML configs and human users, but programmatic consumers (like Rohan's settings model) store the raw physical parameters. Having hasufel accept both eliminates the conversion boilerplate and removes a class of unit-conversion bugs.

### 3.5 `SimulationResult.get_strategic_agent()` Helper

**Problem:** `AbidesOutput.get_strategic_agent_id()` identifies the strategic agent by matching `agent.type == "StrategicAgent"`. This is fragile — it depends on a string convention rather than a typed marker.

**Request:** Add `SimulationResult.get_agents_by_type(type_name)` or a first-class `agent.category` field on `AgentData` (populated from the registry's `category` field). This would let Rohan query `result.get_agents_by_category("strategy")` instead of string-matching on agent type.

### 3.6 Risk Guards Documentation in Template Metadata

**Problem:** Templates do not advertise what risk guard defaults they set. A user choosing `rmsc04` vs. `thin_market` cannot know whether position limits or max drawdown guards are pre-configured.

**Request:** Include `default_risk_guards` in template metadata (or at least document it in the template description string). This is important for the agentic loop, where the LLM writer needs to know whether the simulation environment already has guardrails.

---

## 4. Rohan Internal Simplifications

These are changes entirely within Rohan's codebase that do not require hasufel changes.

### 4.1 Eliminate Dual Parameter Vocabularies

**Problem:** Rohan's `AgentSettings` stores oracle and value agent parameters using raw physical names (`kappa`, `lambda_a`, `sigma_s`, `fund_vol`), but hasufel's builder expects human-friendly names (`mean_reversion_half_life`, `mean_wakeup_gap`, `fund_vol`). This dual vocabulary means:
- `config_builder.py` has 35 lines of unit-conversion helpers (`_kappa_to_half_life`, `_lambda_to_interval`).
- The field names in `OracleSettings` and `ValueAgentSettings` don't match anything in hasufel's API, making the mapping non-obvious.

**Recommendation:** Update `AgentSettings` to use hasufel-native names as the primary representation. If backward compatibility with existing `.env` files or stored presets is needed, use `Field(alias=...)` or a migration validator.

### 4.2 Register `StrategicAgentAdapter` via `@register_agent`

**Problem:** The adapter is injected post-compilation via `config_add_agents()`, bypassing hasufel's config system. This means it doesn't get seed derivation, risk guards, or typed result extraction from the normal pipeline.

**Recommendation:** Create a `StrategicAgentConfig(BaseAgentConfig)` that holds only serializable parameters (symbol, starting_cash, wake_up_freq, order_book_depth) and uses `_prepare_constructor_kwargs()` to inject the non-serializable `strategy` instance at compile time.

This unlocks: `run_simulation()` usage, `ResultProfile.QUANT` extraction, `AgentData` in `SimulationResult`, and proper risk guard integration.

### 4.3 Consolidate `SimulationMetrics` / `MarketMetrics` Duplication

**Problem:** `SimulationMetrics` and `MarketMetrics` have 11 identical fields. Both models exist because `SimulationMetrics` was the original aggregate model and `MarketMetrics` was added for baseline comparison. But they're structurally identical.

**Recommendation:** Use a single `MarketMetrics` model for both. `SimulationMetrics` becomes an alias or is removed.

### 4.4 Replace `build_simulation_config()` Backward-Compat Wrapper

**Problem:** `build_simulation_config()` exists as a "backward-compatible wrapper" around `create_simulation_builder()` but returns a tuple `(SimulationConfig, oracle_instance)` that is no longer used anywhere in the codebase — `SimulationRunnerAbides` calls `create_simulation_builder()` directly.

**Recommendation:** If no external code depends on this function, remove it to reduce the public API surface.

---

## 5. Template System Assessment

### 5.1 Template Coverage

The current templates cover three useful market regimes:

| Template | Regime | Noise | Value | Momentum | MM | Use Case |
|---|---|---|---|---|---|---|
| `rmsc04` | Reference | 1000 | 102 | 12 | 2 | Default benchmark, balanced market |
| `liquid_market` | High liquidity | 5000 | 200 | 25 | 4 | Stress-test market impact in deep markets |
| `thin_market` | Low liquidity | 100 | 20 | 0 | 0 | Illiquid conditions, large spreads |

**Gap analysis for Rohan's agentic loop:**

- **Missing: High-volatility regime** — No template models a regime with elevated `fund_vol` or frequent megashocks. This is critical for testing strategy robustness.
- **Missing: Adversarial regime** — A template where the market maker is aggressive (tight spreads, high backstop) to test adverse selection.
- **Missing: Short-duration preset** — Templates default to full-day simulations. A `quick_test` template with a 10-minute window and reduced agent counts would accelerate the inner loop of strategy refinement.

**Overlay templates** (`with_momentum`, `with_execution`) are a strong composition primitive. They enable adding agent groups without replacing the base market environment.

### 5.2 Template Composability for Agentic Use

The `SimulationBuilder.from_template()` composability is well-designed. Stacking `from_template("rmsc04").from_template("with_execution")` is intuitive. However, the interaction between overlays and existing agent groups needs documentation:

- Does `from_template("with_momentum")` on top of `rmsc04` (which already has momentum agents) add 12 more, or replace the existing 12?
- What happens when overlays conflict (e.g., two overlays both set `noise.count`)?

**Recommendation:** Document overlay semantics clearly. Ideally, last-wins for same-name agent groups, additive for new groups.

### 5.3 Template Recommendations for Agentic Scenario Generation

Templates are a strong foundation for agentic scenario composition. An LLM agent selecting market conditions could:

1. Query `list_templates()` for available regimes
2. Select a base template matching the desired market conditions
3. Stack overlays for additional agent groups
4. Override specific parameters via `enable_agent(..., param=value)`

For this workflow to be robust, templates need the structured metadata requested in §3.3 (description, regime tags). Currently, an LLM would need to parse the template name or description text, which is brittle.

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

**Hasufel alternative:**
```
run_simulation(config, profile=ResultProfile.QUANT)
  → SimulationResult.markets[symbol] → MarketSummary (L1 close, VWAP, volume)
  → SimulationResult.agents[i] → AgentData (PnL, holdings)
  → SimulationResult.markets[symbol].l1_series → numpy array
```

**Assessment:** The hasufel path provides a subset of Rohan's metrics directly. The advanced microstructure metrics (VPIN, resilience, LOB imbalance) are computed by Rohan and documented in the functional spec (`docs/functional/metrics_definition.md`). These should remain in Rohan unless hasufel provides equivalent functionality (see request §3.2).

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

**Prerequisites:** Template metadata (§3.3), structured parameter descriptions from `get_full_manifest()`.

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

This could be codified as a rule-based explainer tool that supplements the LLM explainer agent.

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

**Mitigation:** Use a `category="internal"` or `hidden=True` flag to exclude it from discovery queries. If hasufel doesn't support this, the registration description should clearly mark it as "framework-internal, not user-selectable."

---

## 9. Prioritised Recommendations

### Tier 1 — High Impact, Low Effort

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| R1 | Add `extra_agents` parameter to `run_simulation()` or `compile()` | **Hasufel** | Low | Unblocks Rohan's use of `SimulationResult` and `ResultProfile.QUANT` |
| R2 | Add `scenario_description` and `regime_tags` to template metadata | **Hasufel** | Low | Enables agentic scenario selection |
| R3 | Remove `build_simulation_config()` backward-compat wrapper | **Rohan** | Trivial | Reduces API surface, dead code |

### Tier 2 — High Impact, Medium Effort

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| R4 | Register `StrategicAgentAdapter` via `@register_agent` | **Rohan** | Medium | Unlocks `run_simulation()`, risk guards, typed results |
| R5 | Adopt `run_batch()` for parallel baseline runs | **Rohan** | Low | Reduces iteration latency |
| R6 | Consolidate `SimulationMetrics` / `MarketMetrics` duplication | **Rohan** | Low | Cleaner models |
| R7 | Accept raw physical parameters (`kappa`, `lambda_a`) alongside human-readable strings | **Hasufel** | Medium | Eliminates 35 lines of conversion boilerplate in Rohan |

### Tier 3 — Strategic, Higher Effort

| # | Action | Owner | Effort | Impact |
|---|---|---|---|---|
| R8 | Factor `compute_metrics()` into a standalone hasufel function | **Hasufel** | Medium | Canonical metrics implementation; enables tool-based extraction |
| R9 | Migrate `AbidesOutput` to use `SimulationResult` (after R1 or R4) | **Rohan** | Medium | Eliminates ~155 lines, insulates from internal dict changes |
| R10 | Add high-volatility and quick-test templates | **Hasufel** | Low | Covers gaps in Rohan's scenario matrix |
| R11 | Build template-based scenario generation tool (§7.1) | **Rohan** | Medium | New agentic capability |
| R12 | Build rule-based metric explanation tool (§7.3) | **Rohan** | Medium | Augments LLM explainer with deterministic interpretation |
| R13 | Add `agent.category` to `AgentData` and `get_agents_by_category()` to `SimulationResult` | **Hasufel** | Low | Replaces fragile string-matching for agent identification |

---

## Appendix A: File-Level Impact Map

| Rohan File | Lines | Change Impact |
|---|---|---|
| `config_builder.py` | 325 | R3 removes dead wrapper; R7 eliminates unit-conversion helpers (−35 lines) |
| `simulation_runner_abides.py` | 82 | R1/R4 enable `run_simulation()` path, potentially replacing entire file |
| `abides_output.py` | 309 | R9 replaces L1/L2 extraction with `SimulationResult` accessors (−155 lines) |
| `strategic_agent_adapter.py` | 460 | R4 adds `@register_agent` + `BaseAgentConfig` (+30 lines) |
| `simulation_metrics.py` | 128 | R6 merges `SimulationMetrics`/`MarketMetrics` (−30 lines) |
| `simulation_service.py` | 200 | R5 adds `run_batch()` call path |
| `agent_settings.py` | 283 | R7 would update field names to hasufel-native vocabulary |

**Net estimated change:** −185 lines removed, +50 lines added = **−135 lines** total reduction in integration code.

## Appendix B: Hasufel Documentation Quality Score

| Document | Completeness | Accuracy | Agentic Utility | Score |
|---|---|---|---|---|
| `ABIDES_REFERENCE.md` | High — covers all critical traps | Verified against code | High — crash table is directly usable | 9/10 |
| `ABIDES_CONFIG_SYSTEM.md` | High — full builder API + templates | Verified against code | High — discoverability API section | 9/10 |
| `ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md` | High — two patterns + scaffold | Verified against code | High — copy-paste ready | 8/10 |
| `ABIDES_LLM_INTEGRATION_GOTCHAS.md` | High — exhaustive None/NaN catalog | Verified against code | Very high — directly targets LLM failure modes | 10/10 |
| `ABIDES_DATA_EXTRACTION.md` | Medium — covers L1/L2 but not all `ResultProfile` fields | Partially verified | Medium — missing `summary_dict()` examples | 7/10 |
| `PARALLEL_SIMULATION_GUIDE.md` | High — covers `run_batch()` + manual parallelism | Verified against code | Medium — mostly operational, less agentic | 8/10 |

**Overall documentation score: 8.5/10** — Exceptional for a simulation library. The LLM gotchas document alone would prevent most common integration failures.
