# ROHAN — Scenario Customizability Implementation Plan

This document is the authoritative plan for improving ABIDES scenario customizability across five phases. Phases are ordered by strategic importance and dependency. **Each phase will be planned in detail before execution.**

**Dependency**: `abides-rohan>=v1.2.7-rohan` (provides `ExternalDataOracle`, `BatchDataProvider`, `PointDataProvider`, `DataFrameProvider`, `InterpolationStrategy`).

> [!IMPORTANT]
> **No backward compatibility constraints.** The database can be dropped and recreated at any time. Saved scenarios, presets, and serialized configs can be broken freely — we prioritize clean design over migration paths.

---

## Phase 1: Historical Data Integration

**Goal:** Build a plug-and-play data pipeline that connects external market data to ABIDES simulations via the `ExternalDataOracle`.

### 1.1 Data Model Design

Before any implementation, design the full data model covering all layers of the pipeline:

- **Data Transport Objects (DTOs):** Pydantic models for how fundamental value data flows through the system — from ingestion (raw CSV rows, DB query results, API responses) through normalization to the oracle interface. Must include rich metadata (symbol, source provenance, date range, sampling frequency).
- **Normalization Pipeline:** All data flows through a normalization step regardless of source: timestamps → nanosecond-precision DatetimeIndex, prices → integer cents, optional re-centering around `r_bar`, validation (no large gaps, monotonic timestamps, positive prices). **Must explicitly handle Corporate Actions (split/dividend adjustments) strictly prior to injection, and define an interpolation strategy (e.g., Last Observation Carried Forward (LOCF) to avoid look-ahead bias common in linear interpolation). Also needs Market Hours Filtering to either exclude or explicitly map out-of-hours data.**
- **Database Schema:** SQLAlchemy models for storing curated historical datasets with metadata. This is the primary production storage path.
- **Canonical CSV Format:** A simple, well-defined CSV schema that `CsvDataProvider` expects and that the DB can export. Two columns: `timestamp` (ISO-8601 nanosecond precision), `price_cents` (integer).
- **API Response Mapping:** Define how raw API responses (e.g., Alpaca bar data, Polygon tick data) map to the internal DTOs. Thin adapters handle provider-specific quirks.

### 1.2 Pluggable Data Provider Interface

Define a `FundamentalDataProvider` protocol with three concrete implementations:

| Provider | Backend | Use Case |
|---|---|---|
| `CsvDataProvider` | Local filesystem | Dev/examples, bundled datasets |
| `DatabaseDataProvider` | PostgreSQL / SQLite | Production storage, curated library |
| `ApiDataProvider` | HTTP (Alpaca, Polygon, etc.) | Live data ingestion from external APIs |

Each provider returns a normalized `pd.Series` (DatetimeIndex, int cents) ready for `ExternalDataOracle`. All providers implement `list_available()` to support UI discovery.

### 1.3 Data Source Research & Acquisition

Dedicated research step — **no shortcuts with fake data**:

1. Survey free/freemium intraday data sources: Yahoo Finance (1-min OHLCV), Alpaca free tier (IEX routing), Polygon.io basic, FirstRate Data, Kaggle datasets, SEC MIDAS.
2. Evaluate each against requirements: timestamp granularity (ideally ≤ 1s), price type, coverage of notable events, licensing.
3. **Fallback:** If no adequate free source exists, write ABIDES generation scripts that produce realistic fundamental value paths using `SparseMeanRevertingOracle` with calibrated parameters and export the oracle's internal series. Clearly labeled as "ABIDES-generated" in provenance metadata.
4. Document the provenance of every dataset.

### 1.4 Extend `OracleSettings` & `AbidesConfigMapper`

- Add `OracleType` enum (`SYNTHETIC` / `HISTORICAL`) and `ProviderType` enum (`CSV` / `DATABASE` / `API`) to `OracleSettings`.
- Add a nested `HistoricalOracleSettings` model containing provider-specific fields (dataset ID, file path, API key, interpolation strategy).
- Update `AbidesConfigMapper._build_oracle()` to dispatch on oracle type → provider type, resolving the provider and constructing the appropriate oracle.

### 1.5 UI Integration

- Oracle type toggle (Synthetic / Historical) in the sidebar.
- If Synthetic: existing oracle parameter controls (unchanged).
- If Historical: provider type selector, then context-appropriate input (file uploader for CSV, dataset dropdown for DB, API key + ticker for API).
- Preview chart showing the fundamental value series before execution.

### 1.6 Verification

- Unit tests for each provider with fixture data.
- Integration test: full simulation with `ExternalDataOracle` completes successfully.
- Validate value agents behave coherently with historical data (no price explosions).
- UI smoke test: select historical oracle → pick dataset → run → results render.

---

## Phase 2: Agent Configuration Refactor

**Goal:** Replace the hardcoded agent construction in `AbidesConfigMapper` with a registry-based "building blocks" architecture. **Adding a new agent type = one Python file.** Zero changes to framework code or UI.

### 2.1 Agent Plugin Protocol & Registry

- Define an `AgentPlugin` protocol. Each plugin provides:
  1. **Typed Settings Schema** — a Pydantic `BaseModel` with all agent-specific parameters, defaults, validators, and Field metadata (descriptions, bounds, units). Single source of truth for what the agent accepts. **Must include strict quantitative boundary validation (e.g., `0 < pov <= 1`, `lambda_a > 0`, `kappa >= 0`).**
  2. **Builder Function** — constructs ABIDES agent instances from the typed settings and a shared build context (market times, ticker, random state, etc.). **Must ensure deterministic random seed injection so agent behavior is perfectly reproducible across identical scenarios.**
  3. **UI Metadata** — display name, description, category. The UI introspects the Pydantic schema to auto-generate widgets. No manual UI code per agent.
- **Key design decision:** The `ExchangeAgent` is infrastructure (always exactly one, always required). It should not be toggleable. All other agents are participant plugins with an `enabled` toggle.
- Built-in agents (Noise, Value, Momentum, AdaptiveMarketMaker) are refactored from inline blocks in `_build_agents()` into standalone plugin files.
- Future agents (PredatoryHFT, RL agents, etc.) just implement the protocol and register — zero framework changes.

### 2.2 Dynamic Configuration

- Refactor `AgentSettings` to hold agent populations keyed by registered plugin name, each wrapping `enabled: bool` + the plugin's typed settings.
- Move `OracleSettings` out of `AgentSettings` (it's not an agent — it's a data source).
- The existing settings classes (`NoiseAgentSettings`, etc.) become the `settings_schema` for their respective plugins, with the addition of `enabled`.

### 2.3 Dynamic UI Generation

- The UI loops over registered participant plugins.
- For each: enable/disable toggle, then auto-generated config form by introspecting the Pydantic schema (field types → widget types, Field constraints → widget bounds, Field descriptions → help text).
- Adding a new agent type to the system = writing one Python file with the plugin protocol. The UI, config serialization, and regime translator all discover it automatically.

### 2.4 Latency-Based Adversarial Attacks

The ABIDES `LatencyModel` is set at kernel initialization and is **immutable during a simulation run**. Adversarial latency attacks are modeled as **scenario-level** configuration: each adversarial scenario specifies its `LatencyModelSettings` as part of the config override *before* the simulation starts. The strategy experiences degraded conditions organically (delayed fills, stale market data) without ever being "told" about the latency.

### 2.5 Prerequisite Bug Fixes

Apply at the start of Phase 2 (prerequisites for regime translation):

1. **Deep merge** in `scenario_executor_node`: replace `dict.update()` with recursive deep merge so partial `config_override` works on nested agent settings.
2. **`noise_mkt_close` alignment**: fix hardcoded `"16:00:00"` to `max(end_time, "16:00:00")`.
3. **`LatencyType` enum hygiene**: remove accidental trailing commas on enum values.

### 2.6 Verification

- Unit tests for registry: register, discover, build.
- Unit tests for each migrated plugin: verify output matches current `_build_agents()`.
- Regression test: full simulation with new architecture produces valid results.
- Deep merge tests: partial config overrides work correctly.

---

## Phase 3: UI Restructuring & Market Regime Abstraction

**Goal:** Break the `0_Terminal.py` monolith into maintainable components, and introduce the `MarketRegime` abstraction that translates financial concepts into ABIDES parameters.

### 3.1 Componentize the Streamlit Monolith

Refactor `0_Terminal.py` (~1,882 lines) into a component architecture under `src/rohan/ui/components/`. Pages become thin composition layers (~200 lines each) that import and compose focused component functions. Key components: agent config panel, oracle config, simulation controls, execution panel, results display, regime selector.

### 3.2 `MarketRegime` Data Models

Implement `MarketRegime`, `RegimeType`, and `ScenarioSource` as specified in `adversarial_scenario_system.md` §B.1. Ten regime types mapping financial conditions (e.g., Flash Crash, Momentum Cascade, Informed Trading Surge) to their ABIDES mechanisms.
**Addition:** Incorporate a `STRUCTURAL_BREAK` regime type. While `MEAN_REVERSION_BREAKDOWN` drops the OU parameter `kappa`, a structural break explicitly shifts the baseline fundamental value (`r_bar`) mid-simulation, accurately modeling macroeconomic shocks or earnings surprises.

### 3.3 `RegimeTranslator` Implementation

As specified in `adversarial_scenario_system.md` §B.2–B.4:
- Severity-parameterized interpolation between mild and extreme parameter sets. **Crucial Quant Requirement:** Acknowledge nonlinear market microstructure impacts. E.g., moving from 100 to 50 noise agents degrades liquidity much faster than moving from 1000 to 950. The translator must allow for **nonlinear scalar mapping** (e.g., exponential or logarithmic decay) rather than strict linear interpolation.
- Composable regimes with directional conflict resolution.
- Replace existing presets with `MarketRegime` equivalents.

### 3.4 Regime UI Surface

- Regime type dropdown + severity slider.
- "Compose regimes" button for multi-regime scenarios.
- "Preview parameters" expander showing the translated config override.
- Toggle between regime mode and raw parameter mode for power users.

### 3.5 Verification

- Unit tests for `RegimeTranslator` at multiple severity levels.
- Composition and conflict resolution tests.
- UI smoke test: select regime → adjust severity → verify parameter preview.

---

## Phase 4: Dual-Simulation Comparison Mode

**Goal:** Configure two scenarios with locked shared parameters and different agent populations, run both, and visually compare results.

### 4.1 Comparison State Management

- "Compare Mode" toggle in the Terminal.
- Shared parameters (date, time, seed, ticker, cash) rendered once and locked.
- Per-scenario parameters: agent populations, oracle settings, regime, latency.

### 4.2 Dual Execution

- Sequential or parallel execution of both scenarios.
- Graceful partial failure: if one scenario crashes, show the other's results with an error message.

### 4.3 Overlaid Results & Delta Analysis

- Overlaid Plotly charts (price, PnL, inventory, spread, volume) with scenario-colored traces.
- Delta Summary table: 6 scoring axes × (Scenario A, Scenario B, Δ absolute, Δ %).
- Highlight significant deltas.
- **Statistical Rigor (Quant Extension):** Visuals are insufficient for rigorous algorithmic trading analysis. Include baseline statistical tests comparing the two scenarios (e.g., compute the **Kolmogorov-Smirnov (KS) statistic** to compare fill slippage distributions, and run **Welch's t-test** on terminal PnL paths across multiple seeds if stochasticity is present).

### 4.4 Verification

- Integration test: two scenarios, same seed, different agents → both complete, results differ.
- Edge case: one scenario fails → graceful degradation.

---

## Phase 5: Architecture Polish & Adversary Integration

**Goal:** Harden the architecture and connect the regime layer to the LLM-based adversary agent.

### 5.1 Configuration Validation Layer

Cross-field validation rules:
- Exchange agent always enabled.
- Historical oracle requires a valid data source.
- Agent count ceiling (memory/performance guard).
- `enabled=True` with `num_agents=0` → auto-disable or warn.
- Regime severity bounds validated for simulation stability.
- **Quantitative Consistency Guards:** E.g., ensure `max_size` of momentum agents isn't excessively larger than the natural L1 depth generated by the configured number of market makers, which could artificially break the exchange matching engine.

### 5.2 Adversary Node Integration

As specified in `adversarial_scenario_system.md` §C.1–C.6:
- `generate_adversarial_scenarios()` as a standalone function.
- Thin graph wrapper in the refinement loop.
- Adversarial metadata plumbing through `ScenarioResult`.
- UI: `⚔️ Adversarial` badge, rationale, predicted vs. actual scoring axis pressure.

### 5.3 Verification

- Adversary tests with mock LLM responses.
- End-to-end: refinement loop with adversarial scenarios.

---

## Cross-Cutting Concerns

### No Backward Compatibility Constraints
The database can be dropped and recreated freely. Saved scenarios and presets can be broken. This project prioritizes clean architecture over migration paths.

### Testing Strategy
Each phase includes a verification section. All new code follows existing patterns: `pytest`, `hypothesis` for property-based tests, `pyright`, `ruff`.

### Documentation
Each phase updates `docs/technical/` with relevant reference docs. Key new docs: data pipeline reference, agent plugin development guide.
