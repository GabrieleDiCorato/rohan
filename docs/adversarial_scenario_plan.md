# ROHAN — Adversarial Scenario Generation Plan

## Product Context

ROHAN's product direction (documented in `implementation_plan_abides_capabilities.md`) defines three priorities: **scenario control quality**, **interpretation depth**, and **deterministic scoring**. The refinement loop is PoC scaffolding — the lasting product is a system where a user submits a strategy and the system finds the market conditions that break it, then explains *why* with code-level precision.

This plan extends ROHAN with the ability to **automatically generate adversarial market scenarios** targeting a specific strategy's weaknesses. It builds directly on the foundation laid by the implementation plan's Phases 1–3 (deterministic scoring, tool-equipped explainer, structured feedback).

---

## Prerequisites

This work begins **after** the implementation plan's Phases 1–3 are complete. The adversarial system depends on:

| Dependency | Source | Why required |
|---|---|---|
| Deterministic 6-axis scoring | Phase 1, Steps 1–3 | The adversary needs quantitative targets: "find scenarios that minimize profitability score." LLM-scored numbers are too noisy for targeted adversarial search. |
| Per-scenario fixed seeds | Phase 1, Step 5 | Adversarial scenarios must be reproducible across iterations — same market conditions each time so score changes reflect strategy changes, not seed variance. |
| Tool-equipped ReAct explainer | Phase 2, Steps 8–9 | Adversarial results are worthless without deep investigation. The explainer must trace *why* the strategy failed — fills, inventory trajectory, adverse selection, counterparty mix. |
| Rich `ScenarioResult` (PnL/inventory curves) | Phase 2, Step 10 | The adversary's rationale must be verifiable against simulation evidence. |
| 6-axis UI and structured feedback | Phase 3, Steps 11–14 | Adversarial results surface through the same UI and feedback channels. |

---

## Phase A — Configuration Surface

**Goal:** Make the ABIDES parameter space reachable through clean overrides. Scoped tightly — only what the regime layer and adversary need.

### A.1: Fix deep merge in `scenario_executor_node`

**Problem:** The executor at `nodes.py` L300–303 uses shallow `dict.update()`:

```python
merged = base_settings.model_dump()
merged.update(scenario.config_override)
settings = SimulationSettings.model_validate(merged)
```

This means overriding any nested agent setting (e.g., `agents.momentum.num_agents`) requires providing the *entire* `agents` subtree in `config_override`. Every adversarial scenario touches nested agent settings.

**Solution:** Replace with recursive deep-merge. Pydantic's `model_validate` handles nested model reconstruction, so this works:

```python
def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

merged = deep_merge(base_settings.model_dump(), scenario.config_override)
settings = SimulationSettings.model_validate(merged)
```

After this, a `config_override` of `{"agents": {"momentum": {"num_agents": 50}}}` works without clobbering all other agent settings.

### A.1b: Fix `LatencyType` enum tuple values

**Problem:** In `latency_settings.py` L11–12, `NO_LATENCY` and `DETERMINISTIC` are accidentally defined with tuple syntax:

```python
NO_LATENCY = ("no_latency",)    # ← tuple syntax (trailing comma)
DETERMINISTIC = ("deterministic",)  # ← tuple syntax
CUBIC = "cubic"                  # ← plain string (correct)
```

Python's `str` enum coerces these to plain strings via `str.__new__`, so `.value` happens to work correctly at runtime. However, this is inconsistent and fragile — future Python or Pydantic versions may not handle this gracefully, and it confuses anyone reading the code.

**Solution:** Remove trailing commas for consistency:

```python
NO_LATENCY = "no_latency"
DETERMINISTIC = "deterministic"
CUBIC = "cubic"
```

**Verified:** `model_dump()` → `model_validate()` round-trips and `config_override` with `latency.type = "cubic"` work correctly today. This is a code hygiene fix, not a blocking bug.

### A.2: Expose strategic agent configuration

**Problem:** Two hardcoded values in `abides_config_mapper.py` L250–251:

```python
wake_up_freq="1S",
order_book_depth=10,
```

These are adversarially meaningful: a slower wake-up penalises latency-dependent strategies; reduced book depth limits information available to L2-based strategies.

**Solution:** Add a `StrategicAgentSettings` model:

```python
class StrategicAgentSettings(BaseModel):
    wake_up_freq: str = "1S"
    order_book_depth: int = Field(default=10, ge=1, le=50)
```

Nest it under `AgentSettings` as `strategic: StrategicAgentSettings`. Wire through `AbidesConfigMapper._build_agents()`.

### A.3: Align `noise_mkt_close` with simulation window

**Problem:** `noise_mkt_close` is hardcoded to `"16:00:00"` in `abides_config_mapper.py` L59, regardless of `end_time`. When adversarial scenarios vary simulation duration, noise agents trade past the strategy's window, creating confusing asymmetries.

**Solution:** Use `max(end_time, "16:00:00")` so noise agents close at or after the simulation window, but never before it.

### A.4: Document the adversarial parameter catalog

Enumerate every ABIDES parameter with its financial effect and adversarial relevance. This catalog becomes the reference for the `RegimeTranslator` in Phase B.

**High-impact adversarial knobs (already exposed but under-used):**

| Parameter | Setting | Default | Adversarial effect |
|---|---|---|---|
| `agents.noise.num_agents` | `NoiseAgentSettings` | 1000 | Liquidity provider. Reducing creates thin markets with wider spreads. |
| `agents.value.num_agents` | `ValueAgentSettings` | 102 | Informed traders. Increasing creates more adverse selection for passive strategies. |
| `agents.value.lambda_a` | `ValueAgentSettings` | 5.7e-12 | Arrival rate. Higher = more frequent informed-trader activity = hostile for uninformed market makers. |
| `agents.momentum.num_agents` | `MomentumAgentSettings` | 12 | Trend followers. Increasing creates momentum cascades that punish mean-reversion strategies. |
| `agents.momentum.max_size` | `MomentumAgentSettings` | 10 | Order size. Larger = more market impact per momentum trade. |
| `agents.momentum.wake_up_freq` | `MomentumAgentSettings` | "37s" | Faster wake-up = more aggressive momentum pressure. |
| `agents.adaptive_market_maker.num_agents` | `AdaptiveMarketMakerSettings` | 2 | Competing MMs. More = tighter spreads, harder for user's MM strategy. Fewer = wider spreads, harder for liquidity-consuming strategies. |
| `agents.adaptive_market_maker.pov` | `AdaptiveMarketMakerSettings` | 0.025 | Participation. Lower = less aggressive MM presence = thinner book. |
| `agents.adaptive_market_maker.spread_alpha` | `AdaptiveMarketMakerSettings` | 0.75 | Spread aggressiveness. Higher = tighter native spreads = harder for user's MM to compete. |
| `agents.adaptive_market_maker.skew_beta` | `AdaptiveMarketMakerSettings` | 0 | Inventory skew. Non-zero creates asymmetric book pressure. |
| `oracle.fund_vol` | `OracleSettings` | 5e-5 | Fundamental volatility. Higher = more volatile fair value = harder to track. |
| `oracle.megashock_mean` | `OracleSettings` | 1000 | Shock size. Larger = bigger sudden price dislocations. |
| `oracle.megashock_lambda_a` | `OracleSettings` | 2.78e-18 | Shock frequency. Higher = more frequent jumps. |
| `starting_cash` | `SimulationSettings` | 10,000,000 (cents) | Capital constraint. Less capital = tighter risk limits, faster drawdown. |
| `latency.type` | `LatencyModelSettings` | DETERMINISTIC | Latency model. Cubic add jitter/delay that punish speed-dependent strategies. |

**Not varied in current presets** (untapped adversarial value):
- `value.kappa` — agent mean-reversion speed (Bayesian belief update rate). Higher = agent's estimate reverts to `r_bar` faster = stronger conviction about fundamental value = more aggressive informed trading.
- `sigma_n` — value agent observation noise variance. **Not independently configurable** — derived as `r_bar / 100` in `abides_config_mapper.py` L156. Controls how precisely agents observe the fundamental price. Lower = more informed agents = stronger adverse selection. Currently changes silently when `r_bar` is overridden. Should be exposed as an explicit field in `ValueAgentSettings` (default: `r_bar / 100`) to enable independent control.
- `adaptive_market_maker.level_spacing` — depth spacing. Wider = less depth per tick level.
- `adaptive_market_maker.backstop_quantity` — reserve liquidity. Zero = MM can fully withdraw.
- `exchange.pipeline_delay_ns` — exchange processing delay. Non-zero creates execution uncertainty.
- `starting_cash` — never varied; capital constraint is a powerful stress axis.

**Important coupling:** `value.r_bar` is used by both the oracle (as the fundamental price mean) and value agents (as their belief anchor). Overriding `agents.value.r_bar` via `config_override` changes **both** the oracle's fundamental mean and the agents' target — `abides_config_mapper.py` L273 reads `r_bar` from `ValueAgentSettings` for oracle construction. Additionally, `sigma_n = r_bar / 100` means changing `r_bar` silently scales agent observation precision.

---

## Phase B — Market Regime Abstraction Layer

**Goal:** Create a financially meaningful interface that maps market conditions in natural language to ABIDES configurations. This is the core product investment for scenario control quality.

### B.1: `MarketRegime` model

A new Pydantic model representing a market condition in financial terms:

```python
class RegimeType(str, Enum):
    NORMAL = "normal"
    HIGH_VOLATILITY = "high_volatility"
    LOW_LIQUIDITY = "low_liquidity"
    FLASH_CRASH = "flash_crash"
    MOMENTUM_CASCADE = "momentum_cascade"
    SPREAD_WIDENING = "spread_widening"
    INFORMED_TRADING_SURGE = "informed_trading_surge"
    MEAN_REVERSION_BREAKDOWN = "mean_reversion_breakdown"
    LATENCY_DEGRADATION = "latency_degradation"
    CROWDED_TRADE_UNWIND = "crowded_trade_unwind"

class ScenarioSource(str, Enum):
    USER = "user"
    PRESET = "preset"
    ADVERSARIAL = "adversarial"

class MarketRegime(BaseModel):
    regime_type: RegimeType
    severity: float = Field(ge=0.0, le=1.0)
    description: str = ""
    rationale: str = ""                          # why this targets the strategy
    source: ScenarioSource = ScenarioSource.USER
```

**Financial grounding for each regime:**

| Regime | Real-world analogue | Primary ABIDES mechanism |
|---|---|---|
| `NORMAL` | Typical trading session | Default parameters, baseline reference |
| `HIGH_VOLATILITY` | VIX spike, earnings season | Oracle `fund_vol` ↑, megashock frequency ↑, momentum agents ↑ |
| `LOW_LIQUIDITY` | Pre-market, holidays, small-caps | Noise agents ↓, value agents ↓, MM count ↓, MM `pov` ↓ |
| `FLASH_CRASH` | May 6, 2010; Aug 24, 2015 | Oracle megashock_mean ↑↑, noise ↓, MM withdrawn (`pov` → 0), momentum ↑ |
| `MOMENTUM_CASCADE` | GME Jan 2021, quant momentum factor | Momentum agents ↑↑, `max_size` ↑, `wake_up_freq` ↓ (faster) |
| `SPREAD_WIDENING` | Risk-off, credit events | MM `spread_alpha` ↑, MM count ↓, noise ↓ |
| `INFORMED_TRADING_SURGE` | Pre-announcement activity, M&A leaks | Value agents ↑↑, `lambda_a` ↑ (more frequent wake-ups), value agent `kappa` ↑ (stronger conviction about fair value → more aggressive mean-reversion trading → more adverse selection for uninformed strategies) |
| `MEAN_REVERSION_BREAKDOWN` | Regime change, structural break | Oracle `kappa` ↓↓ (fundamental price becomes a near-random walk — structural regime change where the OU mean-reversion vanishes), `fund_vol` ↑, momentum ↑. **Note:** this models the *fundamental* losing its mean-reverting property. A separate adversarial variant could model *agents* losing conviction (lowering `ValueAgentSettings.kappa` while oracle kappa stays normal) — a "loss of informed trader confidence" scenario. |
| `LATENCY_DEGRADATION` | Exchange slowdowns, flash events | Latency → CUBIC with `jitter` ↑, exchange `pipeline_delay_ns` ↑ |
| `CROWDED_TRADE_UNWIND` | Aug 2007 quant quake | Momentum agents ↑↑ (amplify directional pressure after a negative shock), oracle `megashock_mean` ↑, value agents ↓ (reduced price discovery), MM withdrawn. **Limitation:** ABIDES models the *dynamics* of a cascading sell-off (directional momentum pressure with no liquidity buffer) but not the *mechanism* (pre-existing crowded positions, stop-loss triggers, margin calls forcing liquidation). The regime approximates the market conditions *during* an unwind, not the cause. |

### B.2: `RegimeTranslator`

A deterministic mapping layer. Each `RegimeType` has a template function: `severity ∈ [0, 1]` → `config_override: dict`.

**Interpolation model:** Linear interpolation between a "mild" baseline and an "extreme" parameter set. Each regime has a **base threshold** (default: `severity = 0.1`) below which it produces a minimally noticeable version of the condition, and above which parameters interpolate linearly toward the extreme. `severity = 0.0` is *reserved for the `NORMAL` regime* and produces an empty override. For all other regimes, `severity` should be ≥ 0.1 to produce a meaningful condition. `severity = 1.0` represents the most extreme condition that still produces a stable simulation (validated empirically).

**Why not `severity = 0.0` → empty for all types:** If every regime degenerates to defaults at `severity = 0.0`, the regime type label becomes meaningless at low severities, and the `NORMAL` regime is redundant. The base threshold ensures that even `LOW_LIQUIDITY(severity=0.1)` produces a *noticeably* thinner market, distinguishable from `NORMAL`.

**Nonlinear effects note:** Severity is a *configuration interpolant*, NOT a linear stress intensity. Doubling `severity` does not double the stress on the strategy — agent interactions are nonlinear (e.g., halving noise agents from 1000→500 has less impact than halving from 100→50; momentum agents crowd each other out). The empirical calibration step (stability guardrails) should map severity curves to actual measured stress metrics and adjust templates if major nonlinearities are discovered.

```python
class RegimeTranslator:
    def translate(self, regime: MarketRegime) -> dict[str, Any]:
        """Convert a MarketRegime to a config_override dict."""
        template = self._templates[regime.regime_type]
        return template(regime.severity)

    def translate_composite(self, regimes: list[MarketRegime]) -> dict[str, Any]:
        """Merge multiple regimes with conflict resolution."""
        overrides = [self.translate(r) for r in regimes]
        return self._merge_overrides(overrides)
```

**Conflict resolution for composition:**

Each regime template tags its parameters with a **stress direction** — whether increasing or decreasing the value creates more adversarial conditions. The merge function uses this direction to resolve conflicts:

| Agent / parameter type | Stress direction | Merge rule | Rationale |
|---|---|---|---|
| `noise.num_agents` | decrease | `min()` | Fewer noise agents = thinner market |
| `adaptive_market_maker.num_agents` | decrease | `min()` | Fewer MMs = wider spreads, less depth |
| `adaptive_market_maker.pov` | decrease | `min()` | Lower participation = thinner book |
| `value.num_agents` | **context-dependent** | see below | More = more adverse selection; fewer = less price discovery |
| `momentum.num_agents` | increase | `max()` | More momentum = stronger trend pressure |
| `momentum.max_size` | increase | `max()` | Larger orders = more impact |
| `oracle.fund_vol`, `megashock_*` | increase | `max()` | More volatile/shocking fundamentals |
| `latency.*` | increase | `max()` | Worse latency = more execution uncertainty |
| `spread_alpha` | increase | `max()` | Tighter native spreads = harder for MM strategies |

**`value.num_agents` conflict handling:** When two regimes disagree on value agent direction (e.g., `INFORMED_TRADING_SURGE` increases them, `LOW_LIQUIDITY` decreases them), the regime with the higher severity takes precedence for this parameter. This is the one non-commutative rule — it reflects the financial reality that informed trading surge and liquidity drought are partially contradictory conditions.

The merge is commutative for all parameters except `value.num_agents` when stress directions conflict.

**Example template — `LOW_LIQUIDITY`:**

```python
def _low_liquidity(self, severity: float) -> dict:
    return {
        "agents": {
            "noise": {"num_agents": int(lerp(1000, 50, severity))},
            "value": {"num_agents": int(lerp(102, 10, severity))},
            "adaptive_market_maker": {
                "num_agents": max(1, int(lerp(2, 1, severity))),
                "pov": lerp(0.025, 0.003, severity),
            },
        },
    }
```

**Stability guardrails:** Templates must ensure simulations run to completion. Each template's extreme (`severity=1.0`) is tested empirically — if ABIDES crashes or produces degenerate output (e.g., zero ticks), the extreme is pulled back. Document these limits.

### B.3: Regime composition

A single adversarial scenario can combine regimes. Examples:
- `LOW_LIQUIDITY(0.7) + MOMENTUM_CASCADE(0.6)` = "momentum event in a thin market" — orders walk through a thin book with trend pressure
- `HIGH_VOLATILITY(0.8) + LATENCY_DEGRADATION(0.5)` = "volatile market with slow execution" — strategy can't react fast enough to large moves
- `INFORMED_TRADING_SURGE(0.7) + SPREAD_WIDENING(0.6)` = "adverse selection with wide spreads" — passive orders get picked off at worse prices

The adversary agent (Phase C) can output both single regimes and compositions.

### B.4: Replace presets with regimes

Current presets → `MarketRegime` equivalents:

| Preset | Regime mapping |
|---|---|
| Default (Balanced Market) | `NORMAL(severity=0.0)` |
| High Volatility | `HIGH_VOLATILITY(severity=0.7)` |
| Low Liquidity | `LOW_LIQUIDITY(severity=0.7)` |
| Market Maker Stress Test | `MOMENTUM_CASCADE(severity=0.6) + SPREAD_WIDENING(severity=0.4)` |
| Momentum Dominated | `MOMENTUM_CASCADE(severity=0.8)` |

Validate backward compatibility: old preset configs and new regime-translated configs produce statistically equivalent simulation outcomes (same agent counts, same oracle params).

### B.5: `ScenarioConfig` evolution

Extend to carry the regime alongside the override:

```python
class ScenarioConfig(BaseModel):
    name: str = "default"
    regime: MarketRegime | None = None              # semantic description
    config_override: dict[str, Any] = Field(default_factory=dict)  # raw or translated
```

When `regime` is present and `config_override` is empty, the executor calls `RegimeTranslator.translate()` to populate `config_override`. When both are present, regime-translated values serve as the base and `config_override` provides fine-grained overrides on top. When only `config_override` is present, behavior is unchanged (backward compatible).

### B.6: UI integration

**Terminal sidebar:**
- Regime type dropdown + severity slider replaces (or augments) the parameter-heavy form
- "Preview parameters" expander shows the translated config
- Power users toggle between regime mode and raw parameter mode
- "Compose regimes" button adds additional regime rows

**Refinement Lab sidebar:**
- Scenario selection gains a "Generate Adversarial" section (Phase C)
- Each scenario displays its regime type, severity, and source badge (`USER` / `PRESET` / `ADVERSARIAL`)
- Scenario results show adversary rationale when `source == ADVERSARIAL`

---

## Phase C — Adversary Agent

**Goal:** An LLM agent that reads a strategy's code and goal, identifies its assumptions and fragilities, and generates targeted adversarial scenarios as `MarketRegime` objects. Runs once per session. Fixed for all iterations.

### C.1: Graph integration

New `adversary_node` inserted after the first successful validation:

```
writer → validator → [adversary (once)] → executor → explainer → aggregator → ...
                                                                    ↓
                                                              writer (next iteration)
```

**Graph wiring:** In `build_refinement_graph()`, the `validation_router`'s `"execute"` edge targets `"adversary"` instead of `"executor"`, and an unconditional edge connects `"adversary"` → `"executor"`. This is a two-line change to the existing graph.

**Iteration gating:** The adversary node runs on *every* iteration but performs an early-return check:

```python
def adversary_node(state: RefinementState) -> dict:
    if state.get("iteration_number", 1) > 1:
        return {}  # no-op — scenarios already generated in iteration 1
    # ... generate adversarial scenarios ...
```

This is simpler than adding a 4-way conditional edge router or dynamic graph topology. The no-op check is 2 lines, zero cost, and keeps the graph topology static. The standalone API (Phase C.6) works because standalone callers either don't set `iteration_number` (defaulting to 1 → runs) or explicitly set it to 1.

**Inputs consumed:** `current_code`, `current_class_name`, `goal`, `scenarios` (user-selected base scenarios), `num_adversarial_scenarios`

**Output:** `scenarios` — the original user scenarios + `N` generated adversarial scenarios appended; `weakness_analysis` (persisted in state for UI display)

**Configuration:** `num_adversarial_scenarios: int` in `RefinementState` (default: 3, user-configurable in UI). Requires adding a `num_adversarial_scenarios` parameter to `run_refinement()` and including it in `initial_state` alongside `goal`, `max_iterations`, and `scenarios`.

### C.2: Adversary prompt design

The adversary receives all necessary context to perform financial reasoning about the strategy's attack surface:

**System prompt includes:**
1. The `StrategicAgent` protocol documentation — what callbacks exist, what data the strategy can access
2. The `MarketRegime` catalog — all regime types with their financial descriptions and ABIDES mechanism descriptions (from Phase B table)
3. The 6-axis scoring system — profitability, risk-adjusted, volatility impact, spread impact, liquidity impact, execution quality
4. Instructions: "You are a quantitative risk analyst tasked with finding market conditions that expose weaknesses in a trading strategy. Analyse the strategy code, identify its assumptions, then generate adversarial scenarios that violate those assumptions."

**Human prompt includes:**
1. The strategy source code
2. The user's goal
3. The number of adversarial scenarios to generate

**Output model:**

```python
class StrategyWeaknessAnalysis(BaseModel):
    assumptions: list[str]          # market conditions the strategy implicitly relies on
    missing_controls: list[str]     # risk controls absent from the code
    order_profile: str              # aggressive/passive/mixed
    information_usage: list[str]    # which MarketState fields the strategy actually uses
    vulnerability_summary: str      # one-paragraph fragility assessment

class AdversarialScenario(BaseModel):
    regime: MarketRegime            # with rationale populated
    predicted_weak_axes: list[str]  # which scoring axes will be pressured
    attack_narrative: str           # one sentence: "X happens, causing Y"

class AdversaryOutput(BaseModel):
    weakness_analysis: StrategyWeaknessAnalysis
    scenarios: list[AdversarialScenario]
```

### C.3: Strategy weakness analysis

The adversary performs a structured code analysis via prompt (not AST). It identifies:

| Analysis dimension | What it looks for | Adversarial implication |
|---|---|---|
| **Price assumptions** | Mean-reversion logic, trend detection, fundamental value references | Mean-reversion → attack with `MOMENTUM_CASCADE`. Trend-following → attack with `MEAN_REVERSION_BREAKDOWN`. |
| **Liquidity assumptions** | Order placement without depth checks, fixed order sizes, no spread guards | Passive strategies → attack with `LOW_LIQUIDITY` or `SPREAD_WIDENING`. Large orders → attack with thin books. |
| **Risk controls** | Stop-losses, position limits, drawdown guards, time-of-day limits | Missing stop-loss → attack with `FLASH_CRASH`. No position limit → attack with sustained momentum. |
| **Order type profile** | Market vs. limit orders, aggressive vs. passive execution | Limit-heavy → attack with `INFORMED_TRADING_SURGE` (adverse selection). Market-heavy → attack with `LOW_LIQUIDITY` (slippage). |
| **Information usage** | Which `MarketState` fields are accessed (`bid_depth`, `spread`, `inventory`, etc.) | Unused information = blind spots. If no spread monitoring → attack with `SPREAD_WIDENING`. If no inventory tracking → attack with `CROWDED_TRADE_UNWIND`. |
| **Timing dependencies** | Wake-frequency assumptions, time-remaining checks | Speed-dependent → attack with `LATENCY_DEGRADATION`. No end-of-day unwinding → attack with late-session stress. |

### C.4: Scenario generation quality controls

**Diversity constraint:** The adversary must produce scenarios targeting *different* scoring axes. If all 3 scenarios attack profitability, the stress test has no breadth. The prompt instructs: "Each scenario must target a different primary scoring axis or a different combination of axes."

**Severity calibration:** Adversarial scenarios should be challenging but not degenerate. A `severity=1.0` flash crash with zero liquidity will kill any strategy — that's not a useful test. The prompt instructs: "Use severity 0.5–0.85 for most scenarios. Reserve severity > 0.85 for a single extreme scenario. The goal is to find the boundary where the strategy breaks, not to create impossible conditions."

**Redundancy check:** If a user-selected scenario already covers a regime (e.g., user selected "High Volatility" and the adversary also generates `HIGH_VOLATILITY`), the adversary should generate a different scenario or compose the regime with another dimension (e.g., `HIGH_VOLATILITY + LOW_LIQUIDITY`).

### C.5: Adversarial metadata plumbing

**Problem:** The adversary produces rich metadata per scenario (`regime_type`, `severity`, `rationale`, `predicted_weak_axes`, `attack_narrative`), but `ScenarioResult` has no fields for this. The explainer node receives `scenario_results` but *not* the original `ScenarioConfig` objects — the adversary's rationale is structurally unreachable downstream.

**Solution — extend `ScenarioResult` with adversarial fields:**

```python
class ScenarioResult(BaseModel):
    # ... existing fields ...
    # Adversarial metadata (populated by executor from ScenarioConfig.regime)
    regime_type: str | None = None
    regime_severity: float | None = None
    scenario_source: str | None = None       # "user" | "preset" | "adversarial"
    adversary_rationale: str | None = None
    predicted_weak_axes: list[str] | None = None
    attack_narrative: str | None = None
```

The executor node copies adversarial metadata from `ScenarioConfig.regime` into `ScenarioResult` during result construction. This ensures the explainer and aggregator can access it without reading the `scenarios` list separately.

**Persistence:** Adversarial scenarios are stored in `RefinementState.scenarios` with `source=ADVERSARIAL`. They persist across iterations. If the refinement session is saved, adversarial scenarios are saved alongside user scenarios.

**UI treatment:**
- Adversarial scenarios display with a distinct badge (e.g., `⚔️ Adversarial`)
- Each shows: regime type, severity, the adversary's rationale, predicted weak axes
- After execution: actual scores vs. predicted axis pressure — a "prediction accuracy" indicator
- The weakness analysis is shown once at the top of the adversarial section

**Explainer integration:** When the explainer (ReAct agent) analyses an adversarial scenario, the `ScenarioResult.attack_narrative` and `predicted_weak_axes` are included in the explainer prompt. This focuses the investigation: "The adversary predicted this strategy would fail on profitability because X. Investigate whether this prediction is accurate and explain the causal mechanism."

**Aggregator integration:** The aggregator reads adversarial metadata from `ScenarioResult`. Its cross-scenario synthesis prompt includes: "For adversarial scenarios, evaluate whether the predicted weakness materialized. If it did, recommend specific code changes to address it. If it didn't, note the strategy's resilience to that condition."

### C.6: Standalone operation path

The adversary node must be usable outside the refinement graph. In the future product:

```
User submits strategy → generate_adversarial_scenarios(code, goal, N) → list[ScenarioConfig]
                                                                          ↓
                                                          scenario_executor_node(code, scenarios)
                                                                          ↓
                                                          explainer_node(results, code)
                                                                          ↓
                                                          Stress-test report
```

**Implementation:** The core logic lives in a standalone function `generate_adversarial_scenarios(code, class_name, goal, base_scenarios, num_scenarios) -> AdversaryOutput`. The graph-integrated `adversary_node` is a thin wrapper:

```python
def adversary_node(state: RefinementState) -> dict:
    if state.get("iteration_number", 1) > 1:
        return {}
    result = generate_adversarial_scenarios(
        code=state["current_code"],
        class_name=state["current_class_name"],
        goal=state["goal"],
        base_scenarios=state.get("scenarios", []),
        num_scenarios=state.get("num_adversarial_scenarios", 3),
    )
    # Convert AdversaryOutput.scenarios to ScenarioConfig objects via RegimeTranslator
    new_configs = [RegimeTranslator().to_scenario_config(s) for s in result.scenarios]
    return {
        "scenarios": state.get("scenarios", []) + new_configs,
        "weakness_analysis": result.weakness_analysis,
    }
```

Standalone callers invoke `generate_adversarial_scenarios()` directly — no `RefinementState` construction needed. The separation ensures the adversary logic has zero coupling to graph infrastructure.

---

## Phase D — Historical Data Integration

**Goal:** Extend the adversarial repertoire to include real market events — replay actual intraday price dynamics through the oracle.

### D.1: `ExternalFileOracle` implementation

**Status (verified):** `ExternalFileOracle` does **not** exist in the `abides-rohan` fork. In `abides_markets/oracles/__init__.py`, both `DataOracle` and `ExternalFileOracle` imports are commented out, and **no source file** (`external_file_oracle.py`) exists on disk. Only `SparseMeanRevertingOracle`, `MeanRevertingOracle`, and the base `Oracle` are available.

This means Phase D requires **writing an oracle from scratch** in the `abides-rohan` fork, not merely uncommenting an import. The oracle must conform to the `Oracle` base class interface (primarily `observe_price(symbol, current_time, random_state) -> int` and provide fundamental value lookups for agents).

**Scope of fork contribution:**
1. Implement `ExternalFileOracle(Oracle)` — loads a time-series CSV, provides `observe_price()` via interpolation/nearest-lookup on the loaded data.
2. Add unit tests for the new oracle in the `abides-rohan` test suite.
3. Release a new tag (e.g., `v1.3.0-rohan`) and update `pyproject.toml` dependency.

This is a non-trivial fork contribution. The API surface is small (one class, one primary method) but correctness is critical — the oracle is the global source of truth for agent beliefs.

**`OracleSettings` extension:**

```python
class OracleType(str, Enum):
    SYNTHETIC = "synthetic"         # SparseMeanRevertingOracle (current)
    HISTORICAL = "historical"       # ExternalFileOracle (new)

class OracleSettings(BaseModel):
    oracle_type: OracleType = OracleType.SYNTHETIC
    # Synthetic oracle params (existing):
    kappa: float = 1.67e-16
    sigma_s: float = 0
    fund_vol: float = 5e-5
    megashock_lambda_a: float = 2.77778e-18
    megashock_mean: int = 1000
    megashock_var: int = 50_000
    # Historical oracle params (new):
    data_source: str | None = None   # path or event name
```

**`AbidesConfigMapper._build_oracle()` dispatch:**

```python
def _build_oracle(self, ...):
    if oracle_settings.oracle_type == OracleType.HISTORICAL:
        return ExternalFileOracle(...)
    else:
        return SparseMeanRevertingOracle(...)
```

### D.2: Curated stress event library

Bundle 5–10 notable intraday events as CSV files in a `data/events/` directory:

| Event | Date | Character | Why adversarial |
|---|---|---|---|
| Flash Crash | 2010-05-06, 14:30-15:00 ET | Rapid 6% decline + V-shaped recovery | Tests crash resilience + recovery behavior |
| GME Squeeze | 2021-01-28, 09:30-10:30 ET | Extreme upward momentum + reversal | Tests momentum sensitivity + reversal handling |
| ETF Dislocation | 2015-08-24, 09:30-10:00 ET | Circuit breakers, price gaps, illiquidity | Tests gap risk + halt behavior |
| COVID Open | 2020-03-16, 09:30-10:00 ET | Extreme volatility from open | Tests high-vol survival |
| Calm Day (baseline) | 2021-02-05 | Normal trading | Baseline for comparison |

**Data format:** CSV with columns `timestamp_ns, fundamental_value` (cents). Preprocessed to match ABIDES's `SparseMeanRevertingOracle` interface.

**Price level normalization:** Historical data is centered around `r_bar` (default 100,000 cents = $1,000) so that agent population calibration remains valid. The *shape* of the price path is preserved; the *level* is normalized.

### D.3: `MarketRegime` extension for historical events

```python
class MarketRegime(BaseModel):
    regime_type: RegimeType
    severity: float = Field(ge=0.0, le=1.0)
    description: str = ""
    rationale: str = ""
    source: ScenarioSource = ScenarioSource.USER
    oracle_source: OracleType = OracleType.SYNTHETIC     # new
    historical_event: str | None = None                   # new — event name or path
```

Historical regimes still compose with agent population overrides. "Flash Crash with reduced MM presence" = historical flash crash price path + `adaptive_market_maker.num_agents=1`.

The `RegimeTranslator` gains a `_historical_events` registry mapping event names to `(data_path, default_agent_config)` pairs.

### D.4: Historical data validation

`HistoricalDataLoader` class:
- Validates CSV format (required columns, monotonic timestamps, positive prices)
- Checks coverage: data must cover the simulation window (`start_time` to `end_time`)
- Normalizes price levels to `r_bar` scale (preserving returns, not absolute levels)
- Caches validated data for reuse across simulations

### D.5: User upload support (sub-phase, deferred)

Terminal sidebar gains a file upload widget for custom CSVs. Uploaded data is validated, previewed (price chart), and stored. Deferred until the curated library proves the concept.

---

## Verification

### Phase A
- Existing test suite (417+) passes without modification.
- Integration test: override `{"agents": {"momentum": {"num_agents": 50}}}` via deep-merged `config_override` — verify momentum agents increase to 50 while all other agent settings remain at defaults.
- Verify `wake_up_freq="2S"` and `order_book_depth=5` overrides propagate to the `StrategicAgentAdapter` construction.

### Phase B
- Each `RegimeType` at severity 0.0, 0.5, 1.0 produces valid `SimulationSettings` and completes a 30-minute simulation without crash.
- `NORMAL(severity=0.0)` produces `config_override == {}`. Non-NORMAL regimes at `severity=0.1` produce measurably different settings from defaults.
- Composed regimes (`LOW_LIQUIDITY + MOMENTUM_CASCADE`) merge without key conflicts. Composed regimes with conflicting `value.num_agents` direction resolve correctly per stress-direction rules.
- Migrated presets produce identical `SimulationSettings` to the original hardcoded dicts.

### Phase C
- End-to-end: submit a simple mean-reversion strategy → adversary generates 3 scenarios → executor runs them → adversarial scenario profitability scores are lower than the default scenario's scores on at least 2 of 3 adversarial scenarios.
- Adversary rationale references specific code patterns from the strategy (e.g., "the strategy buys when price < moving average, assuming mean reversion").
- Generated scenarios target at least 2 different primary scoring axes across the 3 scenarios.
- Adversary output validates against `AdversaryOutput` Pydantic model (structured output, not free text).

### Phase D
- Simulation using `ExternalFileOracle` with a curated CSV runs to completion and produces metrics within expected ranges.
- Price chart from the historical simulation visually matches the shape of the input CSV data.
- All 5+ curated events simulate successfully with default agent populations.
- Normalized price levels compare correctly (returns match, absolute levels centered on `r_bar`).

---

## Decisions Log

| Decision | Chosen | Over | Rationale |
|----------|--------|------|-----------|
| **Phase ordering** | A → B → C → D | Parallel or different ordering | Each phase builds on the previous. The regime layer (B) must exist before the adversary (C) can output structured scenarios. Historical data (D) extends the adversary's toolkit but isn't needed for the core mechanism. |
| **Dependency on impl plan Phases 1–3** | Hard prerequisite | Parallel development | Deterministic scoring and rich observation are load-bearing infrastructure. Building the adversary without them produces superficial, ungrounded results. |
| **Config surface scope** | Minimal targeted fixes | Full exposure of every ABIDES dial | The `RegimeTranslator` abstracts over raw parameters. Over-exposing individual knobs adds UX complexity without proportional value. The regime layer is the user-facing interface; raw params are the power-user escape hatch. |
| **MarketRegime as structured type** | Pydantic model with enum regime types | Freeform dicts or plain-text descriptions | Structured output constrains the LLM, enables deterministic translation and composition, and makes the adversary's reasoning auditable. |
| **Adversary output format** | `list[MarketRegime]` → deterministic `RegimeTranslator` | Raw `config_override` dicts generated by LLM | Separation of concerns: the LLM reasons in financial concepts (momentum cascade, liquidity drought), the translator does parameter engineering. LLMs are unreliable with precise numeric parameterisation — let them choose *what* to stress, not *how many* noise agents to use. |
| **Adversary timing** | Once per session, iteration 1 | Per-iteration or post-hoc only | Per user direction. Fixed scenarios ensure score comparability across iterations. Post-hoc becomes the product path when the refinement loop is removed. |
| **Regime composition model** | Per-agent-type stress-direction merge | Blanket min/max or sequential override | Blanket min() for agent counts breaks when composing regimes that *increase* agents adversarially (e.g., `INFORMED_TRADING_SURGE` increases value agents; composing with `LOW_LIQUIDITY` via min() would nullify the surge). Per-agent-type rules with stress-direction tags handle all composition cases correctly. |
| **Historical data priority** | Phase D (last) | Earlier or integrated from start | Synthetic regime templates cover the most valuable adversarial use cases. Historical adds realism but requires data pipeline infrastructure. The adversarial *mechanism* works without it. |
| **Product architecture** | Adversary node designed for standalone invocation | Tightly coupled to refinement graph | The refinement loop is PoC scaffolding. The adversary + explainer + scoring pipeline becomes the standalone stress-test product. Decoupled design avoids rework. |
| **New agent types** | Not in scope | Custom adversarial agents (predatory HFT, whale) | Adding agent types requires forking `abides-rohan`. The existing 5 agent types, fully parameterized via regimes, can simulate whale-like behavior (extreme momentum size), informed trading surges (value agent λ_a), and liquidity withdrawal (MM count/POV). New agent types are a separate, larger workstream. |
| **Deterministic timed events** | Via oracle megashock parameters | Custom event injection agent | ABIDES `SparseMeanRevertingOracle` mega-shocks are Poisson-distributed — not precisely timed. For the synthetic oracle, we control the *statistical character* of shocks (frequency, magnitude) but not their exact timing. Precise timing comes via historical data in Phase D. This is an acceptable trade-off: the adversary cares about regime character, not exact timestamps. |
| **`ExternalFileOracle` scope** | Build from scratch in `abides-rohan` fork | Assume it exists and uncomment | Verified: no source file exists, only commented-out imports. Phase D requires a fork contribution — small API surface but high correctness bar. |
| **`sigma_n` exposure** | Expose as explicit field (default: `r_bar / 100`) | Keep hidden coupling | Agent observation precision is a powerful adversarial lever. Hidden coupling with `r_bar` creates unintended side effects when adversarial scenarios change price levels. Explicit field enables independent control. |
