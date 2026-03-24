# ABIDES Agent Assessment

Assessment of the ABIDES agent-based simulation library used in ROHAN, evaluated after the release of `abides-rohan` v1.3.0-rohan.

**Current dependency:** `abides-rohan` v1.2.7-rohan
**Latest available:** `abides-rohan` v1.3.0-rohan
**Assessment date:** 2026-03-24

---

## 1. Agent Inventory

ROHAN uses 5 ABIDES agent types plus a custom strategic agent adapter.

| Agent | Default Count | Role | Configurability |
|---|---|---|---|
| **ExchangeAgent** | 1 | Central LOB matching engine, market hours, order routing | Pipeline/computation delay, book logging depth |
| **NoiseAgent** | 1,000 | Random liquidity providers; background volume | Count only |
| **ValueAgent** | 102 | Informed traders; mean-reversion around fundamental value | Count, `r_bar`, `kappa`, `lambda_a` |
| **AdaptiveMarketMakerAgent** | 2 | Passive liquidity provision via POV-based quoting | 13 parameters (most configurable agent) |
| **MomentumAgent** | 12 | Trend followers; momentum-based order placement | Count, `min_size`, `max_size`, `wake_up_freq` |
| **StrategicAgentAdapter** | 0–1 | User-provided strategy wrapped for ABIDES | `wake_up_freq` and `order_book_depth` **hardcoded** |

### Oracle

The simulation's fundamental price process is driven by one of two oracle types:

| Oracle | Use Case | Key Parameters |
|---|---|---|
| **SparseMeanRevertingOracle** | Synthetic (default) | `kappa`, `sigma_s`, `fund_vol`, megashock params |
| **ExternalDataOracle** | Historical / generated data | Provider type (CSV/DB/API), interpolation strategy |

---

## 2. Upstream Bug Fixes (v1.2.7 → v1.3.0)

The ABIDES fork received a substantial remediation between v1.2.7-rohan and v1.3.0-rohan. ROHAN should upgrade to benefit from these fixes.

### 2.1 Critical (simulation correctness)

| Fix | Component | Impact |
|---|---|---|
| Exchange query fall-through | `exchange_agent.py` | Queries could be silently dropped, producing stale market data |
| Subscription cancellation mutation | `kernel.py` | Modifying subscription list during iteration caused missed or double deliveries |
| `kernel.has_run` flag | `kernel.py` | Kernel could re-run without proper reset |
| TradingAgent `KeyError` / `self.symbol` | `trading_agent.py` | Crashes when accessing `known_bids`/`known_asks` before first update |
| Oracle config mutation | `sparse_mean_reverting_oracle.py` | Shared config dict mutated across agents |

### 2.2 High Priority (agent behavior)

| Fix | Component | Impact |
|---|---|---|
| Agent `wakeup()` return checks | noise, value, background agents | Agents trading before market hours known, causing `None` arithmetic |
| MomentumAgent float prices | `momentum_agent.py` | Float price calculations instead of integer cents |
| AdaptiveMarketMaker `None` mid / bare `except` | `adaptive_market_maker_agent.py` | Silent failures when midpoint unavailable; bare except swallowing real errors |
| Exchange `log_orders` inversion | `exchange_agent.py` | Log flag had inverted semantics |
| **Hardcoded "ABM" ticker removed** | `exchange_agent.py` | Exchange agent no longer requires ticker to be "ABM" |
| Kernel division by zero | `kernel.py` | Edge case in timing calculations |

### 2.3 Medium Priority

| Fix | Component | Impact |
|---|---|---|
| `order_book` `is True` vs `is not None` | `order_book.py` | Falsy-value confusion on book state checks |
| ValueAgent negative price clamp | `value_agent.py` | Could produce negative limit prices |
| Kernel `ValueError` tuples / `time=0` truthiness | `kernel.py` | Edge cases with zero-valued timestamps treated as falsy |
| AdaptiveMarketMaker integer division | `adaptive_market_maker_agent.py` | Float division where integer cents required |

### 2.4 Low Priority

| Fix | Component | Impact |
|---|---|---|
| ExternalDataOracle LINEAR interpolation memory bomb | `external_data_oracle.py` | Float precision causing massive reindex allocations |
| Megashock edge case | `sparse_mean_reverting_oracle.py` | Boundary condition in shock generation |
| AdaptiveMarketMaker `KeyError` | `adaptive_market_maker_agent.py` | Missing key access on market data dicts |
| NoiseAgent `kernel_stopping` `KeyError` | `noise_agent.py` | Crash during shutdown when no trades occurred |
| CoreBackgroundAgent wakeup return protocol | `core_background_agent.py` | Inconsistent return value from `wakeup()` |

### 2.5 Performance

| Improvement | Impact |
|---|---|
| O(log N) binary search for order book insert/cancel/modify | ~10× faster book operations at scale |
| `heapq` replaces `PriorityQueue` in kernel | Eliminates mutex overhead in single-threaded event loop |
| Subscription publishing scoped to affected symbol | Eliminates full-symbol iteration on every order event |
| 6 redundant `deepcopy()` calls removed from exchange | Reduced per-order allocation overhead |

---

## 3. Issues Resolved (Removed from Active Tracking)

These issues from previous assessments have been verified as fixed.

### 3.1 LatencyType Enum Tuple Values

**Previous issue:** Trailing commas on `LatencyType` enum values (e.g., `NO_LATENCY = ("no_latency",)`) created accidental tuples instead of strings.

**Status:** ✅ **Fixed** in ROHAN v0.2.1. All three enum values are now plain strings.

```python
# Current (correct)
NO_LATENCY = "no_latency"
DETERMINISTIC = "deterministic"
CUBIC = "cubic"
```

### 3.2 Hardcoded "ABM" in ABIDES ExchangeAgent

**Previous issue:** The ABIDES `ExchangeAgent` had hardcoded references to the ticker symbol "ABM", preventing use of custom tickers.

**Status:** ✅ **Fixed** in `abides-rohan` v1.3.0-rohan. The exchange agent now accepts arbitrary ticker symbols via the `symbols` parameter. ROHAN can remove the TODO in `simulation_settings.py` once upgraded.

### 3.3 Agent ID Hardcoded to 1

**Previous issue:** `run_with_baseline` used hardcoded agent ID `1` instead of the actual `strategic_agent_id`.

**Status:** ✅ **Fixed** in ROHAN v0.2.1. Now uses `res.result.strategic_agent_id`.

### 3.4 Missing ORDER_EXECUTED Explanations

**Previous issue:** Error path in `process_scenario_node` omitted the `explanations` key.

**Status:** ✅ **Fixed** in ROHAN v0.2.1 via `_error_explanation()` helper.

---

## 4. Open Issues (Verified Present)

These issues have been verified as still present in the current codebase.

### 4.1 Shallow Config Merge in Scenario Executor

**File:** `src/rohan/llm/nodes.py`, lines 325–326
**Severity:** High — blocks adversarial regime overrides

```python
merged = base_settings.model_dump()
merged.update(scenario.config_override)  # shallow merge
```

Nested overrides (e.g., `{"agents": {"momentum": {"num_agents": 50}}}`) clobber the entire `agents` dict, resetting all other agent settings to defaults. A recursive deep merge is required.

### 4.2 Hardcoded Strategic Agent Parameters

**File:** `src/rohan/simulation/abides_impl/abides_config_mapper.py`, lines 260–261
**Severity:** Medium — limits strategy evaluation fidelity

```python
wake_up_freq="1S",       # hardcoded 1-second tick
order_book_depth=10,     # hardcoded L2 depth
```

Both parameters are architecturally significant for strategy behavior. A `StrategicAgentSettings` model should expose them as configurable fields, especially for adversarial latency scenarios where tick frequency matters.

### 4.3 Noise Agent Market Close Hardcoded

**File:** `src/rohan/simulation/abides_impl/abides_config_mapper.py`, line 69
**Severity:** Medium — creates asymmetric simulation windows

```python
noise_mkt_close = date + str_to_ns("16:00:00")  # ignores settings.end_time
```

When the simulation window ends before 16:00 (e.g., 11:30), noise agents continue trading for hours after the strategic agent's evaluation window. This creates non-representative volume and price dynamics during the measured period.

**Recommended fix:** Compare parsed time values (not string lexicographic order) and use the later of `end_time` and `"16:00:00"`, or make `noise_mkt_close` an independently configurable field on `SimulationSettings`.

### 4.4 Value Agent sigma_n Coupled to r_bar

**File:** `src/rohan/simulation/abides_impl/abides_config_mapper.py`, line 173
**Severity:** Low — limits adversarial configurability

```python
sigma_n = value_agent_settings.r_bar / 100  # observation noise variance
```

The observation noise for value agents is derived from the fundamental price level, preventing independent control. Changing `r_bar` silently rescales agent precision. An explicit `sigma_n` field in `ValueAgentSettings` would decouple these.

### 4.5 Starting Cash Never Varied

**File:** `src/rohan/ui/utils/presets.py`, all 5 presets
**Severity:** Low — limits stress testing coverage

All presets use identical `starting_cash=10_000_000` ($100,000). Capital constraints are a meaningful stress axis (e.g., undercapitalized market makers, margin-constrained momentum agents). Including varied starting cash in adversarial scenarios would improve coverage.

### 4.6 ABIDES Dependency Outdated

**File:** `pyproject.toml`, line 134
**Severity:** High — missing critical bug fixes

```toml
abides-rohan = { git = "...", rev = "v1.2.7-rohan" }
```

The current dependency is 3 minor versions behind (v1.2.7 → v1.3.0). All fixes in §2 are unavailable. Upgrading to v1.3.0-rohan would resolve the hardcoded ABM ticker (§3.2) and numerous stability bugs.

---

## 5. Missing Agents — Simulation Realism Gaps

The current 5-agent population covers the foundational market microstructure archetypes but lacks several participant types that materially affect price dynamics in real markets.

### 5.1 High Priority — Core Realism

| Missing Agent | Market Role | Impact on Simulation Realism |
|---|---|---|
| **Institutional / Block Trader** | Large directional orders split across time (VWAP, TWAP, POV execution) | Current sim has no large persistent directional flow. Strategies are not tested against sustained buy/sell pressure that moves the market over minutes/hours. |
| **Mean-Reversion Trader** | Fades short-term price moves; provides counter-momentum | The ValueAgent trades on fundamental value, not price mean-reversion. Without a statistical arb agent, the simulation lacks the natural "rubber band" effect that dampens intraday volatility in real markets. |
| **Stop-Loss / Take-Profit Agent** | Places contingent orders triggered by price thresholds | Real markets exhibit cascading liquidation effects. Without stop-loss agents, flash crash and momentum cascade scenarios lack the critical positive feedback loop that makes these events dangerous. |

### 5.2 Medium Priority — Scenario-Specific

| Missing Agent | Market Role | Impact on Simulation Realism |
|---|---|---|
| **HFT / Predatory Trader** | Sub-millisecond latency; queue priority; momentum ignition | Strategies designed for real markets face HFT adversaries. Without predatory agents, market maker strategies appear more profitable than they would be in production. |
| **News / Event-Driven Trader** | Sudden directional burst correlated with oracle fundamental jumps | The oracle's megashock mechanism creates price jumps, but no agents explicitly react to the jump as an informed event. Adding agents that detect and trade oracle regime changes would create more realistic post-shock dynamics. |
| **Retail Flow Agent** | Small orders, arrival clustered around market open/close, tendency to buy at market | Provides the "uninformed" counterparty flow that market makers depend on. NoiseAgent partially fills this role but lacks temporal clustering patterns. |

### 5.3 Low Priority — Advanced Scenarios

| Missing Agent | Market Role | Impact on Simulation Realism |
|---|---|---|
| **Pairs / Stat-Arb Agent** | Trades relative value between correlated instruments | Requires multi-symbol support (currently single-ticker). Important for portfolio-level strategy testing. |
| **Index Arbitrage Agent** | Keeps derivative pricing consistent with underlying | Also requires multi-symbol. Relevant for strategies operating in index-heavy markets. |
| **Market Replay Agent** | Replays historical order flow instead of generating synthetic orders | Would allow hybrid backtesting: historical background flow + synthetic strategic agent. Bridges the gap between pure simulation and pure backtesting. |

---

## 6. Improvement Recommendations for Existing Agents

### 6.1 NoiseAgent

| Current Limitation | Recommendation |
|---|---|
| Single wakeup then done | Add configurable re-wakeup count or continuous trading mode for sustained background noise |
| No temporal clustering | Add open/close clustering parameter to model real intraday volume U-shape |
| Order size model is fixed | Expose `OrderSizeModel` parameters (mean, distribution) to config |

### 6.2 ValueAgent

| Current Limitation | Recommendation |
|---|---|
| `sigma_n` coupled to `r_bar` | Add independent `sigma_n` field (see §4.4) |
| Homogeneous belief | Allow per-agent `r_bar` perturbation to create informed-agent heterogeneity |
| No order-type diversity | Currently limit-orders only; add configurable market-order fraction |

### 6.3 AdaptiveMarketMakerAgent

| Current Limitation | Recommendation |
|---|---|
| No inventory risk limit | Add `max_inventory` parameter to trigger aggressive position reduction |
| No spread widening under stress | Add volatility-responsive spread component |
| POV denominator unclear | Document what "volume" the POV is measured against (last N seconds? cumulative?) |

### 6.4 MomentumAgent

| Current Limitation | Recommendation |
|---|---|
| Fixed lookback window | Add configurable lookback period for momentum signal |
| No momentum decay | Add signal half-life parameter to model momentum exhaustion |
| Homogeneous frequency | Allow per-agent frequency variation to prevent synchronized herding |

### 6.5 StrategicAgentAdapter

| Current Limitation | Recommendation |
|---|---|
| Hardcoded `wake_up_freq` / `order_book_depth` | Create `StrategicAgentSettings` model (see §4.2) |
| No latency-aware testing | Strategic agent should be configurable to experience degraded latency independently of background agents |
| No fill-latency simulation | Add configurable delay between order submission and execution acknowledgment |

---

## 7. Product Assessment

### 7.1 Dashboard Use Case (Simulation Configuration & Visualization)

The dashboard allows graphical selection, configuration, and execution of agent populations with CSV/database data injection and custom strategy metrics.

**Strengths:**
- 5 presets cover common market conditions (normal, high volatility, low liquidity, MM stress, momentum dominated)
- 13 configurable parameters on the most complex agent (AdaptiveMarketMaker)
- Historical data integration via `ExternalDataOracle` with CSV, DB, and API backends
- 6-chart visualization grid (Price, Spread, Volume, PnL, Inventory, Fills)
- Pandera schema validation on L1/L2 data outputs

**Gaps for Dashboard Use Case:**

| Gap | Priority | Effort |
|---|---|---|
| **No per-agent enable/disable toggle** — adding/removing agent types requires code changes | High | Medium (Phase 2 plugin registry) |
| **No agent-level metrics dashboard** — metrics only computed for strategic agent, not background agents | High | Medium |
| **No intraday volume profile** — volume distribution is flat, not realistic U-shaped | Medium | Low (NoiseAgent clustering) |
| **Starting cash not configurable per agent type** — all agents share same capital | Medium | Low |
| **No simulation warm-up indicator** — users can't see when the book is "ready" | Low | Low |
| **No multi-ticker support** — single instrument only | Low | High (requires ABIDES changes) |

### 7.2 Adversarial Stress Testing Use Case (Agentic Framework)

The adversarial framework reads a strategy, designs attack scenarios via LLM, runs them in ABIDES with perfect observability, and provides AI-driven insights.

**Strengths:**
- 10 `MarketRegime` types designed and documented (normal through crowded-trade-unwind)
- LLM adversary performs structured code analysis to identify strategy assumptions
- 6-axis deterministic scoring (profitability, risk-adjusted, volatility, spread, liquidity, execution quality)
- ReAct explainer agent with 8 investigation tools and full simulation data access
- Per-scenario seeds for reproducibility

**Gaps for Adversarial Use Case:**

| Gap | Priority | Effort |
|---|---|---|
| **Shallow config merge blocks regime overrides** (§4.1) | Critical | Low (implement deep merge) |
| **RegimeTranslator not implemented** — regime-to-parameter mapping exists in docs but not in code | High | Medium |
| **No stop-loss cascade agents** — flash crash scenarios lack positive feedback loops | High | Medium (new agent type) |
| **No latency degradation per-agent** — latency model applies uniformly, can't selectively degrade strategy agent | Medium | Medium |
| **No capital constraint stress** — starting cash never varied (§4.5) | Medium | Low |
| **No market impact measurement** — can't isolate the strategic agent's contribution to price movement | Medium | Medium |
| **No order-flow toxicity metrics** — VPIN, adverse selection measurement not computed for background flow | Low | Medium |

---

## 8. Recommended Upgrade Path

### Immediate (v0.3.0)

1. **Upgrade `abides-rohan` to v1.3.0-rohan** — unlocks all §2 fixes; resolves hardcoded ABM ticker
2. **Implement deep merge in `scenario_executor_node`** — unblocks adversarial regime overrides
3. **Remove resolved TODO** in `simulation_settings.py` (ticker no longer hardcoded in ABIDES)

### Near-Term (v0.4.0)

4. **Create `StrategicAgentSettings`** — expose `wake_up_freq`, `order_book_depth` to config
5. **Add `sigma_n` field** to `ValueAgentSettings` — decouple from `r_bar`
6. **Fix `noise_mkt_close`** — align with simulation `end_time`
7. **Implement `RegimeTranslator`** — translate `MarketRegime` enums to ABIDES parameter overrides
8. **Add `StopLossAgent`** — critical for realistic flash crash scenarios

### Medium-Term

9. **Agent plugin registry** (Phase 2 from `scenario_customizability_plan.md`) — zero-code agent addition
10. **Institutional block trader agent** — sustained directional flow
11. **Per-agent starting cash** — capital constraint stress axis
12. **Background agent metrics** — compute metrics for all agents, not just strategic
