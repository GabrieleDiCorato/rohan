# ABIDES Capabilities Exposure — Analysis & Implementation Plan

## Executive Summary

This document presents an independent evaluation of which ABIDES `TradingAgent` capabilities should be exposed through ROHAN's `StrategicAgent` protocol. The analysis is grounded in three criteria: **financial relevance** (does this help AI agents write better strategies?), **architectural integrity** (does ROHAN stay decoupled from ABIDES?), and **complexity budget** (is the added surface area justified by concrete value?).

The current interface (`MarketState`, `OrderAction`, `Order`, `AgentConfig`) is minimal but functional. It supports L1/L2 market data, limit and market orders, simple cancellations, and partial/full fill callbacks. What it lacks are situational awareness fields that force LLM-generated strategies to perform manual calculations (mid-price, PnL, time horizon) and advanced order management capabilities present in real markets.

All capabilities referenced below have been verified against the installed `abides-rohan` v1.2.6. Every method, message type, and callback exists.

---

## Assessment Dimensions

### 1. Financial Markets Perspective

The current API gives a strategy L2 book data, inventory, and cash — enough for a basic market maker or momentum trader. However, professional trading systems universally expose:

- **Mark-to-market PnL** — Without this, the strategy cannot evaluate its own performance mid-simulation. Every real trading desk computes this continuously.
- **Time horizon** — Market close awareness is critical for inventory management. A strategy that doesn't know when the market closes cannot implement end-of-day flattening, TWAP/VWAP scheduling, or urgency-adjusted pricing. This is a fundamental gap.
- **Mid-price and spread** — These are the two most referenced quantities in market microstructure. The LLM currently must compute `(best_bid + best_ask) // 2` and `best_ask - best_bid` in every strategy. Pre-computing them eliminates a frequent source of LLM bugs (e.g., None-handling, off-by-one in integer division).
- **Order modification** — Real markets support modify/replace operations. Without them, strategies must cancel-and-replace manually, introducing a time window where they have no resting order (execution risk). For market-making strategies, this is a significant handicap.
- **Hidden and post-only orders** — These are standard order types on most electronic venues. Post-only orders are essential for maker/taker fee management. Hidden (iceberg) orders reduce information leakage for large positions.

**Verdict:** The enrichments proposed are financially sound. The priorities should follow the information hierarchy: awareness fields first (what do I know?), then action capabilities (what can I do?).

### 2. Technical Soundness

The current architecture is clean. ABIDES imports are confined to `src/rohan/simulation/abides_impl/`, and strategy code only imports from `rohan.simulation.models.strategy_api`. This is a well-designed anti-corruption layer.

**Key technical observations:**

- **Computed fields (`mid_price`, `spread`)** use `@computed_field` — zero adapter changes, pure derivations from existing data. This is correct and idiomatic Pydantic v2.
- **`mark_to_market`** is a method on `TradingAgent` that uses ABIDES-internal `last_trade` and `holdings` caches. Calling it from `_build_market_state()` is safe — no new ABIDES types leak through.
- **Time awareness** has a subtle initialization ordering issue: `mkt_open`/`mkt_close` are `None` until the first wakeup-response cycle completes. The plan correctly accounts for this by making the fields `Optional`. Strategies will receive `None` on first tick, then valid values. This mirrors real exchange behavior (market hours are communicated, not assumed).
- **`order_accepted` forwarding** is technically simple but doubles callback volume. Since LLM strategies typically return `[]` for uninteresting events, the overhead is negligible. However, it should be opt-in or documented clearly so the LLM doesn't generate reactive logic on every acceptance.
- **Order modification (modify/partial_cancel/replace)** requires careful `_execute_actions` refactoring. The existing dispatch is a simple if/elif; switching to `match` on `OrderActionType` is cleaner. The backward-compatibility auto-inference validator is sound but adds a subtle implicit behavior that should be well-documented.
- **L2 push subscription** introduces a second data path (`handle_market_data` vs `receive_message`). This is the highest-risk change because it alters the event timing model — strategies would receive callbacks at exchange-event frequency, not wakeup frequency. For LLM-generated code this could cause unexpected behavior.
- **Oracle access** couples to `self.kernel.oracle` — this reaches into ABIDES kernel internals. It works, but it's the most ABIDES-specific change. Gating it behind a flag is correct.

**Verdict:** Technically viable. The anti-corruption layer remains intact for all changes. Risk scales with the change tier — enrichment fields are trivial, order management is moderate, L2 subscription is high.

### 3. Interface Complexity

The current API surface is deliberately small: 5 models, 5 protocol methods, 2 enums. This is LLM-friendly — the Writer prompt can fully describe the contract. Every addition increases the prompt length and the probability of LLM confusion.

**Complexity assessment per change:**

| Change | New Types/Fields | Prompt Impact | LLM Cognitive Load |
|--------|-----------------|---------------|---------------------|
| `mid_price`, `spread` (computed) | 2 read-only properties | Minimal — LLM can ignore them | Very Low |
| `portfolio_value`, `unrealized_pnl` | 2 fields | Minimal | Very Low |
| `time_remaining_ns`, market hours | 1 field + 2 config fields | Low — one sentence | Low |
| `is_hidden`, `is_post_only` | 2 boolean flags | Low — defaults to False | Low |
| `bid_liquidity`, `ask_liquidity` | 2 fields | Low | Low |
| `OrderActionType` enum + factories | 1 enum (6 values), 3 factories | **Moderate** — new concept | **Moderate** |
| `order_accepted` forwarding | 1 new `OrderStatus` value | Low | Low |
| `fundamental_value` + oracle | 1 field + 1 config flag | Low | Low |
| `is_market_closed` | 1 boolean flag | Low | Low |
| `wake_interval_ns` | 1 field + 1 factory | **Moderate** — meta-control | **Moderate** |
| L2 push subscription | 0 (adapter-internal) | None | None |

**Key concern:** The `OrderActionType` enum is the single largest complexity increase. It transforms `OrderAction` from a simple instruction into a discriminated union. While this is a clean pattern, the LLM must understand 6 action types instead of 2 (place or cancel). The convenience factories (`OrderAction.modify(...)`, `OrderAction.replace(...)`) mitigate this.

**Verdict:** Enrichment fields (Tier 1) have negligible complexity cost. Order management (Tier 2) has moderate cost, justified by the capabilities it unlocks. Dynamic wake frequency and L2 subscription (Tier 3) should be deferred — they serve advanced use cases that the current AI loop doesn't exercise.

### 4. ABIDES Coupling

#### 4.1 Import-Level Coupling (Current State)

ABIDES imports exist in exactly 4 files, all inside `abides_impl/`:

| File | ABIDES Imports |
|------|----------------|
| `strategic_agent_adapter.py` | `TradingAgent`, `QuerySpreadResponseMsg`, `LimitOrder`, `Side`, `str_to_ns` |
| `abides_config_mapper.py` | Agents, oracle, models, utilities |
| `abides_output.py` | `parse_logs_df`, `ExchangeAgent`, `OrderBook` |
| `simulation_runner_abides.py` | `abides_core.abides` |

Plus one leak: `analysis_service.py` imports `abides_markets.order_book.ns_date` (timestamp conversion utility).

**Coupling impact of each proposed change:**

- **Computed fields, portfolio value, time, liquidity, market close flag:** No new ABIDES imports. These use data already exposed by `TradingAgent` through existing caches (`holdings`, `known_bids`, `mkt_close`). **Coupling: unchanged.**
- **Hidden/post-only, order modification:** Uses `place_limit_order()`, `modify_order()`, `partial_cancel_order()`, `replace_order()`, `create_limit_order()` — all methods on the parent class. The adapter already inherits from `TradingAgent`. **Coupling: unchanged** (no new imports, just calling more methods on the same base class).
- **Oracle access:** Introduces a runtime reach into `self.kernel.oracle` and calls `observe_price()`. This is a new coupling point to ABIDES kernel internals, but it's guarded by a flag, wrapped in try/except, and only used within `_build_market_state()`. **Coupling: slightly increased**, contained.
- **L2 subscription:** Requires importing `L2SubReqMsg` and overriding `handle_market_data()`. **Coupling: moderately increased** — a new ABIDES message type enters the adapter.

#### 4.2 Semantic Coupling — Does the Protocol Reflect ABIDES or Markets?

Import-level isolation is necessary but insufficient. The deeper question is whether `strategy_api.py` — the protocol that strategies code against — mirrors **ABIDES-specific semantics** or **general market semantics**. If the protocol is just a thin translation of ABIDES's `TradingAgent`, then a second adapter (e.g., FIX gateway, live broker API) would find the interface awkward or impossible to implement.

**Assessment of each protocol concept:**

| Protocol Concept | ABIDES-Specific? | Industry Standard? | FIX Equivalent |
|------------------|-------------------|---------------------|----------------|
| `Side` (BID/ASK) | No | Yes — universal | FIX tag 54 (`Side`): 1=Buy, 2=Sell |
| `OrderType` (LIMIT/MARKET) | No | Yes — FIX OrdType | FIX tag 40: 1=Market, 2=Limit |
| `OrderStatus` (NEW/PARTIAL/FILLED/CANCELLED/REJECTED) | No | Yes — matches FIX OrdStatus | FIX tag 39: 0=New, 1=PartialFill, 2=Filled, 4=Cancelled, 8=Rejected |
| `Order` fields (id, symbol, side, qty, price, status, filled_qty, fill_price) | No | Yes — FIX ExecutionReport | FIX tags 37, 55, 54, 38, 44, 39, 14, 31 |
| `MarketState` (L1 bid/ask, L2 depth, last trade, inventory, cash) | No | Yes — standard market data + position snapshot | FIX MarketDataSnapshotFullRefresh (tag 35=W) + PositionReport (tag 35=AP) |
| `OrderAction` (place/cancel with price/qty/side) | No | Yes — FIX NewOrderSingle + OrderCancelRequest | FIX tag 35=D (new), 35=F (cancel) |
| `cancel_order_id` sentinel for cancel | Slightly | Partial — FIX uses `OrigClOrdID` (tag 41) for cancels, but doesn't have a "cancel all" sentinel | FIX MassCancelRequest (tag 35=q) exists but is structurally separate |
| `AgentConfig` (starting_cash, symbol, latency_ns) | Partially | `starting_cash` is simulation-specific; real brokers don't pass this at session init | No direct FIX equivalent — session setup is different |
| Prices in integer cents | No | Yes — FIX uses `LastPx` (float), but integer-cents is common in practice (e.g., CME) | Adapter would convert float ↔ int |
| Timestamps in nanoseconds | Partial | FIX uses UTC timestamps (tag 60), not ns-since-epoch. But ns precision is standard in modern exchanges | Adapter would convert |
| `on_tick` / `on_market_data` split | Slightly ABIDES-flavored | Conceptually common (timer vs event), but FIX doesn't have "wakeup" — a FIX strategy would be purely event-driven | FIX adapter would drive `on_market_data` from MarketDataIncrementalRefresh; `on_tick` from an internal timer |
| `on_simulation_end` | Simulation-specific | No FIX equivalent — live systems don't "end" | A FIX adapter could call this on session logout or EOD |
| Proposed: `is_hidden`, `is_post_only` | No | Yes — FIX `DisplayMethod` (tag 1084), `ExecInst` (tag 18)='6' (post-only in some venues) | Direct mapping exists |
| Proposed: `modify` / `replace` | No | Yes — FIX OrderCancelReplaceRequest (tag 35=G) is the standard mechanism | FIX `replace` maps cleanly; ABIDES `modify` has no exact FIX equivalent (FIX only does cancel-replace) |
| Proposed: `partial_cancel` | Slightly | Rare — some venues support qty reduction via amend, but FIX doesn't have a dedicated message for it | Would map to FIX OrderCancelReplaceRequest with reduced qty |
| Proposed: `portfolio_value`, `unrealized_pnl` | No | Yes — standard risk/PMS fields | No FIX market data equivalent, but all broker APIs provide this |
| Proposed: `time_remaining_ns` | No | Yes — market hours are public data | Derivable from exchange session schedule |
| Proposed: `fundamental_value` (oracle) | **Yes — ABIDES-specific** | **No** — real markets don't have an oracle. This is a simulation concept | **No FIX mapping.** A FIX adapter would leave this field as `None` permanently |
| Proposed: `wake_interval_ns` | **Yes — ABIDES-specific** | **No** — this is a simulator scheduling concept. In live markets, the strategy controls its own timer | A FIX adapter would need to implement its own timer loop. Doable, but the ns granularity is simulation-flavored |
| Proposed: L2 push subscription | No (adapter-internal) | Yes — FIX MarketDataRequest (tag 35=V) with SubscriptionRequestType=1 (snapshot+updates) | Clean mapping |

**Summary of semantic coupling:**

The core protocol (`MarketState`, `OrderAction`, `Order`, `StrategicAgent`) is essentially a simplified version of standard electronic trading concepts. There is no ABIDES-specific vocabulary leaking into these models. The naming (`BID`/`ASK` rather than ABIDES's internal class names, `LIMIT`/`MARKET` rather than `LimitOrder`/`MarketOrder` classes) confirms that the abstraction is at the right level.

**Concepts that are genuinely ABIDES-flavored:**

1. **`fundamental_value` (oracle)** — This has no real-market equivalent. It is an information-theoretic construct unique to agent-based simulation. A FIX adapter would always return `None`. This is acceptable because it's opt-in (`oracle_access=False` by default), but it means the field is meaningless outside ABIDES.

2. **`wake_interval_ns`** — The concept of a strategy controlling its own polling frequency is a simulator artifact. In a live system, the strategy runs its own event loop. A FIX adapter could honor this as a timer interval, but the semantics are different: in ABIDES, the kernel controls scheduling; in FIX, the strategy is autonomous. This leaks the ABIDES execution model.

3. **`CANCEL_ALL` sentinel (`-1`)** — The use of a magic integer rather than a separate action type is a minor design smell. FIX has `MassCancelRequest` as a distinct message type. The proposed `OrderActionType.CANCEL_ALL` fixes this by making it explicit.

4. **`AgentConfig.starting_cash`** — Simulation-specific. A live broker adapter wouldn't know starting cash at session init. However, this is initialization-time configuration, not runtime state, so it doesn't affect the trading loop.

5. **`on_simulation_end`** — By name, this is simulation-specific. A FIX adapter could map it to session close or EOD. The semantics ("no further orders, compute final metrics") are transferable.

**Concepts that map cleanly to FIX/live trading:**

Everything in Steps 1-3 of the implementation plan, except `fundamental_value`, maps directly to FIX or standard broker API concepts. Specifically:
- `mid_price`, `spread` → derived from L1 data in any venue
- `portfolio_value`, `unrealized_pnl` → standard PMS data
- `time_remaining_ns` → derivable from exchange session schedule
- `is_market_closed` → FIX TradingSessionStatus (tag 35=h)
- `is_hidden`, `is_post_only` → FIX order qualifiers
- `modify`/`replace` → FIX OrderCancelReplaceRequest
- `partial_cancel` → FIX amend with reduced quantity
- `ACCEPTED`/`MODIFIED`/`REPLACED` statuses → FIX ExecutionReport states
- `bid_liquidity`, `ask_liquidity` → derivable from any L2 feed

#### 4.3 FIX Adapter Feasibility Assessment

**Could a `FixGatewayAdapter` implement the `StrategicAgent` protocol as a second backend?**

Yes, with some caveats. The adapter would:

| Protocol Method | FIX Implementation |
|-----------------|-------------------|
| `initialize(config)` | FIX Logon (35=A), ignore `starting_cash` (query from broker), map `symbol` to FIX SecurityID |
| `on_tick(state)` | Internal timer fires, build `MarketState` from cached L1/L2 + position query |
| `on_market_data(state)` | FIX MarketDataIncrementalRefresh (35=X) triggers rebuild of `MarketState` |
| `on_order_update(order)` | FIX ExecutionReport (35=8) maps to `Order` with status translation |
| `on_simulation_end(state)` | FIX Logout (35=5) or EOD event, build final `MarketState` from last cache |
| `_execute_actions` | `PLACE` → NewOrderSingle (35=D), `CANCEL` → OrderCancelRequest (35=F), `REPLACE` → OrderCancelReplaceRequest (35=G), `MODIFY` → same as REPLACE (FIX has no in-place modify), `PARTIAL_CANCEL` → OrderCancelReplaceRequest with reduced qty |

**Fields that would not be populated by a FIX adapter:**
- `fundamental_value` → always `None` (no oracle in real markets)
- `starting_cash` in `AgentConfig` → would need to be queried from broker or set externally

**Fields that change semantics:**
- `wake_interval_ns` → a FIX adapter would implement this as an application-level timer, not kernel-scheduled wakeup. The behavior is equivalent but the control mechanism differs
- `timestamp_ns` → FIX timestamps are UTC microseconds; conversion to ns-since-epoch is trivial

**Structural concerns for a FIX adapter:**
- The `StrategicAgent` protocol is **synchronous** — each callback returns a list of actions immediately. This works in ABIDES (single-threaded event loop) and would work in a FIX adapter (execute actions after each event). No async/await is needed.
- The `MarketState` snapshot model (immutable, rebuilt each callback) is compatible with both backends. A FIX adapter would maintain internal caches and build the snapshot on each event, exactly as the ABIDES adapter does.
- The `Order.order_id` is an `int`. FIX uses `ClOrdID` (string, tag 11). A FIX adapter would need an internal mapping between integer IDs and string ClOrdIDs. This is a minor friction point but not a blocker.

**Verdict on FIX feasibility:** The protocol is portable. A FIX adapter is implementable without changing `strategy_api.py`. The two ABIDES-specific concepts (`fundamental_value`, `wake_interval_ns`) are opt-in and defaulted to `None`/not-used, so a FIX adapter simply ignores them. The `CANCEL_ALL` sentinel is cleaned up by the proposed `OrderActionType` enum. The `Order.order_id` int-vs-string mismatch is the only structural friction, and it's easily bridged with an internal mapping.

**Recommendation:** To future-proof the protocol for a FIX adapter:
1. Keep `fundamental_value` gated behind `oracle_access` (Step 4a). A FIX adapter would never set this flag.
2. If `wake_interval_ns` is implemented, document it as "advisory" — the backend may implement it differently or ignore it.
3. The `ns_date` import leak in `analysis_service.py` should be fixed — replace with a stdlib datetime conversion. This is unrelated to the current plan but is the only ABIDES import outside `abides_impl/`.
4. Consider whether `Order.order_id` should be `str` instead of `int` to align with FIX `ClOrdID`. This is a breaking change and should only be considered if a FIX adapter is actively planned. For now, `int` is fine — ABIDES uses integer order IDs natively.

### 5. Value for the AI Agent Loop

ROHAN's value proposition is the iterative refinement loop: Write → Validate → Simulate → Explain → Judge → Refine. The question is: **which changes most improve the quality of strategies the LLM generates and the quality of explanations the Explainer produces?**

**High-impact for strategy generation:**
- `mid_price`, `spread` → eliminates the #1 source of LLM arithmetic bugs
- `portfolio_value`, `unrealized_pnl` → enables PnL-aware strategies (e.g., "stop trading if drawdown exceeds X")
- `time_remaining_ns` → enables time-structured execution (TWAP, end-of-day flattening)
- `is_market_closed` → prevents "strategy died with open positions" failure mode

**High-impact for explanation quality:**
- `portfolio_value` and `unrealized_pnl` → the Explainer tools already compute PnL post-hoc, but having it in the strategy state means the Explainer can analyze *when the strategy knew it was losing* vs *when it actually was*
- `OrderStatus.ACCEPTED/MODIFIED` → richer order lifecycle gives the Explainer more events to analyze

**Low-impact for current loop:**
- L2 subscription → the Explainer doesn't distinguish between poll-based and push-based data arrival
- Dynamic wake frequency → interesting for latency-sensitive strategies, but the current Writer prompt doesn't guide the LLM toward frequency adaptation
- Oracle access → creates an asymmetric information advantage that complicates fair evaluation across scenarios

---

## Prioritized Implementation Plan

Based on the analysis above, changes are organized into four steps. Each step is independently shippable and backward-compatible. The principle: **maximize information before expanding actions**.

### Step 1 — Situational Awareness (Low Risk, High Value)

**Goal:** Give strategies enough information to make intelligent decisions without manual computation.

**Changes:**

1. **`mid_price` and `spread` computed fields on `MarketState`** — Pure derivations via `@computed_field`. Zero adapter work. Eliminates the most common LLM arithmetic mistake.

2. **`portfolio_value` and `unrealized_pnl` on `MarketState`** — Adapter calls `self.mark_to_market(self.holdings)`. Enables PnL-aware strategies (risk limits, drawdown stops). The Explainer can correlate strategy decisions with PnL trajectory.

3. **`time_remaining_ns` on `MarketState`** and **`mkt_open_ns`/`mkt_close_ns` on `AgentConfig`** — Adapter reads `self.mkt_close` (available after first wakeup cycle). Fields are `Optional` to handle pre-market state. Enables TWAP, urgency pricing, end-of-day logic.

4. **`is_market_closed` flag on `MarketState`** — Override `market_closed()` in adapter to send a final `on_tick` with the flag set. Non-breaking, strategies that don't check it are unaffected. Prevents the "strategy held position through close" failure mode.

**Affected files:** `strategy_api.py` (model additions), `strategic_agent_adapter.py` (4 adapter changes in `_build_market_state` + 1 new override).

**Testing:**
- Unit: computed field edge cases (None inputs, crossed book), default values for new fields
- Integration: verify `portfolio_value > 0` after trades, `time_remaining_ns` decreases, `is_market_closed` fires

**Prompt update:** Add 4 lines to `WRITER_SYSTEM` describing the new fields. Update `EXPLAINER_SYSTEM` to mention PnL-aware analysis.

**Risk:** None. All fields have defaults; existing strategies pass unchanged.

---

### Step 2 — Order Type Enrichment (Low Risk, Moderate Value)

**Goal:** Expose standard order types that enable more sophisticated execution strategies.

**Changes:**

1. **`is_hidden` and `is_post_only` boolean flags on `OrderAction`** — Both default to `False`. Adapter passes them through to `place_limit_order()` which already supports them. Add a `@model_validator` warning when used with `MARKET` orders.

2. **`bid_liquidity` and `ask_liquidity` on `MarketState`** — Adapter calls `self.get_known_liquidity(self.symbol, within=0.005)`. Gives strategies a sense of market depth beyond the L2 book snapshot. Useful for impact-aware sizing.

3. **`OrderStatus.ACCEPTED` + forward `order_accepted` callback** — The adapter already handles this event internally. Forwarding to the strategy as an `on_order_update` provides confirmation that an order entered the book. Useful for strategies that track order lifecycle states.

**Affected files:** `strategy_api.py` (2 fields on `OrderAction`, 2 fields on `MarketState`, 1 enum value), `strategic_agent_adapter.py` (pass flags in `place_limit_order`, compute liquidity, forward `order_accepted`).

**Testing:**
- Unit: `is_hidden`/`is_post_only` validation with MARKET orders
- Integration: hidden order flows through, `ACCEPTED` status received, liquidity fields populated

**Risk:** Low. `order_accepted` forwarding doubles callback frequency, but LLM strategies return `[]` for unhandled events. Monitor simulation performance with a benchmark test.

---

### Step 3 — Advanced Order Management (Moderate Risk, High Value)

**Goal:** Enable order modification semantics that are standard in electronic markets.

**Changes:**

1. **`OrderActionType` enum** (`PLACE`, `CANCEL`, `CANCEL_ALL`, `MODIFY`, `PARTIAL_CANCEL`, `REPLACE`) — Discriminator for `OrderAction`. The default is `PLACE` to preserve backward compatibility.

2. **`new_price` and `new_quantity` fields on `OrderAction`** — Used by `MODIFY` and `REPLACE` actions. `new_quantity` is also the reduction amount for `PARTIAL_CANCEL`.

3. **Convenience factories:** `OrderAction.modify(order_id, ...)`, `OrderAction.partial_cancel(order_id, reduce_by)`, `OrderAction.replace(order_id, ...)` — These are the primary interface the LLM should use. They enforce correct field combinations.

4. **Backward-compatibility validator** — Auto-infers `action_type` from `cancel_order_id` when `action_type` is still `PLACE`. Old-style `OrderAction(cancel_order_id=123)` continues to work.

5. **New `OrderStatus` values:** `MODIFIED`, `PARTIAL_CANCELLED`, `REPLACED` — Reported via `on_order_update`.

6. **Adapter overrides:** `order_modified()`, `order_partial_cancelled()`, `order_replaced()` — Update internal cache and forward to strategy.

7. **Refactored `_execute_actions()`** — `match` on `action_type` instead of if/elif on `cancel_order_id`. New handlers: `_handle_modify`, `_handle_partial_cancel`, `_handle_replace`, `_handle_place`.

**Affected files:** `strategy_api.py` (1 enum, 3 fields, 3 factories, 2 validators, 3 status values), `strategic_agent_adapter.py` (3 new overrides, refactored dispatch, 4 new handler methods).

**Testing:**
- Unit: factory validation (modify requires at least one of price/quantity), backward compat, action type inference
- Integration: full modify flow (place → modify → MODIFIED callback), partial cancel flow, replace flow
- Backward compat: run full existing test suite — must pass without changes

**Risk:** Moderate. This is the largest single change. The `_execute_actions` refactoring touches the core dispatch path. Mitigation: implement the enum and validators first (pure model changes), test exhaustively, then wire the adapter.

**Implementation note:** Consider splitting this into two sub-steps:
- 3a: Add the enum, fields, factories, validators, and status values (model-only, no adapter changes)
- 3b: Wire the adapter overrides and dispatch

---

### Step 4 — Extended Capabilities (Evaluate Before Implementing)

These changes are technically valid but should be evaluated against actual usage patterns before implementation. **Do not implement these in the initial rollout.** Revisit after the first round of refinement-loop runs with Steps 1-3.

#### 4a. Oracle / Fundamental Value (Conditional)

**Value:** Enables strategies that incorporate fair-value estimates. Interesting for mean-reversion and statistical arbitrage strategies.

**Concern:** Creates an information asymmetry that complicates fair evaluation across scenarios. The strategy would have privileged information not available to a real trader. If the evaluation framework compares oracle-enabled vs non-oracle strategies, the comparison is unfair.

**Recommendation:** Implement only if the project introduces *scenario types* where oracle access is an explicit parameter (e.g., "informed trader" scenarios). Gate behind `AgentConfig.oracle_access` flag.

#### 4b. Dynamic Wake Frequency (Conditional)

**Value:** Enables adaptive polling — faster near market events, slower during quiet periods. Relevant for latency-aware market making.

**Concern:** Meta-control capability that adds a new axis of strategy design. The Writer prompt would need to explain frequency adaptation, and the Explainer would need to analyze timing patterns. Current loop doesn't benefit from this.

**Recommendation:** Implement when the refinement loop demonstrates strategies that are bottlenecked by fixed tick frequency. Until then, the fixed 1-second wakeup is sufficient.

#### 4c. L2 Push Subscription (Deferred)

**Value:** Event-driven market data instead of polling. More realistic for high-frequency strategies.

**Concern:** Highest-risk change. Alters the fundamental event timing model. With `freq=0`, the strategy receives a callback on every order book change — potentially hundreds per second. LLM-generated strategies are not designed for this frequency. The callback volume could cause simulation slowdowns, and the strategy would need to distinguish between time-driven (`on_tick`) and event-driven (`on_market_data`) callbacks, which is a subtle concept for generated code.

**Recommendation:** Defer until ROHAN supports latency-aware or high-frequency strategy archetypes. The current polling model is appropriate for the second-scale strategies the LLM generates.

---

## Affected Files Summary

| File | Steps | Nature of Changes |
|------|-------|-------------------|
| `src/rohan/simulation/models/strategy_api.py` | 1, 2, 3 | New fields, computed properties, enum values, factories, validators |
| `src/rohan/simulation/abides_impl/strategic_agent_adapter.py` | 1, 2, 3 | `_build_market_state` enrichment, new lifecycle overrides, refactored dispatch |
| `src/rohan/llm/prompts.py` | 1, 2, 3 | Update Writer and Explainer system prompts to document new capabilities |
| `tests/test_strategy_api_validation.py` | 1, 2, 3 | Unit tests for all new model behavior |
| `tests/test_strategic_agent.py` | 1, 2, 3 | Integration tests with new strategy helpers |

Files that require **no changes:** `strategy_validator.py` (whitelist already covers `strategy_api`), `simulation_settings.py` (no new settings needed at this stage).

---

## Design Invariants (Must Hold After Every Step)

1. **No ABIDES types in `strategy_api.py`** — Protocol models import only from stdlib, `pydantic`, and `typing`.
2. **All new fields have defaults** — Existing strategies compile and run without modification.
3. **`StrategicAgent` Protocol is not extended** — No new methods. New information flows through existing callbacks via enriched models.
4. **ABIDES imports stay inside `abides_impl/`** — The anti-corruption boundary is preserved.
5. **Every step is independently shippable** — No step depends on a later step. Partial implementation is valid.

---

## Testing Strategy

### Per-Step Validation

After each step:
1. Run `pytest tests/` — all existing tests must pass (backward compatibility).
2. Run new unit tests for added model behavior.
3. Run new integration tests with purpose-built strategy helpers.
4. Run one full refinement loop (`run_refinement`) to verify the Writer prompt update produces strategies that use the new capabilities.

### New Test Cases by Step

**Step 1:**

| Test | Validates |
|------|-----------|
| `test_mid_price_both_sides` | `mid_price` = `(bid + ask) // 2` |
| `test_mid_price_one_side_none` | `mid_price` = `None` when either side missing |
| `test_spread_crossed_book` | `spread` is negative when `bid > ask` |
| `test_portfolio_value_defaults` | Defaults to 0, Pydantic validates |
| `test_time_remaining_populated` | Decreases across ticks in integration test |
| `test_market_closed_final_tick` | `is_market_closed` is `True` in the last tick |

**Step 2:**

| Test | Validates |
|------|-----------|
| `test_hidden_market_order_warns` | `is_hidden=True` with `MARKET` raises or warns |
| `test_post_only_passthrough` | `is_post_only=True` reaches `place_limit_order` |
| `test_liquidity_fields_populated` | `bid_liquidity > 0` when book has orders |
| `test_order_accepted_callback` | Strategy receives `OrderStatus.ACCEPTED` |

**Step 3:**

| Test | Validates |
|------|-----------|
| `test_modify_factory_validation` | `modify()` requires at least one of `new_price`, `new_quantity` |
| `test_partial_cancel_factory` | `partial_cancel(id, 50)` creates correct action |
| `test_replace_factory` | `replace(id, side, qty, price)` creates correct action |
| `test_action_type_inferred_cancel` | `OrderAction(cancel_order_id=5)` → `CANCEL` |
| `test_action_type_inferred_cancel_all` | `OrderAction(cancel_order_id=-1)` → `CANCEL_ALL` |
| `test_backward_compat_old_cancel` | Old-style cancellation still works end-to-end |
| `test_modify_order_integration` | Place → modify → `MODIFIED` callback |
| `test_partial_cancel_integration` | Place 100 → partial cancel 50 → `PARTIAL_CANCELLED` |
| `test_replace_order_integration` | Place → replace → `REPLACED` on old, new order active |
