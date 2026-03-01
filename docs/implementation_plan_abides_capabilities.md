# ABIDES Capabilities Exposure — Implementation Plan

## Completed Work (Steps 1–3)

The `StrategicAgent` protocol now exposes the full range of ABIDES `TradingAgent` capabilities that are financially relevant and architecturally safe. All changes are tested (417/417 passing) and documented in `docs/technical_architecture.md` and `docs/development/ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md`.

**What was added:**
- **Situational awareness:** `mid_price`, `spread` (computed), `portfolio_value`, `unrealized_pnl`, `time_remaining_ns`, `is_market_closed`, market hours in `AgentConfig`
- **Order enrichment:** `is_hidden`, `is_post_only`, `bid_liquidity`, `ask_liquidity`, `OrderStatus.ACCEPTED` forwarding
- **Order management:** `OrderActionType` enum, `OrderAction.modify()` / `partial_cancel()` / `replace()` factories, `match`-based dispatch, 4 new adapter lifecycle overrides

**Design invariants held:** zero ABIDES imports in `strategy_api.py`, all new fields have defaults, `StrategicAgent` Protocol signature unchanged, ABIDES imports stay inside `abides_impl/`.

---

## Step 4 — Extended Capabilities (Pending Evaluation)

These capabilities are technically feasible but should only be implemented when usage data from the refinement loop justifies the added complexity.

### 4a. Oracle / Fundamental Value

**What:** Expose the ABIDES oracle's `observe_price()` as `MarketState.fundamental_value`. Gate behind `AgentConfig.oracle_access: bool = False`.

**Why wait:** Creates information asymmetry — the strategy gets a "true price" that doesn't exist in real markets. Complicates fair comparison across scenarios. Only useful if ROHAN introduces explicit "informed trader" scenario types.

**Trigger to implement:** The refinement loop generates mean-reversion or stat-arb strategies that would benefit from a fair-value anchor, and the evaluation framework supports scenario-level configuration.

### 4b. Dynamic Wake Frequency

**What:** Let the strategy return a `wake_interval_ns` to control its polling rate. Faster near events, slower during quiet periods.

**Why wait:** Meta-control capability that the current Writer prompt doesn't guide toward. Adds a new optimization axis the LLM hasn't shown a need for. The fixed 1-second wakeup is sufficient for current strategy archetypes.

**Trigger to implement:** Generated strategies are visibly bottlenecked by fixed tick frequency (e.g., missing short-lived opportunities, or wasting compute during quiet periods).

### 4c. L2 Push Subscription

**What:** Subscribe to exchange-level L2 updates so `on_market_data` fires on every order book change, not just on polling.

**Why wait:** Highest-risk change — alters event timing fundamentally. With `freq=0`, callbacks fire hundreds of times per second. LLM-generated strategies aren't designed for this frequency. Risk of simulation slowdowns and subtle bugs from confusing time-driven vs event-driven callbacks.

**Trigger to implement:** ROHAN supports latency-aware or high-frequency strategy archetypes where sub-second reaction matters.

---

## Housekeeping

- `analysis_service.py` imports `abides_markets.order_book.ns_date` — the only ABIDES import outside `abides_impl/`. Replace with stdlib datetime conversion.
- `Order.order_id` is `int` (matches ABIDES). If a FIX adapter is planned, consider migrating to `str`. Breaking change — defer unless actively needed.
