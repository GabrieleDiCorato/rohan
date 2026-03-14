# ABIDES Custom Agent Implementation Guide

This guide is optimized for AI coding agents implementing custom trading strategies in ABIDES.
Since the ABIDES source code is available in your environment, this guide focuses on **architecture, API contracts, and critical behaviors**, avoiding redundant code snippets.

**CRITICAL PREREQUISITE:** Before building any agent, you **MUST** read [`ABIDES_LLM_INTEGRATION_GOTCHAS.md`](./ABIDES_LLM_INTEGRATION_GOTCHAS.md) to understand how market data is asynchronously populated (and avoids `None`/`KeyError` crashes).

---

## 1. Core Architecture & Assumptions

### 1.1 Time & Pricing
- **Time:** Discrete-event simulation. All time variables are integer **nanoseconds**. There are no datetime objects.
- **Prices:** All prices are integer **cents** ($100.00 = 10_000). Never use floats.
- **Holdings:** Cash is tracked in integer cents (`self.holdings["CASH"]`). Asset holdings are integer share counts (`self.holdings["SYMBOL"]`).

### 1.2 Event-Driven Execution (No Loops)
ABIDES agents do not iterate in a loop (no `while True:` or `on_bar()`).
Execution is solely driven by two callbacks:
1. `wakeup(current_time)`: Triggered by the kernel at an agent's scheduled time.
2. `receive_message(current_time, sender_id, message)`: Triggered when a message (market data, order fill, etc.) arrives.

---

## 2. The Recommended Pattern: Interface + Adapter

Do not tightly couple your trading strategy logic directly to ABIDES internals.
Use the **Adapter Pattern**:

1. **Strategy Interface (Your Code):** Defines a clean API entirely agnostic of ABIDES (e.g., `on_tick()`, `on_fill()`). Takes clean dataclasses (e.g., `MarketSnapshot`).
2. **Strategy Implementation (Your Code):** Implements the logic.
3. **Adapter (Bridge):** A class that inherits from `TradingAgent` (from `abides_markets.agents.trading_agent`). It manages ABIDES events and calls your strategy.

### Why extend `TradingAgent`?
`TradingAgent` implements ~1,200 lines of necessary plumbing: exchange discovery, market hours tracking, bid/ask spread caching, portfolio tracking, and order lifecycle management. If you inherit from `Agent` or `FinancialAgent`, you will have to reimplement all of this.

---

## 3. Agent Lifecycle

1. `kernel_initializing(kernel)`: Agent created. Do not send messages.
2. `kernel_starting(start_time)`: Find the `ExchangeAgent`. Schedule the first `wakeup()`.
3. `wakeup(current_time)`: Main active entry point.
   - *CRITICAL:* Always call `super().wakeup(current_time)`. If it returns `False`, the market hours are unknown or the market is closed. **Do not trade if `False`.**
   - Must schedule the next wakeup: `self.set_wakeup(current_time + self.get_wake_frequency())`.
4. `receive_message(current_time, sender_id, message)`: React to inbound data.
   - *CRITICAL:* Always call `super().receive_message(...)` first. `TradingAgent` uses this to update your portfolio and known bounds.
5. `kernel_stopping()`: Log final holdings and PnL. `TradingAgent` automatically marks the portfolio to market.

---

## 4. Retrieving Market Data

ABIDES requires sending a message to request data. The response arrives asynchronously.

### Option A: Subscriptions (Recommended)
Send a subscription request *once* (e.g., in your first valid `wakeup()`).
- Use `L2SubReqMsg`, `TransactedVolSubReqMsg`, etc.
- `TradingAgent` automatically processes `L2DataMsg` and updates `self.known_bids[symbol]` and `self.known_asks[symbol]`.

### Option B: Point-in-time Queries
Send a query (e.g., `self.get_current_spread(symbol)`).
- The result arrives later as a `QuerySpreadResponseMsg`.
- Your `receive_message()` must intercept this to invoke your strategy.

*(Refer to `ABIDES_LLM_INTEGRATION_GOTCHAS.md` for safe access patterns, as all internal data dictionaries start empty).*

---

## 5. Order Management

All orders are placed via `TradingAgent` helpers:
- `self.place_limit_order(symbol, quantity, side, limit_price)`
- `self.place_market_order(symbol, quantity, side)`

**Open Orders Tracking:**
- Monitored in `self.orders` (Dict by `order_id`).
- Automatically decremented/removed by `TradingAgent.order_executed()` when filled.

**Cancellations:**
- Call `self.cancel_order(order)`. This sends a message. The order remains in `self.orders` until the `OrderCancelledMsg` arrives.

---

## 6. Simulation Configuration

To inject your custom agent, instantiate your adapter and append it to the configuration's agent list.

```python
from abides_markets.configs.rmsc04 import build_config

config = build_config(seed=42)
# Create your adapter (which extends TradingAgent)
my_adapter = MyAbidesAdapter(
    id=len(config["agents"]),
    strategy=MyStrategy(),
    symbol="AAPL"
)
config["agents"].append(my_adapter)

# Run simulation
from abides_core import abides
end_state = abides.run(config)
```

---

## 7. Custom Oracles — Injecting External Data

If you need to backtest against historical data or generative models (like CGANs) instead of algorithmic mean-reverting data, replace the default oracle.

See `abides-markets/abides_markets/oracles/external_data_oracle.py`.
- **Batch mode:** Use `DataFrameProvider` to load full series at initialization.
- **Point mode:** Implement `PointDataProvider` to query a database or generator on demand (uses LRU cache).

Example config modification:
```python
from abides_markets.oracles import ExternalDataOracle, DataFrameProvider
provider = DataFrameProvider({"AAPL": my_historical_series})
config["kernel"]["oracle"] = ExternalDataOracle(
    mkt_open, mkt_close, ["AAPL"], provider=provider
)
```
