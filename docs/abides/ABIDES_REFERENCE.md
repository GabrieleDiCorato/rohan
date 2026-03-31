# ABIDES Integration Reference

Condensed quick-reference for integrating with `abides-hasufel`.
For full details, see the specialized guides linked at the bottom.

---

## Core Mental Model

ABIDES is a **discrete-event simulation**. Time advances by jumping to the next event in a priority queue. There is no loop, no `on_bar()`. The only entry points into your agent are:

- `wakeup(current_time)` — called at scheduled times
- `receive_message(current_time, sender_id, message)` — called when a message arrives

Nothing is synchronous. Calling `get_current_spread()` sends a message to the Exchange; the result arrives *later* in `receive_message()`.

---

## Agent Lifecycle

Custom agents only implement two callbacks:

- **`wakeup(current_time)`**: Schedule next wakeup with `set_wakeup()`. Request data or place orders.
- **`receive_message(current_time, sender_id, message)`**: React to fills, spread responses, market data.

`TradingAgent` handles all other lifecycle phases internally (`kernel_starting`, `kernel_stopping`, etc.).

---

## Critical: Prices Are Integer Cents

All prices everywhere in ABIDES are **positive integers in cents**.

```python
$100.00 → 10_000
$1.50   →    150
$0.01   →      1

# Display only:
f"${price / 100:.2f}"

# Midpoint:
mid = (bid + ask) // 2
```

---

## Critical: Market Data Is None/Empty Until Received

All state fields start unset. They are populated asynchronously.

| Field | Initial state | Populated when |
|-------|-------------|----------------|
| `self.mkt_open` | `None` | `MarketHoursMsg` received |
| `self.mkt_close` | `None` | `MarketHoursMsg` received |
| `self.known_bids[symbol]` | `KeyError` → then `[]` | First `QuerySpreadResponseMsg` or `L2DataMsg` |
| `self.known_asks[symbol]` | `KeyError` → then `[]` | First `QuerySpreadResponseMsg` or `L2DataMsg` |
| `self.last_trade[symbol]` | `KeyError` | First executed trade in session |
| `L1DataMsg.bid / .ask` | `None` | Book side has resting orders |
| `L2DataMsg.bids / .asks` | `[]` | Book side has resting orders (never `None`) |

### Safe access patterns

```python
# Best bid/ask (safe)
bids = self.known_bids.get(symbol, [])
asks = self.known_asks.get(symbol, [])
bid = bids[0][0] if bids else None
ask = asks[0][0] if asks else None

# Midpoint (safe)
mid = (bid + ask) // 2 if bid is not None and ask is not None else None

# Last trade (safe)
last = self.last_trade.get(symbol)

# Ready-to-trade check
def can_trade(self, symbol):
    if not self.mkt_open or self.mkt_closed:
        return False
    bids = self.known_bids.get(symbol, [])
    asks = self.known_asks.get(symbol, [])
    return bool(bids and asks)
```

---

## The Pre-Market Window

The first `wakeup()` fires **before** `mkt_open` is known. `TradingAgent.wakeup()` returns `False` in this state. Always check it:

```python
def wakeup(self, current_time):
    can_trade = super().wakeup(current_time)
    if not can_trade:
        return              # Market hours unknown, or market closed
    # Safe to act here
```

---

## Order Book Data Structures

```python
# L1 — best bid/ask
get_l1_bid_data() → Optional[Tuple[int, int]]   # (price, qty) or None if empty

# L2 — all levels aggregated
get_l2_bid_data(depth=N) → List[Tuple[int, int]]  # [(price, qty), ...] or []
# Index 0 = best (closest to spread). Never None, can be [].

# L3 — all levels, per-order
get_l3_bid_data(depth=N) → List[Tuple[int, List[int]]]  # [(price, [qty1, qty2, ...]), ...]

# TradingAgent caches (L2 format):
self.known_bids[symbol]   # List[Tuple[int,int]], KeyError if never updated
self.known_asks[symbol]   # same
```

---

## Getting Market Data: Pull vs. Subscription

### Pull (one-shot request/response)
```python
# In wakeup(): request
self.get_current_spread(symbol)      # → QuerySpreadResponseMsg
self.get_last_trade(symbol)          # → QueryLastTradeResponseMsg
self.get_transacted_volume(symbol)   # → QueryTransactedVolResponseMsg

# In receive_message(): react
if isinstance(message, QuerySpreadResponseMsg):
    bids = message.bids   # List[Tuple[int,int]], may be []
    asks = message.asks
    last = message.last_trade   # Optional[int]
```

### Subscription (push, recommended)
```python
from abides_markets.messages.marketdata import L2SubReqMsg

# Subscribe once (e.g. first wakeup):
self.request_data_subscription(L2SubReqMsg(
    symbol=symbol,
    freq=int(1e8),   # push every 100ms of sim time
    depth=10,
))

# TradingAgent.handle_market_data() auto-updates:
#   self.known_bids[symbol], self.known_asks[symbol]
#   self.last_trade[symbol], self.exchange_ts[symbol]
```

Available subscription types: `L1SubReqMsg`, `L2SubReqMsg`, `L3SubReqMsg`, `TransactedVolSubReqMsg`, `BookImbalanceSubReqMsg`.

---

## Placing Orders

```python
from abides_markets.orders import Side

# Limit order
self.place_limit_order(
    symbol="ABM",
    quantity=100,          # shares (positive integer)
    side=Side.BID,         # Side.BID or Side.ASK
    limit_price=10_050,    # cents ($100.50)
)

# Market order
self.place_market_order(symbol="ABM", quantity=100, side=Side.BID)

# Cancel all open orders
self.cancel_all_orders()
```

`self.orders: Dict[int, Order]` — open orders keyed by `order_id`. Disappears when fully filled or cancelled.

---

## Custom Agents

Subclass `TradingAgent`, register with `@register_agent`, and declare a `BaseAgentConfig`.
See [`ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md`](./ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md) for the full pattern, copy-paste scaffold, and checklist.

---

## Running Simulations

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation

config = (SimulationBuilder()
    .from_template("rmsc04")
    .enable_agent("my_strategy", count=1, threshold=0.08)
    .seed(42)
    .build())

result = run_simulation(config)  # → SimulationResult
```

`SimulationResult` fields: `result.agents` (per-agent PnL, holdings), `result.markets` (L1 close, liquidity), `result.metadata` (seed, timing). Control depth with `ResultProfile`: `SUMMARY` (default), `QUANT` (adds L1/L2 series), `FULL` (adds raw logs).

See [`ABIDES_CONFIG_SYSTEM.md`](./ABIDES_CONFIG_SYSTEM.md) for builder API, templates, oracle config, and serialization.

---

## External Oracle — Historical / Generated Data

> [!IMPORTANT]
> The `Oracle` base class is an `abc.ABC`. Any custom oracle must implement both `get_daily_open_price(...)` and `observe_price(...)`.

Build the oracle externally and inject it via the builder:

```python
from abides_markets.oracles import ExternalDataOracle, DataFrameProvider

oracle = ExternalDataOracle(
    mkt_open, mkt_close, ["AAPL"],
    data={"AAPL": my_series},     # pd.Series, DatetimeIndex, int cents
)

config = (SimulationBuilder()
    .oracle_instance(oracle)
    .enable_agent("noise", count=500)
    .seed(42)
    .build())
```

See [`ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md` §7](./ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md) for point-mode providers, oracle-less simulations, and `ExternalDataOracleConfig`.

---

## mark_to_market Trap

`mark_to_market()` uses `self.last_trade[symbol]` — **crashes with `KeyError` if no trade has occurred yet**. Always guard:

```python
if self.last_trade.get(self.symbol) is not None:
    value = self.mark_to_market(self.holdings)
```

---

## logEvent Default: No Deep Copy

`logEvent(event_type, event, deepcopy_event=False)` logs agent events to the event log. The default is **no deep copy** — if you log a mutable object (e.g. `self.holdings`) and modify it later, the log entry will reflect the *final* state, not the state at log time.

```python
# BAD — holdings dict will be mutated after logging
self.logEvent("SNAPSHOT", self.holdings)

# GOOD — snapshot captured at log time
self.logEvent("SNAPSHOT", self.holdings, deepcopy_event=True)

# OK — immutable values don't need deepcopy
self.logEvent("TRADE", {"price": price, "qty": qty})
```

---

## Full Reference

- [`ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md`](./ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md) — adapter pattern, scaffold, checklist, testing
- [`ABIDES_CONFIG_SYSTEM.md`](./ABIDES_CONFIG_SYSTEM.md) — builder API, templates, oracle config, serialization
- [`ABIDES_LLM_INTEGRATION_GOTCHAS.md`](./ABIDES_LLM_INTEGRATION_GOTCHAS.md) — all None/NaN traps, safe patterns
- [`ABIDES_DATA_EXTRACTION.md`](./ABIDES_DATA_EXTRACTION.md) — parsing results, L1/L2 book history
- [`PARALLEL_SIMULATION_GUIDE.md`](./PARALLEL_SIMULATION_GUIDE.md) — multiprocessing, RNG hierarchy, log layout

---

| Crash | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'ABM'` on `self.known_bids` | Never received a spread update | Use `.get(symbol, [])` |
| `IndexError` on `self.known_bids[symbol][0]` | Book side is `[]` | Check `if bids:` first |
| `KeyError: 'ABM'` on `self.last_trade` | No trade yet | Use `.get(symbol)` |
| `TypeError: unsupported operand None + int` | `mkt_open` is `None` | Check `super().wakeup()` return |
| Wrong prices by factor of 100 | Forgot prices are cents | All prices are integer cents |
| Agent never trades | `wakeup()` not scheduling next wakeup | Call `self.set_wakeup(next_time)` |
| Orders silently ignored | `place_limit_order` with qty=0 or negative price | Validate before placing |
