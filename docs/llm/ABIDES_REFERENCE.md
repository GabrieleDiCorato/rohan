# ABIDES Integration Reference

Condensed reference for integrating with `abides-jpmc-public`.
Copy this into your project's `.github/copilot-instructions.md` or LLM context.

---

## Core Mental Model

ABIDES is a **discrete-event simulation**. Time advances by jumping to the next event in a priority queue. There is no loop, no `on_bar()`. The only entry points into your agent are:

- `wakeup(current_time)` — called at scheduled times
- `receive_message(current_time, sender_id, message)` — called when a message arrives

Nothing is synchronous. Calling `get_current_spread()` sends a message to the Exchange; the result arrives *later* in `receive_message()`.

---

## Agent Lifecycle (6 phases)

```
kernel_initializing() → kernel_starting() → wakeup() / receive_message() [repeated] → kernel_stopping() → kernel_terminating()
```

- **`kernel_starting`**: Find exchange ID. Set up state. Do not trade.
- **`wakeup`**: Schedule next wakeup with `set_wakeup()`. Request data or place orders.
- **`receive_message`**: React to fills, spread responses, market data.
- **`kernel_stopping`**: Log final results. TradingAgent auto-marks holdings to market.

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

## Custom Agent — Adapter Pattern

Implement `TradingStrategy` (no ABIDES imports), wrap it in `AbidesStrategyAdapter(TradingAgent)`.

```python
class MyStrategy(TradingStrategy):
    def on_tick(self, snapshot: MarketSnapshot, portfolio: PortfolioState) -> List[StrategyOrder]:
        bid, ask = snapshot.best_bid, snapshot.best_ask
        if bid is None or ask is None:
            return []
        mid = (bid + ask) // 2
        return [StrategyOrder("ABM", OrderSide.BUY, 10, OrderType.LIMIT, mid - 50)]

    def get_wakeup_interval_ns(self): return 60_000_000_000  # 1 min
    def on_trading_start(self, s, p): pass
    def on_fill(self, fill, p): return []

# Wire into config:
adapter = AbidesStrategyAdapter(id=len(config["agents"]), strategy=MyStrategy(), symbol="ABM")
config["agents"].append(adapter)
```

`MarketSnapshot` fields: `timestamp`, `symbol`, `best_bid`, `best_ask`, `best_bid_size`, `best_ask_size`, `last_trade_price`, `bid_depth`, `ask_depth`, `mkt_open`, `mkt_close`. All prices `Optional[int]` in cents.

---

## External Oracle — Historical / Generated Data

```python
from abides_markets.oracles import ExternalDataOracle, DataFrameProvider, InterpolationStrategy

# Batch mode (pre-load full series)
oracle = ExternalDataOracle(
    mkt_open, mkt_close, ["AAPL"],
    data={"AAPL": my_series},          # pd.Series, DatetimeIndex, int cents
    interpolation=InterpolationStrategy.FORWARD_FILL,
)

# Point mode (on-demand, memory-bounded — for DB / CGAN)
class MyProvider:
    def get_fundamental_at(self, symbol: str, timestamp: int) -> int: ...

oracle = ExternalDataOracle(
    mkt_open, mkt_close, ["AAPL"],
    provider=MyProvider(),
    cache_size=10_000,
)

# Inject into simulation config:
config["kernel"]["oracle"] = oracle
```

---

## mark_to_market Trap

`mark_to_market()` uses `self.last_trade[symbol]` — **crashes with `KeyError` if no trade has occurred yet**. Always guard:

```python
if self.last_trade.get(self.symbol) is not None:
    value = self.mark_to_market(self.holdings)
```

---

## Full Reference

- `docs/ABIDES_LLM_INTEGRATION_GOTCHAS.md` — all None/NaN traps, safe patterns
- `docs/ABIDES_CUSTOM_AGENT_IMPLEMENTATION_GUIDE.md` — full adapter pattern
- `docs/ABIDES_DATA_EXTRACTION.md` — parsing results (`parse_logs_df`, L1/L2 book history)

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
