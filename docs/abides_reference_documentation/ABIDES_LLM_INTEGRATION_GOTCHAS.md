# ABIDES — LLM Integration Gotchas & Reference

This document is written for **AI coding assistants** integrating with ABIDES.
It catalogs every source of `None`, `NaN`, empty collections, and surprising
API behavior that commonly causes LLM-generated code to crash or misbehave.

---

## Table of Contents

1. [Mental Model — Not a Loop, Not Synchronous](#1-mental-model)
2. [Market Data — All Sources of None / Empty](#2-market-data--all-sources-of-none--empty)
3. [Pre-Market Window — Special State](#3-pre-market-window--special-state)
4. [Order Book Data Structures Explained](#4-order-book-data-structures-explained)
5. [The mark_to_market Trap](#5-the-mark_to_market-trap)
6. [Price Units — Everything is Integer Cents](#6-price-units--everything-is-integer-cents)
7. [Order Lifecycle and Tracking](#7-order-lifecycle-and-tracking)
8. [Correct Safe-Access Patterns](#8-correct-safe-access-patterns)
9. [Subscription vs. Pull-Based Data](#9-subscription-vs-pull-based-data)
10. [Complete State Validity Checklist](#10-complete-state-validity-checklist)

---

## 1. Mental Model

ABIDES is a **discrete-event simulation**, NOT a loop-based system.

```
❌ WRONG mental model:
    while market_open:
        bid, ask = get_spread()   # synchronous
        if bid is not None:
            place_order(...)

✅ CORRECT mental model:
    def wakeup(self, current_time):
        # 1. Request data (sends a message - async)
        self.get_current_spread(symbol)
        # 2. Return — execution ends here

    def receive_message(self, current_time, sender_id, message):
        # 3. Data arrives HERE, possibly nanoseconds later
        if isinstance(message, QuerySpreadResponseMsg):
            # NOW self.known_bids / known_asks are fresh
            place_order(...)
```

**Key consequence**: Every call to `get_current_spread()`, `get_last_trade()`, etc.
sends a *message* to the Exchange. Nothing is returned immediately. The response
arrives asynchronously in `receive_message()`.

---

## 2. Market Data — All Sources of None / Empty

### 2.1 `self.known_bids[symbol]` and `self.known_asks[symbol]`

| Situation | Value |
|-----------|-------|
| Before any spread query response received | **`KeyError`** — key doesn't exist |
| After response, but book side is empty | **`[]`** (empty list) |
| After response, book side has data | `[(price, qty), ...]` |

```python
# ❌ Crashes before first response
bid = self.known_bids["ABM"][0][0]

# ❌ Crashes if book side is empty
bid = self.known_bids["ABM"][0][0]  # IndexError on []

# ✅ Safe access
bids = self.known_bids.get("ABM", [])
bid = bids[0][0] if bids else None
```

### 2.2 `self.last_trade[symbol]`

| Situation | Value |
|-----------|-------|
| Before any trade has occurred | **`KeyError`** — key doesn't exist |
| After any trade in the session | `int` (price in cents) |
| After market close message | Updated to official close price |

```python
# ❌ Crashes before first trade (common at simulation start)
price = self.last_trade["ABM"]

# ✅ Safe access
price = self.last_trade.get("ABM")  # Returns None if missing
```

### 2.3 `self.mkt_open` and `self.mkt_close`

Both start as `None`. They are only populated after the agent receives a
`MarketHoursMsg` from the Exchange (which arrives in response to the automatic
`MarketHoursRequestMsg` sent within `TradingAgent.wakeup()`).

```python
# ❌ Crashes on first wakeup before market hours are known
next_wakeup = self.mkt_open + interval

# ✅ TradingAgent.wakeup() returns a boolean — use it
def wakeup(self, current_time):
    can_trade = super().wakeup(current_time)
    if not can_trade:
        return  # mkt_open/mkt_close not yet known, or market closed
    ...
```

### 2.4 `L1DataMsg.bid` and `L1DataMsg.ask`

The type annotation says `Tuple[int, int]` but in practice these come from
`OrderBook.get_l1_bid_data()` / `get_l1_ask_data()` which return **`None`**
when the book side is empty.

```python
# The actual source in order_book.py:
def get_l1_bid_data(self) -> Optional[Tuple[int, int]]:
    if len(self.bids) == 0:
        return None   # ← this happens before any orders are placed

# ✅ Always guard
if message.bid is not None:
    bid_price, bid_qty = message.bid
```

### 2.5 `L2DataMsg.bids` and `L2DataMsg.asks`

These are always `List[Tuple[int, int]]` — never `None`. But they can be
`[]` (empty list) when the book side has no resting orders.

```python
bids = message.bids  # List — safe, but may be []
if bids:
    best_bid_price = bids[0][0]  # ✅ guarded
    best_bid_qty   = bids[0][1]
```

### 2.6 `QuerySpreadResponseMsg.last_trade`

Type annotation: `Optional[int]`. Value is `None` when no trade has happened
in the session yet.

```python
# ❌ Crashes before first trade
spread_msg.last_trade + 100

# ✅ Safe
if spread_msg.last_trade is not None:
    ...
```

---

## 3. Pre-Market Window — Special State

The **pre-market window** is the period between simulation start and `mkt_open`.
LLM-generated agents frequently crash during this window because:

1. `wakeup()` is called, but `mkt_open` is `None`
2. `known_bids`/`known_asks` dictionaries are empty
3. `last_trade` dictionary is empty
4. The order book itself has no orders yet → all queries return `None` / `[]`

The **correct pattern** (already implemented in `TradingAgent`):

```python
def wakeup(self, current_time):
    can_trade = super().wakeup(current_time)

    # super().wakeup() returns False if:
    #   - mkt_open is None (market hours not yet known)
    #   - mkt_close is None
    #   - self.mkt_closed is True
    if not can_trade:
        return   # ← Do nothing during pre-market

    # Only here is it safe to trade
```

---

## 4. Order Book Data Structures Explained

### L1 — Best bid/ask only

```python
# OrderBook.get_l1_bid_data() → Optional[Tuple[int, int]]
# Returns: (price_in_cents, quantity)  OR  None if book side empty

(100_050, 200)   # best bid: $1000.50 for 200 shares
None             # no bids at all
```

### L2 — All price levels, aggregated by price

```python
# OrderBook.get_l2_bid_data(depth=N) → List[Tuple[int, int]]
# Each tuple: (price_in_cents, total_quantity_at_that_price)
# Ordered from best price inward (closest to spread first)

[(100_100, 500), (100_050, 200), (100_000, 1000)]
# Index 0 = best bid, index -1 = worst bid

[]   # Empty list if no bids at all
```

### L3 — All price levels, per-order quantities

```python
# OrderBook.get_l3_bid_data(depth=N) → List[Tuple[int, List[int]]]
# Each tuple: (price_in_cents, [qty_order1, qty_order2, ...])
# Within a price level, orders are in FIFO priority order

[(100_100, [200, 300]), (100_050, [500])]
```

### TradingAgent cache fields

| Field | Type | Content |
|-------|------|---------|
| `self.known_bids[symbol]` | `List[Tuple[int,int]]` | L2 bid levels, best first. May be `[]`. **KeyError if symbol never queried.** |
| `self.known_asks[symbol]` | `List[Tuple[int,int]]` | L2 ask levels, best first. May be `[]`. **KeyError if symbol never queried.** |
| `self.last_trade[symbol]` | `int` | Last executed price (cents). **KeyError if no trade yet.** |
| `self.exchange_ts[symbol]` | `NanosecondTime` | Exchange timestamp of last L2 update. Populated by subscriptions only. |
| `self.mkt_open` | `Optional[int]` | Nanoseconds. `None` until `MarketHoursMsg` received. |
| `self.mkt_close` | `Optional[int]` | Nanoseconds. `None` until `MarketHoursMsg` received. |

---

## 5. The `mark_to_market` Trap

`TradingAgent.mark_to_market(holdings)` does:

```python
value = self.last_trade[symbol] * shares  # ← KeyError if no trades!
```

**This will crash if called before any trade has occurred in the session.**
The `kernel_stopping()` method calls `mark_to_market()` — so even if your
agent never trades, this runs at the end of every simulation.

The Exchange sends `MarketClosePriceMsg` at market close with all final
prices, which populates `last_trade`. So crashes are only a risk if called
*before* the market has had any trades.

```python
# ✅ Safe pattern
def compute_portfolio_value(self):
    if self.last_trade.get(self.symbol) is None:
        return None  # Not enough data yet
    return self.mark_to_market(self.holdings)
```

---

## 6. Price Units — Everything is Integer Cents

**All prices in ABIDES are integers in cents.** There are no floats for prices.

```python
# $100.00 = 10_000 cents
# $1.50   =    150 cents
# $0.01   =      1 cent

# ❌ Common LLM mistake: treating prices as dollars
if bid_price > 100:      # Wrong — this is 100 cents = $1.00
    ...

# ✅ Correct
if bid_price > 10_000:   # $100.00
    ...

# Converting for display only:
print(f"Bid: ${bid_price / 100:.2f}")
```

**Consequence for midpoint calculation:**

```python
# ✅ Integer midpoint (standard in ABIDES)
midpoint = (bid + ask) // 2

# ✅ Also used in TradingAgent.get_known_bid_ask_midpoint():
midpoint = int(round((bid + ask) / 2))
```

---

## 7. Order Lifecycle and Tracking

### Agent's view of its own orders

```python
self.orders: Dict[int, Order]
# Keys: order_id
# Present: order is open (submitted, not fully filled, not cancelled)
# Absent: order fully filled OR cancelled

# ⚠️ Order may disappear from self.orders BEFORE you receive the
#    OrderExecutedMsg — timing race in the event queue
```

### Order execution flow

```
place_limit_order() → LimitOrderMsg → Exchange
                                         ↓
                               (waits for counterpart)
                                         ↓
                             OrderAcceptedMsg ← agent.order_accepted()
                                         ↓
                             OrderExecutedMsg ← agent.order_executed()
                             (holdings & CASH updated automatically by TradingAgent)
```

### Cancellation subtlety

```python
# Calling cancel_order() does NOT immediately remove from self.orders.
# The order stays in self.orders until OrderCancelledMsg arrives.
# If the order executes BEFORE the cancel reaches the Exchange,
# you receive OrderExecutedMsg instead — no cancel confirmation.
```

---

## 8. Correct Safe-Access Patterns

### Getting bid/ask safely

```python
def _safe_best_bid_ask(self, symbol: str):
    """Returns (bid, ask) in cents, or (None, None) if unavailable."""
    bids = self.known_bids.get(symbol, [])
    asks = self.known_asks.get(symbol, [])
    bid = bids[0][0] if bids else None
    ask = asks[0][0] if asks else None
    return bid, ask
```

### Getting midpoint safely

```python
def _safe_midpoint(self, symbol: str) -> Optional[int]:
    bid, ask = self._safe_best_bid_ask(symbol)
    if bid is None or ask is None:
        return None
    return (bid + ask) // 2
```

### Checking if ready to trade

```python
def _can_trade(self, symbol: str) -> bool:
    """True only if all the data we need is present."""
    if self.mkt_open is None or self.mkt_close is None:
        return False
    if self.mkt_closed:
        return False
    bids = self.known_bids.get(symbol, [])
    asks = self.known_asks.get(symbol, [])
    return bool(bids and asks)
```

### Computing spread

```python
def _spread(self, symbol: str) -> Optional[int]:
    bid, ask = self._safe_best_bid_ask(symbol)
    if bid is None or ask is None:
        return None
    return ask - bid   # in cents; should always be >= 0 in a valid book
```

---

## 9. Subscription vs. Pull-Based Data

ABIDES supports two ways to get market data:

### Pull (request/response)

```python
# In wakeup():
self.get_current_spread(symbol)          # → QuerySpreadResponseMsg
self.get_last_trade(symbol)              # → QueryLastTradeResponseMsg
self.get_transacted_volume(symbol)       # → QueryTransactedVolResponseMsg

# Each sends a message; response arrives in receive_message() LATER.
```

### Subscription (push)

```python
# In wakeup(), first time only:
from abides_markets.messages.marketdata import L2SubReqMsg
self.request_data_subscription(L2SubReqMsg(
    symbol=symbol,
    freq=int(1e8),   # push every 100ms of sim time
    depth=10,        # top 10 levels
))

# Updates arrive as L2DataMsg in receive_message() periodically.
# TradingAgent.handle_market_data() automatically updates:
#   self.known_bids[symbol]
#   self.known_asks[symbol]
#   self.last_trade[symbol]
#   self.exchange_ts[symbol]
```

**Recommendation**: Use **subscriptions** if your strategy needs fresh data on
every tick. Use **pull** if you only need data occasionally.

**Available subscription types:**

| Message Class | Data | Trigger |
|---------------|------|---------|
| `L1SubReqMsg` | Best bid/ask (price + qty) | Periodic |
| `L2SubReqMsg` | All price levels (aggregated) | Periodic |
| `L3SubReqMsg` | All price levels (per-order) | Periodic |
| `TransactedVolSubReqMsg` | Buy/sell volume in lookback window | Periodic |
| `BookImbalanceSubReqMsg` | Imbalance event | Event-driven |

---

## 10. Complete State Validity Checklist

Use this checklist before acting on any market state in your agent:

```python
def _state_is_valid(self, symbol: str) -> bool:
    # 1. Do we know market hours?
    if self.mkt_open is None or self.mkt_close is None:
        return False

    # 2. Is the market still open?
    if self.mkt_closed:
        return False

    # 3. Have we received at least one book update?
    if symbol not in self.known_bids or symbol not in self.known_asks:
        return False

    # 4. Does the book have both sides?
    if not self.known_bids[symbol] or not self.known_asks[symbol]:
        return False

    # 5. Is the spread valid (no crossed book)?
    bid = self.known_bids[symbol][0][0]
    ask = self.known_asks[symbol][0][0]
    if bid >= ask:
        return False  # Crossed book — shouldn't happen in well-formed sim

    return True
```

---

## Quick Reference: What Returns None/Empty and When

| Expression | Type | `None`/empty when |
|------------|------|-------------------|
| `self.mkt_open` | `Optional[int]` | Before `MarketHoursMsg` received |
| `self.mkt_close` | `Optional[int]` | Before `MarketHoursMsg` received |
| `self.known_bids.get(symbol)` | `Optional[List]` | Before first spread response (`None`); or empty book (`[]`) |
| `self.known_asks.get(symbol)` | `Optional[List]` | Before first spread response (`None`); or empty book (`[]`) |
| `self.last_trade.get(symbol)` | `Optional[int]` | Before any trade in the session |
| `L1DataMsg.bid` | `Optional[Tuple]` | When bid side of book is empty |
| `L1DataMsg.ask` | `Optional[Tuple]` | When ask side of book is empty |
| `L2DataMsg.bids` | `List` | Never `None`, but can be `[]` |
| `L2DataMsg.asks` | `List` | Never `None`, but can be `[]` |
| `QuerySpreadResponseMsg.last_trade` | `Optional[int]` | Before any trade |
| `QueryLastTradeResponseMsg.last_trade` | `Optional[int]` | Before any trade |
| `OrderBook.get_l1_bid_data()` | `Optional[Tuple]` | Empty book side |
| `OrderBook.get_l1_ask_data()` | `Optional[Tuple]` | Empty book side |
| `OrderBook.get_l2_bid_data()` | `List` | Never `None`, but can be `[]` |
| `OrderBook.get_l2_ask_data()` | `List` | Never `None`, but can be `[]` |
| `OrderBook.last_trade` | `Optional[int]` | Before any trade |
