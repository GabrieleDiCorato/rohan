# ABIDES Custom Agent Implementation Guide

## Decoupling Interface & Adapter Pattern for Custom Trading Strategies

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [ABIDES Internals — What You Need to Know](#2-abides-internals--what-you-need-to-know)
3. [Class Hierarchy & Inheritance Map](#3-class-hierarchy--inheritance-map)
4. [Agent Lifecycle — The 6 Phases](#4-agent-lifecycle--the-6-phases)
5. [The Kernel Event Loop — How Time Works](#5-the-kernel-event-loop--how-time-works)
6. [Message Protocol & Order Flow](#6-message-protocol--order-flow)
7. [State Inference — The Logging Trap](#7-state-inference--the-logging-trap)
8. [Decoupling Interface Design](#8-decoupling-interface-design)
9. [Adapter Class — Bridging Into ABIDES](#9-adapter-class--bridging-into-abides)
10. [Config Integration — Injecting Your Agent](#10-config-integration--injecting-your-agent)
11. [Extracting Results After Simulation](#11-extracting-results-after-simulation)
12. [Gotchas & Pitfalls](#12-gotchas--pitfalls)
13. [Quick-Start Recipes](#13-quick-start-recipes)

---

## 1. Architecture Overview

ABIDES is a **discrete-event, message-passing simulation** framework. It is *not* a traditional loop-based simulator — time advances by jumping to the next event in a priority queue. This has profound implications for how you design agents.

```mermaid
graph TB
    subgraph "Your Code (Decoupled)"
        Strategy["StrategicAgent Protocol"]
        Impl["Your Strategy Implementation"]
    end

    subgraph "Adapter Layer"
        Adapter["StrategicAgentAdapter<br/>(extends TradingAgent)"]
    end

    subgraph "ABIDES Runtime"
        Kernel["Kernel<br/>(Event Loop + Priority Queue)"]
        Exchange["ExchangeAgent<br/>(Order Book)"]
        Background["Background Agents<br/>(Noise, Value, MM)"]
        Oracle["Oracle<br/>(Fundamental Value)"]
    end

    Impl --> Strategy
    Strategy --> Adapter
    Adapter -->|Messages| Kernel
    Kernel -->|Dispatch| Exchange
    Kernel -->|Dispatch| Background
    Exchange -.->|Order Events| Adapter
    Adapter -.->|State Updates| Strategy
    Oracle -.->|Fundamental| Exchange
```

**Key principle**: Your strategy code should know *nothing* about ABIDES internals. The adapter translates between your clean API and ABIDES's message-based event system.

---

## 2. ABIDES Internals — What You Need to Know

### 2.1 Time Representation

All time in ABIDES is in **nanoseconds** (`NanosecondTime = int`). There are no `datetime` objects inside the simulation — only `int` values representing nanoseconds since epoch.

```python
# Converting human-readable to nanoseconds
from abides_core.utils import str_to_ns
one_second = str_to_ns("1s")      # 1_000_000_000
market_open = str_to_ns("09:30:00")  # nanoseconds from midnight
```

### 2.2 Prices

All prices are in **integer cents** (1 dollar = 100). A stock at $1,000.00 has `price = 100_000`. This avoids floating-point issues in financial calculations.

### 2.3 Cash & Holdings

Cash is tracked as integer cents in `self.holdings["CASH"]`. Share positions are tracked as integers in `self.holdings["SYMBOL"]`.

### 2.4 The Oracle

The oracle provides the "true" fundamental value of assets. Background agents use it to inform their beliefs. Your custom agent may or may not use it — accessing it via `self.kernel.oracle` is available but optional.

---

## 3. Class Hierarchy & Inheritance Map

```mermaid
classDiagram
    class Agent {
        +id: int
        +name: str
        +type: str
        +kernel: Kernel
        +current_time: NanosecondTime
        +log: List
        +kernel_initializing(kernel)
        +kernel_starting(start_time)
        +kernel_stopping()
        +kernel_terminating()
        +wakeup(current_time)
        +receive_message(current_time, sender_id, message)
        +send_message(recipient_id, message)
        +set_wakeup(requested_time)
    }

    class FinancialAgent {
        +dollarize(cents)
    }

    class TradingAgent {
        +holdings: Dict
        +orders: Dict
        +exchange_id: int
        +mkt_open: NanosecondTime
        +mkt_close: NanosecondTime
        +place_limit_order(...)
        +place_market_order(...)
        +cancel_order(order)
        +cancel_all_orders()
        +get_current_spread(symbol)
        +get_last_trade(symbol)
        +get_known_bid_ask(symbol)
        +order_executed(order)
        +order_accepted(order)
        +order_cancelled(order)
        +get_wake_frequency() NanosecondTime
    }

    class StrategicAgentAdapter {
        +strategy: StrategicAgent
        +wakeup(current_time)
        +receive_message(...)
        +order_accepted(order)
        +order_modified(order)
        +order_partial_cancelled(order)
        +order_replaced(old, new)
        +market_closed()
    }

    Agent <|-- FinancialAgent
    FinancialAgent <|-- TradingAgent
    TradingAgent <|-- StrategicAgentAdapter
```

> [!IMPORTANT]
> **Always extend `TradingAgent`**, not `Agent` or `FinancialAgent`. `TradingAgent` handles exchange discovery, market hours, order lifecycle, portfolio tracking, and bid/ask caching. If you skip it, you must reimplement ~1,200 lines of plumbing.

---

## 4. Agent Lifecycle — The 6 Phases

Every agent in ABIDES goes through a strict lifecycle managed by the Kernel:

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: kernel_initializing (kernel)                           │
│   • Kernel reference stored (self.kernel = kernel)              │
│   • No other agents guaranteed to exist yet                     │
│   • DO NOT send messages here                                   │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: kernel_starting (start_time)                           │
│   • All agents exist now                                        │
│   • TradingAgent discovers ExchangeAgent via                    │
│     kernel.find_agents_by_type(ExchangeAgent)                   │
│   • First wakeup scheduled via set_wakeup(start_time)           │
│   • Override get_wake_frequency() to control first-wake delay   │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: wakeup(current_time) — MAIN STRATEGY ENTRY POINT       │
│   • Called when your scheduled alarm fires                      │
│   • First wakeup: TradingAgent requests market hours            │
│   • Subsequent wakeups: your strategy logic runs here           │
│   • MUST call super().wakeup(current_time) first                │
│   • Schedule next wakeup with self.set_wakeup(next_time)        │
├─────────────────────────────────────────────────────────────────┤
│ Phase 4: receive_message (current_time, sender_id, message)     │
│   • Called when incoming messages arrive (order fills, data, ..)│
│   • TradingAgent routes messages to typed handlers              │
│   • Override order_executed(), query_spread(), etc.             │
│   • MUST call super().receive_message(...) first                │
├─────────────────────────────────────────────────────────────────┤
│ Phase 5: kernel_stopping()                                      │
│   • All agents still exist                                      │
│   • TradingAgent logs final holdings, marks to market           │
│   • Good place to compute your strategy metrics                 │
├─────────────────────────────────────────────────────────────────┤
│ Phase 6: kernel_terminating()                                   │
│   • Agent logs written to disk                                  │
│   • No other agents guaranteed to exist                         │
│   • DO NOT send messages here                                   │
└─────────────────────────────────────────────────────────────────┘
```

> [!CAUTION]
> The very first `wakeup()` call does **not** mean the market is open. `TradingAgent.wakeup()` returns a boolean indicating trade-readiness. You must check `self.mkt_open` and `self.mkt_close` are set before trading. Market hours are received asynchronously via a message exchange in `receive_message`.

---

## 5. The Kernel Event Loop — How Time Works

```mermaid
graph LR
    A["Priority Queue<br/>(timestamp, event)"] -->|Pop earliest| B["Dispatch"]
    B -->|WakeupMsg| C["agent.wakeup()"]
    B -->|Other Message| D["agent.receive_message()"]
    C -->|"set_wakeup()"|A
    C -->|"send_message()"| A
    D -->|"send_message()"| A
    D -->|"set_wakeup()"| A
```

**Critical details**:

1. **Time only advances** when the Kernel pops the next event. There is no "tick" — no `on_bar()` callback. You decide your own schedule via `set_wakeup()`.

2. **Computation delay**: After each wakeup/message, the agent's "local clock" advances by `computation_delay` nanoseconds. The agent **cannot act again** until the global clock catches up. This prevents infinite loops of simultaneous events.

3. **Latency**: Messages between agents are delayed by a latency model. Your order doesn't reach the Exchange instantly — it arrives after `latency + computation_delay` nanoseconds.

4. **No synchronous responses**: When you call `get_current_spread(symbol)`, a *request message* is sent to the Exchange. The response arrives asynchronously later as a `QuerySpreadResponseMsg` inside `receive_message()`. This is a fundamental architectural constraint.

---

## 6. Message Protocol & Order Flow

### 6.1 Placing an Order

```
Your Agent                    Kernel                     Exchange
    │                           │                           │
    ├── place_limit_order() ──► │                           │
    │   (creates LimitOrderMsg) │                           │
    │                           ├── deliver after latency ─►│
    │                           │                           ├── OrderAcceptedMsg
    │                           │◄──────────────────────────┤
    │◄── receive_message() ─────┤                           │
    │    (order_accepted)       │                           │
    │                           │                           │
    │                           │  ... time passes ...      │
    │                           │                           │
    │                           │                           ├── OrderExecutedMsg
    │                           │◄──────────────────────────┤  (partial or full fill)
    │◄── receive_message() ─────┤                           │
    │    (order_executed)       │                           │
```

### 6.2 Key Message Types

| Direction | Message | Purpose |
|-----------|---------|---------|
| Agent → Exchange | `LimitOrderMsg`, `MarketOrderMsg` | Place orders |
| Agent → Exchange | `CancelOrderMsg` | Cancel an open order |
| Agent → Exchange | `QuerySpreadMsg` | Request current bid/ask |
| Agent → Exchange | `QueryLastTradeMsg` | Request last trade price |
| Agent → Exchange | `MarketHoursRequestMsg` | Ask for open/close times |
| Agent → Exchange | `L2SubReqMsg` | Subscribe to L2 data feed |
| Exchange → Agent | `OrderAcceptedMsg` | Order entered the book |
| Exchange → Agent | `OrderExecutedMsg` | Order (partially) filled |
| Exchange → Agent | `OrderCancelledMsg` | Order cancelled |
| Exchange → Agent | `QuerySpreadResponseMsg` | Bid/ask response |
| Exchange → Agent | `MarketHoursMsg` | Open/close times |
| Exchange → Agent | `L2DataMsg` | L2 subscription update |
| Exchange → Agent | `MarketClosedMsg` | Market is closed |

### 6.3 Data You Can Access

| Source | Access Pattern | Timing |
|--------|---------------|--------|
| Current spread | `get_current_spread(symbol)` → async response in `receive_message` | On-demand |
| Last trade | `get_last_trade(symbol)` → async response | On-demand |
| L2 order book | Subscribe via `request_data_subscription(L2SubReqMsg(...))` | Periodic push |
| Holdings | `self.holdings` (dict) | Always available |
| Open orders | `self.orders` (dict of order_id → Order) | Always available |
| Cached bid/ask | `self.known_bids[symbol]`, `self.known_asks[symbol]` | After any spread query |
| Mark-to-market | `self.mark_to_market(self.holdings)` | Anytime after first trade |

---

## 7. State Inference — The Logging Trap

> [!WARNING]
> **Agent state is partially inferred from logs.** This is one of the most subtle aspects of ABIDES. The kernel and post-simulation analysis code extract metrics like PnL, trade count, and final valuations from the *event log* that agents maintain, not from explicit state queries.

Key log events recorded by `TradingAgent`:

| Event Type | When Logged | What It Contains |
|------------|-------------|-----------------|
| `STARTING_CASH` | `kernel_starting` | Initial cash (int cents) |
| `FINAL_CASH_POSITION` | `kernel_stopping` | Final cash position |
| `ENDING_CASH` | `kernel_stopping` | Mark-to-market value |
| `HOLDINGS_UPDATED` | After each fill | Full holdings dict |
| `ORDER_SUBMITTED` | When placing orders | Order dict |
| `ORDER_EXECUTED` | On fill notification | Order dict with fill_price |
| `ORDER_CANCELLED` | On cancel confirmation | Order dict |
| `MARK_TO_MARKET` | On M2M computation | Symbol × quantity × price |

**Implication for your adapter**: If you want post-simulation analysis tools to work, you must either:
1. Call `super()` methods faithfully so `TradingAgent` logs correctly, OR
2. Add your own log events via `self.logEvent("YOUR_EVENT", data)`.

---

## 8. Decoupling Interface Design

This is the **clean API** your strategy code implements. It knows nothing about ABIDES.

```python
"""strategy_api.py — Pure strategy abstraction, zero ABIDES dependency."""

from __future__ import annotations

from enum import Enum
from typing import Protocol

from pydantic import BaseModel, Field, computed_field, model_validator


class Side(str, Enum):
    BID = "BID"
    ASK = "ASK"


class OrderType(str, Enum):
    LIMIT = "LIMIT"
    MARKET = "MARKET"


class OrderStatus(str, Enum):
    ACCEPTED = "ACCEPTED"       # Exchange accepted the order
    NEW = "NEW"                 # Live and resting on the book
    PARTIAL = "PARTIAL"         # Partially filled
    FILLED = "FILLED"           # Fully filled
    CANCELLED = "CANCELLED"     # Cancelled (by strategy or exchange)
    REJECTED = "REJECTED"       # Exchange rejected
    MODIFIED = "MODIFIED"       # Order price/quantity changed
    PARTIAL_CANCELLED = "PARTIAL_CANCELLED"  # Quantity reduced
    REPLACED = "REPLACED"       # Atomically replaced by a new order


class Order(BaseModel):
    """A live or historical order visible to the strategy."""

    order_id: int
    symbol: str
    side: Side
    quantity: int = Field(ge=1, description="Number of shares")
    price: int = Field(description="Limit price in cents (0 for market orders)")
    order_type: OrderType
    status: OrderStatus
    filled_quantity: int = Field(default=0, ge=0, description="Shares filled so far")
    fill_price: int | None = Field(
        default=None, description="Price at which the last fill occurred (cents)"
    )


class MarketState(BaseModel):
    """Snapshot of the market visible to the strategy at a given instant."""

    # ── Core market data ─────────────────────────────────────────────────
    timestamp_ns: int = Field(description="Nanoseconds since epoch")
    best_bid: int | None = Field(default=None, description="Best bid price in cents")
    best_ask: int | None = Field(default=None, description="Best ask price in cents")
    bid_depth: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Order book bid levels: [(price_cents, qty), ...], best first",
    )
    ask_depth: list[tuple[int, int]] = Field(
        default_factory=list,
        description="Order book ask levels: [(price_cents, qty), ...], best first",
    )
    last_trade: int | None = Field(default=None, description="Last trade price in cents")

    # ── Portfolio ────────────────────────────────────────────────────────
    inventory: int = Field(description="Signed position in shares")
    cash: int = Field(description="Available cash in cents")
    open_orders: list[Order]
    portfolio_value: int = Field(default=0, description="Mark-to-market value in cents")
    unrealized_pnl: int = Field(default=0, description="portfolio_value - starting_cash, in cents")

    # ── Situational awareness ────────────────────────────────────────────
    time_remaining_ns: int | None = Field(
        default=None, description="Nanoseconds until market close (None if unknown)"
    )
    is_market_closed: bool = Field(
        default=False, description="True when the market has closed"
    )

    # ── Near-touch liquidity ─────────────────────────────────────────────
    bid_liquidity: int = Field(default=0, description="Bid volume within 0.5% of best bid")
    ask_liquidity: int = Field(default=0, description="Ask volume within 0.5% of best ask")

    # ── Computed convenience ─────────────────────────────────────────────
    @computed_field
    @property
    def mid_price(self) -> int | None:
        """Midpoint price in cents, or None if either side is missing."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) // 2
        return None

    @computed_field
    @property
    def spread(self) -> int | None:
        """Bid-ask spread in cents, or None if either side is missing."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


CANCEL_ALL: int = -1  # Sentinel: cancel_order_id = CANCEL_ALL cancels all open orders.


class OrderActionType(str, Enum):
    """Discriminator for order action intent."""
    PLACE = "PLACE"
    CANCEL = "CANCEL"
    CANCEL_ALL = "CANCEL_ALL"
    MODIFY = "MODIFY"
    PARTIAL_CANCEL = "PARTIAL_CANCEL"
    REPLACE = "REPLACE"


class OrderAction(BaseModel):
    """An order instruction returned by the strategy.

    Examples:
        OrderAction(side=Side.BID, quantity=100, price=10050)       # Place limit order
        OrderAction.cancel(order_id=123)                            # Cancel one order
        OrderAction.cancel_all()                                    # Cancel all
        OrderAction.modify(order_id=123, new_price=10050)           # Change price
        OrderAction.partial_cancel(order_id=123, reduce_by=50)      # Reduce quantity
        OrderAction.replace(order_id=123, side=Side.BID, quantity=100, price=10050)
    """

    action_type: OrderActionType = Field(default=OrderActionType.PLACE)
    side: Side = Field(default=Side.BID)
    quantity: int = Field(default=1, ge=1)
    price: int | None = Field(default=None, description="Limit price in cents")
    order_type: OrderType = Field(default=OrderType.LIMIT)
    cancel_order_id: int | None = Field(
        default=None, description="Target order ID for CANCEL/MODIFY/PARTIAL_CANCEL/REPLACE"
    )
    is_hidden: bool = Field(default=False, description="Iceberg order (LIMIT only)")
    is_post_only: bool = Field(default=False, description="Reject if would fill (LIMIT only)")
    new_price: int | None = Field(default=None, description="New price for MODIFY/REPLACE")
    new_quantity: int | None = Field(
        default=None, description="New qty for MODIFY/REPLACE, or reduction for PARTIAL_CANCEL"
    )

    # Validators: _validate_price_for_order_type, _validate_hidden_post_only, _infer_action_type
    # (see source for full implementations)

    # ── Convenience factories ────────────────────────────────────────────
    @classmethod
    def cancel(cls, order_id: int) -> OrderAction: ...

    @classmethod
    def cancel_all(cls) -> OrderAction: ...

    @classmethod
    def modify(cls, order_id: int, *, new_price=None, new_quantity=None) -> OrderAction: ...

    @classmethod
    def partial_cancel(cls, order_id: int, reduce_by: int) -> OrderAction: ...

    @classmethod
    def replace(cls, order_id: int, *, side: Side, quantity: int, price: int) -> OrderAction: ...


class AgentConfig(BaseModel):
    """Configuration passed to StrategicAgent.initialize at simulation start."""

    starting_cash: int = Field(description="Initial cash in cents")
    symbol: str = Field(description="Ticker symbol to trade")
    latency_ns: int = Field(description="Simulated network latency in nanoseconds")
    mkt_open_ns: int | None = Field(default=None, description="Market open (nanoseconds)")
    mkt_close_ns: int | None = Field(default=None, description="Market close (nanoseconds)")


class StrategicAgent(Protocol):
    """Protocol for LLM-generated trading strategies."""

    def initialize(self, config: AgentConfig) -> None:
        """Called once at simulation start."""
        ...

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        """Called on every periodic wakeup (time-driven)."""
        ...

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        """Called when new market data (L1/L2) is received."""
        ...

    def on_order_update(self, update: Order) -> list[OrderAction]:
        """Called on order fills, cancellations, modifications, and replacements."""
        ...

    def on_simulation_end(self, final_state: MarketState) -> None:
        """Called once at the end of the simulation (market is closed)."""
        ...
```

### Design Rationale

| Decision | Why |
|----------|-----|
| `pydantic` models | Runtime validation and clear schema definition |
| `Protocol` | Loose coupling — agents can be any class implementing the interface |
| `OrderAction` | Unifies placement, cancellation, and modification into a single return type |
| `OrderActionType` | Explicit action intent; backward-compatible via `_infer_action_type` validator |
| `MarketState` | Consolidates portfolio, market data, and situational signals into one snapshot |
| `computed_field` | `mid_price` and `spread` derived from L1 data without manual computation |
| `on_market_data` | Explicit hook for data-driven strategies (vs time-driven `on_tick`) |
| Factories | `cancel()`, `modify()`, `replace()` ensure valid field combinations |

---

## 9. Adapter Class — Bridging Into ABIDES

```python
"""strategic_agent_adapter.py — Bridges the StrategicAgent protocol into ABIDES."""

import logging

from abides_core.utils import str_to_ns
from abides_markets.agents import TradingAgent
from abides_markets.messages.query import QuerySpreadResponseMsg
from abides_markets.orders import LimitOrder, Side as AbidesSide

from rohan.simulation.models.strategy_api import (
    CANCEL_ALL, AgentConfig, MarketState, Order, OrderAction, OrderActionType,
    OrderStatus, OrderType, Side, StrategicAgent,
)

logger = logging.getLogger(__name__)


class StrategicAgentAdapter(TradingAgent):
    """
    ABIDES TradingAgent that wraps a StrategicAgent implementation.

    Lifecycle mapping:
        kernel_starting  → strategy.initialize()
        wakeup           → strategy.on_tick()
        receive_message  → strategy.on_market_data()
        order_accepted / order_executed / order_cancelled
           / order_modified / order_partial_cancelled
           / order_replaced  → strategy.on_order_update()
        market_closed    → strategy.on_tick(is_market_closed=True)
        kernel_stopping  → strategy.on_simulation_end()

    Features:
        • Incremental _open_orders_cache (avoids rebuilding every tick)
        • Cumulative fill tracking via _filled_quantities
        • Portfolio valuation via mark_to_market()
        • Near-touch liquidity via get_known_liquidity()
        • match-based action dispatch (PLACE/CANCEL/MODIFY/...)
    """

    def __init__(self, id, strategy, symbol, starting_cash, wake_up_freq="1S",
                 order_book_depth=10, log_orders=True, random_state=None):
        super().__init__(id=id, name=f"StrategicAgent_{id}", type="StrategicAgent",
                         starting_cash=starting_cash, log_orders=log_orders,
                         random_state=random_state)
        self.strategy = strategy
        self.symbol = symbol
        self.wake_up_freq_ns = str_to_ns(wake_up_freq)
        self.order_book_depth = order_book_depth
        self._open_orders_cache: dict[int, Order] = {}
        self._filled_quantities: dict[int, int] = {}

    # ── Lifecycle ──────────────────────────────────────────────────

    def kernel_starting(self, start_time):
        super().kernel_starting(start_time)
        cfg = AgentConfig(
            starting_cash=self.starting_cash, symbol=self.symbol, latency_ns=0,
            mkt_open_ns=self.mkt_open,   # None until MarketHoursMsg arrives
            mkt_close_ns=self.mkt_close, # None until MarketHoursMsg arrives
        )
        self.strategy.initialize(cfg)
        self.set_wakeup(start_time + self.wake_up_freq_ns)

    def kernel_stopping(self):
        state = self._build_market_state(self.current_time)
        self.strategy.on_simulation_end(state)
        super().kernel_stopping()

    # ── Wakeup & Message Handling ─────────────────────────────────

    def wakeup(self, current_time):
        super().wakeup(current_time)
        state = self._build_market_state(current_time)
        actions = self.strategy.on_tick(state)
        self._execute_actions(actions)
        self.get_current_spread(self.symbol, depth=self.order_book_depth)
        self.set_wakeup(current_time + self.wake_up_freq_ns)

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)
        if isinstance(message, QuerySpreadResponseMsg):
            self._handle_market_data(current_time)

    # ── Order lifecycle overrides ─────────────────────────────────

    def order_accepted(self, order):     # Exchange accepted → ACCEPTED
        super().order_accepted(order)
        self._cache_add(order)
        actions = self.strategy.on_order_update(
            self._abides_to_proto(order, status=OrderStatus.ACCEPTED))
        self._execute_actions(actions)

    def order_executed(self, order):     # Fill → PARTIAL or FILLED
        # ... cumulative fill tracking, cache update, protocol conversion ...
        ...

    def order_cancelled(self, order):    # → CANCELLED
        ...

    def order_modified(self, order):     # → MODIFIED
        ...

    def order_partial_cancelled(self, order):  # → PARTIAL_CANCELLED
        ...

    def order_replaced(self, old_order, new_order):  # → REPLACED
        ...

    def market_closed(self):             # → on_tick(is_market_closed=True)
        super().market_closed()
        state = self._build_market_state(self.current_time)
        state = state.model_copy(update={"is_market_closed": True})
        actions = self.strategy.on_tick(state)
        self._execute_actions(actions)

    # ── Internal helpers ──────────────────────────────────────────

    def _build_market_state(self, current_time) -> MarketState:
        """Build MarketState from TradingAgent caches.

        Includes: L1/L2 data, portfolio (inventory, cash, portfolio_value,
        unrealized_pnl), time_remaining_ns, bid/ask_liquidity.
        """
        ...

    def _execute_actions(self, actions: list[OrderAction]) -> None:
        """Dispatch on action_type using match statement."""
        for action in actions:
            match action.action_type:
                case OrderActionType.CANCEL:          self._handle_cancel(...)
                case OrderActionType.CANCEL_ALL:      self._handle_cancel(CANCEL_ALL)
                case OrderActionType.MODIFY:          self._handle_modify(action)
                case OrderActionType.PARTIAL_CANCEL:  self._handle_partial_cancel(action)
                case OrderActionType.REPLACE:         self._handle_replace(action)
                case OrderActionType.PLACE:           self._handle_place(action)

    def _handle_place(self, action):  # LIMIT with is_hidden/is_post_only, or MARKET
        ...
    def _handle_modify(self, action):  # Modify price/quantity of existing order
        ...
    def _handle_partial_cancel(self, action):  # Reduce quantity
        ...
    def _handle_replace(self, action):  # Atomic cancel + place
        ...
```

> **Note**: The above is a structural overview. See the full implementation in
> `src/rohan/simulation/abides_impl/strategic_agent_adapter.py` (461 lines).

---

## 10. Config Integration — Injecting Your Agent

ABIDES simulations are configured via Python functions that return a config dictionary. To inject your custom agent, you add it to the agent list. Below is the minimal pattern:

```python
"""my_simulation.py — Running a simulation with your custom agent."""

import numpy as np
from abides_core import abides
from abides_markets.configs.rmsc04 import build_config

from rohan.simulation.models.strategy_api import (
    StrategicAgent, MarketState, OrderAction, Side, OrderType, AgentConfig, Order
)
from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter


# ── Step 1: Implement your strategy ──────────────────────────────

class SimpleSpreadStrategy(StrategicAgent):
    """Example: place limit orders around the midpoint."""

    def initialize(self, config: AgentConfig):
        self.symbol = config.symbol
        self.tick_size = 10_000_000_000  # 10 seconds

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        if state.mid_price is None:
            return []

        spread_offset = 50  # 50 cents = $0.50

        return [
            OrderAction(
                side=Side.BID,
                quantity=10,
                order_type=OrderType.LIMIT,
                price=state.mid_price - spread_offset,
            ),
            OrderAction(
                side=Side.ASK,
                quantity=10,
                order_type=OrderType.LIMIT,
                price=state.mid_price + spread_offset,
            ),
        ]

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_order_update(self, update: Order) -> list[OrderAction]:
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        # unrealized_pnl tracks portfolio_value - starting_cash automatically
        print(f"Strategy PnL: ${final_state.unrealized_pnl / 100:.2f}")


# ── Step 2: Build config and inject adapter ──────────────────────

def run_simulation(seed: int = 42):
    # Build the standard background agent config
    config = build_config(seed=seed, end_time="11:00:00")

    # Create master RNG for reproducibility
    master_rng = np.random.RandomState(seed)

    # Create your strategy and wrap it in the adapter
    strategy = SimpleSpreadStrategy()

    # The adapter needs a unique agent ID (append after all other agents)
    next_id = len(config["agents"])

    adapter_agent = StrategicAgentAdapter(
        id=next_id,
        strategy=strategy,
        symbol="ABM",
        starting_cash=10_000_000,
        name="MyCustomStrategy",
        random_state=np.random.RandomState(
            seed=master_rng.randint(low=0, high=2**32, dtype="uint64")
        ),
    )

    # Inject into the agent list
    config["agents"].append(adapter_agent)

    # ... rest of setup (latency model update etc.) ...
```

> [!NOTE]
> **Agent ID must be unique and sequential.** ABIDES uses agent IDs as indices into arrays (`agent_current_times`, `agent_computation_delays`, `agent_latency`). Your agent's `id` must equal its position in the `agents` list. Use `len(config["agents"])` before appending.

> [!WARNING]
> **The latency model may need resizing.** The `build_config` functions create a latency model sized for their agent count. When appending a new agent, ensure the latency model covers the new agent count. The simplest fix: rebuild the latency model after appending, or use the default scalar latency. See below:

```python
from abides_markets.utils import generate_latency_model

# After appending your agent:
new_count = len(config["agents"])
latency_rng = np.random.RandomState(seed=seed + 1)
config["agent_latency_model"] = generate_latency_model(new_count, latency_rng)
```

---

## 11. Extracting Results After Simulation

The simulation returns an `end_state` dictionary with all agents and kernel state:

```python
end_state = abides.run(config)

# Access all agents
agents = end_state["agents"]

# Your adapter agent (by index)
my_agent = agents[your_agent_id]

# Portfolio metrics (from TradingAgent)
print(f"Final cash: ${my_agent.holdings['CASH'] / 100:.2f}")
print(f"Share position: {my_agent.holdings.get('ABM', 0)}")

# Event log (list of (timestamp, event_type, event_data) tuples)
for timestamp, event_type, event_data in my_agent.log[-10:]:
    print(f"  {event_type}: {event_data}")

# Exchange agent order book data
exchange = agents[0]

# Summary log from kernel
# Written to log/<log_dir>/summary_log.bz2 if skip_log=False
```

---

## 12. Gotchas & Pitfalls

### 12.1 The Async Mind Shift

**Nothing is synchronous.** When you call `self.get_current_spread(symbol)`, the data arrives *later* in `receive_message()`. You cannot do:

```python
# ❌ WRONG — This does not work
self.get_current_spread("ABM")
bid = self.known_bids["ABM"][0][0]  # May be stale or crash
```

Instead, use the **state machine pattern** (as done by `ValueAgent` and `NoiseAgent`):

```python
# ✅ Correct — Request data, then act on receipt
def wakeup(self, current_time):
    super().wakeup(current_time)
    self.get_current_spread(self.symbol)
    self.state = "AWAITING_SPREAD"

def receive_message(self, current_time, sender_id, message):
    super().receive_message(current_time, sender_id, message)
    if self.state == "AWAITING_SPREAD" and isinstance(message, QuerySpreadResponseMsg):
        # NOW self.known_bids and self.known_asks are fresh
        self.place_orders()
        self.state = "AWAITING_WAKEUP"
```

**Our adapter avoids this** by:
- Subscribing to L2 data (push-based, always fresh).
- Using `known_bids`/`known_asks` caches populated by subscription updates.
- The first few ticks may have `None` bid/ask values — strategy must handle this.

### 12.2 Random State Requirements

Every agent **must** have a `random_state` parameter (a `np.random.RandomState` instance). The base `Agent.__init__` will raise `ValueError` if it's `None`. Always create one from a seeded RNG for reproducibility.

### 12.3 Order ID Collisions

ABIDES uses a global `Order._order_id_counter`. Across agents, order IDs are unique. Don't manually set `order_id` unless implementing a replay scenario.

### 12.4 Cash is Always Cents

`$100.00` = `10_000` cents. `$1,000.00` = `100_000` cents. Forgetting this will produce wildly incorrect PnL.

### 12.5 Computation Delay

The default `computation_delay` is typically 1–50 nanoseconds. This is the time an agent is "busy" after each event. If your strategy needs to send multiple messages in one wake cycle, they all get sent at `current_time + computation_delay`. Use `self.delay(additional_ns)` to stagger them.

### 12.6 Log Events for Post-Analysis

If you want custom metrics in the summary log, use:

```python
self.logEvent("MY_SIGNAL", signal_value, append_summary_log=True)
```

The `append_summary_log=True` flag makes it available in the global summary log, not just the per-agent log.

---

## 13. Quick-Start Recipes

### 13.1 Strategy That Trades Once Per Minute

```python
class MinuteTrader(StrategicAgent):
    def initialize(self, config):
        pass

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        # Logic to check time or just rely on wakeup frequency
        if state.best_ask and state.inventory == 0:
            return [OrderAction(
                side=Side.BID,
                quantity=100,
                order_type=OrderType.MARKET,
                price=None
            )]
        return []

    # ... implement remaining protocol methods ...
```

### 13.2 Strategy Using L2 Depth

```python
def on_market_data(self, state: MarketState) -> list[OrderAction]:
    """React to every L2 update."""
    if len(state.bid_depth) >= 3 and len(state.ask_depth) >= 3:
        # Sum volume at top 3 bid levels
        bid_vol = sum(qty for _, qty in state.bid_depth[:3])
        ask_vol = sum(qty for _, qty in state.ask_depth[:3])

        if bid_vol > 2 * ask_vol:
            # Heavy bid imbalance → buy signal
            return [OrderAction(
                side=Side.BID,
                quantity=10,
                order_type=OrderType.MARKET,
                price=None
            )]
    return []
```

### 13.3 Order Management — Modify, Partial Cancel, Replace

```python
def on_order_update(self, update: Order) -> list[OrderAction]:
    """React to order events with advanced order management."""
    if update.status == OrderStatus.PARTIAL:
        # Reduce remaining quantity by half after partial fill
        remaining = update.quantity - update.filled_quantity
        if remaining > 10:
            return [OrderAction.partial_cancel(update.order_id, reduce_by=remaining // 2)]

    if update.status == OrderStatus.ACCEPTED:
        # Modify price closer to midpoint after acceptance
        return [OrderAction.modify(update.order_id, new_price=update.price - 10)]

    return []
```

### 13.4 Portfolio-Aware Risk Management

```python
def on_tick(self, state: MarketState) -> list[OrderAction]:
    """Use situational awareness to manage risk."""
    # Flatten positions near market close
    if state.time_remaining_ns is not None and state.time_remaining_ns < 60_000_000_000:  # <1 min
        if state.inventory != 0:
            side = Side.ASK if state.inventory > 0 else Side.BID
            return [OrderAction(
                side=side, quantity=abs(state.inventory),
                order_type=OrderType.MARKET, price=None,
            )]

    # Stop trading if losing too much
    if state.unrealized_pnl < -500_00:  # -$500
        return [OrderAction.cancel_all()]

    return []
```

### 13.5 Extending the Adapter

If you need additional ABIDES functionality (e.g., oracle access), subclass the adapter:

```python
class ExtendedAdapter(StrategicAgentAdapter):
    def kernel_starting(self, start_time):
        super().kernel_starting(start_time)
        # Access oracle fundamental value
        self.oracle = self.kernel.oracle
```

---

## Summary

| Layer | File | Responsibility |
|-------|------|---------------|
| **Strategy Protocol** | `strategy_api.py` | Pure financial logic, no ABIDES dependency. Enums, models, computed fields, order factories, validators. |
| **Adapter** | `strategic_agent_adapter.py` | Translates all ABIDES lifecycle events, manages order cache, dispatches actions (place/cancel/modify/replace). |
| **Config** | `simulation_runner.py` | Wires adapter into ABIDES config, runs simulation |

The key insight: **ABIDES agents are event reactors, not loop runners.** Your strategy sees clean snapshots and returns order actions. The adapter handles the event-driven complexity, async message passing, and lifecycle management that makes ABIDES work.
