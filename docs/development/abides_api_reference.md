# ABIDES API Reference for LLMs

**Purpose**: Quick reference for using ABIDES (abides-core + abides-markets) in external projects.
**Audience**: LLMs implementing features that use ABIDES simulation framework.

---

## Quick Start

```python
from abides_core import abides
from abides_markets.configs import rmsc04

# Run simulation
config = rmsc04.build_config(seed=0, start_time="09:30:00", end_time="16:00:00")
end_state = abides.run(config)

# Access results
order_book = end_state["agents"][0].order_books["ABM"]
```

---

## Core Concepts

### Critical Facts
- **Time**: All timestamps are nanoseconds (int64)
- **Prices**: All prices are cents (int: $100.50 = 10050)
- **Exchange**: Always agent ID 0
- **Async**: All inter-agent communication is message-based with latency

### Imports
```python
# Core
from abides_core import abides, Agent, Message, NanosecondTime
from abides_core.utils import str_to_ns, ns_date, fmt_ts
from abides_core.generators import PoissonTimeGenerator, ConstantTimeGenerator

# Markets
from abides_markets.agents import (
    ExchangeAgent, TradingAgent, ValueAgent, NoiseAgent,
    MomentumAgent, AdaptiveMarketMakerAgent
)
from abides_markets.orders import Side, LimitOrder, MarketOrder
from abides_markets.order_book import OrderBook
from abides_markets.messages import (
    MarketDataMsg, OrderExecutedMsg, OrderAcceptedMsg
)
from abides_markets.configs import rmsc03, rmsc04
from abides_markets.utils import dollarize, generate_latency_model
```

---

## Time API

```python
from abides_core.utils import str_to_ns, ns_date, fmt_ts

# Duration conversion
str_to_ns("10S")       # → 10_000_000_000 (10 seconds)
str_to_ns("1min")      # → 60_000_000_000
str_to_ns("30ms")      # → 30_000_000
str_to_ns("1H")        # → 3_600_000_000_000

# Time of day (from midnight)
str_to_ns("09:30:00")  # Market open

# Date handling
ns_date("2020-06-03")  # Midnight on date
market_open = ns_date("2020-06-03") + str_to_ns("09:30:00")

# Format for display
fmt_ts(timestamp)      # "YYYY-MM-DD HH:MM:SS.microseconds"
```

---

## Configuration API

### Build Configuration
```python
config = {
    "seed": int,                      # Random seed
    "start_time": int,                # Nanoseconds
    "stop_time": int,                 # Nanoseconds
    "agents": List[Agent],            # All agents (Exchange first)
    "agent_latency_model": dict,      # Communication latency
    "default_computation_delay": int, # Agent processing delay
    "custom_properties": dict,
    "log_dir": str,
}
```

### Pre-built Configs
```python
from abides_markets.configs import rmsc03, rmsc04

# rmsc04: 1 Exchange + 2 MMs + 102 Value + 12 Momentum + 1000 Noise
config = rmsc04.build_config(
    seed=0,
    ticker="ABM",
    start_time="09:30:00",
    end_time="16:00:00",
    num_value_agents=100,
    num_noise_agents=1000,
    mm_pov=0.025,                  # Market maker participation
    mm_window_size="adaptive",     # or float for fixed spread
    exchange_log_orders=True,
    book_logging=True,
)
```

### Add Custom Agents
```python
from abides_markets.utils import config_add_agents

base_config = rmsc04.build_config(seed=0)
new_agents = [MyAgent(id=len(base_config["agents"]) + i) for i in range(10)]
config = config_add_agents(base_config, new_agents)
```

---

## Agent API

### Base Agent (abides-core)
```python
class Agent:
    # Lifecycle
    def kernel_starting(self, start_time: NanosecondTime) -> None
    def kernel_stopping(self) -> None
    def wakeup(self, current_time: NanosecondTime) -> None
    def receive_message(self, current_time: NanosecondTime,
                       sender_id: int, message: Message) -> None

    # Communication
    def send_message(self, recipient_id: int, message: Message, delay: int = 0)
    def set_wakeup(self, requested_time: NanosecondTime)
    def delay(self, additional_delay: int)

    # Logging
    def logEvent(self, event_type: str, event_data)
```

### TradingAgent (abides-markets)
```python
class TradingAgent(Agent):
    # Attributes
    holdings: Dict[str, int]  # {"CASH": int, "ABM": int, ...}
    orders: Dict[int, LimitOrder]
    known_bids: Dict[str, List[Tuple[int, int]]]  # [(price, qty)]
    known_asks: Dict[str, List[Tuple[int, int]]]

    # Order placement
    def place_limit_order(self, symbol: str, quantity: int,
                         side: Side, limit_price: int,
                         order_id: int = None, is_hidden: bool = False)
    def place_market_order(self, symbol: str, quantity: int, side: Side)
    def cancel_order(self, order: LimitOrder)

    # Market data
    def get_current_spread(self, symbol: str, depth: int = 1)  # Async request
    def subscribe_to_market_data(self, symbol: str, depth: int = 1)
    def get_known_bid_ask(self, symbol: str, best: bool = True)
        # → (bid_price, ask_price) or (None, None)
    def get_known_liquidity(self, symbol: str, within: float = 0.00)
        # → (bid_liquidity, ask_liquidity)

    # Portfolio
    def mark_to_market(self, holdings: Dict[str, int],
                      use_midpoint: bool = False) → int  # Total value
    def get_holdings(self, symbol: str) → int
    def get_cash() → int
```

### Custom Agent Template
```python
from abides_markets.agents import TradingAgent
from abides_markets.orders import Side
from abides_core.utils import str_to_ns

class MyAgent(TradingAgent):
    def __init__(self, id, symbol, starting_cash, **params):
        super().__init__(id=id, name=f"MyAgent_{id}", type="MyAgent",
                        starting_cash=starting_cash)
        self.symbol = symbol
        self.state = "INIT"

    def kernel_starting(self, start_time):
        super().kernel_starting(start_time)
        self.set_wakeup(start_time + str_to_ns("10S"))

    def wakeup(self, current_time):
        super().wakeup(current_time)
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_SPREAD"

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)

        from abides_markets.messages import QuerySpreadResponseMsg
        if isinstance(message, QuerySpreadResponseMsg):
            bid, ask = self.get_known_bid_ask(self.symbol)
            if bid and ask:
                # Trading logic here
                self.place_limit_order(self.symbol, 100, Side.BID, bid + 1)
            self.set_wakeup(current_time + str_to_ns("10S"))
```

---

## OrderBook API

```python
# Access from simulation results
order_book = end_state["agents"][0].order_books["ABM"]

# L1 (Best bid/ask)
best_bid = order_book.get_best_bid()    # → int (cents) or None
best_ask = order_book.get_best_ask()    # → int (cents) or None
spread = order_book.get_spread()        # → int (cents) or None
mid = order_book.get_mid_price()        # → int (cents) or None

# L1 History
snapshots = order_book.get_L1_snapshots()
# → List[[timestamp, best_bid, best_ask, bid_size, ask_size, last_trade, volume]]

# L2 (Aggregated by price)
bids = order_book.get_l2_bid_data(depth=10)  # → [(price, total_qty)]
asks = order_book.get_l2_ask_data(depth=10)

# L3 (Individual orders)
bids = order_book.get_l3_bid_data(depth=10)  # → [(price, [qty1, qty2, ...])]
asks = order_book.get_l3_ask_data(depth=10)

# Volume
bid_vol, ask_vol = order_book.get_transacted_volume(lookback_period="10min")

# Imbalance
imbalance, side = order_book.get_imbalance(depth=5)
# imbalance: float in [-1, 1], side: Side.BID or Side.ASK

# Last trade
last_price = order_book.last_trade  # int (cents) or None
```

---

## Message Types

### Receiving Messages
```python
def receive_message(self, current_time, sender_id, message):
    from abides_markets.messages import (
        QuerySpreadResponseMsg, OrderAcceptedMsg, OrderExecutedMsg,
        OrderPartialExecutedMsg, OrderCancelledMsg, MarketDataMsg
    )

    if isinstance(message, QuerySpreadResponseMsg):
        # message.bids: [(price, qty), ...]
        # message.asks: [(price, qty), ...]
        # message.last_trade: int or None
        pass
    elif isinstance(message, OrderExecutedMsg):
        # message.order_id, message.quantity, message.price
        pass
    elif isinstance(message, OrderAcceptedMsg):
        # message.order_id
        pass
```

### Sending Orders
```python
# Orders are sent via TradingAgent methods, not direct messages:
self.place_limit_order(symbol, quantity, Side.BID, limit_price)
self.place_market_order(symbol, quantity, Side.ASK)
self.cancel_order(order)
```

---

## Pre-built Agents

### ExchangeAgent
```python
ExchangeAgent(
    id=0,                          # Always 0
    mkt_open=market_open_ns,
    mkt_close=market_close_ns,
    symbols=["ABM"],
    random_state=np.random.RandomState(seed),
    book_logging=True,
    log_orders=True,
)
```

### ValueAgent
```python
ValueAgent(
    id=1,
    symbol="ABM",
    starting_cash=10_000_000,      # $100k in cents
    sigma_n=1_000_000,             # Noise in value estimate
    r_bar=100_000,                 # Mean fundamental value
    kappa=1.67e-15,                # Mean reversion rate
    random_state=np.random.RandomState(seed),
)
```

### NoiseAgent
```python
NoiseAgent(
    id=2,
    symbol="ABM",
    starting_cash=10_000_000,
    wakeup_time=market_open_ns,
    random_state=np.random.RandomState(seed),
)
```

### MomentumAgent
```python
MomentumAgent(
    id=3,
    symbol="ABM",
    starting_cash=10_000_000,
    min_size=1,
    max_size=10,
    wake_up_freq=str_to_ns("20S"),
    random_state=np.random.RandomState(seed),
)
```

### AdaptiveMarketMakerAgent
```python
AdaptiveMarketMakerAgent(
    id=4,
    symbol="ABM",
    starting_cash=10_000_000,
    pov=0.025,                     # 2.5% participation
    window_size="adaptive",        # or float (e.g., 0.001)
    num_ticks=10,                  # Levels per side
    level_spacing=5,               # Ticks between levels
    wake_up_freq=str_to_ns("10S"),
    skew_beta=0,                   # Inventory skew
    random_state=np.random.RandomState(seed),
)
```

---

## Common Patterns

### Run Simulation and Extract Data
```python
import pandas as pd

config = rmsc04.build_config(seed=0, start_time="09:30:00", end_time="16:00:00")
end_state = abides.run(config)

# Order book analysis
ob = end_state["agents"][0].order_books["ABM"]
df = pd.DataFrame(ob.get_L1_snapshots(),
                  columns=["time", "bid", "ask", "bid_sz", "ask_sz", "trade", "vol"])
df["spread"] = df["ask"] - df["bid"]
df["mid"] = (df["bid"] + df["ask"]) / 2

# Agent analysis
for agent in end_state["agents"][1:]:
    if hasattr(agent, "holdings"):
        pnl = agent.mark_to_market(agent.holdings) - agent.starting_cash
        print(f"{agent.type} {agent.id}: PnL=${pnl/100:.2f}")
```

### Parameter Sweep
```python
results = []
for seed in range(10):
    for num_noise in [500, 1000, 2000]:
        config = rmsc04.build_config(seed=seed, num_noise_agents=num_noise)
        end_state = abides.run(config)
        ob = end_state["agents"][0].order_books["ABM"]
        avg_spread = np.mean([a-b for _, b, a, *_ in ob.get_L1_snapshots() if b and a])
        results.append({"seed": seed, "noise": num_noise, "spread": avg_spread})
```

### Custom Configuration from Scratch
```python
import numpy as np

def build_config(seed=0):
    date = "20200603"
    mkt_open = ns_date(date) + str_to_ns("09:30:00")
    mkt_close = ns_date(date) + str_to_ns("16:00:00")

    agents = [
        ExchangeAgent(id=0, mkt_open=mkt_open, mkt_close=mkt_close,
                     symbols=["ABM"], random_state=np.random.RandomState(seed)),
    ]

    for i in range(100):
        agents.append(NoiseAgent(
            id=i+1, symbol="ABM", starting_cash=10_000_000,
            wakeup_time=mkt_open, random_state=np.random.RandomState(seed+i+1)
        ))

    return {
        "seed": seed,
        "start_time": mkt_open,
        "stop_time": mkt_close,
        "agents": agents,
        "agent_latency_model": generate_latency_model(len(agents), "deterministic"),
        "default_computation_delay": 0,
        "custom_properties": {},
        "log_dir": None,
    }
```

---

## Utilities

### Price Conversion
```python
from abides_markets.utils import dollarize

dollarize(10050)              # "$100.50"
dollarize([10050, 10100])     # ["$100.50", "$101.00"]
```

### Log Parsing
```python
from abides_core.utils import parse_logs_df

logs_df = parse_logs_df(end_state)
# Columns: EventTime, AgentID, AgentType, EventType, Event
value_logs = logs_df[logs_df["AgentType"] == "ValueAgent"]
order_logs = logs_df[logs_df["EventType"] == "ORDER_SUBMITTED"]
```

### Random Generators
```python
from abides_core.generators import PoissonTimeGenerator, ConstantTimeGenerator

poisson_gen = PoissonTimeGenerator(mean_interval=str_to_ns("10S"))
next_time = poisson_gen.next(current_time)  # Random ~10S later

const_gen = ConstantTimeGenerator(interval=str_to_ns("10S"))
next_time = const_gen.next(current_time)    # Exactly 10S later
```

---

## Debugging

### Enable Logging
```python
config = rmsc04.build_config(
    exchange_log_orders=True,
    log_orders=True,
    stdout_log_level="DEBUG",  # or "INFO", "WARNING"
)
```

### Agent Logging
```python
self.logEvent("DEBUG", f"Spread: {ask-bid}, Position: {self.holdings.get(symbol, 0)}")
```

### Common Issues
1. **No market data**: Call `self.get_current_spread(symbol)` before using `get_known_bid_ask()`
2. **Orders not executing**: Check if price crosses spread for immediate execution
3. **Agent not waking**: Ensure `set_wakeup(time)` is called in `wakeup()` or `receive_message()`
4. **Negative cash**: Check order sizes don't exceed available capital

---

## Key Enums and Constants

```python
from abides_markets.orders import Side

Side.BID  # Buy side
Side.ASK  # Sell side

# Default symbol
"ABM"

# Exchange ID
0
```

---

## Complete Example: Custom Strategy

```python
from abides_core import abides
from abides_core.utils import str_to_ns, ns_date
from abides_markets.agents import TradingAgent
from abides_markets.orders import Side
from abides_markets.configs import rmsc04
from abides_markets.utils import config_add_agents
import numpy as np

class SimpleStrategy(TradingAgent):
    def __init__(self, id, symbol, starting_cash, spread_threshold):
        super().__init__(id=id, name=f"Strategy_{id}", type="Strategy",
                        starting_cash=starting_cash)
        self.symbol = symbol
        self.spread_threshold = spread_threshold
        self.state = "INIT"

    def kernel_starting(self, start_time):
        super().kernel_starting(start_time)
        self.set_wakeup(start_time + str_to_ns("30S"))

    def wakeup(self, current_time):
        super().wakeup(current_time)
        self.get_current_spread(self.symbol)
        self.state = "AWAITING_SPREAD"

    def receive_message(self, current_time, sender_id, message):
        super().receive_message(current_time, sender_id, message)

        if self.state == "AWAITING_SPREAD":
            bid, ask = self.get_known_bid_ask(self.symbol)
            if bid and ask and (ask - bid) >= self.spread_threshold:
                mid = (bid + ask) // 2
                self.place_limit_order(self.symbol, 100, Side.BID, mid)
                self.place_limit_order(self.symbol, 100, Side.ASK, mid)

            self.set_wakeup(current_time + str_to_ns("30S"))
            self.state = "AWAITING_WAKEUP"

# Use in simulation
config = rmsc04.build_config(seed=0)
agents = [SimpleStrategy(
    id=len(config["agents"]) + i,
    symbol="ABM",
    starting_cash=10_000_000,
    spread_threshold=20
) for i in range(5)]
config = config_add_agents(config, agents)
end_state = abides.run(config)

# Analyze
for agent in end_state["agents"]:
    if agent.type == "Strategy":
        pnl = agent.mark_to_market(agent.holdings) - agent.starting_cash
        print(f"Agent {agent.id}: PnL = ${pnl/100:.2f}")
```

---

## File Locations

**Core**: `abides-core/abides_core/`
- `kernel.py`, `agent.py`, `message.py`, `utils.py`, `generators.py`

**Markets**: `abides-markets/abides_markets/`
- `agents/`: `exchange_agent.py`, `trading_agent.py`, `value_agent.py`, etc.
- `order_book.py`, `orders.py`
- `messages/`: `market.py`, `order.py`, `query.py`
- `configs/`: `rmsc03.py`, `rmsc04.py`
- `utils/`: Various utilities

---

## Quick Reference Card

| Operation | Code |
|-----------|------|
| Time to ns | `str_to_ns("10S")` |
| Date to ns | `ns_date("2020-06-03")` |
| Price to $ | `dollarize(10050)` → "$100.50" |
| Run sim | `abides.run(config)` |
| Get order book | `end_state["agents"][0].order_books["ABM"]` |
| Best bid/ask | `ob.get_best_bid()`, `ob.get_best_ask()` |
| Place order | `self.place_limit_order(symbol, qty, Side.BID, price)` |
| Request spread | `self.get_current_spread(symbol)` |
| Get cached data | `self.get_known_bid_ask(symbol)` |
| Schedule wakeup | `self.set_wakeup(time)` |
| Log event | `self.logEvent(type, data)` |

---

**End of API Reference**
