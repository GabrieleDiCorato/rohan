# Analyzing ABIDES Output

## 1. Recommended: `SimulationResult`

`run_simulation()` returns a typed, immutable `SimulationResult` with structured data:

```python
from abides_markets.config_system import SimulationBuilder
from abides_markets.simulation import run_simulation, ResultProfile

config = (SimulationBuilder()
    .from_template("rmsc04")
    .seed(42)
    .build())

result = run_simulation(config, profile=ResultProfile.QUANT)
```

**`SimulationResult` fields:**

| Field | Content |
|-------|---------|
| `result.metadata` | Seed, tickers, sim start/end (ns), wall-clock elapsed |
| `result.agents` | List of `AgentData` — per-agent `pnl_cents`, `pnl_pct`, `final_holdings`, `mark_to_market_cents` |
| `result.markets` | Dict of `MarketSummary` — per-symbol `l1_close`, `liquidity`, optional `l1_series`/`l2_series` |
| `result.logs` | Raw agent log DataFrame (only with `ResultProfile.FULL`) |

**`ResultProfile` controls depth:**
- `SUMMARY` (default) — metadata, per-agent PnL, liquidity metrics (KB-scale)
- `QUANT` — adds L1/L2 time-series as numpy arrays
- `FULL` — adds raw agent log DataFrame

```python
# Per-agent PnL
for agent in result.agents:
    print(f"{agent.agent_name}: ${agent.pnl_cents / 100:.2f} ({agent.pnl_pct:.2%})")

# Human-readable summary
print(result.summary())
```

---

## 2. Low-Level Extraction (via `abides.run()`)

For advanced use cases (gymnasium environments, custom Kernel access), you can work with the raw `end_state` dict returned by `abides.run()`. This gives access to complete agent logs and order book history.

```python
from abides_markets.config_system import compile
from abides_core import abides

runtime = compile(config)           # fresh runtime dict — consumed once
end_state = abides.run(runtime)
```

### Extracting Agent Logs

All events logged by agents during the simulation (order submissions, executions, holdings updates) can be extracted into a single DataFrame.

```python
from abides_core.utils import parse_logs_df

# end_state is the dictionary returned by abides.run()
logs_df = parse_logs_df(end_state)

# Example: Get all executed orders for a specific agent type
executions = logs_df[
    (logs_df.agent_type == "MyCustomAgent") &
    (logs_df.EventType == "ORDER_EXECUTED")
]

# Example: Get all submitted orders
submissions = logs_df[logs_df.EventType == "ORDER_SUBMITTED"]
```

### Extracting Order Book History

By convention, the first agent in `end_state["agents"]` is the `ExchangeAgent`. You can access its internal `OrderBook` to retrieve historical snapshots.

**Note:** The ExchangeAgent must have `log_orders=True` (usually the default in standard configs) for history to be populated.

```python
from abides_core.utils import ns_date

# 1. Get the order book for the target symbol (e.g., "AAPL")
order_book = end_state["agents"][0].order_books["AAPL"]

# 2. Extract Level 1 snapshots (Best Bid & Ask at every tick)
L1 = order_book.get_L1_snapshots()

# Convert to DataFrames
best_bids = pd.DataFrame(L1["best_bids"], columns=["time", "price", "qty"])
best_asks = pd.DataFrame(L1["best_asks"], columns=["time", "price", "qty"])

# Convert raw ns timestamps (ns from 1970) to ns from midnight for easier plotting
best_bids["time"] = best_bids["time"].apply(lambda x: x - ns_date(x))
```

### Extracting Level 2 History

You can also extract deeper order book snapshots up to a specified depth.

```python
# Extract top 10 levels of the order book
L2 = order_book.get_L2_snapshots(nlevels=10)

# L2 returns a dict with:
# "times": list of timestamps
# "bids": numpy array of shape (time_steps, nlevels, 2) -> [price, qty]
# "asks": numpy array of shape (time_steps, nlevels, 2) -> [price, qty]

# Example: Plotting the 5th best bid/ask over time
times = [t - ns_date(t) for t in L2["times"]]
plt.plot(times, L2["bids"][:, 4, 0])  # bids[time_idx, depth_idx, 0=price]
plt.plot(times, L2["asks"][:, 4, 0])  # asks[time_idx, depth_idx, 0=price]
```

## 3. Time Utilities

ABIDES operates in nanoseconds. Use these core utilities when parsing timestamps:

```python
from abides_core.utils import str_to_ns, ns_date, fmt_ts

# Convert string to ns
t_start = str_to_ns("09:30:00")

# Format ns to readable string
print(fmt_ts(t_start))  # "1970-01-01 09:30:00"

# Strip date component to just get offset from midnight
ns_from_midnight = t_start - ns_date(t_start)
```

## 4. Data Types and Conventions

When analyzing ABIDES output DataFrames, keep these core conventions in mind to avoid calculation errors.

### Timestamps
- **Unit**: All times are integer nanoseconds (`1e9` per second).
- **Epoch**: The simulation runs on a specific calendar date (e.g., `2021-02-05`). A raw timestamp of `1612535400000000000` is the absolute nanosecond time since 1970.
- **Conversion to Time of Day**: Use `t - ns_date(t)` to strip the calendar day. The result is pure nanoseconds since midnight (e.g., `09:30:00` = `34_200_000_000_000`).

### Prices and Volumes
- **Prices**: All prices are **integer cents** (e.g., `$100.50` = `10050`). ABIDES never uses floats for core price representations.
- **Volumes (shares)**: Always integer quantities.

### NaN Values and Empty Books
When extracting L1 or L2 history, you will encounter `NaN` (`np.nan`) or `0`:

| Data Structure | Missing / Empty Value Representation |
|----------------|--------------------------------------|
| **L1 History** (`best_bids` DataFrame) | If the order book side becomes empty, the `price` and `qty` columns for that tick will be `NaN`. Use `.dropna()` or `.fillna()` when plotting. |
| **L2 History Arrays** (`L2["bids"]` numpy array) | If the book has fewer resting levels than the requested `nlevels`, the deeper unpopulated rows of the array (`[:, depth, :]`) will be padded with zeros (price: `0`, qty: `0`). Always filter out `price == 0`. |
| **Agent Logs** (`parse_logs_df`) | Agents that do not use `limit_price` (e.g., `MarketOrder` submissions) will have `NaN` in the `price` column of their log row. |

### L3 `.get_l3_itch()` DataFrame
If you generate an ITCH-like L3 history (`df = order_book.get_l3_itch()`), it outputs event-by-event rows.
- **`type` column**: `ADD`, `DELETE`, `CANCEL`, `EXECUTE`, `REPLACE`
- **`side` column**: `B` (Bid), `S` (Sell/Ask)

**Warning:** The ITCH format logs `side`, but for an `EXECUTE` event, the row's `side` may be set to `NaN` depending on the log parser configuration. Handle `pd.isna(df['side'])` robustly.

---

## 5. SimulationResult (High-Level API)

When using `run_simulation()`, you get an immutable `SimulationResult`
with typed accessors instead of raw dicts:

```python
from abides_markets.simulation import run_simulation

result = run_simulation(config)

# Human-readable narrative
print(result.summary())

# Structured dict for dashboards / JSON APIs
data = result.summary_dict()
# → {"metadata": {...}, "markets": {...}, "agent_leaderboard": [...], "warnings": [...]}
```

### Key fields in `summary_dict()`

| Key | Contents |
|-----|----------|
| `metadata` | seed, tickers, wall-clock elapsed time |
| `markets` | Per-symbol: bid/ask/spread (cents), last trade, VWAP, volume, liquidity % |
| `agent_leaderboard` | Top 10 agents by PnL (id, type, pnl_cents, pnl_pct, mark_to_market) |
| `warnings` | Anomaly strings: one-sided market, zero trades, etc. |

### VWAP

`LiquidityMetrics.vwap_cents` is now computed automatically from order-book
execution history. It is an integer-cents volume-weighted average price
across all trades in the session.

---

## 6. Rich Metrics API

For consumers who need agent-level analytics, microstructure indicators, and
per-fill execution analysis without reimplementing computations from raw logs,
use `compute_rich_metrics()`:

```python
from abides_markets.simulation import (
    compute_rich_metrics,
    ResultProfile,
    run_simulation,
)

result = run_simulation(config, profile=ResultProfile.QUANT)
rich = compute_rich_metrics(result)

# Per-agent analytics
for agent in rich.agents:
    print(f"Agent {agent.agent_id}: PnL={agent.total_pnl_cents}¢ "
          f"Sharpe={agent.sharpe_ratio} trades={agent.trade_count}")

# Microstructure indicators per symbol
for sym, mkt in rich.markets.items():
    m = mkt.microstructure
    print(f"{sym}: LOB imbalance={m.lob_imbalance_mean:.3f} "
          f"two-sided={m.pct_time_two_sided:.1f}%")

# Tier 3: per-fill slippage and adverse selection (opt-in)
rich = compute_rich_metrics(
    result,
    include_fills=True,
    adverse_selection_windows=["100ms", "1s"],
)
for fill in (rich.fills or []):
    print(f"  fill@{fill.price_cents}¢ slip={fill.slippage_bps}bps "
          f"AS={fill.adverse_selection_bps}")
```

See [ABIDES_METRICS_API.md](ABIDES_METRICS_API.md) for the full metric
catalogue and mathematical definitions.
