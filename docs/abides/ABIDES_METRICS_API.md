# ABIDES Metrics API Reference

Complete reference for the `abides_markets.simulation` metrics surface.
Covers standalone compute functions, the `compute_rich_metrics()` convenience
API, and the Pydantic models that carry the results.

> **Audience:** LLM tool-calling agents, quantitative researchers, and
> financial-industry practitioners building on ABIDES-Hasufel.

> **Data conventions:** All prices are **integer cents** (`$100.00 = 10_000`).
> All timestamps are **nanoseconds since Unix epoch** (`int64`).
> Basis-point fields use `1 bp = 0.01%` (`10_000 bps = 100%`).
> Fields that cannot be computed from the available data degrade to `None`
> (or `0` / `0.0` for counters and percentages).

---

## Quick Start

```python
from abides_markets.simulation import (
    compute_rich_metrics,
    ResultProfile,
    run_simulation,
)

# 1. Run a simulation with enough extraction depth
result = run_simulation(config, profile=ResultProfile.QUANT)

# 2. Compute all metrics in one call
rich = compute_rich_metrics(result)

# 3. Access structured results
for agent in rich.agents:
    print(f"[{agent.agent_id}] {agent.agent_type}: "
          f"PnL={agent.total_pnl_cents}¢  Sharpe={agent.sharpe_ratio}")

for sym, mkt in rich.markets.items():
    m = mkt.microstructure
    print(f"{sym}: imbalance={m.lob_imbalance_mean}  "
          f"resilience={m.resilience_mean_ns}ns")
```

---

## Data Availability by Profile

Richer `ResultProfile` flags yield richer metrics. All fields degrade
gracefully when the required data is absent — no exceptions are raised.

| Metric group | Minimum profile | Absent → |
|:---|:---|:---|
| `total_pnl_cents`, `end_inventory` | `SUMMARY` | Always present |
| `sharpe_ratio`, `max_drawdown_cents` | `QUANT` (includes `EQUITY_CURVE`) | `None` |
| `lob_imbalance_*`, `resilience_mean_ns`, `pct_time_two_sided` | `QUANT` (includes `L1_SERIES`) | `None` / `0.0` |
| `vwap_cents`, `trade_count`, `inventory_std` | `QUANT` (includes `TRADE_ATTRIBUTION`) | `None` / `0` |
| `slippage_bps`, `adverse_selection_bps` | `QUANT` + `include_fills=True` | `None` |
| `fill_rate_pct`, `order_to_trade_ratio`, `market_ott_ratio`, `order_lifecycles` | `FULL` (includes `AGENT_LOGS`) | `None` / `[]` |

**Recommendation:** Use `ResultProfile.QUANT` for analysis workloads.
Add `ResultProfile.AGENT_LOGS` (i.e. `ResultProfile.FULL`) when
order-level fill-rate, OTT metrics, or per-order lifecycle tracking are needed.

---

## `compute_rich_metrics()`

```python
def compute_rich_metrics(
    result: SimulationResult,
    *,
    include_fills: bool = False,
    adverse_selection_windows: Sequence[str | int] = (),
) -> RichSimulationMetrics
```

**Parameters:**

| Parameter | Type | Description |
|:---|:---|:---|
| `result` | `SimulationResult` | Output of `run_simulation()`. |
| `include_fills` | `bool` | If `True`, produce per-fill `FillRecord` entries (Tier 3). Default `False`. |
| `adverse_selection_windows` | `Sequence[str \| int]` | Look-ahead windows for adverse-selection computation. Accepts `"100ms"`, `"1s"`, `"5s"`, or raw nanosecond integers. |

**Returns:** `RichSimulationMetrics` (see model reference below).

---

## Return Models

### `RichSimulationMetrics`

Top-level return type of `compute_rich_metrics()`.

| Field | Type | Description |
|:---|:---|:---|
| `markets` | `dict[str, MarketSummary]` | Per-symbol market data with `microstructure` populated. |
| `agents` | `list[RichAgentMetrics]` | Per-agent enriched analytics (one entry per `TradingAgent`). |
| `fills` | `list[FillRecord] \| None` | Per-fill execution records; `None` unless `include_fills=True`. |

---

### `MicrostructureMetrics`

Attached to each `MarketSummary` under the `microstructure` field.
All fields are `None` when the required L1 series or agent logs are absent.

| Field | Type | Definition | Reference |
|:---|:---|:---|:---|
| `lob_imbalance_mean` | `float \| None` | Mean L1 order-book imbalance: $\frac{Q_{\text{bid}} - Q_{\text{ask}}}{Q_{\text{bid}} + Q_{\text{ask}}}$. Range $[-1, 1]$. Near 0 = balanced; positive = buying pressure. | Cont, Kukanov & Stoikov (2014) |
| `lob_imbalance_std` | `float \| None` | Standard deviation of L1 imbalance. High values = erratic LOB composition. | |
| `resilience_mean_ns` | `float \| None` | Mean spread recovery time after shock events (spread > $\mu + 2\sigma$), in nanoseconds. `None` if no shocks detected. Longer = more fragile market. | Foucault, Kadan & Kandel (2013) |
| `market_ott_ratio` | `float \| None` | Market-wide order-to-trade ratio: $N_{\text{submissions}} / N_{\text{fills}}$. MiFID II RTS 9 reference threshold: 4:1. Requires `AGENT_LOGS`. | MiFID II RTS 9 |
| `pct_time_two_sided` | `float` | Percentage of L1 observations with both bid and ask present (0–100). Complement of `pct_time_no_bid`/`pct_time_no_ask`. | |

---

### `RichAgentMetrics`

One entry per agent. Combines data from `AgentData`, `EquityCurve`,
`TradeAttribution`, and `AGENT_LOGS`.

| Field | Type | Definition | Source |
|:---|:---|:---|:---|
| `agent_id` | `int` | Unique agent identifier. | `AgentData` |
| `agent_type` | `str` | Agent class name. | `AgentData` |
| `agent_name` | `str` | Display name. | `AgentData` |
| `total_pnl_cents` | `int` | Mark-to-market PnL: $\text{ending\_cash} + \sum(\text{inventory}_s \times P_{\text{last},s}) - \text{starting\_cash}$. Always present. | `AgentData.pnl_cents` |
| `sharpe_ratio` | `float \| None` | Annualised Sharpe ratio from equity curve: $\frac{\bar{r}}{\sigma_r} \times \sqrt{\frac{T_{\text{ns/year}}}{\Delta t_{\text{median}}}}$. `None` if < 30 observations or $\sigma_r = 0$. | `EQUITY_CURVE` |
| `max_drawdown_cents` | `int \| None` | Peak-to-trough drawdown in cents: $\max_t(\max_{s \le t} \text{NAV}_s - \text{NAV}_t)$. 0 if monotonically increasing. | `EQUITY_CURVE` |
| `fill_rate_pct` | `float \| None` | Percentage of submitted orders that received at least one fill: $N_{\text{executed}} / N_{\text{submitted}} \times 100$. Requires `AGENT_LOGS`. | `AGENT_LOGS` |
| `order_to_trade_ratio` | `float \| None` | Per-agent OTT: $N_{\text{submitted}} / N_{\text{executed}}$. `None` if no executions. Requires `AGENT_LOGS`. | `AGENT_LOGS` |
| `vwap_cents` | `int \| None` | Per-agent volume-weighted average fill price: $\sum(P_i \times Q_i) / \sum Q_i$. Integer cents, floor-divided. | `TRADE_ATTRIBUTION` |
| `trade_count` | `int` | Number of fills involving this agent (both passive and aggressive sides). 0 if `TRADE_ATTRIBUTION` absent. | `TRADE_ATTRIBUTION` |
| `end_inventory` | `dict[str, int]` | Final position per symbol (shares), excluding `CASH`. Empty dict if agent holds only cash. | `AgentData.final_holdings` |
| `inventory_std` | `float \| None` | Standard deviation of intraday inventory reconstructed fill-by-fill. High values = large positional swings. `None` if < 2 fills. | `TRADE_ATTRIBUTION` |
| `order_lifecycles` | `list[OrderLifecycle] \| None` | Per-order lifecycle records. `None` when `AGENT_LOGS` profile is not active. See `OrderLifecycle` below. | `AGENT_LOGS` |

---

### `OrderLifecycle`

One entry per submitted order, reconstructed from `ORDER_SUBMITTED`,
`ORDER_EXECUTED`, and `ORDER_CANCELLED` log events. Populated on
`RichAgentMetrics.order_lifecycles` when `ResultProfile.AGENT_LOGS` is active.

| Field | Type | Definition |
|:---|:---|:---|
| `order_id` | `int` | Unique order identifier. |
| `agent_id` | `int` | Agent that submitted the order. |
| `submitted_at_ns` | `int` | Timestamp of the `ORDER_SUBMITTED` event (nanoseconds, Unix epoch). |
| `status` | `Literal["filled", "partially_filled", "cancelled", "resting"]` | Terminal status. `"filled"` when `filled_qty == submitted_qty`; `"partially_filled"` when `0 < filled_qty < submitted_qty`; `"cancelled"` when an `ORDER_CANCELLED` event was observed; `"resting"` when no terminal event occurred by end of simulation. |
| `filled_qty` | `int` | Total quantity filled across all execution events. |
| `submitted_qty` | `int` | Quantity specified in the original `ORDER_SUBMITTED` event. |
| `resting_time_ns` | `int \| None` | Elapsed time from submission to terminal state (nanoseconds). `None` when the order is still `"resting"` at simulation end. |
| `fill_events` | `list[tuple[int, int, int]]` | Per-fill details as `(time_ns, price_cents, qty)` tuples. Empty list if no fills occurred. |

---

### `FillRecord`

One entry per agent per fill (each `TradeAttribution` produces two records —
one for the passive/maker side, one for the aggressive/taker side).
Only populated when `include_fills=True`.

| Field | Type | Definition |
|:---|:---|:---|
| `time_ns` | `int` | Fill timestamp (nanoseconds, Unix epoch). |
| `agent_id` | `int` | Agent involved in this fill. |
| `side` | `str` | `"BUY"` or `"SELL"` from this agent's perspective. |
| `price_cents` | `int` | Fill price in integer cents. |
| `quantity` | `int` | Number of shares filled. |
| `slippage_bps` | `int \| None` | Signed slippage vs contemporaneous L1 mid-price. For a BUY: $\text{bps} = (P_{\text{fill}} - P_{\text{mid}}) \times 10{,}000 / P_{\text{mid}}$. Positive = paid above mid (costly). For a SELL the sign is negated so positive = received below mid (costly). `None` when no two-sided L1 quote exists. |
| `adverse_selection_bps` | `dict[str, int \| None]` | Mid-price move at configurable look-ahead windows. Keys are window labels (e.g. `"100ms"`, `"1s"`). For a BUY: positive = price rose after fill (favorable); negative = adverse selection. For a SELL the sign is inverted. `None` per window when the mid-price lookup fails. |

---

## Standalone Compute Functions

Each function below can be called independently with plain Python data.
They do not require a running simulation.

### Market-Level

| Function | Signature | Returns | Description |
|:---|:---|:---|:---|
| `compute_vwap` | `(trades: Sequence[tuple[int, int]])` | `int \| None` | VWAP from `(price_cents, qty)` pairs. Floor-divided. |
| `compute_liquidity_metrics` | `(trades, *, pct_time_no_bid, pct_time_no_ask, total_exchanged_volume, last_trade_cents)` | `LiquidityMetrics` | Session-level stats. |
| `compute_mean_spread` | `(l1: L1Snapshots)` | `float \| None` | Time-averaged quoted spread (cents). |
| `compute_effective_spread` | `(fills, l1: L1Snapshots)` | `float \| None` | Mean $2 \times \|P_{\text{fill}} - P_{\text{mid}}\|$ (cents). |
| `compute_volatility` | `(l1: L1Snapshots)` | `float \| None` | Annualised mid-price return volatility. Requires ≥ 30 two-sided observations. |
| `compute_avg_liquidity` | `(l1: L1Snapshots)` | `tuple[float \| None, float \| None]` | `(avg_bid_qty, avg_ask_qty)` at best price. |
| `compute_lob_imbalance` | `(l1: L1Snapshots)` | `tuple[float \| None, float \| None]` | `(mean, std)` of $(Q_b - Q_a)/(Q_b + Q_a)$. |
| `compute_resilience` | `(l1: L1Snapshots, *, window_frac)` | `float \| None` | Mean spread recovery time (ns). |
| `compute_vpin` | `(fills, l1, *, n_buckets, min_fills)` | `float \| None` | VPIN (Easley et al. 2012). |
| `compute_market_ott_ratio` | `(n_submissions, n_fills)` | `float \| None` | Market-wide order-to-trade ratio. |

### Agent-Level

| Function | Signature | Returns | Description |
|:---|:---|:---|:---|
| `compute_agent_pnl` | `(holdings, starting_cash_cents, last_trade_prices, ...)` | `AgentData` | Mark-to-market PnL from plain holdings dict. |
| `compute_execution_metrics` | `(fills, target_quantity, filled_quantity, ...)` | `ExecutionMetrics` | VWAP slippage, participation rate, implementation shortfall. |
| `compute_equity_curve` | `(fill_events: Sequence[tuple[int, int, int]], *, l1: L1Snapshots \| None = None)` | `EquityCurve \| None` | NAV time-series from `(time_ns, nav_cents, peak_nav_cents)` tuples. When `l1` is provided, one observation per two-sided L1 tick (carry-forward). |
| `compute_sharpe_ratio` | `(curve: EquityCurve \| None)` | `float \| None` | Annualised Sharpe from equity curve. |
| `compute_inventory_std` | `(fills: Sequence[tuple[str, int]])` | `float \| None` | Inventory volatility from `(side, quantity)` pairs. |
| `compute_order_fill_rate` | `(n_executed, n_submitted)` | `float \| None` | Order-level fill rate (0–100). |

### Per-Fill (Tier 3)

| Function | Signature | Returns | Description |
|:---|:---|:---|:---|
| `compute_fill_slippage` | `(fill_price_cents, fill_time_ns, side, l1)` | `int \| None` | Signed slippage vs L1 mid (basis points). |
| `compute_adverse_selection` | `(fill_price_cents, fill_time_ns, side, l1, window_ns)` | `int \| None` | Mid-price move at look-ahead window (basis points). |

### Structural Helpers

| Function | Signature | Returns | Description |
|:---|:---|:---|:---|
| `compute_l1_close` | `(book_log2)` | `L1Close` | Last L1 snapshot from raw book log. |
| `compute_l1_series` | `(book_log2)` | `L1Snapshots` | Full L1 time-series from raw book log. |
| `compute_l2_series` | `(book_log2)` | `L2Snapshots` | Full sparse L2 series from raw book log. |
| `compute_trade_attribution` | `(exec_entries)` | `list[TradeAttribution]` | Per-execution causal records. |
| `compute_metrics` | `(*, book_log2, exec_trades, agent_holdings, ...)` | `dict[str, Any]` | Canonical metric set from raw data (low-level orchestrator). |

---

## Existing Models (from `SimulationResult`)

These models are populated by `run_simulation()` and also used by
`compute_rich_metrics()`.

### `MarketSummary`

Per-symbol container. The `microstructure` field is populated by
`compute_rich_metrics()` and is `None` when accessed directly from
`SimulationResult`.

| Field | Type | Profile requirement |
|:---|:---|:---|
| `symbol` | `str` | Always |
| `l1_close` | `L1Close` | Always |
| `liquidity` | `LiquidityMetrics` | Always |
| `l1_series` | `L1Snapshots \| None` | `L1_SERIES` |
| `l2_series` | `L2Snapshots \| None` | `L2_SERIES` |
| `trades` | `list[TradeAttribution] \| None` | `TRADE_ATTRIBUTION` |
| `microstructure` | `MicrostructureMetrics \| None` | Populated by `compute_rich_metrics()` |

### `LiquidityMetrics`

| Field | Type | Description |
|:---|:---|:---|
| `pct_time_no_bid` | `float` | % of session with empty bid (0–100). |
| `pct_time_no_ask` | `float` | % of session with empty ask (0–100). |
| `total_exchanged_volume` | `int` | Total shares traded. |
| `last_trade_cents` | `int \| None` | Last trade price (cents). |
| `vwap_cents` | `int \| None` | Session VWAP (cents). |

### `AgentData`

| Field | Type | Description |
|:---|:---|:---|
| `agent_id` | `int` | Unique ID. |
| `agent_type` | `str` | Class name. |
| `agent_name` | `str` | Display name. |
| `agent_category` | `str` | Registry category. |
| `final_holdings` | `dict[str, int]` | `{"CASH": cents, "TICKER": shares}`. |
| `starting_cash_cents` | `int` | Initial cash (cents). |
| `mark_to_market_cents` | `int` | Portfolio value at close (cents). |
| `pnl_cents` | `int` | Absolute PnL (cents). |
| `pnl_pct` | `float` | PnL as % of starting cash. |
| `execution_metrics` | `ExecutionMetrics \| None` | For execution-category agents only. |
| `equity_curve` | `EquityCurve \| None` | Requires `EQUITY_CURVE` profile flag. |

### `ExecutionMetrics`

| Field | Type | Description |
|:---|:---|:---|
| `target_quantity` | `int` | Intended shares. |
| `filled_quantity` | `int` | Shares filled. |
| `fill_rate_pct` | `float` | `filled / target × 100`. |
| `avg_fill_price_cents` | `int \| None` | Average fill price (cents). |
| `vwap_cents` | `int \| None` | Session VWAP (cents). |
| `vwap_slippage_bps` | `int \| None` | Slippage vs VWAP (bps). |
| `participation_rate_pct` | `float \| None` | Agent volume / total volume × 100. |
| `arrival_price_cents` | `int \| None` | Mid at first order (cents). |
| `implementation_shortfall_bps` | `int \| None` | Shortfall vs arrival (bps). |

### `EquityCurve`

| Field | Type | Description |
|:---|:---|:---|
| `times_ns` | `list[int]` | Timestamp per observation. One entry per fill in fill-only mode; one entry per two-sided L1 tick when `l1` is provided to `compute_equity_curve()`. |
| `nav_cents` | `list[int]` | NAV at each observation (cents). In L1-sampled mode, the most recent fill NAV is carried forward. |
| `peak_nav_cents` | `list[int]` | High-water mark at each observation (cents). |
| `max_drawdown_cents` | `int` | Property: max peak-to-trough (cents). |

---

## Sign Conventions

### Slippage (`slippage_bps`)

| Direction | Positive value means | Negative value means |
|:---|:---|:---|
| BUY | Paid above mid (costly) | Paid below mid (favorable) |
| SELL | Received below mid (costly) | Received above mid (favorable) |

### Adverse Selection (`adverse_selection_bps`)

| Direction | Positive value means | Negative value means |
|:---|:---|:---|
| BUY | Price rose after fill (favorable) | Price fell after fill (adverse) |
| SELL | Price fell after fill (favorable) | Price rose after fill (adverse) |

---

## Diagnostic Patterns

Common interpretation patterns for practitioners and LLM agents:

| Observation | Diagnosis | Suggested action |
|:---|:---|:---|
| High `total_pnl_cents`, high `lob_imbalance_std` | Profitable but destabilising the book. | Reduce order aggressiveness or size. |
| Low `fill_rate_pct`, low `total_pnl_cents` | Strategy too passive; orders not crossing spread. | Increase limit price aggressiveness or use market orders. |
| High `order_to_trade_ratio` | Excessive cancellations / quote-stuffing. | Reduce quote update frequency. |
| High `inventory_std`, high `max_drawdown_cents` | Large unhedged positional risk. | Tighten inventory limits or add mean-reversion. |
| Negative `adverse_selection_bps` at all windows | Consistently trading against future price moves. | Improve signal timing or add latency to entry. |
| `pct_time_two_sided` < 50% | Market frequently one-sided; poor liquidity. | Add market-maker agents or increase their aggressiveness. |
| `resilience_mean_ns` increasing | Market healing slower after shocks. | Strategy may be degrading microstructure quality. |
| `market_ott_ratio` > 4 | Exceeds MiFID II reference threshold. | Review quoting strategy for regulatory compliance. |

---

## Import Paths

All public symbols are re-exported from `abides_markets.simulation`:

```python
from abides_markets.simulation import (
    # High-level API
    compute_rich_metrics,
    RichSimulationMetrics,
    RichAgentMetrics,
    MicrostructureMetrics,
    OrderLifecycle,
    FillRecord,

    # Standalone compute functions
    compute_fill_slippage,
    compute_adverse_selection,
    compute_vwap,
    compute_sharpe_ratio,
    compute_lob_imbalance,
    compute_resilience,
    compute_vpin,
    compute_market_ott_ratio,
    compute_inventory_std,
    compute_order_fill_rate,
    compute_mean_spread,
    compute_effective_spread,
    compute_volatility,
    compute_avg_liquidity,
    compute_agent_pnl,
    compute_execution_metrics,
    compute_equity_curve,
    compute_liquidity_metrics,
    compute_l1_close,
    compute_l1_series,
    compute_l2_series,
    compute_trade_attribution,
    compute_metrics,

    # Result models
    SimulationResult,
    MarketSummary,
    AgentData,
    ExecutionMetrics,
    EquityCurve,
    TradeAttribution,
    LiquidityMetrics,
    L1Close,
    L1Snapshots,
    L2Snapshots,
    ResultProfile,
)
```
