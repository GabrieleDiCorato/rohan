# Performance Metrics & Evaluation Framework

Canonical functional reference for all quantitative metrics in the Rohan simulation
framework. Metrics are organised into three groups: **agent performance**,
**market microstructure**, and **market impact** (strategy vs. baseline deltas).

> **Data conventions:** All prices are integer cents. Times in the L1 `time`
> column are nanoseconds since midnight; `EventTime` is nanoseconds since epoch.

---

## 1. Agent Performance

These metrics assess the standalone performance of the strategic agent.
All fields live on `AgentMetrics`.

### 1.1 Profit & Loss (PnL)

Total PnL decomposes into realized and unrealized components.

- **Realized PnL**: Cash change from closed trades.
  $$PnL_{\text{realized}} = Cash_{\text{end}} - Cash_{\text{start}}$$
- **Unrealized PnL**: Mark-to-market value of residual inventory at end of
  simulation, valued at the last observed two-sided mid-price.
  $$PnL_{\text{unrealized}} = Inventory_{\text{end}} \times P_{\text{mid,end}}$$
- **Total PnL**:
  $$PnL_{\text{total}} = PnL_{\text{realized}} + PnL_{\text{unrealized}}$$

> `compute_agent_metrics()` auto-detects the strategic agent via
> `AbidesOutput.get_strategic_agent_id()` (matching `agent.type == "StrategicAgent"`)
> and reads `starting_cash` from the agent object.

**Fields:** `total_pnl`, `initial_cash`, `ending_cash`.

---

### 1.2 Execution Quality

- **Fill Rate**: Fraction of submitted orders that were executed.
  A low fill rate indicates passive orders are not crossing the spread or
  prices are moving adversely.
  $$\text{FillRate} = \frac{N_{\text{executed}}}{N_{\text{submitted}}}$$

- **Trade Count**: Total number of `ORDER_EXECUTED` events for the strategic
  agent. Indicates activity level and turnover.

- **Traded Volume**: Total shares executed, summed over all fills.
  $$V_{\text{total}} = \sum_i \text{quantity}_i$$

- **Effective Spread**: Average cost of immediacy measured relative to the
  mid-price at the moment of each fill.
  $$\text{EffSpread} = \frac{1}{N_{\text{fills}}} \sum_i 2\,\left| P_{\text{fill},i} - P_{\text{mid},i} \right|$$
  $P_{\text{mid},i}$ is taken from the L1 snapshot nearest in time to fill $i$
  (join on `EventTime` ↔ `time`). Units: cents. Returns `None` if no two-sided
  L1 rows exist.

**Fields:** `fill_rate`, `trade_count`, `traded_volume`, `effective_spread`.

---

### 1.3 Risk

- **Sharpe Ratio**: Annualised risk-adjusted return computed from the
  mark-to-market PnL curve reconstructed fill-by-fill.
  $$\text{Sharpe} = \frac{\bar{r}}{\sigma_r} \times \sqrt{\frac{T_{\text{ns/year}}}{\Delta t_{\text{median}}}}$$
  Returns `None` if fewer than 30 observations or $\sigma_r = 0$ (flat PnL).

- **Max Drawdown**: Maximum peak-to-trough decline of the PnL curve. Always
  $\geq 0$; units: cents.
  $$\text{MaxDrawdown} = \max_t\!\left(\max_{s \leq t} PnL_s - PnL_t\right)$$
  Returns `None` if fewer than 2 observations.

- **Inventory Std**: Standard deviation of the agent's intraday inventory
  (share count), reconstructed fill-by-fill from `ORDER_EXECUTED` events.
  High values indicate large, volatile positional exposure.
  Returns `None` if fewer than 2 fills.

**Fields:** `sharpe_ratio`, `max_drawdown`, `inventory_std`.

---

## 2. Market Microstructure

These metrics describe the quality and dynamics of the limit order book.
They are computed from the **L1 snapshot DataFrame** with columns:
`time` (ns since midnight), `bid_price`, `bid_qty`, `ask_price`, `ask_qty`
(prices in cents).

All fields live on `MarketMetrics`.

### NaN semantics

`bid_price` or `ask_price` is `NaN` when that side of the book is **genuinely
empty** — no resting orders at any price level. This is an economically meaningful
state and must **never be forward-filled**.

A **two-sided row** is one where both `bid_price` and `ask_price` are non-NaN.
Most microstructure metrics are defined only over two-sided rows.

---

### 2.1 Volatility

Annualised standard deviation of mid-price returns, computed only from
two-sided snapshots.

1. Filter: `two_sided = l1.dropna(subset=["bid_price", "ask_price"])`.
2. Compute mid-price: $P_{\text{mid},t} = (P_{\text{bid},t} + P_{\text{ask},t}) / 2$.
3. Compute returns: $r_t = P_{\text{mid},t} / P_{\text{mid},t-1} - 1$ (no `fillna`).
4. Annualise from median inter-snapshot interval $\Delta t$ (ns):
   $$\sigma_{\text{ann}} = \text{std}(r) \times \sqrt{\frac{T_{\text{ns/year}}}{\Delta t_{\text{median}}}}$$

Returns `None` if fewer than 30 two-sided return observations. Using
`pct_change` without `fillna` ensures one-sided states do not contribute
artificial zero returns.

**Field:** `volatility`.

---

### 2.2 Spread & Liquidity

All computed from two-sided rows.

- **Mean Spread**: Time-averaged quoted spread.
  $$\bar{s} = \frac{1}{N} \sum_t (P_{\text{ask},t} - P_{\text{bid},t})$$
  Units: cents.

- **Average Bid Liquidity**: Mean quantity resting at best bid.
  $$\bar{Q}_{\text{bid}} = \frac{1}{N} \sum_t Q_{\text{bid},t}$$

- **Average Ask Liquidity**: Same for the ask side.
  $$\bar{Q}_{\text{ask}} = \frac{1}{N} \sum_t Q_{\text{ask},t}$$

**Fields:** `mean_spread`, `avg_bid_liquidity`, `avg_ask_liquidity`.

---

### 2.3 LOB Imbalance

Measures directional pressure in the limit order book
(Cont, Kukanov & Stoikov, 2014).

$$I_t = \frac{Q_t^{\text{bid}} - Q_t^{\text{ask}}}{Q_t^{\text{bid}} + Q_t^{\text{ask}}} \in [-1,\, 1]$$

- $I_t > 0$: more quantity on the bid (buying pressure).
- $I_t < 0$: more quantity on the ask (selling pressure).
- $I_t \approx 0$: balanced book.

Computed on all rows where both `bid_qty` and `ask_qty` are non-zero.

- **`lob_imbalance_mean`**: Mean of $I_t$. Persistent positive values indicate
  sustained directional bias; a well-functioning market maker keeps this near 0.
- **`lob_imbalance_std`**: Std of $I_t$. High values indicate erratic LOB
  composition.

Returns `None` if no valid rows exist.

**Fields:** `lob_imbalance_mean`, `lob_imbalance_std`.

---

### 2.4 VPIN

Volume-Synchronized Probability of Informed Trading
(Easley, López de Prado & O'Hara, 2012). Measures the probability that
order flow is driven by informed traders.

**Algorithm:**

1. Classify each fill as buy- or sell-initiated using the **Lee-Ready tick test**:
   - $P_{\text{fill}} > P_{\text{mid}}$ → buy;
   - $P_{\text{fill}} < P_{\text{mid}}$ → sell;
   - $P_{\text{fill}} = P_{\text{mid}}$ → tick rule (compare to previous trade).
2. Partition fills into equal-volume buckets of size
   $V_{\text{bucket}} = V_{\text{total}} / 50$.
3. For each bucket $\tau$:
   $$\text{OI}_\tau = \frac{|V_\tau^{\text{buy}} - V_\tau^{\text{sell}}|}{V_{\text{bucket}}}$$
4. $$\text{VPIN} = \frac{1}{n} \sum_{\tau=1}^{n} \text{OI}_\tau \in [0,\, 1]$$

**Interpretation:**
- Low ($\approx 0$): balanced order flow; low adverse-selection risk.
- High ($\approx 1$): heavily directional; high informed-trading probability —
  a strategy with high VPIN should reduce order aggressiveness.

Returns `None` if fewer than 20 fills.

**Field:** `vpin`.

---

### 2.5 Market Resilience

How quickly does the bid-ask spread recover after a shock
(Foucault, Kadan & Kandel, 2013)?

**Algorithm:**

1. Compute spread series $s_t = P_{\text{ask},t} - P_{\text{bid},t}$ on
   two-sided rows.
2. Compute rolling mean $\bar{s}$ and rolling std $\sigma_s$
   (window = min(100 ticks, 10 % of data)).
3. Identify **shock events**: $s_t > \bar{s} + 2\sigma_s$.
4. For each shock, record the **recovery time**: elapsed nanoseconds until
   $s_t \leq \bar{s} + \sigma_s$.
5. Report the mean recovery time.

**Interpretation:**
- Long recovery time: the market heals slowly after spread widening — fragile,
  low-liquidity conditions. A strategy that increases this value is degrading
  market quality.

Returns `None` if no shock events are detected.

**Field:** `resilience_mean_ns` (units: nanoseconds).

---

### 2.6 Market OTT Ratio

Market-wide order-to-trade ratio (MiFID II RTS 9).

$$\text{OTT} = \frac{N_{\text{submissions}}}{N_{\text{fills}}}$$

Counted across **all agents** in the simulation.

**Interpretation:**
- Elevated OTT indicates quote-stuffing or excessive cancellations.
- Complement to the per-agent OTT (which measures only the strategic agent's
  own behaviour). MiFID II reference threshold: 4:1.

Returns `None` if no fills occurred.

**Field:** `market_ott_ratio`.

---

## 3. Market Impact (Strategy vs. Baseline)

Each delta metric is the percentage change from a **Baseline Run** (strategic
agent disabled or replaced by noise) to the **Strategy Run** (active agent).

$$\Delta X\,(\%) = \frac{X_{\text{strategy}} - X_{\text{baseline}}}{X_{\text{baseline}}} \times 100$$

Returns `None` if either value is `None` or if the baseline value is zero.

All fields live on `MarketImpact`.

| Field | Metric | Positive means the strategy... |
|---|---|---|
| `volatility_delta_pct` | Volatility | ...increases price noise |
| `spread_delta_pct` | Mean spread | ...widens spreads (worse liquidity) |
| `bid_liquidity_delta_pct` | Avg bid liquidity | ...adds bid-side depth |
| `ask_liquidity_delta_pct` | Avg ask liquidity | ...adds ask-side depth |
| `lob_imbalance_delta_pct` | LOB imbalance mean | ...increases directional book bias |
| `vpin_delta_pct` | VPIN | ...increases informed-flow probability |
| `resilience_delta_pct` | Mean recovery time | ...degrades market resilience |
| `ott_ratio_delta_pct` | OTT ratio | ...increases cancellation noise |

---

## 4. Comparative Evaluation

The `ComparisonResult` object encapsulates strategy metrics, baseline metrics,
and their deltas in a single object.

| Category | Key indicators | Actionable signal |
|:---|:---|:---|
| **Alpha generation** | `total_pnl`, `sharpe_ratio` | Is the strategy profitable on a risk-adjusted basis? |
| **Execution quality** | `fill_rate`, `effective_spread`, `traded_volume` | Is the strategy efficiently accessing liquidity? |
| **Inventory risk** | `max_drawdown`, `inventory_std` | Is the strategy taking on excessive positional risk? |
| **Market quality** | `spread_delta_pct`, `volatility_delta_pct` | Is the strategy improving or degrading the book? |
| **Microstructure** | `vpin_delta_pct`, `resilience_delta_pct`, `ott_ratio_delta_pct` | Is the strategy introducing toxicity or fragility? |

### Diagnostic feedback patterns

- *High PnL, high `volatility_delta_pct`*: strategy is profitable but
  destabilising the market.
- *Low `fill_rate`, low PnL*: strategy is too passive; orders are not crossing
  the spread.
- *High `vpin`*: aggressive directional flow — consider reducing order
  aggressiveness or size.
- *High `inventory_std`, high `max_drawdown`*: strategy is taking on large
  unhedged positions.
- *Positive `resilience_delta_pct`*: strategy is slowing spread recovery —
  consider reducing order frequency or size.
