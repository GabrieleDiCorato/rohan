# Analytics Improvement Plan

**Date:** 2026-02-22
**Module:** `rohan.framework.analysis_service`
**Status:** Draft

---

## 1. Problem Statement

The `AnalysisService.compute_metrics()` method frequently returns `volatility = None`.
Several fields in `SimulationMetrics` and `AgentMetrics` are declared but never
computed (`effective_spread`, `traded_volume`, `sharpe_ratio`, `max_drawdown`,
`inventory_std`). The current analytics toolbox lacks microstructure-grade metrics
that would give the LLM Explainer better signal for strategy refinement.

---

## 2. Root Cause Analysis: NaN Volatility

### 2.1 Data Flow

```
OrderBook.append_book_log2()         ← called on every LOB mutation
    → book_log2: list[dict]          ← {QuoteTime, bids, asks}
        → get_L1_snapshots()         ← iterates book_log2, extracts best bid/ask
            → best_bids, best_asks   ← both same length, same timestamps
                → pd.merge(on="time", how="outer")   ← functionally inner (same times)
                    → L1 DataFrame   ← columns: time, bid_price, bid_qty, ask_price, ask_qty
```

### 2.2 What NaN Actually Means

**Investigation result:** The `best_bids` and `best_asks` lists always have
**identical timestamps** because they are both extracted from the same
`book_log2` entries (each entry snapshots both sides of the book
simultaneously). The outer merge does **not** introduce any NaN values.

NaN values originate from ABIDES' `safe_first()` function inside
`get_L1_snapshots()`:

```python
def safe_first(x):
    return x[0] if len(x) > 0 else np.array([None, None])
```

When a side of the book is empty (`get_l2_bid_data()` returns `[]` because no
price level has `qty > 0`), the snapshot records `[time, None, None]` for that
side. This becomes NaN in the DataFrame.

**A NaN `bid_price` means there are genuinely zero resting bid orders on the
book at that instant.** The same applies symmetrically for `ask_price`.

| State | `bid_price` | `ask_price` | Meaning |
|-------|-------------|-------------|---------|
| Normal | value | value | Both sides active |
| No bids | NaN | value | Cannot sell — bid side empty |
| No asks | value | NaN | Cannot buy — ask side empty |
| Empty book | NaN | NaN | Total illiquidity |

These are **economically meaningful** states — they represent real illiquidity
events, not data artifacts.

### 2.3 Why Forward-Fill Is Wrong

A naïve `ffill()` would carry the last observed price into periods where that
side of the book is genuinely empty. This:

- Produces a **fictitious mid-price** where no trade could actually execute.
- Masks illiquidity events that are critical for microstructure analysis (VPIN,
  resilience, spread dynamics).
- Generates **artificial zero returns** (ffilled price → next ffilled price =
  same value → 0% return), deflating volatility instead of fixing it.
- Corrupts the spread calculation (spread should be undefined, not stale).

### 2.4 Correct Fix

The volatility calculation at `analysis_service.py:34-48` does the following:

```python
l1["mid_price"] = (l1["bid_price"] + l1["ask_price"]) / 2   # NaN when either side empty
l1["returns"]   = l1["mid_price"].pct_change().fillna(0)     # .fillna(0) only helps row 0
vol = float(l1["returns"].std()) * np.sqrt(obs_per_year)     # NaN if too many NaN returns
```

**The fix:** compute mid-price and returns **only from rows where both sides of
the book are present** (i.e., the book is two-sided and tradeable):

```python
two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()
two_sided["mid_price"] = (two_sided["bid_price"] + two_sided["ask_price"]) / 2
two_sided["returns"] = two_sided["mid_price"].pct_change()     # no fillna(0)
returns = two_sided["returns"].dropna()
```

This is semantically correct because:

- Volatility measures the dispersion of **tradeable** price changes.
- One-sided book states have no meaningful mid-price; including them adds noise.
- Excluding them preserves the time-differential structure of `pct_change()`
  (each return is computed from the previous two-sided snapshot, which may be
  further apart in time — this is fine for realized-vol estimation).
- The annualisation factor already accounts for observation frequency via
  `time_col.diff().median()`, which should be computed on the same filtered
  timestamps.

The same `two_sided` subset applies to spread and liquidity computations.

### 2.5 Preserving the Full Timeline

The unfiltered `l1` DataFrame (with NaN rows) is preserved for:

- LOB imbalance (one-sided states are themselves informative)
- Resilience measurement (needs to see when the book goes one-sided and when
  it recovers)
- VPIN (trade classification uses mid-price, but the empty-book states inform
  toxicity detection)

---

## 3. Existing Stub Fields — Implementation Plan

### 3.1 Market-Wide (`SimulationMetrics`)

| Field | Current | Implementation |
|-------|---------|---------------|
| `volatility` | Broken (NaN) | Fix per §2.4 |
| `mean_spread` | Fragile (NaN possible) | Compute on `two_sided` subset |
| `effective_spread` | Always `None` | $2 \times \|P_\text{fill} - P_\text{mid,nearest}\|$ averaged across all fills. Extract fills from `result.get_logs_df()` filtering `EventType == "ORDER_EXECUTED"`. Match each fill timestamp to the nearest L1 mid-price. |
| `avg_bid_liquidity` | Fragile | Compute on `two_sided` subset (or on rows where `bid_qty` is not NaN — one-sided liquidity is still meaningful) |
| `avg_ask_liquidity` | Fragile | Same |
| `traded_volume` | Always `None` | Sum filled quantities from `ORDER_EXECUTED` log events |

### 3.2 Per-Agent (`AgentMetrics`)

| Field | Current | Implementation |
|-------|---------|---------------|
| `total_pnl` | Working | No change |
| `sharpe_ratio` | Always `None` | Build mark-to-market PnL curve: at each two-sided L1 snapshot, compute `cash_t + inventory_t × mid_t`. Derive per-period returns from the PnL curve. Annualised Sharpe = $\frac{\bar{r}}{\sigma_r} \times \sqrt{N_\text{year}}$ |
| `max_drawdown` | Always `None` | From the PnL curve: running-max series → drawdown = `running_max - pnl`. Report `max(drawdown)` in cents. |
| `inventory_std` | Always `None` | From the agent's fill log, reconstruct an inventory time series (start at `start_inventory`, +1 on buy fill, -1 on sell fill). Compute `std()` of that series. |
| `avg_hold_time_ns` | Always `None` | Deferred — requires FIFO/LIFO matching of individual fills to compute per-share holding periods. Complex and low-priority for the LLM feedback loop. |
| `fill_rate` | Working | No change |
| `order_to_trade_ratio` | Working | No change |

### 3.3 Implementation Notes

**Agent PnL curve reconstruction:** The `compute_agent_metrics` method currently
receives `result` (SimulationOutput) and `agent_id`. To build a PnL curve, we
need:

1. The agent's **fill events** (from `result.get_logs_by_agent()[agent_id]`),
   each with timestamp, side, qty, price.
2. The **L1 mid-price** series (from `result.get_order_book_l1()`), filtered to
   two-sided rows as in §2.4.

From fills we reconstruct `cash_t` and `inventory_t` at each fill event. We then
interpolate (or join) onto the L1 timeline to get `pnl_t = cash_t + inventory_t × mid_t`.

**Sharpe ratio risk:** Very short simulations (5 min) may have too few data
points for a meaningful Sharpe. We should return `None` if `len(returns) < 30`
or similar threshold.

---

## 4. New Microstructure Metrics

### 4.1 LOB Imbalance

**Definition:**

$$I_t = \frac{Q_t^\text{bid} - Q_t^\text{ask}}{Q_t^\text{bid} + Q_t^\text{ask}}$$

where $Q_t^\text{bid}$ and $Q_t^\text{ask}$ are the quantities at best
bid/ask at time $t$.

**Multi-level variant (from L2 data):**

$$I_t^{(n)} = \frac{\sum_{k=1}^{n} Q_{t,k}^\text{bid} - \sum_{k=1}^{n} Q_{t,k}^\text{ask}}{\sum_{k=1}^{n} Q_{t,k}^\text{bid} + \sum_{k=1}^{n} Q_{t,k}^\text{ask}}$$

**New fields on `SimulationMetrics`:**

- `lob_imbalance_mean: float | None` — mean of $I_t$ across all two-sided snapshots
- `lob_imbalance_std: float | None` — std of $I_t$

**Rationale:** LOB imbalance is one of the strongest short-term predictors of
price direction (Cont, Kukanov & Stoikov, 2014). It tells the LLM whether a
strategy is biasing the book — a market-making strategy should keep imbalance
near zero; a directional strategy will naturally create imbalance.

**Data source:** L1 (`bid_qty`, `ask_qty`) — available. Can optionally compute
the multi-level variant from L2 for deeper analysis.

### 4.2 VPIN (Volume-Synchronized Probability of Informed Trading)

**Definition (Easley, López de Prado & O'Hara, 2012):**

1. Classify each fill as buy-initiated or sell-initiated using the **Lee-Ready
   tick test**: if $P_\text{fill} > P_\text{mid}$, it's a buy; if
   $P_\text{fill} < P_\text{mid}$, it's a sell. (At the mid, use the tick rule:
   compare to the previous trade price.)
2. Accumulate fills into **volume buckets** of size $V_\text{bucket}$. Each
   bucket contains exactly $V_\text{bucket}$ shares (a single fill may span two
   buckets).
3. For each bucket $\tau$:
   $$\text{OI}_\tau = \frac{|V_\tau^\text{buy} - V_\tau^\text{sell}|}{V_\text{bucket}}$$
4. VPIN = rolling mean of $\text{OI}_\tau$ over $n$ buckets:
   $$\text{VPIN} = \frac{1}{n} \sum_{\tau=1}^{n} \text{OI}_\tau$$

**New field on `SimulationMetrics`:**

- `vpin: float | None` — final VPIN value (or mean over the simulation)

**Bucket sizing:** $V_\text{bucket}$ = total volume / 50 buckets (standard
practice). $n$ = 50 for the rolling window (i.e., we report the final VPIN).

**Rationale:** VPIN measures the probability of informed trading — high VPIN
means aggressive directional flow is dominating, indicating either a toxic
strategy or an adverse selection risk. This is directly actionable: the LLM
should reduce aggressiveness when VPIN is high.

**Data source:** Fill events from `result.get_logs_df()` + L1 mid-prices.
Both available.

### 4.3 Market Resilience

**Definition:** How quickly does the spread recover after an aggressive event
(large spread widening)?

**Algorithm:**

1. Compute the spread series: $s_t = \text{ask}_t - \text{bid}_t$ on two-sided rows.
2. Compute rolling mean $\bar{s}$ and rolling std $\sigma_s$ (window = 100
   ticks or 10% of data, whichever is smaller).
3. Identify **shock events**: timestamps where $s_t > \bar{s} + k \cdot \sigma_s$
   (default $k = 2$).
4. For each shock, measure the **recovery time**: the time (in ns) until the
   spread returns to $\bar{s} + 1 \cdot \sigma_s$.
5. Report the mean recovery time.

**New field on `SimulationMetrics`:**

- `resilience_mean_ns: float | None` — average spread recovery time in nanoseconds

**Rationale:** Resilience is a classic LOB quality metric (Foucault, Kadan &
Kandel, 2013). A strategy that degrades resilience is harmful to market quality.
This gives the LLM a direct measure of whether the strategy is destabilizing the
book.

**Data source:** L1 spread series (two-sided rows). Available.

### 4.4 OTT Ratio (Market-Wide)

**Definition (MiFID II RTS 9):**

$$\text{OTT} = \frac{N_\text{submissions}}{N_\text{fills}}$$

where $N_\text{submissions}$ and $N_\text{fills}$ are counted across **all
agents** in the simulation.

**New field on `SimulationMetrics`:**

- `market_ott_ratio: float | None` — market-wide order-to-trade ratio

**Rationale:** An elevated OTT indicates quote-stuffing or excessive order
cancellation. MiFID II caps OTT at 4:1 for certain instruments. The per-agent
OTT (already computed) measures the strategy's own behaviour; the market-wide
OTT measures whether the strategy is increasing noise for everyone.

**Data source:** `result.get_logs_df()` filtering by `EventType`. Available.

---

## 5. Model Updates

### 5.1 `SimulationMetrics` — New Fields

```python
# --- Microstructure metrics ---
lob_imbalance_mean: float | None = Field(default=None,
    description="Mean L1 order-book imbalance (Qbid-Qask)/(Qbid+Qask)")
lob_imbalance_std: float | None = Field(default=None,
    description="Std of L1 order-book imbalance")
vpin: float | None = Field(default=None,
    description="Volume-synchronised probability of informed trading")
resilience_mean_ns: float | None = Field(default=None,
    description="Mean spread recovery time after shocks (ns)")
market_ott_ratio: float | None = Field(default=None,
    description="Market-wide order-to-trade ratio (submissions/fills)")
```

### 5.2 `MarketMetrics` — Mirror new fields

Add the same five fields so that `_to_market()` copies them through for
baseline comparison. `MarketImpact` gains corresponding `_delta_pct` fields.

### 5.3 `AgentMetrics` — No new fields

All fields already declared. We fill the existing stubs.

---

## 6. Downstream Integration

### 6.1 LLM Prompts

Update `framework/prompts.py` — `INTERPRETER_PROMPT_TEMPLATE` — add a
**Microstructure** section:

```
### Microstructure
- **LOB Imbalance**: {lob_imbalance_mean} (σ = {lob_imbalance_std})
- **VPIN**: {vpin}
- **Resilience**: {resilience_mean_ns}
- **Market OTT**: {market_ott_ratio}
```

Update `format_interpreter_prompt()` to inject these values (N/A when None).

The same change mirrors in `llm/prompts.py` for the Explainer's context.

### 6.2 `MarketImpact` — New delta fields

```python
lob_imbalance_delta_pct: float | None = ...
vpin_delta_pct: float | None = ...
resilience_delta_pct: float | None = ...
ott_ratio_delta_pct: float | None = ...
```

These are computed in `_run_comparison()` (iteration_pipeline.py) and in
`nodes.py` using the same `_pct()` helper.

### 6.3 Streamlit UI

Add a new expander section **"Microstructure"** in `0_Terminal.py` and
`1_Refinement_Lab.py` that renders the new metrics with `st.metric()`.

### 6.4 LangChain Tools

Add a new explainer tool `get_microstructure_stats()` that returns VPIN, LOB
imbalance, resilience, and OTT as a formatted string for the Explainer agent's
tool-use loop.

---

## 7. Execution Plan

### Phase 1 — Fix NaN Volatility + Harden Existing Metrics

**Priority: Critical**

1. Refactor `compute_metrics()`: filter L1 to two-sided rows before computing
   mid-price, returns, and spread. Compute annualisation factor from the
   filtered timestamps.
2. Compute `effective_spread` from fill events + nearest L1 mid.
3. Compute `traded_volume` from fill events.
4. Update existing tests — assert `volatility is not None` for the 5-minute
   config.

### Phase 2 — Fill Agent Metric Stubs

**Priority: High**

5. Implement `sharpe_ratio` from the mark-to-market PnL curve.
6. Implement `max_drawdown` from the PnL curve.
7. Implement `inventory_std` from the fill event log.
8. Add unit tests with synthetic fixtures.

### Phase 3 — New Microstructure Metrics

**Priority: High**

9. Implement LOB Imbalance (L1-based).
10. Implement VPIN (fill log + L1 mid).
11. Implement Market Resilience (spread recovery).
12. Implement market-wide OTT ratio (log-based).
13. Add new fields to `SimulationMetrics`, `MarketMetrics`, `MarketImpact`.

### Phase 4 — Downstream Wiring

**Priority: Medium**

14. Update `INTERPRETER_PROMPT_TEMPLATE` and `format_interpreter_prompt()`.
15. Update `_run_comparison()` in iteration_pipeline.py and in llm/nodes.py.
16. Add explainer tool `get_microstructure_stats()`.
17. Update Streamlit UI with microstructure section.
18. End-to-end integration test: run simulation → verify all metrics non-None
    → verify prompt contains new metrics.

### Phase 5 — Pluggable Metric Registry (Optional, Deferred)

**Priority: Low — future enhancement**

19. Define a `MetricDescriptor` dataclass with name, label, description, unit,
    category, format_fn.
20. Auto-discover metrics from Pydantic `Field` metadata on
    `SimulationMetrics` / `AgentMetrics`.
21. Refactor `format_interpreter_prompt()` to iterate the registry.
22. Refactor Streamlit UI to dynamically render from the registry.
23. Adding a new metric becomes: add field to model → implement in
    `compute_metrics()` → done. Prompts and UI pick it up automatically.

---

## 8. Testing Strategy

| Layer | What | How |
|-------|------|-----|
| Unit | Each metric computation | Synthetic L1/L2 DataFrames with known values |
| Unit | NaN-handling edge cases | DataFrame with all-NaN rows, single-row, empty |
| Integration | Full pipeline | 30-min sim → assert all metrics non-None |
| Regression | Existing tests pass | `pytest tests/test_analysis.py -v` |
| Smoke | LLM prompt rendering | `format_interpreter_prompt()` with full metrics → no crash, values present |

---

## 9. Risk & Considerations

- **Short simulations (< 5 min):** May not have enough fills for VPIN or enough
  spread shocks for resilience. Return `None` with sensible thresholds rather
  than computing garbage.
- **No market makers:** Without MM agents, the book will be sparse and one-sided
  more often. The two-sided filter in §2.4 handles this gracefully — metrics are
  computed only from tradeable states.
- **Sharpe denominator:** If $\sigma_r = 0$ (flat PnL), Sharpe is undefined.
  Return `None`.
- **VPIN bucket sizing:** If total volume is too low for 50 buckets, fall back
  to fewer buckets or return `None`.
- **Performance:** L2 data for multi-level LOB imbalance is expensive to compute.
  Start with L1-only; add L2 as an opt-in parameter if needed.
