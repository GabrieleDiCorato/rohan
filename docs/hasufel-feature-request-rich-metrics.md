# Feature Request: Rich Simulation Metrics API

## Summary

Extend `compute_metrics()` (or add a companion `compute_rich_metrics()`) to
return **agent-level metrics** and **standard microstructure indicators** that
every hasufel consumer currently has to reimplement from raw logs.

## Motivation

`compute_metrics()` today returns ~5 market-wide statistics (bid/ask
availability, total volume, VWAP, last trade). The `SimulationResult` already
holds all the raw data needed for much deeper analysis — L1/L2 snapshots, full
order logs, per-agent end-state — but consumers must parse and compute
everything themselves.

In practice, this means every downstream project:

1. **Reconstructs agent PnL from order logs** instead of using the
   pre-computed `result.agents[i].pnl_cents` that the kernel already tracks.
2. **Reimplements standard microstructure metrics** (LOB imbalance, spread
   resilience, VPIN bucket calibration, order-to-trade ratio) with subtle
   differences across implementations.
3. **Computes per-fill execution quality** (slippage vs. mid, adverse
   selection) by manually joining fill events to L1 snapshots.

This is duplicative, error-prone, and creates a fragmented metric landscape
across the hasufel ecosystem. Centralising these computations in hasufel
would give consumers a single, tested, versioned analytics surface.

## Proposed Scope

### Tier 1 — Agent-Level Metrics (highest value)

Return per-agent statistics alongside the existing market summary. These are
fundamental simulation outputs that every consumer needs.

| Metric | Description |
|--------|-------------|
| `total_pnl` | Mark-to-market PnL: ending_cash + inventory × last_mid − initial_cash |
| `sharpe_ratio` | Annualised Sharpe of PnL returns |
| `max_drawdown` | Peak-to-trough drawdown (cents) |
| `fill_rate` | Fills / submissions |
| `order_to_trade_ratio` | Submissions / fills |
| `vwap_cents` | Per-agent volume-weighted average fill price |
| `trade_count` | Number of fills |
| `end_inventory` | Final position (shares) |
| `inventory_std` | Position volatility over simulation |

The kernel already tracks `pnl_cents` per agent at end-state — this just needs
to be surfaced through the metrics API rather than requiring consumers to
walk the order log.

### Tier 2 — Standard Microstructure Indicators

These are well-established academic and regulatory metrics that belong in a
market simulator's analytics layer. They complement the existing market
summary.

| Metric | Description |
|--------|-------------|
| `lob_imbalance_mean` / `_std` | $(Q_{bid} - Q_{ask}) / (Q_{bid} + Q_{ask})$ at L1 |
| `resilience_mean_ns` | Mean spread recovery time after shock (spread > $\mu + 2\sigma$) |
| `market_ott_ratio` | Market-wide order-to-trade ratio (MiFID II standard) |
| `pct_time_two_sided` | Fraction of observations with both bid and ask present |

### Tier 3 — Per-Fill Execution Analysis

Per-fill metrics are broadly useful for execution quality research and strategy
evaluation. These could live in a separate `compute_execution_analysis()`
function or be opt-in via a flag.

| Metric | Description |
|--------|-------------|
| `slippage_bps` | Signed slippage per fill vs. contemporaneous mid-price |
| `adverse_selection_bps` | Mid-price move against fill direction at configurable look-ahead windows (e.g. 100ms, 500ms, 1s, 5s) |
| `order_lifecycle` | Per-order status: filled / cancelled / resting, with resting duration |

## Suggested API Shape

```python
# Option A: extend compute_metrics() return type
result = compute_metrics(book_log2, exec_trades, agent_states=result.agents)
result.market     # MarketSummary (existing + Tier 2 additions)
result.agents     # list[AgentMetrics]           ← NEW
result.fills      # list[FillRecord] | None      ← NEW (Tier 3, opt-in)

# Option B: separate function
from abides_markets.simulation.metrics import compute_rich_metrics

rich = compute_rich_metrics(
    simulation_result,          # SimulationResult has everything needed
    include_fills=True,         # opt-in for Tier 3
    adverse_selection_windows=["100ms", "500ms", "1s"],
)
```

Option B is cleaner — it takes the full `SimulationResult` (which already
bundles L1, logs, and agent end-state) rather than requiring callers to
destructure it into separate arguments.

## What This Unblocks

- **Consumers drop 200+ lines** of metric computation code per project.
- **Agent-level results become first-class** — today the kernel computes agent
  PnL internally but `compute_metrics()` discards it, forcing reconstruction.
- **Metric definitions are versioned** — when the computation changes (e.g.
  VPIN bucket size, resilience threshold), it's in one place.
- **New consumers get rich analytics out of the box** — lower barrier to
  building strategy evaluation, backtesting, and research tools on hasufel.

## Context

This request comes from building [Rohan](https://github.com/GabrieleDiCorato/rohan),
an LLM-driven strategy refinement platform that uses hasufel as its simulation
backend. Rohan currently maintains its own `AnalysisService` (~400 lines) to
compute all of the above from hasufel's raw data. Happy to contribute the
implementations upstream if that's useful.
