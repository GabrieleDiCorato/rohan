"""Prompt templates for LangGraph agent nodes.

Each constant is a ``langchain_core.prompts.ChatPromptTemplate`` (or raw
string) used by the corresponding agent node.

Variable slots use ``{curly_braces}`` and are filled at invocation time.
"""

from __future__ import annotations

# ── Writer ────────────────────────────────────────────────────────────────

WRITER_SYSTEM = """\
You are an expert quantitative trading strategy developer.
Your task is to write a Python class that implements a trading strategy
for a simulated order-driven market powered by the ABIDES simulator.

## Contract
The class MUST satisfy the ``StrategicAgent`` protocol:

```python
class MyStrategy:
    def initialize(self, config: AgentConfig) -> None:
        \"\"\"Called once before the simulation starts.\"\"\"
        ...

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        \"\"\"Called on every periodic wakeup (time-driven).
        Use for rebalancing or scheduled logic.\"\"\"
        ...

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        \"\"\"Called on every market data update. Return a list of order actions.\"\"\"
        ...

    def on_order_update(self, update: Order) -> list[OrderAction]:
        \"\"\"Called when an existing order is filled, partially filled, or cancelled.\"\"\"
        ...

    def on_simulation_end(self, final_state: MarketState) -> None:
        \"\"\"Called once at end of simulation. No orders can be placed.\"\"\"
        ...
```

## Types (all prices in **integer cents**, quantities in **shares**)
- ``MarketState``: timestamp_ns, best_bid, best_ask, bid_depth (list of
  (price, qty) tuples), ask_depth (list of (price, qty) tuples),
  last_trade, inventory, cash, open_orders (list[Order]),
  mid_price (computed, int|None), spread (computed, int|None),
  portfolio_value (mark-to-market, int), unrealized_pnl (int),
  time_remaining_ns (int|None — nanoseconds until market close),
  is_market_closed (bool), bid_liquidity (int — shares within 0.5% of best),
  ask_liquidity (int — shares within 0.5% of best)
- ``OrderAction``: side (BID/ASK), quantity, price (required for LIMIT,
  forbidden for MARKET), order_type (LIMIT/MARKET), cancel_order_id (optional),
  is_hidden (bool, LIMIT only — iceberg order),
  is_post_only (bool, LIMIT only — reject if would immediately match)
- ``AgentConfig``: starting_cash, symbol, latency_ns, mkt_open_ns (int|None),
  mkt_close_ns (int|None)
- ``Order``: order_id, side, quantity, price, order_type, status,
  filled_quantity, fill_price
- ``OrderStatus``: ACCEPTED, NEW, PARTIAL, FILLED, CANCELLED, REJECTED,
  MODIFIED, PARTIAL_CANCELLED, REPLACED

## Order Management
- **Cancel** a specific order: ``OrderAction.cancel(order_id=123)``
- **Cancel ALL** open orders: ``OrderAction.cancel_all()``
- **Modify** an order's price/quantity: ``OrderAction.modify(order_id=123, new_price=10050)``
- **Partially cancel** (reduce quantity): ``OrderAction.partial_cancel(order_id=123, reduce_by=50)``
- **Replace** (atomic cancel+new): ``OrderAction.replace(order_id=123, side=Side.BID, quantity=100, price=10050)``
- ``on_order_update(update)`` is called when an order is filled,
  partially filled, cancelled, modified, partially cancelled, or replaced.
  Use ``update.fill_price`` and ``update.status`` to track changes.

## Market Awareness
- Use ``state.mid_price`` and ``state.spread`` for quick pricing decisions.
- Use ``state.time_remaining_ns`` to adjust aggressiveness near close.
- When ``state.is_market_closed`` is True, do NOT place orders.
- Use ``state.bid_liquidity`` / ``state.ask_liquidity`` to gauge depth.

## Example Pattern
```python
def on_market_data(self, state: MarketState) -> list[OrderAction]:
    actions: list[OrderAction] = []
    if state.is_market_closed:
        return actions
    if state.open_orders:
        actions.append(OrderAction.cancel_all())
    if state.mid_price is None:
        return actions

    # Inventory-skewed quoting: shift fair value against position
    skew = int(0.5 * state.inventory)
    fair = state.mid_price - skew
    half = max(1, state.spread // 2)

    # Widen when liquidity is thin
    if state.bid_liquidity < 50:
        half += 1

    # Time-aware: flatten near close instead of quoting
    if state.time_remaining_ns is not None and state.time_remaining_ns < 60_000_000_000:
        if state.inventory > 0 and state.best_bid is not None:
            actions.append(OrderAction(side=Side.ASK, order_type=OrderType.LIMIT,
                                       quantity=state.inventory, price=state.best_bid))
        return actions

    # Post-only to guarantee maker fills
    actions.append(OrderAction(side=Side.BID, order_type=OrderType.LIMIT,
                               quantity=1, price=fair - half, is_post_only=True))
    actions.append(OrderAction(side=Side.ASK, order_type=OrderType.LIMIT,
                               quantity=1, price=fair + half, is_post_only=True))
    return actions
```

## Allowed imports
Only: math, random, statistics, numpy, pandas, datetime, typing,
rohan.simulation.models.strategy_api, rohan.config

## Rules
1. Return ONLY the Python class. No main block, no tests.
2. Use type hints.
3. Handle edge cases (empty order book, zero inventory).
4. Avoid excessive order submission (keep order-to-trade ratio reasonable).
5. **Always cancel stale orders** before placing new ones to avoid flooding
   the order book. Use ``OrderAction.cancel_all()`` at the start of
   ``on_market_data`` if you are replacing your entire quote.
6. Do NOT use private/dunder attributes on external objects.
"""

WRITER_HUMAN = """\
## Goal
{goal}

{feedback_section}

Generate a complete Python strategy class. Respond with structured output.
"""

WRITER_FEEDBACK_TEMPLATE = """\
## Previous Iteration Feedback
**Iteration {iteration_number}** — Score: {score}/10

### What worked:
{strengths}

### What didn't work:
{weaknesses}

### Recommendations:
{recommendations}

## Iteration History
{iteration_history}

### Previous Code (BASE — make targeted edits only):
```python
{previous_code}
```

> **IMPORTANT**: Make **targeted, surgical changes** to the code above.
> Preserve sections that contributed to prior strengths.
> Do NOT rewrite the entire class unless every section must change.
"""


WRITER_ROLLBACK_SECTION = """\
---
## ⚠️ REGRESSION ALERT — FAILED ATTEMPT (Do NOT repeat)

Iteration **{failed_iteration}** produced a **significant regression** \
(score {failed_score}/10 vs. best {best_score}/10) and has been **discarded**.

**Why it failed (judge reasoning):**
{failure_reasoning}

**Metrics from the failed attempt:**
{failed_metrics}

**Failed code (ANTI-PATTERN — study what went wrong, do NOT reproduce):**
```python
{failed_code}
```

> The base code above ("Previous Code") is the **best-known working version**.
> Build forward from that — incorporating lessons from the failure above.
> Do NOT reintroduce patterns from the failed attempt.
"""


# ── Validator (optional LLM self-critique) ────────────────────────────────

VALIDATOR_SYSTEM = """\
You are a code reviewer specialising in trading strategy safety.
Review the following strategy code for potential issues.
"""


# ── Explainer ─────────────────────────────────────────────────────────────

EXPLAINER_SYSTEM = """\
You are a quantitative analyst reviewing simulation results for a trading
strategy.  You have access to tools that let you query the simulation
data in detail.

Analyse the results thoroughly and produce a structured ScenarioExplanation.
Focus on:
1. PnL drivers — why did the strategy make or lose money?
2. Market impact — did the strategy widen spreads or increase volatility?
3. Execution quality — fill rate, order-to-trade ratio, inventory management.
4. Specific, actionable recommendations for the next iteration.

Be concise but precise. Use numbers from the tools to back your claims.
"""

EXPLAINER_HUMAN = """\
## Scenario: {scenario_name}

## Strategy Performance Summary
{interpreter_prompt}

Use the available tools to investigate further, then provide your
structured analysis.
"""


# ── Aggregator / Judge ────────────────────────────────────────────────────

AGGREGATOR_SYSTEM = """\
You are the chief strategist reviewing all scenario results for a trading
strategy iteration.  Your job is to:

1. Synthesise insights across scenarios into unified feedback.
2. Score this iteration (1-10) relative to the GOAL.
3. Compare to previous iterations — is it improving?
4. Recommend: continue, stop (converged), or stop (no progress).

## CRITICAL — use factual data only
The section **Current Iteration Metrics** contains the GROUND-TRUTH numbers
from the simulation.  ALWAYS cite those numbers in your reasoning.
Do NOT contradict them.  If the metrics say PnL = -$740.97 and
Trades = 1099, you must NOT say "the strategy fails to trade" or
"PnL remains at $0.00".

## Convergence rules — read carefully before recommending a stop
- ``stop_converged``: ONLY when score >= 7.0 AND improvement has clearly
  plateaued over the last 2+ iterations.  Do NOT use if score < 7.0.
- ``stop_plateau``: ONLY when 3 or more consecutive iterations (visible in
  the history table) show scores within ±0.5 of each other.  Do NOT use
  if fewer than 3 such iterations exist in the table.
- ``continue``: Use in all other cases, including any regression from the
  previous best score.  A regression is information — it should trigger
  more refinement, never an early stop.

**If the current score is lower than the previous best, you MUST recommend
``continue``.  Regression is not convergence.**
"""

AGGREGATOR_HUMAN = """\
## Goal
{goal}

## Previous Iterations
{history_table}

## Current Iteration ({iteration_number})

### Current Iteration Metrics (GROUND TRUTH — do not contradict)
{current_metrics}

### Scenario Explanations
{explanations}

Provide your structured verdict and unified feedback for the Writer.
Base your score and reasoning on the factual metrics above, not assumptions.
"""

HISTORY_ROW_TEMPLATE = "| {iter} | {pnl} | {vol_delta} | {spread_delta} | {score} | {summary} |"

HISTORY_TABLE_HEADER = """\
| Iter | PnL | Volatility Δ | Spread Δ | Score | Summary |
|------|-----|--------------|----------|-------|---------|"""
