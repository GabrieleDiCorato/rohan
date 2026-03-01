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

### Previous Code:
```python
{previous_code}
```
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

Convergence criteria:
- ``stop_converged``: Score ≥ 8 AND improvement plateaued (diminishing returns).
- ``stop_plateau``: Score similar (±0.5) for 3+ consecutive iterations.
- ``continue``: Otherwise, keep refining.
"""

AGGREGATOR_HUMAN = """\
## Goal
{goal}

## Previous Iterations
{history_table}

## Current Iteration ({iteration_number})

### Scenario Explanations
{explanations}

Provide your structured verdict and unified feedback for the Writer.
"""

HISTORY_ROW_TEMPLATE = "| {iter} | {pnl} | {vol_delta} | {spread_delta} | {score} | {summary} |"

HISTORY_TABLE_HEADER = """\
| Iter | PnL | Volatility Δ | Spread Δ | Score | Summary |
|------|-----|--------------|----------|-------|---------|"""
