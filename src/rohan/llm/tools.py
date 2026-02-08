"""LangChain tools for the Explainer agent.

Each tool wraps an analysis function and operates on the simulation
output stored in the current graph state.  The tools are stateless —
they receive a :class:`~rohan.simulation.models.SimulationOutput`
reference via closure.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from langchain_core.tools import tool

if TYPE_CHECKING:
    from rohan.simulation.models.simulation_output import SimulationOutput

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool factory — creates tool instances bound to a specific SimulationOutput
# ---------------------------------------------------------------------------


def make_explainer_tools(output: SimulationOutput) -> list[Any]:
    """Return a list of LangChain tools closed over *output*.

    This allows the Explainer agent to query simulation data without
    needing to pass ``output`` in every call.
    """

    @tool
    def get_order_book_snapshot(timestamp_ns: int) -> str:
        """Get the L1 order-book snapshot closest to *timestamp_ns*.

        Returns a human-readable string with bid/ask price and qty.
        """
        l1 = output.get_order_book_l1()
        if l1.empty:
            return "No order book data available."

        # Find closest timestamp
        idx = (l1["time"] - timestamp_ns).abs().idxmin()
        row = l1.loc[idx]
        return f"Time: {row['time']}\n" f"Bid: {row['bid_price']} (qty={row['bid_qty']})\n" f"Ask: {row['ask_price']} (qty={row['ask_qty']})\n" f"Mid: {(row['bid_price'] + row['ask_price']) / 2:.0f}"

    @tool
    def get_agent_trades(agent_id: int) -> str:
        """Get all trades (fills) executed by a specific agent.

        Returns a summary table of fills with timestamps, sides, prices,
        and quantities.
        """
        logs = output.get_logs_df()
        if logs.empty:
            return "No log data available."

        # Filter for the agent and fill events
        agent_mask = logs["AgentID"] == agent_id
        fill_mask = logs["EventType"].str.contains("FILL|ORDER_EXECUTED", case=False, na=False)
        trades = logs[agent_mask & fill_mask]
        if trades.empty:
            return f"No trades found for agent {agent_id}."

        result: str = trades.to_string(max_rows=50, max_cols=10)
        return result

    @tool
    def compute_pnl_curve(agent_id: int = 1) -> str:  # noqa: ARG001
        """Compute a rough PnL curve over time for the given agent.

        Returns descriptive statistics (start, end, min, max, mean).
        """
        l1 = output.get_order_book_l1()
        if l1.empty:
            return "No order book data for PnL curve."

        l1c = l1.copy()
        l1c["mid"] = (l1c["bid_price"] + l1c["ask_price"]) / 2

        # Simple PnL proxy: cumulative mid-price change
        l1c["cum_return"] = l1c["mid"].pct_change().fillna(0).cumsum()
        stats = l1c["cum_return"].describe()
        return f"PnL curve stats:\n{stats.to_string()}"

    @tool
    def get_price_stats() -> str:
        """Get summary statistics for mid prices (min, max, mean, std, count)."""
        l1 = output.get_order_book_l1()
        if l1.empty:
            return "No order book data."
        mid = (l1["bid_price"] + l1["ask_price"]) / 2
        return f"Mid-price stats:\n{mid.describe().to_string()}"

    @tool
    def get_spread_stats() -> str:
        """Get summary statistics for the bid-ask spread."""
        l1 = output.get_order_book_l1()
        if l1.empty:
            return "No order book data."
        spread = l1["ask_price"] - l1["bid_price"]
        return f"Spread stats (cents):\n{spread.describe().to_string()}"

    @tool
    def query_logs(agent_type: str = "", event_type: str = "", limit: int = 20) -> str:
        """Query agent logs with optional filters.

        Parameters
        ----------
        agent_type : str
            Filter by agent type (e.g. "ExchangeAgent", "ValueAgent").
            Empty string means no filter.
        event_type : str
            Filter by event type (e.g. "ORDER_SUBMITTED", "FILL").
            Empty string means no filter.
        limit : int
            Maximum rows to return.
        """
        logs = output.get_logs_df()
        if logs.empty:
            return "No log data."

        mask = pd.Series(True, index=logs.index)
        if agent_type:
            mask &= logs["AgentType"].str.contains(agent_type, case=False, na=False)
        if event_type:
            mask &= logs["EventType"].str.contains(event_type, case=False, na=False)

        filtered = logs[mask].head(limit)
        if filtered.empty:
            return f"No logs matching agent_type={agent_type!r}, event_type={event_type!r}."
        result: str = filtered.to_string(max_cols=10)
        return result

    @tool
    def get_volume_profile() -> str:
        """Get the traded volume profile over time (binned)."""
        logs = output.get_logs_df()
        if logs.empty:
            return "No log data."

        fills = logs[logs["EventType"].str.contains("FILL|ORDER_EXECUTED", case=False, na=False)]
        if fills.empty:
            return "No fills recorded."

        return f"Total fill events: {len(fills)}\n{fills['AgentType'].value_counts().to_string()}"

    return [
        get_order_book_snapshot,
        get_agent_trades,
        compute_pnl_curve,
        get_price_stats,
        get_spread_stats,
        query_logs,
        get_volume_profile,
    ]
