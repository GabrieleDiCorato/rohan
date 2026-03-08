"""Parameterized investigation tools for the Explainer ReAct agent.

Each tool operates on a serialized ``RichAnalysisBundle`` JSON string,
parsed once at tool-creation time.  Tools are closure-bound — no live
``SimulationOutput`` is required.

The factory :func:`make_investigation_tools` returns 8 LangChain tools
ready to be passed to ``create_react_agent``.
"""

from __future__ import annotations

import logging
from bisect import bisect_left
from typing import Any

from langchain_core.tools import tool

from rohan.framework.analysis_models import RichAnalysisBundle

logger = logging.getLogger(__name__)

# Maximum characters per tool response to keep LLM context manageable.
_MAX_RESPONSE_CHARS = 4000

_NO_DATA = "No analysis data available."


def _truncate(text: str, limit: int = _MAX_RESPONSE_CHARS) -> str:
    """Truncate *text* and append an indicator if it exceeds *limit*."""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... (truncated)"


def _ns_to_human(ns: int) -> str:
    """Format nanoseconds-since-midnight as ``HH:MM:SS.mmm``."""
    total_s = ns / 1e9
    h = int(total_s // 3600)
    m = int((total_s % 3600) // 60)
    s = total_s % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def make_investigation_tools(rich_json: str | None) -> list[Any]:
    """Return 8 LangChain tools closed over a parsed ``RichAnalysisBundle``.

    Parameters
    ----------
    rich_json:
        JSON string produced by ``RichAnalysisBundle.model_dump_json()``.
        If *None* or empty, every tool returns a "no data" message.
    """

    bundle: RichAnalysisBundle | None = None
    if rich_json:
        try:
            bundle = RichAnalysisBundle.model_validate_json(rich_json)
        except Exception:
            logger.warning("Failed to parse rich_analysis_json; tools will return no-data.")

    # -- helpers for time filtering ---
    def _in_range(ts: int, start: int | None, end: int | None) -> bool:
        if start is not None and ts < start:
            return False
        return not (end is not None and ts > end)

    # ── Tool definitions ──────────────────────────────────────────────

    @tool
    def query_fills(
        start_ns: int | None = None,
        end_ns: int | None = None,
        side: str | None = None,
        limit: int = 50,
    ) -> str:
        """Get per-fill execution data with optional time-range and side filters.

        Parameters
        ----------
        start_ns : optional
            Start timestamp filter (nanoseconds since midnight).
        end_ns : optional
            End timestamp filter (nanoseconds since midnight).
        side : optional
            Filter by 'BUY' or 'SELL'.
        limit : int
            Maximum number of fills to return (default 50).
        """
        if bundle is None:
            return _NO_DATA
        fills = bundle.fills
        if side:
            fills = [f for f in fills if f.side.upper() == side.upper()]
        fills = [f for f in fills if _in_range(f.timestamp_ns, start_ns, end_ns)]
        if not fills:
            return "No fills match the given filters."
        fills = fills[:limit]
        lines = [f"Fills ({len(fills)} shown):"]
        lines.append("Time | Side | Price | Qty | Mid@Fill | Slippage(bps) | Counterparty")
        lines.append("-" * 80)
        for f in fills:
            mid_s = f"{f.mid_at_fill:.0f}" if f.mid_at_fill is not None else "N/A"
            slip_s = f"{f.slippage_bps:.1f}" if f.slippage_bps is not None else "N/A"
            cp_s = f.counterparty_type or "unknown"
            lines.append(f"{_ns_to_human(f.timestamp_ns)} | {f.side} | {f.price} | {f.qty} | {mid_s} | {slip_s} | {cp_s}")
        return _truncate("\n".join(lines))

    @tool
    def query_pnl_curve(
        start_ns: int | None = None,
        end_ns: int | None = None,
        limit: int = 100,
    ) -> str:
        """Get mark-to-market PnL over time with optional time-range filter.

        Parameters
        ----------
        start_ns : optional
            Start timestamp filter.
        end_ns : optional
            End timestamp filter.
        limit : int
            Maximum data points to return (default 100).
        """
        if bundle is None:
            return _NO_DATA
        pts = [p for p in bundle.pnl_curve if _in_range(p.timestamp_ns, start_ns, end_ns)]
        if not pts:
            return "No PnL data in the given range."
        # Downsample if needed
        if len(pts) > limit:
            step = len(pts) / limit
            pts = [pts[int(i * step)] for i in range(limit)]
        lines = [f"PnL curve ({len(pts)} points):"]
        lines.append("Time | MTM PnL (cents)")
        lines.append("-" * 40)
        for p in pts:
            lines.append(f"{_ns_to_human(p.timestamp_ns)} | {p.mark_to_market_pnl:+.0f}")
        # Add summary stats
        all_pnl = [p.mark_to_market_pnl for p in pts]
        lines.append(f"\nMin: {min(all_pnl):+.0f}  Max: {max(all_pnl):+.0f}  Final: {all_pnl[-1]:+.0f}")
        return _truncate("\n".join(lines))

    @tool
    def query_inventory(
        start_ns: int | None = None,
        end_ns: int | None = None,
        limit: int = 100,
    ) -> str:
        """Get position trajectory over time with optional time-range filter.

        Parameters
        ----------
        start_ns : optional
            Start timestamp filter.
        end_ns : optional
            End timestamp filter.
        limit : int
            Maximum data points to return (default 100).
        """
        if bundle is None:
            return _NO_DATA
        pts = [p for p in bundle.inventory_trajectory if _in_range(p.timestamp_ns, start_ns, end_ns)]
        if not pts:
            return "No inventory data in the given range."
        if len(pts) > limit:
            step = len(pts) / limit
            pts = [pts[int(i * step)] for i in range(limit)]
        lines = [f"Inventory trajectory ({len(pts)} points):"]
        lines.append("Time | Position (shares)")
        lines.append("-" * 40)
        for p in pts:
            lines.append(f"{_ns_to_human(p.timestamp_ns)} | {p.position:+d}")
        positions = [p.position for p in pts]
        lines.append(f"\nMin: {min(positions):+d}  Max: {max(positions):+d}  Final: {positions[-1]:+d}")
        return _truncate("\n".join(lines))

    @tool
    def query_adverse_selection(window_label: str | None = None) -> str:
        """Get adverse selection metrics (mid-price move against fill direction).

        Parameters
        ----------
        window_label : optional
            Specific window to query (e.g. '100ms', '500ms', '1s', '5s').
            If omitted, returns all available windows plus the default metric.
        """
        if bundle is None:
            return _NO_DATA
        lines = ["Adverse Selection Analysis:"]
        if bundle.adverse_selection_bps is not None:
            lines.append(f"  Default window: {bundle.adverse_selection_bps:.2f} bps")
        if window_label:
            val = bundle.adverse_selection_by_window.get(window_label)
            if val is not None:
                lines.append(f"  Window {window_label}: {val:.2f} bps")
            else:
                available = ", ".join(bundle.adverse_selection_by_window.keys()) or "none"
                lines.append(f"  Window '{window_label}' not found. Available: {available}")
        else:
            if bundle.adverse_selection_by_window:
                lines.append("  By window:")
                for label, val in sorted(bundle.adverse_selection_by_window.items()):
                    lines.append(f"    {label}: {val:.2f} bps")
        if len(lines) == 1:
            return "No adverse selection data available."
        return "\n".join(lines)

    @tool
    def query_book_at_time(timestamp_ns: int, n_levels: int = 5) -> str:
        """Get the L2 order book snapshot nearest to a specific timestamp.

        Parameters
        ----------
        timestamp_ns : int
            Target timestamp (nanoseconds since midnight).
        n_levels : int
            Number of price levels to show per side (default 5).
        """
        if bundle is None:
            return _NO_DATA
        if not bundle.l2_snapshots:
            return "No L2 snapshots available."
        # Binary search for nearest snapshot
        times = [s.timestamp_ns for s in bundle.l2_snapshots]
        idx = bisect_left(times, timestamp_ns)
        # Pick closest of idx-1 and idx
        if idx >= len(times):
            idx = len(times) - 1
        elif idx > 0 and abs(times[idx - 1] - timestamp_ns) <= abs(times[idx] - timestamp_ns):
            idx = idx - 1
        snap = bundle.l2_snapshots[idx]
        delta_ms = (snap.timestamp_ns - timestamp_ns) / 1e6
        lines = [
            f"L2 Snapshot at {_ns_to_human(snap.timestamp_ns)} (delta: {delta_ms:+.1f}ms from query):",
            "",
            "  BIDS (price | qty):",
        ]
        for p, q in snap.bids[:n_levels]:
            lines.append(f"    {p:>8d} | {q:>6d}")
        if not snap.bids:
            lines.append("    (empty)")
        lines.append("  ASKS (price | qty):")
        for p, q in snap.asks[:n_levels]:
            lines.append(f"    {p:>8d} | {q:>6d}")
        if not snap.asks:
            lines.append("    (empty)")
        if snap.bids and snap.asks:
            spread = snap.asks[0][0] - snap.bids[0][0]
            mid = (snap.asks[0][0] + snap.bids[0][0]) / 2
            lines.append(f"\n  Spread: {spread} cents  Mid: {mid:.0f}")
        return "\n".join(lines)

    @tool
    def query_counterparties() -> str:
        """Get breakdown of which agent types the strategy traded against."""
        if bundle is None:
            return _NO_DATA
        if not bundle.counterparty_breakdown:
            return "No counterparty data available."
        lines = ["Counterparty Breakdown:"]
        lines.append("Agent Type | Trades | Avg Size | Total Volume")
        lines.append("-" * 55)
        total_trades = 0
        total_vol = 0
        for c in sorted(bundle.counterparty_breakdown, key=lambda x: x.trade_count, reverse=True):
            lines.append(f"{c.agent_type:<20s} | {c.trade_count:>6d} | {c.avg_size:>8.1f} | {c.total_volume:>12d}")
            total_trades += c.trade_count
            total_vol += c.total_volume
        lines.append("-" * 55)
        lines.append(f"{'TOTAL':<20s} | {total_trades:>6d} |          | {total_vol:>12d}")
        return "\n".join(lines)

    @tool
    def query_order_lifecycle(status: str | None = None, limit: int = 50) -> str:
        """Get order submission/fill/cancel statistics and timing.

        Parameters
        ----------
        status : optional
            Filter by status ('filled', 'cancelled', 'resting').
        limit : int
            Maximum records to return (default 50).
        """
        if bundle is None:
            return _NO_DATA
        records = bundle.order_lifecycle
        if status:
            records = [r for r in records if r.status.lower() == status.lower()]
        if not records:
            return f"No order lifecycle records{f' with status={status!r}' if status else ''}."
        records = records[:limit]
        lines = [f"Order Lifecycle ({len(records)} shown):"]
        lines.append("OrderID | Submitted | Status | Resting(ms) | Filled/Submitted")
        lines.append("-" * 70)
        for r in records:
            rest_s = f"{r.resting_time_ns / 1e6:.1f}" if r.resting_time_ns is not None else "N/A"
            lines.append(f"{r.order_id:>7d} | {_ns_to_human(r.submitted_at_ns)} | {r.status:<9s} | {rest_s:>11s} | {r.filled_qty}/{r.submitted_qty}")
        # Summary stats
        all_recs = bundle.order_lifecycle
        filled = sum(1 for r in all_recs if r.status == "filled")
        cancelled = sum(1 for r in all_recs if r.status == "cancelled")
        resting = sum(1 for r in all_recs if r.status == "resting")
        lines.append(f"\nTotals: {len(all_recs)} orders — {filled} filled, {cancelled} cancelled, {resting} resting")
        if filled > 0:
            fill_rate = filled / len(all_recs) * 100
            lines.append(f"Fill rate: {fill_rate:.1f}%")
        return _truncate("\n".join(lines))

    @tool
    def get_simulation_summary() -> str:
        """Get high-level simulation summary: fill count, PnL range, inventory range, adverse selection, time span."""
        if bundle is None:
            return _NO_DATA
        lines = ["Simulation Summary:"]

        # Fills
        lines.append(f"  Total fills: {len(bundle.fills)}")
        if bundle.fills:
            buys = sum(1 for f in bundle.fills if f.side == "BUY")
            sells = len(bundle.fills) - buys
            lines.append(f"  Buys: {buys}  Sells: {sells}")
            t_min = bundle.fills[0].timestamp_ns
            t_max = bundle.fills[-1].timestamp_ns
            lines.append(f"  First fill: {_ns_to_human(t_min)}  Last fill: {_ns_to_human(t_max)}")

        # PnL
        if bundle.pnl_curve:
            pnls = [p.mark_to_market_pnl for p in bundle.pnl_curve]
            lines.append(f"  PnL range: {min(pnls):+.0f} to {max(pnls):+.0f} cents")
            lines.append(f"  Final PnL: {pnls[-1]:+.0f} cents (${pnls[-1] / 100:+.2f})")

        # Inventory
        if bundle.inventory_trajectory:
            positions = [p.position for p in bundle.inventory_trajectory]
            lines.append(f"  Inventory range: {min(positions):+d} to {max(positions):+d} shares")
            lines.append(f"  Final inventory: {positions[-1]:+d} shares")

        # Adverse selection
        if bundle.adverse_selection_bps is not None:
            lines.append(f"  Adverse selection (default window): {bundle.adverse_selection_bps:.2f} bps")
        if bundle.adverse_selection_by_window:
            windows = ", ".join(f"{k}={v:.2f}bps" for k, v in sorted(bundle.adverse_selection_by_window.items()))
            lines.append(f"  Adverse selection by window: {windows}")

        # Counterparties
        if bundle.counterparty_breakdown:
            cp_summary = ", ".join(f"{c.agent_type}({c.trade_count})" for c in bundle.counterparty_breakdown)
            lines.append(f"  Counterparties: {cp_summary}")

        # Orders
        if bundle.order_lifecycle:
            total = len(bundle.order_lifecycle)
            filled = sum(1 for r in bundle.order_lifecycle if r.status == "filled")
            lines.append(f"  Orders: {total} total, {filled} filled ({filled / total * 100:.0f}% fill rate)")

        # Data coverage
        if bundle.mid_price_series:
            lines.append(f"  Mid-price observations: {len(bundle.mid_price_series)}")
        if bundle.l2_snapshots:
            lines.append(f"  L2 snapshots: {len(bundle.l2_snapshots)}")

        return "\n".join(lines)

    return [
        query_fills,
        query_pnl_curve,
        query_inventory,
        query_adverse_selection,
        query_book_at_time,
        query_counterparties,
        query_order_lifecycle,
        get_simulation_summary,
    ]
