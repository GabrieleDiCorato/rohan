"""Tests for investigation tools (Step 9 — make_investigation_tools).

All tests work with fully-serialised ``RichAnalysisBundle`` JSON —
no live SimulationOutput is needed.
"""

from __future__ import annotations

from rohan.framework.analysis_models import (
    CounterpartySummary,
    FillRecord,
    InventoryPoint,
    L2Snapshot,
    MidPricePoint,
    OrderLifecycleRecord,
    PnLPoint,
    RichAnalysisBundle,
)
from rohan.llm.tools import _NO_DATA, make_investigation_tools

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _sample_bundle() -> RichAnalysisBundle:
    """Return a small but realistic ``RichAnalysisBundle`` for testing."""
    return RichAnalysisBundle(
        fills=[
            FillRecord(timestamp_ns=1_000_000_000, side="BUY", price=10050, qty=100, mid_at_fill=10025.0, slippage_bps=2.5, counterparty_type="NoiseAgent"),
            FillRecord(timestamp_ns=2_000_000_000, side="SELL", price=10040, qty=50, mid_at_fill=10045.0, slippage_bps=-0.5, counterparty_type="ValueAgent"),
            FillRecord(timestamp_ns=3_000_000_000, side="BUY", price=10060, qty=75, mid_at_fill=10055.0, slippage_bps=0.5, counterparty_type="NoiseAgent"),
        ],
        pnl_curve=[
            PnLPoint(timestamp_ns=1_000_000_000, mark_to_market_pnl=0),
            PnLPoint(timestamp_ns=2_000_000_000, mark_to_market_pnl=500),
            PnLPoint(timestamp_ns=3_000_000_000, mark_to_market_pnl=-200),
        ],
        inventory_trajectory=[
            InventoryPoint(timestamp_ns=1_000_000_000, position=100),
            InventoryPoint(timestamp_ns=2_000_000_000, position=50),
            InventoryPoint(timestamp_ns=3_000_000_000, position=125),
        ],
        adverse_selection_bps=1.5,
        adverse_selection_by_window={"100ms": 0.8, "500ms": 1.2, "1s": 1.5, "5s": 2.1},
        counterparty_breakdown=[
            CounterpartySummary(agent_type="NoiseAgent", trade_count=2, avg_size=87.5, total_volume=175),
            CounterpartySummary(agent_type="ValueAgent", trade_count=1, avg_size=50.0, total_volume=50),
        ],
        order_lifecycle=[
            OrderLifecycleRecord(order_id=1, submitted_at_ns=900_000_000, status="filled", resting_time_ns=100_000_000, filled_qty=100, submitted_qty=100),
            OrderLifecycleRecord(order_id=2, submitted_at_ns=1_800_000_000, status="filled", resting_time_ns=200_000_000, filled_qty=50, submitted_qty=50),
            OrderLifecycleRecord(order_id=3, submitted_at_ns=2_500_000_000, status="cancelled", resting_time_ns=500_000_000, filled_qty=0, submitted_qty=200),
            OrderLifecycleRecord(order_id=4, submitted_at_ns=2_800_000_000, status="filled", resting_time_ns=200_000_000, filled_qty=75, submitted_qty=75),
        ],
        mid_price_series=[
            MidPricePoint(timestamp_ns=500_000_000, mid_price=10025.0),
            MidPricePoint(timestamp_ns=1_500_000_000, mid_price=10035.0),
            MidPricePoint(timestamp_ns=2_500_000_000, mid_price=10050.0),
        ],
        l2_snapshots=[
            L2Snapshot(timestamp_ns=1_000_000_000, bids=[(10020, 200), (10010, 300)], asks=[(10050, 150), (10060, 250)]),
            L2Snapshot(timestamp_ns=2_000_000_000, bids=[(10030, 180), (10020, 250)], asks=[(10060, 120), (10070, 200)]),
            L2Snapshot(timestamp_ns=3_000_000_000, bids=[(10040, 160), (10030, 280)], asks=[(10070, 100), (10080, 220)]),
        ],
    )


def _bundle_json() -> str:
    return _sample_bundle().model_dump_json()


def _get_tool(tools: list, name: str):
    return next(t for t in tools if t.name == name)


# Tools with required parameters need non-empty dicts when invoked.
_REQUIRED_ARGS: dict[str, dict] = {
    "query_book_at_time": {"timestamp_ns": 0},
}


def _invoke_tool(t, extra_args: dict | None = None):
    """Invoke a tool with its required defaults (if any) plus *extra_args*."""
    args = {**_REQUIRED_ARGS.get(t.name, {}), **(extra_args or {})}
    return t.invoke(args)


# ═══════════════════════════════════════════════════════════════════════════
# Factory Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMakeInvestigationTools:
    def test_returns_eight_tools(self):
        tools = make_investigation_tools(_bundle_json())
        assert isinstance(tools, list)
        assert len(tools) == 8

    def test_expected_tool_names(self):
        tools = make_investigation_tools(_bundle_json())
        names = {t.name for t in tools}
        expected = {
            "query_fills",
            "query_pnl_curve",
            "query_inventory",
            "query_adverse_selection",
            "query_book_at_time",
            "query_counterparties",
            "query_order_lifecycle",
            "get_simulation_summary",
        }
        assert names == expected

    def test_none_json_returns_no_data_tools(self):
        tools = make_investigation_tools(None)
        assert len(tools) == 8
        for t in tools:
            assert _invoke_tool(t) == _NO_DATA

    def test_empty_string_returns_no_data_tools(self):
        tools = make_investigation_tools("")
        for t in tools:
            assert _invoke_tool(t) == _NO_DATA

    def test_invalid_json_returns_no_data_tools(self):
        tools = make_investigation_tools("{bad json!!}")
        for t in tools:
            assert _invoke_tool(t) == _NO_DATA


# ═══════════════════════════════════════════════════════════════════════════
# query_fills
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryFills:
    def test_returns_all_fills(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_fills").invoke({})
        assert "Fills (3 shown)" in result
        assert "BUY" in result
        assert "SELL" in result

    def test_filter_by_side(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_fills").invoke({"side": "BUY"})
        assert "Fills (2 shown)" in result
        assert "SELL" not in result.split("\n", 3)[-1]  # No SELL in data rows

    def test_filter_by_time_range(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_fills").invoke({"start_ns": 1_500_000_000, "end_ns": 2_500_000_000})
        assert "Fills (1 shown)" in result

    def test_limit(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_fills").invoke({"limit": 1})
        assert "Fills (1 shown)" in result

    def test_no_match(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_fills").invoke({"start_ns": 99_000_000_000})
        assert "No fills match" in result

    def test_shows_counterparty(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_fills").invoke({})
        assert "NoiseAgent" in result
        assert "ValueAgent" in result


# ═══════════════════════════════════════════════════════════════════════════
# query_pnl_curve
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryPnlCurve:
    def test_returns_pnl_data(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_pnl_curve").invoke({})
        assert "PnL curve" in result
        assert "+500" in result
        assert "-200" in result

    def test_time_filter(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_pnl_curve").invoke({"end_ns": 1_500_000_000})
        assert "1 points" in result

    def test_summary_stats(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_pnl_curve").invoke({})
        assert "Min:" in result
        assert "Max:" in result
        assert "Final:" in result


# ═══════════════════════════════════════════════════════════════════════════
# query_inventory
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryInventory:
    def test_returns_inventory_data(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_inventory").invoke({})
        assert "Inventory trajectory" in result
        assert "+100" in result
        assert "+50" in result

    def test_time_filter(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_inventory").invoke({"start_ns": 2_000_000_000})
        assert "2 points" in result

    def test_summary_stats(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_inventory").invoke({})
        assert "Min:" in result
        assert "Max:" in result
        assert "Final:" in result


# ═══════════════════════════════════════════════════════════════════════════
# query_adverse_selection
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryAdverseSelection:
    def test_returns_all_windows(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_adverse_selection").invoke({})
        assert "Default window" in result
        assert "100ms" in result
        assert "5s" in result

    def test_specific_window(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_adverse_selection").invoke({"window_label": "500ms"})
        assert "500ms" in result
        assert "1.20" in result

    def test_unknown_window(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_adverse_selection").invoke({"window_label": "999ms"})
        assert "not found" in result
        assert "Available" in result

    def test_empty_data(self):
        bundle = RichAnalysisBundle()
        tools = make_investigation_tools(bundle.model_dump_json())
        result = _get_tool(tools, "query_adverse_selection").invoke({})
        assert "No adverse selection data" in result


# ═══════════════════════════════════════════════════════════════════════════
# query_book_at_time
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryBookAtTime:
    def test_exact_timestamp(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_book_at_time").invoke({"timestamp_ns": 1_000_000_000})
        assert "L2 Snapshot" in result
        assert "BIDS" in result
        assert "ASKS" in result
        assert "10020" in result  # bid price
        assert "10050" in result  # ask price

    def test_nearest_snapshot(self):
        tools = make_investigation_tools(_bundle_json())
        # Between 1s and 2s — should snap to nearest (1s or 2s)
        result = _get_tool(tools, "query_book_at_time").invoke({"timestamp_ns": 1_400_000_000})
        assert "L2 Snapshot" in result
        assert "delta:" in result

    def test_n_levels(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_book_at_time").invoke({"timestamp_ns": 1_000_000_000, "n_levels": 1})
        assert "10020" in result  # best bid
        assert "10010" not in result  # second level should be excluded

    def test_spread_and_mid(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_book_at_time").invoke({"timestamp_ns": 1_000_000_000})
        assert "Spread:" in result
        assert "Mid:" in result

    def test_no_snapshots(self):
        bundle = RichAnalysisBundle()
        tools = make_investigation_tools(bundle.model_dump_json())
        result = _get_tool(tools, "query_book_at_time").invoke({"timestamp_ns": 0})
        assert "No L2 snapshots" in result


# ═══════════════════════════════════════════════════════════════════════════
# query_counterparties
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryCounterparties:
    def test_returns_breakdown(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_counterparties").invoke({})
        assert "Counterparty Breakdown" in result
        assert "NoiseAgent" in result
        assert "ValueAgent" in result
        assert "TOTAL" in result

    def test_empty_data(self):
        bundle = RichAnalysisBundle()
        tools = make_investigation_tools(bundle.model_dump_json())
        result = _get_tool(tools, "query_counterparties").invoke({})
        assert "No counterparty data" in result


# ═══════════════════════════════════════════════════════════════════════════
# query_order_lifecycle
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryOrderLifecycle:
    def test_returns_all_orders(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_order_lifecycle").invoke({})
        assert "Order Lifecycle (4 shown)" in result
        assert "filled" in result
        assert "cancelled" in result

    def test_filter_by_status(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_order_lifecycle").invoke({"status": "cancelled"})
        assert "Order Lifecycle (1 shown)" in result

    def test_limit(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_order_lifecycle").invoke({"limit": 2})
        assert "Order Lifecycle (2 shown)" in result

    def test_fill_rate_summary(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_order_lifecycle").invoke({})
        assert "Fill rate:" in result
        assert "75.0%" in result  # 3 filled out of 4

    def test_no_matching_status(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "query_order_lifecycle").invoke({"status": "resting"})
        assert "No order lifecycle records" in result


# ═══════════════════════════════════════════════════════════════════════════
# get_simulation_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestGetSimulationSummary:
    def test_returns_summary(self):
        tools = make_investigation_tools(_bundle_json())
        result = _get_tool(tools, "get_simulation_summary").invoke({})
        assert "Simulation Summary" in result
        assert "Total fills: 3" in result
        assert "Buys: 2" in result
        assert "Sells: 1" in result
        assert "PnL range:" in result
        assert "Inventory range:" in result
        assert "Adverse selection" in result
        assert "Counterparties:" in result
        assert "Orders:" in result
        assert "Mid-price observations: 3" in result
        assert "L2 snapshots: 3" in result

    def test_empty_bundle(self):
        bundle = RichAnalysisBundle()
        tools = make_investigation_tools(bundle.model_dump_json())
        result = _get_tool(tools, "get_simulation_summary").invoke({})
        assert "Simulation Summary" in result
        assert "Total fills: 0" in result
