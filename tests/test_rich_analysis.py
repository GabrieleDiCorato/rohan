"""Tests for rich analysis toolbox (Step 8).

Covers:
- FillRecord / PnLPoint / InventoryPoint / etc. serialisation round-trip
- All 7 new AnalysisService methods (unit + edge cases)
- 3 new chart methods (plot_pnl_curve, plot_inventory, plot_fills_vs_mid)
- compute_rich_analysis convenience method
- Integration test with a real 5-minute ABIDES simulation
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandera.typing import DataFrame

from rohan.framework.analysis_models import (
    CounterpartySummary,
    FillRecord,
    InventoryPoint,
    OrderLifecycleRecord,
    PnLPoint,
    RichAnalysisBundle,
)
from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import SimulationOutput
from rohan.simulation.models.schemas import (
    AgentLogsSchema,
    OrderBookL1Schema,
    OrderBookL2Schema,
)

# ═══════════════════════════════════════════════════════════════════════════
# Mock infrastructure
# ═══════════════════════════════════════════════════════════════════════════


def _make_agent(agent_id: int, agent_type: str, log: list | None = None, holdings: dict | None = None) -> SimpleNamespace:
    """Create a mock agent with the right attributes."""
    return SimpleNamespace(
        id=agent_id,
        type=agent_type,
        name=f"{agent_type}_{agent_id}",
        log=log or [],
        holdings=holdings or {"CASH": 10_000_000},
    )


def _make_fill_entry(ts_ns: int, fill_price: int, qty: int, side: str, order_id: int = 1) -> tuple:
    """Create a raw log tuple for ORDER_EXECUTED."""
    return (ts_ns, "ORDER_EXECUTED", {"fill_price": fill_price, "quantity": qty, "side": side, "order_id": order_id})


def _make_submit_entry(ts_ns: int, qty: int, order_id: int = 1) -> tuple:
    """Create a raw log tuple for ORDER_SUBMITTED."""
    return (ts_ns, "ORDER_SUBMITTED", {"quantity": qty, "order_id": order_id})


def _make_cancel_entry(ts_ns: int, order_id: int = 1) -> tuple:
    """Create a raw log tuple for ORDER_CANCELLED."""
    return (ts_ns, "ORDER_CANCELLED", {"order_id": order_id})


class MockOutput(SimulationOutput):
    """Mock SimulationOutput with configurable agents and L1 data."""

    def __init__(
        self,
        agents: list | None = None,
        l1_df: pd.DataFrame | None = None,
        logs_df: pd.DataFrame | None = None,
        l2_df: pd.DataFrame | None = None,
        strategic_agent_id: int | None = 1,
    ):
        self.strategic_agent_id = strategic_agent_id
        self.end_state: dict[str, Any] = {"agents": agents or []}
        self._l1 = l1_df if l1_df is not None else pd.DataFrame()
        self._logs = logs_df if logs_df is not None else pd.DataFrame()
        self._l2 = l2_df if l2_df is not None else pd.DataFrame()

    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        return self._l1  # pyright: ignore[reportReturnType]

    def get_order_book_l2(self, n_levels: int) -> DataFrame[OrderBookL2Schema]:  # noqa: ARG002
        return self._l2  # pyright: ignore[reportReturnType]

    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        return self._logs  # pyright: ignore[reportReturnType]

    def get_logs_by_agent(self) -> dict[int, Any]:
        return {a.id: a.log for a in self.end_state["agents"]}


# ═══════════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════════

# Timestamps: simple ns-since-midnight values for mock data.
_T = [1_000_000_000 * i for i in range(1, 11)]  # 10 time points at 1s intervals


def _standard_l1(n: int = 10) -> pd.DataFrame:
    """L1 with stable mid ~10_000, spread=100."""
    times = [1_000_000_000 * (i + 1) for i in range(n)]
    mid = 10_000
    return pd.DataFrame(
        {
            "time": times,
            "bid_price": [float(mid - 50)] * n,
            "ask_price": [float(mid + 50)] * n,
            "bid_qty": [100.0] * n,
            "ask_qty": [100.0] * n,
            "timestamp": pd.date_range("1970-01-01 09:30", periods=n, freq="1s"),
        }
    )


def _standard_output(
    agent_fills: list[tuple] | None = None,
    agent_submissions: list[tuple] | None = None,
    agent_cancellations: list[tuple] | None = None,
    other_agents: list | None = None,
    l1_df: pd.DataFrame | None = None,
) -> MockOutput:
    """Build a MockOutput with one strategic agent and standard L1."""
    fills = agent_fills or []
    subs = agent_submissions or []
    cancels = agent_cancellations or []

    strategic_log = subs + fills + cancels
    strategic = _make_agent(1, "StrategicAgent", log=strategic_log)

    # Default counterparty: ExchangeAgent (id=0) with matching opposite-side fills
    exchange = _make_agent(0, "ExchangeAgent")
    if other_agents is None:
        # Create matching counterparty fills (opposite side)
        counter_log = []
        for entry in fills:
            payload = entry[2]
            side = payload.get("side", "")
            opp = "Side.ASK" if "BID" in str(side).upper() else "Side.BID"
            counter_log.append(_make_fill_entry(entry[0], payload["fill_price"], payload["quantity"], opp, payload.get("order_id", 99)))
        counter = _make_agent(2, "NoiseAgent", log=counter_log)
        agents = [exchange, strategic, counter]
    else:
        agents = [exchange, strategic, *other_agents]

    return MockOutput(
        agents=agents,
        l1_df=l1_df if l1_df is not None else _standard_l1(),
        strategic_agent_id=1,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Model serialisation tests
# ═══════════════════════════════════════════════════════════════════════════


class TestModelSerialisation:
    """Verify Pydantic models round-trip through JSON."""

    def test_fill_record_round_trip(self):
        fr = FillRecord(timestamp_ns=100, side="BUY", price=10000, qty=5, mid_at_fill=9990.0, slippage_bps=1.0, counterparty_type="NoiseAgent")
        restored = FillRecord.model_validate_json(fr.model_dump_json())
        assert restored == fr

    def test_pnl_point_round_trip(self):
        p = PnLPoint(timestamp_ns=200, mark_to_market_pnl=-50.5)
        assert PnLPoint.model_validate_json(p.model_dump_json()) == p

    def test_inventory_point_round_trip(self):
        ip = InventoryPoint(timestamp_ns=300, position=-3)
        assert InventoryPoint.model_validate_json(ip.model_dump_json()) == ip

    def test_order_lifecycle_round_trip(self):
        ol = OrderLifecycleRecord(order_id=7, submitted_at_ns=100, status="filled", resting_time_ns=50, filled_qty=10, submitted_qty=10)
        assert OrderLifecycleRecord.model_validate_json(ol.model_dump_json()) == ol

    def test_counterparty_summary_round_trip(self):
        cs = CounterpartySummary(agent_type="NoiseAgent", trade_count=5, avg_size=10.0, total_volume=50)
        assert CounterpartySummary.model_validate_json(cs.model_dump_json()) == cs

    def test_rich_bundle_round_trip(self):
        bundle = RichAnalysisBundle(
            fills=[FillRecord(timestamp_ns=1, side="BUY", price=100, qty=1)],
            pnl_curve=[PnLPoint(timestamp_ns=1, mark_to_market_pnl=0.0)],
            inventory_trajectory=[InventoryPoint(timestamp_ns=0, position=0)],
            adverse_selection_bps=2.5,
            counterparty_breakdown=[CounterpartySummary(agent_type="X", trade_count=1, avg_size=1.0, total_volume=1)],
            order_lifecycle=[OrderLifecycleRecord(order_id=1, submitted_at_ns=0, status="filled")],
        )
        restored = RichAnalysisBundle.model_validate_json(bundle.model_dump_json())
        assert restored == bundle

    def test_empty_bundle_round_trip(self):
        bundle = RichAnalysisBundle()
        restored = RichAnalysisBundle.model_validate_json(bundle.model_dump_json())
        assert restored == bundle
        assert restored.fills == []
        assert restored.adverse_selection_bps is None


# ═══════════════════════════════════════════════════════════════════════════
# _parse_fills tests
# ═══════════════════════════════════════════════════════════════════════════


class TestParseFills:
    """Tests for the extracted _parse_fills helper."""

    def test_basic_buy(self):
        raw = [_make_fill_entry(0, 10000, 5, "Side.BID", order_id=1)]
        result = AnalysisService._parse_fills(raw)
        assert len(result) == 1
        assert result[0]["side"] == "BUY"
        assert result[0]["signed_qty"] == 5
        assert result[0]["cash_delta"] == -50000  # -10000 * 5
        assert result[0]["price"] == 10000
        assert result[0]["order_id"] == 1

    def test_basic_sell(self):
        raw = [_make_fill_entry(0, 10000, 3, "Side.ASK")]
        result = AnalysisService._parse_fills(raw)
        assert len(result) == 1
        assert result[0]["side"] == "SELL"
        assert result[0]["signed_qty"] == -3
        assert result[0]["cash_delta"] == 30000  # +10000 * 3

    def test_empty_fills(self):
        assert AnalysisService._parse_fills([]) == []

    def test_invalid_payload_skipped(self):
        bad = [(0, "ORDER_EXECUTED", "not_a_dict")]
        assert AnalysisService._parse_fills(bad) == []

    def test_missing_fields_skipped(self):
        bad = [(0, "ORDER_EXECUTED", {"fill_price": 100})]  # missing qty, side
        assert AnalysisService._parse_fills(bad) == []

    def test_sorts_by_time(self):
        raw = [
            _make_fill_entry(2_000_000_000, 10000, 1, "Side.BID"),
            _make_fill_entry(1_000_000_000, 10000, 1, "Side.ASK"),
        ]
        result = AnalysisService._parse_fills(raw)
        assert result[0]["time"] < result[1]["time"]


# ═══════════════════════════════════════════════════════════════════════════
# get_fill_analysis tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetFillAnalysis:
    """Tests for get_fill_analysis."""

    def test_basic_fill_with_slippage(self):
        # Buy at 10050 when mid is 10000 → slippage = +1 * (10050 - 10000) / 10000 * 10000 = +50 bps
        output = _standard_output(agent_fills=[_make_fill_entry(_T[0], 10050, 1, "Side.BID")])
        fills = AnalysisService.get_fill_analysis(output, agent_id=1)
        assert len(fills) == 1
        assert fills[0].side == "BUY"
        assert fills[0].price == 10050
        assert fills[0].mid_at_fill is not None
        assert fills[0].slippage_bps is not None
        assert fills[0].slippage_bps > 0  # bought above mid = adverse

    def test_sell_slippage(self):
        # Sell at 9950 when mid is 10000 → slippage = -1 * (9950 - 10000) / 10000 * 10000 = +50 bps (adverse)
        output = _standard_output(agent_fills=[_make_fill_entry(_T[0], 9950, 1, "Side.ASK")])
        fills = AnalysisService.get_fill_analysis(output, agent_id=1)
        assert len(fills) == 1
        assert fills[0].slippage_bps is not None
        assert fills[0].slippage_bps > 0  # sold below mid = adverse

    def test_counterparty_type_resolved(self):
        output = _standard_output(agent_fills=[_make_fill_entry(_T[0], 10000, 1, "Side.BID")])
        fills = AnalysisService.get_fill_analysis(output, agent_id=1)
        assert len(fills) == 1
        assert fills[0].counterparty_type == "NoiseAgent"

    def test_no_fills_returns_empty(self):
        output = _standard_output(agent_fills=[])
        fills = AnalysisService.get_fill_analysis(output, agent_id=1)
        assert fills == []

    def test_agent_not_found_returns_empty(self):
        output = _standard_output()
        fills = AnalysisService.get_fill_analysis(output, agent_id=999)
        assert fills == []

    def test_no_end_state_returns_empty(self):
        output = MockOutput()
        delattr(output, "end_state")
        fills = AnalysisService.get_fill_analysis(output, agent_id=1)
        assert fills == []

    def test_multiple_fills(self):
        fills_data = [
            _make_fill_entry(_T[0], 10050, 2, "Side.BID", order_id=1),
            _make_fill_entry(_T[1], 9950, 3, "Side.ASK", order_id=2),
            _make_fill_entry(_T[2], 10020, 1, "Side.BID", order_id=3),
        ]
        output = _standard_output(agent_fills=fills_data)
        fills = AnalysisService.get_fill_analysis(output, agent_id=1)
        assert len(fills) == 3
        assert fills[0].side == "BUY"
        assert fills[1].side == "SELL"


# ═══════════════════════════════════════════════════════════════════════════
# get_pnl_curve tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetPnlCurve:
    """Tests for get_pnl_curve."""

    def test_basic_pnl_curve(self):
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills)
        pnl = AnalysisService.get_pnl_curve(output, agent_id=1, initial_cash=10_000_000)
        assert len(pnl) > 0
        assert all(isinstance(p, PnLPoint) for p in pnl)

    def test_no_fills_returns_empty(self):
        output = _standard_output(agent_fills=[])
        pnl = AnalysisService.get_pnl_curve(output, agent_id=1)
        assert pnl == []

    def test_agent_not_found_returns_empty(self):
        output = _standard_output(agent_fills=[_make_fill_entry(_T[0], 10000, 1, "Side.BID")])
        pnl = AnalysisService.get_pnl_curve(output, agent_id=999)
        assert pnl == []

    def test_pnl_at_first_point_after_fill(self):
        # Buy 1 share at 10000 with initial_cash=10_000_000
        # After fill: cash = 10_000_000 - 10_000 = 9_990_000, inv = 1
        # MTM = 9_990_000 + 1 * 10_000 = 10_000_000 → PnL = 0
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills)
        pnl = AnalysisService.get_pnl_curve(output, agent_id=1, initial_cash=10_000_000)
        # First point after fill should have PnL near 0 (bought at mid)
        assert len(pnl) > 0
        assert abs(pnl[0].mark_to_market_pnl) < 1000  # close to zero

    def test_empty_l1_returns_empty(self):
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills, l1_df=pd.DataFrame())
        pnl = AnalysisService.get_pnl_curve(output, agent_id=1)
        assert pnl == []


# ═══════════════════════════════════════════════════════════════════════════
# get_inventory_trajectory tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetInventoryTrajectory:
    """Tests for get_inventory_trajectory."""

    def test_basic_trajectory(self):
        fills = [
            _make_fill_entry(_T[0], 10000, 2, "Side.BID"),
            _make_fill_entry(_T[1], 10000, 1, "Side.ASK"),
        ]
        output = _standard_output(agent_fills=fills)
        traj = AnalysisService.get_inventory_trajectory(output, agent_id=1)
        # Initial (0) + 2 fills = 3 points
        assert len(traj) == 3
        assert traj[0].position == 0  # initial
        assert traj[1].position == 2  # after buy 2
        assert traj[2].position == 1  # after sell 1

    def test_all_buys(self):
        fills = [
            _make_fill_entry(_T[0], 10000, 1, "Side.BID"),
            _make_fill_entry(_T[1], 10000, 1, "Side.BID"),
            _make_fill_entry(_T[2], 10000, 1, "Side.BID"),
        ]
        output = _standard_output(agent_fills=fills)
        traj = AnalysisService.get_inventory_trajectory(output, agent_id=1)
        assert traj[-1].position == 3

    def test_no_fills_returns_empty(self):
        output = _standard_output(agent_fills=[])
        traj = AnalysisService.get_inventory_trajectory(output, agent_id=1)
        assert traj == []

    def test_single_fill(self):
        fills = [_make_fill_entry(_T[0], 10000, 5, "Side.ASK")]
        output = _standard_output(agent_fills=fills)
        traj = AnalysisService.get_inventory_trajectory(output, agent_id=1)
        assert len(traj) == 2
        assert traj[0].position == 0
        assert traj[1].position == -5  # sell


# ═══════════════════════════════════════════════════════════════════════════
# get_adverse_selection tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetAdverseSelection:
    """Tests for get_adverse_selection."""

    def test_no_fills_returns_none(self):
        output = _standard_output(agent_fills=[])
        assert AnalysisService.get_adverse_selection(output, agent_id=1) is None

    def test_stable_mid_returns_near_zero(self):
        # With a constant mid-price, no adverse selection
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills)
        result = AnalysisService.get_adverse_selection(output, agent_id=1, window_ns=2_000_000_000)
        assert result is not None
        assert abs(result) < 1  # near zero with constant mid

    def test_agent_not_found_returns_none(self):
        output = _standard_output(agent_fills=[_make_fill_entry(_T[0], 10000, 1, "Side.BID")])
        assert AnalysisService.get_adverse_selection(output, agent_id=999) is None

    def test_empty_l1_returns_none(self):
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills, l1_df=pd.DataFrame())
        assert AnalysisService.get_adverse_selection(output, agent_id=1) is None

    def test_adverse_buy_price_drops(self):
        # Buy when mid at 10000, future mid drops to 9800 → adverse
        l1 = pd.DataFrame(
            {
                "time": [1_000_000_000, 2_000_000_000],
                "bid_price": [9950.0, 9750.0],
                "ask_price": [10050.0, 9850.0],
                "bid_qty": [100.0, 100.0],
                "ask_qty": [100.0, 100.0],
                "timestamp": pd.date_range("1970-01-01 09:30", periods=2, freq="1s"),
            }
        )
        fills = [_make_fill_entry(1_000_000_000, 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills, l1_df=l1)
        result = AnalysisService.get_adverse_selection(output, agent_id=1, window_ns=1_500_000_000)
        assert result is not None
        assert result > 0  # adverse: bought and price dropped


# ═══════════════════════════════════════════════════════════════════════════
# get_order_lifecycle tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetOrderLifecycle:
    """Tests for get_order_lifecycle."""

    def test_filled_order(self):
        subs = [_make_submit_entry(_T[0], qty=10, order_id=1)]
        fills = [_make_fill_entry(_T[1], 10000, 10, "Side.BID", order_id=1)]
        output = _standard_output(agent_fills=fills, agent_submissions=subs)
        lc = AnalysisService.get_order_lifecycle(output, agent_id=1)
        assert len(lc) == 1
        assert lc[0].status == "filled"
        assert lc[0].filled_qty == 10
        assert lc[0].resting_time_ns is not None
        assert lc[0].resting_time_ns > 0

    def test_cancelled_order(self):
        subs = [_make_submit_entry(_T[0], qty=10, order_id=1)]
        cancels = [_make_cancel_entry(_T[2], order_id=1)]
        output = _standard_output(agent_submissions=subs, agent_cancellations=cancels)
        lc = AnalysisService.get_order_lifecycle(output, agent_id=1)
        assert len(lc) == 1
        assert lc[0].status == "cancelled"
        assert lc[0].filled_qty == 0

    def test_resting_order(self):
        subs = [_make_submit_entry(_T[0], qty=10, order_id=1)]
        output = _standard_output(agent_submissions=subs)
        lc = AnalysisService.get_order_lifecycle(output, agent_id=1)
        assert len(lc) == 1
        assert lc[0].status == "resting"
        assert lc[0].resting_time_ns is None

    def test_no_orders_returns_empty(self):
        output = _standard_output()
        lc = AnalysisService.get_order_lifecycle(output, agent_id=1)
        assert lc == []

    def test_multiple_orders(self):
        subs = [
            _make_submit_entry(_T[0], qty=10, order_id=1),
            _make_submit_entry(_T[1], qty=5, order_id=2),
        ]
        fills = [_make_fill_entry(_T[2], 10000, 10, "Side.BID", order_id=1)]
        cancels = [_make_cancel_entry(_T[3], order_id=2)]
        output = _standard_output(agent_fills=fills, agent_submissions=subs, agent_cancellations=cancels)
        lc = AnalysisService.get_order_lifecycle(output, agent_id=1)
        assert len(lc) == 2
        statuses = {r.order_id: r.status for r in lc}
        assert statuses[1] == "filled"
        assert statuses[2] == "cancelled"


# ═══════════════════════════════════════════════════════════════════════════
# get_counterparty_breakdown tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGetCounterpartyBreakdown:
    """Tests for get_counterparty_breakdown."""

    def test_single_counterparty(self):
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills)
        breakdown = AnalysisService.get_counterparty_breakdown(output, agent_id=1)
        assert len(breakdown) >= 1
        noise_entry = next((c for c in breakdown if c.agent_type == "NoiseAgent"), None)
        assert noise_entry is not None
        assert noise_entry.trade_count == 1

    def test_no_fills_returns_empty(self):
        output = _standard_output(agent_fills=[])
        breakdown = AnalysisService.get_counterparty_breakdown(output, agent_id=1)
        assert breakdown == []

    def test_unmatched_fills_show_unknown(self):
        # No counterparty agents → fills classified as Unknown
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills, other_agents=[])
        breakdown = AnalysisService.get_counterparty_breakdown(output, agent_id=1)
        assert len(breakdown) >= 1
        types = {c.agent_type for c in breakdown}
        # Either Unknown or empty (no matching counterparties)
        assert "Unknown" in types or len(types) == 0 or breakdown == []

    def test_multiple_counterparty_types(self):
        fills = [
            _make_fill_entry(_T[0], 10000, 1, "Side.BID", order_id=1),
            _make_fill_entry(_T[1], 10000, 1, "Side.BID", order_id=2),
        ]
        # Two different counterparties
        noise = _make_agent(
            2,
            "NoiseAgent",
            log=[
                _make_fill_entry(_T[0], 10000, 1, "Side.ASK", order_id=10),
            ],
        )
        value = _make_agent(
            3,
            "ValueAgent",
            log=[
                _make_fill_entry(_T[1], 10000, 1, "Side.ASK", order_id=11),
            ],
        )
        output = _standard_output(agent_fills=fills, other_agents=[noise, value])
        breakdown = AnalysisService.get_counterparty_breakdown(output, agent_id=1)
        types = {c.agent_type for c in breakdown}
        assert "NoiseAgent" in types
        assert "ValueAgent" in types


# ═══════════════════════════════════════════════════════════════════════════
# query_book_depth tests
# ═══════════════════════════════════════════════════════════════════════════


class TestQueryBookDepth:
    """Tests for query_book_depth."""

    def test_returns_dict_with_bids_asks(self):
        l2 = pd.DataFrame(
            {
                "time": [1_000_000_000, 1_000_000_000, 1_000_000_000, 1_000_000_000],
                "level": [0, 1, 0, 1],
                "side": ["bid", "bid", "ask", "ask"],
                "price": [9950.0, 9900.0, 10050.0, 10100.0],
                "qty": [100, 80, 90, 70],
                "timestamp": pd.date_range("1970-01-01 09:30", periods=4, freq="1s"),
            }
        )
        output = MockOutput(l2_df=l2)
        book = AnalysisService.query_book_depth(output, timestamp_ns=1_000_000_000, n_levels=2)
        assert "bids" in book
        assert "asks" in book
        assert len(book["bids"]) == 2
        assert len(book["asks"]) == 2

    def test_empty_l2_returns_empty(self):
        output = MockOutput()
        book = AnalysisService.query_book_depth(output, timestamp_ns=1_000_000_000)
        assert book == {"bids": [], "asks": []}


# ═══════════════════════════════════════════════════════════════════════════
# compute_rich_analysis tests
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeRichAnalysis:
    """Tests for the compute_rich_analysis convenience method."""

    def test_returns_rich_bundle(self):
        fills = [
            _make_fill_entry(_T[0], 10050, 2, "Side.BID", order_id=1),
            _make_fill_entry(_T[2], 9950, 1, "Side.ASK", order_id=2),
        ]
        subs = [
            _make_submit_entry(_T[0] - 100, qty=2, order_id=1),
            _make_submit_entry(_T[2] - 100, qty=1, order_id=2),
        ]
        output = _standard_output(agent_fills=fills, agent_submissions=subs)
        bundle = AnalysisService.compute_rich_analysis(output, agent_id=1, initial_cash=10_000_000)

        assert isinstance(bundle, RichAnalysisBundle)
        assert len(bundle.fills) == 2
        assert len(bundle.pnl_curve) > 0
        assert len(bundle.inventory_trajectory) > 0
        assert len(bundle.order_lifecycle) == 2

    def test_json_serialisable(self):
        fills = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        output = _standard_output(agent_fills=fills)
        bundle = AnalysisService.compute_rich_analysis(output, agent_id=1)
        json_str = bundle.model_dump_json()
        restored = RichAnalysisBundle.model_validate_json(json_str)
        assert restored.fills == bundle.fills
        assert restored.pnl_curve == bundle.pnl_curve

    def test_no_fills_returns_empty_bundle(self):
        output = _standard_output(agent_fills=[])
        bundle = AnalysisService.compute_rich_analysis(output, agent_id=1)
        assert bundle.fills == []
        assert bundle.pnl_curve == []
        assert bundle.inventory_trajectory == []
        assert bundle.adverse_selection_bps is None


# ═══════════════════════════════════════════════════════════════════════════
# Chart tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCharts:
    """Tests for the 3 new chart methods."""

    def test_plot_pnl_curve_returns_figure(self):
        points = [PnLPoint(timestamp_ns=i * 1_000_000_000, mark_to_market_pnl=float(i * 100)) for i in range(5)]
        fig = AnalysisService.plot_pnl_curve(points)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_pnl_curve_empty(self):
        fig = AnalysisService.plot_pnl_curve([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_inventory_returns_figure(self):
        points = [
            InventoryPoint(timestamp_ns=0, position=0),
            InventoryPoint(timestamp_ns=1_000_000_000, position=5),
            InventoryPoint(timestamp_ns=2_000_000_000, position=3),
            InventoryPoint(timestamp_ns=3_000_000_000, position=-2),
        ]
        fig = AnalysisService.plot_inventory(points)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_inventory_empty(self):
        fig = AnalysisService.plot_inventory([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fills_vs_mid_returns_figure(self):
        fills = [
            FillRecord(timestamp_ns=1_000_000_000, side="BUY", price=10050, qty=1, mid_at_fill=10000.0),
            FillRecord(timestamp_ns=2_000_000_000, side="SELL", price=9950, qty=1, mid_at_fill=10000.0),
        ]
        fig = AnalysisService.plot_fills_vs_mid(fills)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fills_vs_mid_empty(self):
        fig = AnalysisService.plot_fills_vs_mid([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fills_no_mid(self):
        """Fills without mid_at_fill should still render."""
        fills = [FillRecord(timestamp_ns=1_000_000_000, side="BUY", price=10000, qty=1)]
        fig = AnalysisService.plot_fills_vs_mid(fills)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# _agent_risk_metrics backward compatibility
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentRiskMetricsRefactor:
    """Verify _agent_risk_metrics still works after refactoring."""

    def test_with_raw_fills(self):
        """Original call path (raw tuples) must still work."""
        raw = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        l1 = _standard_l1(50)
        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()
        sharpe, max_dd, inv_std = AnalysisService._agent_risk_metrics(raw, two_sided, 10_000_000)
        # With only 1 fill and constant mid, values should be finite or None
        assert sharpe is None or isinstance(sharpe, float)  # at minimum doesn't crash

    def test_with_parsed_fills(self):
        """New call path (pre-parsed dicts) must work."""
        raw = [_make_fill_entry(_T[0], 10000, 1, "Side.BID")]
        parsed = AnalysisService._parse_fills(raw)
        l1 = _standard_l1(50)
        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()
        sharpe, max_dd, inv_std = AnalysisService._agent_risk_metrics(raw, two_sided, 10_000_000, parsed_fills=parsed)
        assert sharpe is None or isinstance(sharpe, float)

    def test_empty_fills_returns_none_triple(self):
        l1 = _standard_l1()
        two_sided = l1.dropna(subset=["bid_price", "ask_price"]).copy()
        sharpe, max_dd, inv_std = AnalysisService._agent_risk_metrics([], two_sided, 10_000_000)
        assert (sharpe, max_dd, inv_std) == (None, None, None)


# ═══════════════════════════════════════════════════════════════════════════
# Prompt enrichment tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPromptEnrichment:
    """Verify format_interpreter_prompt accepts rich_analysis_json."""

    def test_prompt_with_rich_analysis(self):
        from rohan.framework.prompts import format_interpreter_prompt
        from rohan.simulation import AgentMetrics, ComparisonResult, MarketImpact, MarketMetrics, RunSummary

        agent = AgentMetrics(agent_id=1, total_pnl=1000.0, trade_count=5)
        market = MarketMetrics()
        impact = MarketImpact()
        comparison = ComparisonResult(strategy_metrics=agent, strategy_market_metrics=market, baseline_metrics=market, market_impact=impact)
        summary = RunSummary(comparison=comparison)

        bundle = RichAnalysisBundle(
            adverse_selection_bps=3.5,
            counterparty_breakdown=[CounterpartySummary(agent_type="NoiseAgent", trade_count=3, avg_size=10.0, total_volume=30)],
            fills=[FillRecord(timestamp_ns=1, side="BUY", price=100, qty=1, slippage_bps=2.0)],
        )

        prompt = format_interpreter_prompt(summary, goal="Test", rich_analysis_json=bundle.model_dump_json())
        assert "3.5 bps" in prompt
        assert "NoiseAgent" in prompt
        assert "2.0 bps" in prompt

    def test_prompt_without_rich_analysis(self):
        from rohan.framework.prompts import format_interpreter_prompt
        from rohan.simulation import AgentMetrics, ComparisonResult, MarketImpact, MarketMetrics, RunSummary

        agent = AgentMetrics(agent_id=1)
        market = MarketMetrics()
        impact = MarketImpact()
        comparison = ComparisonResult(strategy_metrics=agent, strategy_market_metrics=market, baseline_metrics=market, market_impact=impact)
        summary = RunSummary(comparison=comparison)

        prompt = format_interpreter_prompt(summary, goal="Test")
        assert "N/A" in prompt  # adverse selection should show N/A


# ═══════════════════════════════════════════════════════════════════════════
# Integration test (real ABIDES simulation)
# ═══════════════════════════════════════════════════════════════════════════


class TestRichAnalysisIntegration:
    """Integration test with a real 5-minute ABIDES simulation."""

    @pytest.fixture
    def simulation_output(self):
        """Run a real simulation for integration testing."""
        from rohan.config import SimulationSettings
        from rohan.simulation import SimulationService

        settings = SimulationSettings()
        settings.start_time = "09:30:00"
        settings.end_time = "09:35:00"
        settings.seed = 42
        settings.agents.noise.num_agents = 100
        settings.agents.value.num_agents = 20
        settings.agents.momentum.num_agents = 2
        settings.agents.adaptive_market_maker.num_agents = 1

        service = SimulationService()
        result = service.run_simulation(settings)
        if result.error is not None:
            raise result.error
        return result.result

    def test_compute_metrics_still_works(self, simulation_output):
        """Existing compute_metrics must not break."""
        analyzer = AnalysisService()
        metrics = analyzer.compute_metrics(simulation_output)
        assert metrics is not None
        assert metrics.mean_spread is not None

    def test_compute_agent_metrics_still_works(self, simulation_output):
        """Existing compute_agent_metrics must not break."""
        analyzer = AnalysisService()
        # Agent 1 is typically the first trading agent (agent 0 is ExchangeAgent).
        # HasufelOutput lacks end_state, so compute_agent_metrics returns an
        # empty AgentMetrics via its hasattr guard — verify it doesn't crash.
        metrics = analyzer.compute_agent_metrics(simulation_output, 1)
        assert metrics is not None

    def test_plots_still_work(self, simulation_output):
        """Existing plot methods must not break."""
        analyzer = AnalysisService()
        fig = analyzer.plot_price_series(simulation_output)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        fig = analyzer.plot_spread(simulation_output)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
