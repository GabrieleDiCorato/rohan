import pandas as pd
import pytest
from pandera.typing import DataFrame

from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import SimulationOutput
from rohan.simulation.models.schemas import AgentLogsSchema, OrderBookL1Schema, OrderBookL2Schema


class MockAgent:
    def __init__(self, id, holdings, log):
        self.id = id
        self.holdings = holdings
        self.log = log


class MockOutput(SimulationOutput):
    def __init__(self, agents, l1_df, *, strategic_agent_id=None):
        self.end_state = {"agents": agents}
        self.strategic_agent_id = strategic_agent_id
        self.l1 = l1_df

    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        return self.l1

    def get_order_book_l2(self, n_levels) -> DataFrame[OrderBookL2Schema]:
        raise NotImplementedError()

    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        raise NotImplementedError()

    def get_logs_by_agent(self) -> dict:
        raise NotImplementedError()


def test_compute_agent_metrics_pnl():
    # All monetary values in integer cents.
    # Agent 1: 100 000¢ CASH ($1 000), 10 NVDA shares
    # Price: Bid 10 000¢ ($100), Ask 10 200¢ ($102) => Mid 10 100¢
    # initial_cash = 50 000¢ ($500)
    # PnL = (100 000 + 10 * 10 100) - 50 000 = 151 000¢

    agent = MockAgent(id=1, holdings={"CASH": 100000, "NVDA": 10}, log=[])

    l1_df = pd.DataFrame(
        {
            "bid_price": [10000.0],
            "ask_price": [10200.0],
            "bid_qty": [10.0],
            "ask_qty": [10.0],
            "time": [0],
            "timestamp": [pd.Timestamp("2021-01-01")],
        }
    )

    output = MockOutput([agent], l1_df, strategic_agent_id=1)
    metrics = AnalysisService.compute_agent_metrics(output, 1, initial_cash=50000)

    assert metrics.agent_id == 1
    assert metrics.initial_cash == 50000
    assert metrics.ending_cash == 100000
    assert metrics.total_pnl == 100000 + 10 * 10100.0 - 50000  # 151 000
    assert metrics.end_inventory == 10


def test_compute_agent_metrics_trades():
    log: list[tuple[int, str, dict]] = [
        (100, "ORDER_SUBMITTED", {}),
        (101, "ORDER_SUBMITTED", {}),
        (102, "ORDER_EXECUTED", {}),  # 1 fill
    ]

    agent = MockAgent(id=2, holdings={"CASH": 0}, log=log)
    output = MockOutput([agent], pd.DataFrame())

    metrics = AnalysisService.compute_agent_metrics(output, 2)

    assert metrics.trade_count == 1
    assert metrics.fill_rate == 0.5  # 1 exec / 2 sub
    assert metrics.order_to_trade_ratio == 2.0  # 2 sub / 1 exec


def test_compute_agent_metrics_via_strategic_id():
    """Typical caller pattern: read strategic_agent_id from output, then pass it."""
    strategic = MockAgent(
        id=10,
        holdings={"CASH": 60000, "NVDA": 5},
        log=[
            (100, "ORDER_SUBMITTED", {}),
            (101, "ORDER_EXECUTED", {}),
        ],
    )
    noise = MockAgent(id=1, holdings={"CASH": 100000}, log=[])

    l1_df = pd.DataFrame(
        {
            "bid_price": [10000.0],
            "ask_price": [10200.0],
            "bid_qty": [10.0],
            "ask_qty": [10.0],
            "time": [0],
            "timestamp": [pd.Timestamp("2021-01-01")],
        }
    )

    output = MockOutput([noise, strategic], l1_df, strategic_agent_id=10)

    # Caller retrieves ID from output, passes it explicitly
    assert output.strategic_agent_id is not None
    metrics = AnalysisService.compute_agent_metrics(
        output,
        output.strategic_agent_id,
        initial_cash=50000,
    )

    assert metrics.agent_id == 10
    assert metrics.initial_cash == 50000
    assert metrics.ending_cash == 60000
    assert metrics.end_inventory == 5
    assert metrics.total_pnl == 60000 + 5 * 10100.0 - 50000
    assert metrics.trade_count == 1
    assert metrics.fill_rate == 1.0


def test_compute_agent_metrics_missing_agent_raises():
    """ValueError when the requested agent_id is not in the output."""
    agent = MockAgent(id=1, holdings={"CASH": 0}, log=[])
    output = MockOutput([agent], pd.DataFrame())

    with pytest.raises(ValueError, match="Agent 999 not found"):
        AnalysisService.compute_agent_metrics(output, 999)
