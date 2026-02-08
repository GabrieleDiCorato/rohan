import pandas as pd
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
    def __init__(self, agents, l1_df):
        self.end_state = {"agents": agents}
        self.l1 = l1_df

    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        return self.l1

    def get_order_book_l2(self, n_levels) -> DataFrame[OrderBookL2Schema]:
        raise NotImplementedError()

    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        raise NotImplementedError()

    def get_logs_by_agent(self) -> dict:
        raise NotImplementedError()


class MockEvent:
    def __init__(self, event_type):
        self.event_type = event_type

    def __str__(self):
        return str(self.event_type)


def test_compute_agent_metrics_pnl():
    # Setup mock data — all monetary values in integer cents.
    # Agent 1: 100000 cents CASH ($1000), 10 NVDA shares
    # Price: Bid 10000¢ ($100), Ask 10200¢ ($102) => Mid 10100¢
    # initial_cash = 50000¢ ($500)
    # PnL = (100000 + 10*10100) - 50000 = 151000¢ ($1510)

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

    output = MockOutput([agent], l1_df)

    metrics = AnalysisService.compute_agent_metrics(output, 1, initial_cash=50000)

    assert metrics.agent_id == 1
    assert metrics.initial_cash == 50000
    assert metrics.ending_cash == 100000
    assert metrics.total_pnl == 100000 + 10 * 10100.0 - 50000  # 151000.0
    assert metrics.end_inventory == 10


def test_compute_agent_metrics_trades():
    # Setup logs with events
    # We simulate events as objects with event_type attribute (like ABIDES)
    # or tuples if your code handles it. My code handles both.

    class EventType:
        ORDER_SUBMITTED = "ORDER_SUBMITTED"
        ORDER_EXECUTED = "ORDER_EXECUTED"

    log: list[tuple[int, str, dict] | tuple[int, str]] = [
        (100, "ORDER_SUBMITTED", {}),
        (101, "ORDER_SUBMITTED", {}),
        (102, "ORDER_EXECUTED", {}),  # 1 fill
    ]

    # Using tuple format for simplicity as my code checks string equality
    # Wait, my code: str(e.event_type) == "..." OR e[1] == "..."
    # So tuples are supported.

    agent = MockAgent(id=2, holdings={"CASH": 0}, log=log)

    output = MockOutput([agent], pd.DataFrame())

    metrics = AnalysisService.compute_agent_metrics(output, 2)

    assert metrics.trade_count == 1
    assert metrics.fill_rate == 0.5  # 1 exec / 2 sub
    assert metrics.order_to_trade_ratio == 2.0  # 2 sub / 1 exec
