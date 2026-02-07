import pandas as pd

from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import SimulationOutput


class MockAgent:
    def __init__(self, id, holdings, log):
        self.id = id
        self.holdings = holdings
        self.log = log


class MockOutput(SimulationOutput):
    def __init__(self, agents, l1_df):
        self.end_state = {"agents": agents}
        self.l1 = l1_df

    def get_order_book_l1(self):
        return self.l1

    def get_order_book_l2(self, n_levels):
        raise NotImplementedError()

    def get_logs_df(self):
        raise NotImplementedError()

    def get_logs_by_agent(self):
        raise NotImplementedError()


class MockEvent:
    def __init__(self, event_type):
        self.event_type = event_type

    def __str__(self):
        return str(self.event_type)


def test_compute_agent_metrics_pnl():
    # Setup mock data
    # Agent 1: 1000 CASH, 10 NVDA
    # Price: Bid 100, Ask 102 => Mid 101

    agent = MockAgent(id=1, holdings={"CASH": 1000, "NVDA": 10}, log=[])

    l1_df = pd.DataFrame(
        {
            "bid_price": [100],
            "ask_price": [102],
            "bid_qty": [10],
            "ask_qty": [10],
            "time": [0],
            "timestamp": [pd.Timestamp("2021-01-01")],
        }
    )

    output = MockOutput([agent], l1_df)

    metrics = AnalysisService.compute_agent_metrics(output, 1)

    assert metrics.agent_id == 1
    assert metrics.realized_pnl == 1000.0  # Cash
    assert metrics.unrealized_pnl == 10 * 101.0  # 1010.0
    assert metrics.total_pnl == 2010.0
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
