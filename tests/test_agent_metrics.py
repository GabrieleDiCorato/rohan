from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest
from pandera.typing import DataFrame

from rohan.framework.analysis_service import AnalysisService
from rohan.simulation import SimulationOutput
from rohan.simulation.models.schemas import AgentLogsSchema, OrderBookL1Schema, OrderBookL2Schema

# ---------------------------------------------------------------------------
# Constants shared by the risk-metric fixture tests (items 8)
# ---------------------------------------------------------------------------
# Timestamps are nanoseconds since the Unix epoch (1970-01-01 00:00:00).
# Using 09:30 on day 0 of that epoch means ns_date() returns 0, so
# t_midnight == ts_ns — both fills and L1 rows are on the same intraday scale.
_ONE_MIN_NS: int = 60 * 10**9
_T0: int = 34_200 * 10**9  # 09:30:00 in ns since epoch 1970-01-01
_T_FILL: int = _T0 - _ONE_MIN_NS  # 09:29:00 — one minute before all L1 rows
_FILL_PRICE: int = 10_000  # cents ($100)

# Mid-price sequence (35 values, cents).  Designed so that:
#   • PnL (= qty * (mid - FILL_PRICE)) rises, then falls → max_drawdown > 0.
#   • Return series has σ > 0 → Sharpe ratio is computable.
_VARYING_MIDS: list[int] = [
    10000,
    10200,
    10300,
    10100,
    9900,
    10000,
    10200,
    10400,
    10300,
    10100,
    9900,
    9800,
    9700,
    10000,
    10200,
    10400,
    10600,
    10400,
    10200,
    10000,
    9800,
    9600,
    9500,
    9700,
    9900,
    10100,
    10300,
    10500,
    10700,
    10600,
    10400,
    10200,
    10000,
    9900,
    10000,
]


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


# ---------------------------------------------------------------------------
# Fixture helpers for risk-metric tests
# ---------------------------------------------------------------------------


def _make_l1_df(mids: Sequence[float]) -> pd.DataFrame:
    """Build a two-sided L1 DataFrame from a list of mid prices.

    All rows are valid (both bid and ask non-NaN).  Times start at _T0 and
    advance by one minute per row, giving a well-defined median interval for
    the Sharpe annualisation step.
    """
    n = len(mids)
    mid = np.array(mids, dtype=float)
    half_spread = 50.0  # 50-cent half-spread
    return pd.DataFrame(
        {
            "time": np.array([_T0 + i * _ONE_MIN_NS for i in range(n)], dtype=int),
            "bid_price": mid - half_spread,
            "ask_price": mid + half_spread,
            "bid_qty": np.full(n, 50.0),
            "ask_qty": np.full(n, 50.0),
            "timestamp": pd.date_range("1970-01-01 09:30", periods=n, freq="1min"),
        }
    )


def _agent_with_single_buy() -> MockAgent:
    """MockAgent with exactly one BUY fill for 1 share at _FILL_PRICE.

    The fill is timestamped at _T_FILL (09:29:00), which is before every
    row in any L1 DataFrame produced by _make_l1_df().  This ensures the
    entire fill history is applied before the first PnL snapshot.
    """
    log = [
        (
            _T_FILL,
            "ORDER_EXECUTED",
            {"fill_price": _FILL_PRICE, "quantity": 1, "side": "BID", "agent_id": 99},
        )
    ]
    return MockAgent(id=99, holdings={"CASH": 0, "NVDA": 1}, log=log)


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


# ---------------------------------------------------------------------------
# Item 8 — risk metric tests with synthetic fixtures
# ---------------------------------------------------------------------------


def test_agent_risk_metrics_non_none():
    """sharpe_ratio, max_drawdown, and inventory_std are all computable when
    there are ≥30 two-sided L1 rows and at least one fill.

    The price sequence in _VARYING_MIDS is carefully chosen so that:
    - pnl_returns has σ > 0 → Sharpe is not None.
    - PnL rises then falls → max peak-to-trough > 0 → max_drawdown is not None.
    - 1 fill creates an inventory change → inventory_std is not None.

    Verification (1 BUY at fill_price=10000, initial_cash=10000):
      pnl[i] = 1 * (mid[i] - 10000)
      Peak: 700 cents at mid=10700; trough: −500 cents at mid=9500.
      max_drawdown = 1200 cents  (running_max 700 − trough of −500).
    """
    l1_df = _make_l1_df(_VARYING_MIDS)
    agent = _agent_with_single_buy()
    output = MockOutput([agent], l1_df, strategic_agent_id=99)
    metrics = AnalysisService.compute_agent_metrics(output, 99, initial_cash=_FILL_PRICE)

    assert metrics.sharpe_ratio is not None, "Sharpe ratio should be computable with 35 two-sided rows"

    assert metrics.max_drawdown is not None, "Max drawdown should be non-None when PnL has a trough"
    assert metrics.max_drawdown >= 0, "Max drawdown is always non-negative"

    assert metrics.inventory_std is not None, "Inventory std should be computable from 1 fill"
    assert metrics.inventory_std >= 0, "Inventory std is always non-negative"


def test_agent_sharpe_none_for_flat_pnl():
    """sharpe_ratio and max_drawdown are both None when PnL is perfectly flat.

    If the agent buys at exactly the mid price and mid never changes,
    mark-to-market PnL = 0 at every L1 snapshot:
    - pnl_returns = [0, 0, ..., 0]  →  σ = 0  →  Sharpe = None.
    - running_max = 0, drawdown = 0, max_dd_val = 0  →  max_drawdown = None.

    inventory_std is still computed (inventory goes 0 → 1 after the fill).
    """
    l1_df = _make_l1_df([_FILL_PRICE] * 35)  # constant mid = FILL_PRICE
    agent = _agent_with_single_buy()
    output = MockOutput([agent], l1_df, strategic_agent_id=99)
    metrics = AnalysisService.compute_agent_metrics(output, 99, initial_cash=_FILL_PRICE)

    assert metrics.sharpe_ratio is None, "Sharpe should be None when σ_pnl = 0"
    assert metrics.max_drawdown is None, "Max drawdown should be None when PnL never falls below its peak"
    assert metrics.inventory_std is not None, "Inventory std is still computable (inventory: 0 → 1)"
