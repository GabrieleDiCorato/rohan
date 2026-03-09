"""Unit tests for StrategicAgentAdapter order lifecycle callbacks (TEST-3).

Tests ``order_modified``, ``order_partial_cancelled``, ``order_replaced``,
and ``market_closed`` at the unit level with mocked ABIDES internals.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from abides_markets.orders import Side as AbidesSide

from rohan.simulation.abides_impl.strategic_agent_adapter import StrategicAgentAdapter
from rohan.simulation.models.strategy_api import (
    AgentConfig,
    MarketState,
    Order,
    OrderAction,
    OrderStatus,
    Side,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingStrategy:
    """Strategy that records every callback invocation for assertions."""

    def __init__(self):
        self.order_updates: list[Order] = []
        self.tick_states: list[MarketState] = []

    def initialize(self, config: AgentConfig) -> None:
        pass

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        self.tick_states.append(state)
        return []

    def on_market_data(self, _state: MarketState) -> list[OrderAction]:
        return []

    def on_order_update(self, order: Order) -> list[OrderAction]:
        self.order_updates.append(order)
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass


def _make_adapter(strategy=None) -> tuple[StrategicAgentAdapter, RecordingStrategy]:
    """Create an adapter with a recording strategy and minimal ABIDES state."""
    strat = strategy or RecordingStrategy()
    adapter = StrategicAgentAdapter(
        id=1,
        strategy=strat,
        symbol="ABM",
        starting_cash=10_000_000,
        random_state=np.random.RandomState(42),
    )
    adapter._initialized = True
    # ABIDES TradingAgent state that callbacks depend on
    adapter.orders = {}
    adapter.holdings = {"CASH": 10_000_000}
    adapter.known_bids = {}
    adapter.known_asks = {}
    adapter.current_time = 1_000_000_000
    adapter.mkt_close = 2_000_000_000
    return adapter, strat


def _mock_order(order_id=100, quantity=50, price=10000, side=AbidesSide.BID, symbol="ABM"):
    """Create a mock ABIDES LimitOrder-like object."""
    return SimpleNamespace(
        order_id=order_id,
        quantity=quantity,
        limit_price=price,
        side=side,
        symbol=symbol,
    )


# ---------------------------------------------------------------------------
# order_modified
# ---------------------------------------------------------------------------


class TestOrderModified:
    """Tests for the order_modified callback."""

    @patch("abides_markets.agents.TradingAgent.order_modified")
    def test_cache_updated_after_modify(self, _mock_super):
        adapter, strategy = _make_adapter()
        order = _mock_order(order_id=10, price=9500)

        # Pre-populate cache as if order was accepted
        adapter._open_orders_cache[10] = adapter._abides_to_proto(order)

        adapter.order_modified(order)

        # Cache should be updated with new order state
        assert 10 in adapter._open_orders_cache
        cached = adapter._open_orders_cache[10]
        assert cached.price == 9500

    @patch("abides_markets.agents.TradingAgent.order_modified")
    def test_strategy_notified_with_modified_status(self, _mock_super):
        adapter, strategy = _make_adapter()
        order = _mock_order(order_id=10)
        adapter._open_orders_cache[10] = adapter._abides_to_proto(order)

        adapter.order_modified(order)

        assert len(strategy.order_updates) == 1
        assert strategy.order_updates[0].status == OrderStatus.MODIFIED
        assert strategy.order_updates[0].order_id == 10

    @patch("abides_markets.agents.TradingAgent.order_modified")
    def test_filled_quantity_preserved_after_modify(self, _mock_super):
        adapter, strategy = _make_adapter()
        order = _mock_order(order_id=10)
        adapter._filled_quantities[10] = 20
        adapter._open_orders_cache[10] = adapter._abides_to_proto(order)

        adapter.order_modified(order)

        update = strategy.order_updates[0]
        assert update.filled_quantity == 20


# ---------------------------------------------------------------------------
# order_partial_cancelled
# ---------------------------------------------------------------------------


class TestOrderPartialCancelled:
    """Tests for the order_partial_cancelled callback."""

    @patch("abides_markets.agents.TradingAgent.order_partial_cancelled")
    def test_cache_updated(self, _mock_super):
        adapter, strategy = _make_adapter()
        order = _mock_order(order_id=20, quantity=30)
        adapter._open_orders_cache[20] = adapter._abides_to_proto(order)

        adapter.order_partial_cancelled(order)

        assert 20 in adapter._open_orders_cache

    @patch("abides_markets.agents.TradingAgent.order_partial_cancelled")
    def test_strategy_notified_with_partial_cancelled_status(self, _mock_super):
        adapter, strategy = _make_adapter()
        order = _mock_order(order_id=20)
        adapter._open_orders_cache[20] = adapter._abides_to_proto(order)

        adapter.order_partial_cancelled(order)

        assert len(strategy.order_updates) == 1
        assert strategy.order_updates[0].status == OrderStatus.PARTIAL_CANCELLED
        assert strategy.order_updates[0].order_id == 20

    @patch("abides_markets.agents.TradingAgent.order_partial_cancelled")
    def test_filled_quantity_preserved(self, _mock_super):
        adapter, strategy = _make_adapter()
        order = _mock_order(order_id=20)
        adapter._filled_quantities[20] = 10
        adapter._open_orders_cache[20] = adapter._abides_to_proto(order)

        adapter.order_partial_cancelled(order)

        assert strategy.order_updates[0].filled_quantity == 10


# ---------------------------------------------------------------------------
# order_replaced
# ---------------------------------------------------------------------------


class TestOrderReplaced:
    """Tests for the order_replaced callback."""

    @patch("abides_markets.agents.TradingAgent.order_replaced")
    def test_old_order_removed_from_cache(self, _mock_super):
        adapter, strategy = _make_adapter()
        old_order = _mock_order(order_id=30, price=8000)
        new_order = _mock_order(order_id=31, price=8500)
        adapter._open_orders_cache[30] = adapter._abides_to_proto(old_order)

        adapter.order_replaced(old_order, new_order)

        assert 30 not in adapter._open_orders_cache

    @patch("abides_markets.agents.TradingAgent.order_replaced")
    def test_new_order_added_to_cache(self, _mock_super):
        adapter, strategy = _make_adapter()
        old_order = _mock_order(order_id=30, price=8000)
        new_order = _mock_order(order_id=31, price=8500)
        adapter._open_orders_cache[30] = adapter._abides_to_proto(old_order)

        adapter.order_replaced(old_order, new_order)

        assert 31 in adapter._open_orders_cache
        assert adapter._open_orders_cache[31].price == 8500

    @patch("abides_markets.agents.TradingAgent.order_replaced")
    def test_strategy_notified_with_replaced_status(self, _mock_super):
        adapter, strategy = _make_adapter()
        old_order = _mock_order(order_id=30)
        new_order = _mock_order(order_id=31)
        adapter._open_orders_cache[30] = adapter._abides_to_proto(old_order)

        adapter.order_replaced(old_order, new_order)

        assert len(strategy.order_updates) == 1
        assert strategy.order_updates[0].status == OrderStatus.REPLACED
        assert strategy.order_updates[0].order_id == 30

    @patch("abides_markets.agents.TradingAgent.order_replaced")
    def test_filled_quantities_cleaned_up(self, _mock_super):
        adapter, strategy = _make_adapter()
        old_order = _mock_order(order_id=30)
        new_order = _mock_order(order_id=31)
        adapter._filled_quantities[30] = 15
        adapter._open_orders_cache[30] = adapter._abides_to_proto(old_order)

        adapter.order_replaced(old_order, new_order)

        assert 30 not in adapter._filled_quantities


# ---------------------------------------------------------------------------
# market_closed
# ---------------------------------------------------------------------------


class TestMarketClosed:
    """Tests for the market_closed callback."""

    @patch("abides_markets.agents.TradingAgent.market_closed")
    def test_strategy_receives_tick_with_market_closed(self, _mock_super):
        adapter, strategy = _make_adapter()
        adapter.current_time = 1_500_000_000

        adapter.market_closed()

        assert len(strategy.tick_states) == 1
        assert strategy.tick_states[0].is_market_closed is True

    @patch("abides_markets.agents.TradingAgent.market_closed")
    def test_market_closed_not_called_before_init(self, _mock_super):
        adapter, strategy = _make_adapter()
        adapter._initialized = False

        adapter.market_closed()

        # on_tick should NOT be called
        assert len(strategy.tick_states) == 0


# ---------------------------------------------------------------------------
# _abides_to_proto conversion
# ---------------------------------------------------------------------------


class TestAbidesToProto:
    """Tests for the _abides_to_proto static method."""

    def test_bid_side_conversion(self):
        order = _mock_order(side=AbidesSide.BID)
        proto = StrategicAgentAdapter._abides_to_proto(order)
        assert proto.side == Side.BID

    def test_ask_side_conversion(self):
        order = _mock_order(side=AbidesSide.ASK)
        proto = StrategicAgentAdapter._abides_to_proto(order)
        assert proto.side == Side.ASK

    def test_fill_price_passed_through(self):
        order = _mock_order()
        proto = StrategicAgentAdapter._abides_to_proto(order, fill_price=9900)
        assert proto.fill_price == 9900

    def test_zero_quantity_clamped_to_one(self):
        order = _mock_order(quantity=0)
        proto = StrategicAgentAdapter._abides_to_proto(order)
        assert proto.quantity >= 1
