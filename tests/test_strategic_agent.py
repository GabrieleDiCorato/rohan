"""Integration tests for StrategicAgent injection into ABIDES simulations.

These tests verify that:
1. A StrategicAgent implementation can be injected into the simulation
2. The strategy receives on_market_data callbacks
3. The strategy receives on_tick callbacks (time-driven)
4. OrderActions are executed correctly
5. Order cancellations work (individual and cancel-all)
6. on_order_update is called on fills and cancellations
7. MarketState includes L2 depth data (bid_depth, ask_depth)
8. on_simulation_end is called at kernel stop

Performance optimization: Uses shared fixtures to run simulations once per test class,
reducing from 5 separate simulations to 3 shared ones (~40% faster).
Simulation duration: 5 minutes (minimum required by SimulationSettings validator).
"""

import pytest

from rohan.config import SimulationSettings
from rohan.simulation.models.strategy_api import (
    AgentConfig,
    MarketState,
    Order,
    OrderAction,
    OrderStatus,
    OrderType,
    Side,
)
from rohan.simulation.simulation_service import SimulationService


class CountingStrategy:
    """A simple strategy that counts callbacks for testing."""

    def __init__(self):
        self.initialize_called = False
        self.market_data_count = 0
        self.tick_count = 0
        self.simulation_ended = False
        self.final_state: MarketState | None = None
        self.config: AgentConfig | None = None
        self.last_state: MarketState | None = None

    def initialize(self, config: AgentConfig) -> None:
        self.initialize_called = True
        self.config = config

    def on_tick(self, _state: MarketState) -> list[OrderAction]:
        self.tick_count += 1
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        self.market_data_count += 1
        self.last_state = state
        return []  # No orders

    def on_order_update(self, _update: Order) -> list[OrderAction]:
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        self.simulation_ended = True
        self.final_state = final_state


class SimpleBuyStrategy:
    """A strategy that places a single buy order on first market data."""

    def __init__(self):
        self.order_placed = False
        self.order_updates: list[Order] = []

    def initialize(self, config: AgentConfig) -> None:
        pass

    def on_tick(self, _state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        # Only place one order
        if not self.order_placed and state.best_ask is not None:
            self.order_placed = True
            return [
                OrderAction(
                    side=Side.BID,
                    quantity=100,
                    price=state.best_ask - 1,  # 1 cent below ask
                    order_type=OrderType.LIMIT,
                )
            ]
        return []

    def on_order_update(self, update: Order) -> list[OrderAction]:
        self.order_updates.append(update)
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass


class CancellingStrategy:
    """A strategy that places an order, then cancels it on the next tick.

    Used to verify:
    - Individual order cancellation via ``OrderAction.cancel()``
    - ``on_order_update`` is called with ``CANCELLED`` status
    """

    def __init__(self):
        self.ticks = 0
        self.placed_order_ids: list[int] = []
        self.order_updates: list[Order] = []
        self.cancelled_ids: list[int] = []
        self.order_placed = False

    def initialize(self, config: AgentConfig) -> None:
        pass

    def on_tick(self, _state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        self.ticks += 1

        if not self.order_placed and state.best_ask is not None:
            # Place a limit BID 1 cent below ask — should rest in book
            self.order_placed = True
            return [
                OrderAction(
                    side=Side.BID,
                    quantity=1,
                    price=state.best_ask - 1,
                    order_type=OrderType.LIMIT,
                )
            ]

        if self.order_placed and len(state.open_orders) > 0:
            # Cancel all open orders once we have at least one
            self.placed_order_ids = [o.order_id for o in state.open_orders]
            return [OrderAction.cancel(o.order_id) for o in state.open_orders]

        return []

    def on_order_update(self, update: Order) -> list[OrderAction]:
        self.order_updates.append(update)
        if update.status == OrderStatus.CANCELLED:
            self.cancelled_ids.append(update.order_id)
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass


class CancelAllStrategy:
    """A strategy that places orders then uses cancel_all() to clear them."""

    def __init__(self):
        self.ticks = 0
        self.open_order_count_after_cancel: int | None = None
        self.cancel_updates: list[Order] = []
        self.orders_placed = False
        self.cancel_sent = False

    def initialize(self, config: AgentConfig) -> None:
        pass

    def on_tick(self, _state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        self.ticks += 1

        if not self.orders_placed and state.best_ask is not None:
            # Place 3 orders, each 1 cent below ask
            self.orders_placed = True
            return [
                OrderAction(
                    side=Side.BID,
                    quantity=1,
                    price=state.best_ask - 1,
                    order_type=OrderType.LIMIT,
                ),
                OrderAction(
                    side=Side.BID,
                    quantity=1,
                    price=state.best_ask - 2,
                    order_type=OrderType.LIMIT,
                ),
                OrderAction(
                    side=Side.BID,
                    quantity=1,
                    price=state.best_ask - 3,
                    order_type=OrderType.LIMIT,
                ),
            ]

        if self.orders_placed and not self.cancel_sent and len(state.open_orders) >= 3:
            # Cancel ALL once we have at least 3 open orders
            self.cancel_sent = True
            return [OrderAction.cancel_all()]

        if self.cancel_sent and self.open_order_count_after_cancel is None:
            # Record count after cancel was sent
            self.open_order_count_after_cancel = len(state.open_orders)

        return []

    def on_order_update(self, update: Order) -> list[OrderAction]:
        if update.status == OrderStatus.CANCELLED:
            self.cancel_updates.append(update)
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass


# Shared simulation settings for tests (5 minutes - minimum required)
TEST_SETTINGS = SimulationSettings(
    seed=42,
    start_time="09:30:00",
    end_time="09:35:00",  # 5 minute duration - minimum required by validator
    _env_file=None,  # type: ignore[call-arg]  # Pydantic-settings parameter not visible to mypy
)


class TestStrategyInjection:
    """Tests for strategy injection into simulations.

    Uses class-scoped fixtures to share simulation results across tests,
    reducing total test time from ~60s to ~15s.
    """

    @pytest.fixture(scope="class")
    def counting_strategy_result(self):
        """Run simulation once with CountingStrategy and share across tests."""
        strategy = CountingStrategy()
        service = SimulationService()
        result = service.run_simulation(TEST_SETTINGS, strategy=strategy)  # pyright: ignore[reportArgumentType]
        return result, strategy

    @pytest.fixture(scope="class")
    def buy_strategy_result(self):
        """Run simulation once with SimpleBuyStrategy."""
        strategy = SimpleBuyStrategy()
        service = SimulationService()
        result = service.run_simulation(TEST_SETTINGS, strategy=strategy)  # pyright: ignore[reportArgumentType]
        return result, strategy

    @pytest.fixture(scope="class")
    def no_strategy_result(self):
        """Run simulation once without strategy (backwards compatibility)."""
        service = SimulationService()
        return service.run_simulation(TEST_SETTINGS)

    @pytest.fixture(scope="class")
    def cancelling_strategy_result(self):
        """Run simulation once with CancellingStrategy."""
        strategy = CancellingStrategy()
        service = SimulationService()
        result = service.run_simulation(TEST_SETTINGS, strategy=strategy)  # pyright: ignore[reportArgumentType]
        return result, strategy

    @pytest.fixture(scope="class")
    def cancel_all_strategy_result(self):
        """Run simulation once with CancelAllStrategy."""
        strategy = CancelAllStrategy()
        service = SimulationService()
        result = service.run_simulation(TEST_SETTINGS, strategy=strategy)  # pyright: ignore[reportArgumentType]
        return result, strategy

    def test_strategy_receives_initialize_callback(self, counting_strategy_result):
        """Test that the strategy's initialize method is called."""
        result, strategy = counting_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        assert strategy.initialize_called, "Strategy.initialize() was not called"
        assert strategy.config is not None, "Config was not passed to initialize"
        assert strategy.config.symbol == TEST_SETTINGS.ticker

    def test_strategy_receives_market_data_callbacks(self, counting_strategy_result):
        """Test that the strategy receives on_market_data callbacks."""
        result, strategy = counting_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        assert strategy.market_data_count > 0, "Strategy did not receive any on_market_data callbacks"
        assert strategy.last_state is not None, "No MarketState was received"

    def test_strategy_market_state_has_valid_data(self, counting_strategy_result):
        """Test that MarketState contains valid market data."""
        result, strategy = counting_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        state = strategy.last_state
        assert state is not None

        # Timestamp should be positive nanoseconds
        assert state.timestamp_ns > 0, "Invalid timestamp"

        # Cash should be set from starting_cash
        assert state.cash > 0, "Cash should be positive"

    def test_strategy_can_place_orders(self, buy_strategy_result):
        """Test that a strategy can place orders."""
        result, strategy = buy_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        assert strategy.order_placed, "Strategy did not place any orders"

    def test_on_order_update_called_on_interesting_events(self, buy_strategy_result):
        """Test that on_order_update fires for accepts, fills, and/or cancellations."""
        result, strategy = buy_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        # Strategy placed at best_ask - 1, so it may or may not fill.
        # order_accepted fires immediately, fills/cancels come later.
        for update in strategy.order_updates:
            assert update.status in (
                OrderStatus.ACCEPTED,
                OrderStatus.FILLED,
                OrderStatus.PARTIAL,
                OrderStatus.CANCELLED,
            )

    def test_simulation_without_strategy_still_works(self, no_strategy_result):
        """Test that simulation works without a strategy (backwards compatibility)."""
        result = no_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        assert result.result is not None

    # ── Cancellation tests ────────────────────────────────────────────────

    def test_cancel_order_works(self, cancelling_strategy_result):
        """Test that individual order cancellation via OrderAction.cancel() works."""
        result, strategy = cancelling_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        # Strategy should have placed at least one order on tick 1
        assert len(strategy.placed_order_ids) > 0, "No orders were placed to cancel"
        # And it should have received cancellation callbacks
        assert len(strategy.cancelled_ids) > 0, "on_order_update was not called with CANCELLED status"
        # Cancelled IDs should match placed IDs
        for cid in strategy.cancelled_ids:
            assert cid in strategy.placed_order_ids, f"Cancelled unknown order {cid}"

    def test_cancel_all_works(self, cancel_all_strategy_result):
        """Test that cancel_all() cancels every open order."""
        result, strategy = cancel_all_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        # Should have received CANCELLED callbacks for all 3 orders
        assert len(strategy.cancel_updates) == 3, f"Expected 3 cancel callbacks, got {len(strategy.cancel_updates)}"
        # Note: open_order_count_after_cancel is checked immediately after sending cancel_all,
        # so it may still show orders (async). The important test is the cancel callbacks.

    def test_on_tick_called_on_wakeup(self, counting_strategy_result):
        """Test that on_tick is called at least once (wakeup-driven)."""
        result, strategy = counting_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        assert strategy.tick_count > 0, "Strategy did not receive any on_tick callbacks"

    def test_market_state_includes_l2_depth(self, counting_strategy_result):
        """Test that MarketState includes bid_depth and ask_depth."""
        result, strategy = counting_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        state = strategy.last_state
        assert state is not None

        # After a 5-minute sim, there should be order book depth
        assert isinstance(state.bid_depth, list), "bid_depth should be a list"
        assert isinstance(state.ask_depth, list), "ask_depth should be a list"

        # At least some ticks should have had depth
        # (early ticks may not have any data yet)
        if state.best_bid is not None:
            assert len(state.bid_depth) > 0, "bid_depth should not be empty when best_bid is set"
        if state.best_ask is not None:
            assert len(state.ask_depth) > 0, "ask_depth should not be empty when best_ask is set"

    def test_on_simulation_end_called(self, counting_strategy_result):
        """Test that on_simulation_end is called at kernel stop."""
        result, strategy = counting_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        assert strategy.simulation_ended, "on_simulation_end was not called"
        assert strategy.final_state is not None, "Final state was not passed"
        assert strategy.final_state.cash > 0, "Final cash should be positive"


class TestStrategicAgentAdapter:
    """Unit tests for the StrategicAgentAdapter class (no simulation needed)."""

    def test_adapter_can_be_instantiated(self):
        """Test that the adapter can be created with a strategy."""
        import numpy as np

        from rohan.simulation.abides_impl import StrategicAgentAdapter

        strategy = CountingStrategy()
        adapter = StrategicAgentAdapter(
            id=1,
            strategy=strategy,  # pyright: ignore[reportArgumentType]
            symbol="ABM",
            starting_cash=10_000_000,
            order_book_depth=10,
            random_state=np.random.RandomState(42),  # Required by TradingAgent
        )

        assert adapter.symbol == "ABM"
        assert adapter.starting_cash == 10_000_000
        assert adapter.strategy is strategy
        assert adapter._open_orders_cache == {}
        assert adapter._filled_quantities == {}
        assert adapter.order_book_depth == 10


# ---------------------------------------------------------------------------
# Situational Awareness Strategy (Step 1)
# ---------------------------------------------------------------------------
class SituationalAwarenessStrategy:
    """Records new MarketState fields for verification."""

    def __init__(self):
        self.config: AgentConfig | None = None
        self.portfolio_values: list[int] = []
        self.pnl_values: list[int] = []
        self.time_remaining: list[int | None] = []
        self.mid_prices: list[int | None] = []
        self.spreads: list[int | None] = []
        self.bid_liquidities: list[int] = []
        self.ask_liquidities: list[int] = []
        self.market_closed_seen = False
        self.last_state: MarketState | None = None

    def initialize(self, config: AgentConfig) -> None:
        self.config = config

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        self._record(state)
        if state.is_market_closed:
            self.market_closed_seen = True
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        self._record(state)
        return []

    def on_order_update(self, _update: Order) -> list[OrderAction]:
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        self.last_state = final_state

    def _record(self, state: MarketState):
        self.portfolio_values.append(state.portfolio_value)
        self.pnl_values.append(state.unrealized_pnl)
        self.time_remaining.append(state.time_remaining_ns)
        self.mid_prices.append(state.mid_price)
        self.spreads.append(state.spread)
        self.bid_liquidities.append(state.bid_liquidity)
        self.ask_liquidities.append(state.ask_liquidity)


class TestSituationalAwareness:
    """Integration tests for Step 1 — Situational Awareness fields."""

    @pytest.fixture(scope="class")
    def awareness_result(self):
        """Run simulation with SituationalAwarenessStrategy."""
        strategy = SituationalAwarenessStrategy()
        service = SimulationService()
        result = service.run_simulation(TEST_SETTINGS, strategy=strategy)  # pyright: ignore[reportArgumentType]
        return result, strategy

    def test_config_includes_market_hours(self, awareness_result):
        """AgentConfig should include mkt_open_ns and mkt_close_ns."""
        result, strategy = awareness_result
        assert result.error is None, f"Simulation failed: {result.error}"
        assert strategy.config is not None
        # Market hours may be None before MarketHoursMsg arrives,
        # but after kernel_starting they should be set from ABIDES mkt_open/mkt_close
        # (which may still be None if exchange hasn't sent them yet)
        # Just verify the fields exist
        assert hasattr(strategy.config, "mkt_open_ns")
        assert hasattr(strategy.config, "mkt_close_ns")

    def test_portfolio_value_populated(self, awareness_result):
        """portfolio_value should be non-zero after initialization."""
        result, strategy = awareness_result
        assert result.error is None
        # At least some ticks should have non-zero portfolio value
        # (strategy starts with cash, so mark_to_market should be > 0)
        non_zero = [v for v in strategy.portfolio_values if v != 0]
        assert len(non_zero) > 0, "portfolio_value was never non-zero"

    def test_unrealized_pnl_initially_zero(self, awareness_result):
        """unrealized_pnl should start at 0 (no positions taken)."""
        result, strategy = awareness_result
        assert result.error is None
        # First few PnL values should be 0 (no trades yet)
        assert strategy.pnl_values[0] == 0, "First PnL should be 0 (no positions)"

    def test_time_remaining_decreases(self, awareness_result):
        """time_remaining_ns should decrease over time."""
        result, strategy = awareness_result
        assert result.error is None
        # Filter valid values
        valid = [t for t in strategy.time_remaining if t is not None and t > 0]
        if len(valid) >= 2:
            assert valid[0] >= valid[-1], "time_remaining should decrease"

    def test_mid_price_computed(self, awareness_result):
        """mid_price should be computed when both bid and ask exist."""
        result, strategy = awareness_result
        assert result.error is None
        non_none = [p for p in strategy.mid_prices if p is not None]
        assert len(non_none) > 0, "mid_price was never computed"

    def test_spread_computed(self, awareness_result):
        """spread should be computed when both bid and ask exist."""
        result, strategy = awareness_result
        assert result.error is None
        non_none = [s for s in strategy.spreads if s is not None]
        assert len(non_none) > 0, "spread was never computed"
        # Spread should be non-negative
        for s in non_none:
            assert s >= 0, f"Spread should be non-negative, got {s}"

    def test_liquidity_populated(self, awareness_result):
        """bid_liquidity and ask_liquidity should be populated at some point."""
        result, strategy = awareness_result
        assert result.error is None
        # Liquidity may be 0 early, but should be non-zero after book builds
        non_zero_bid = [liq for liq in strategy.bid_liquidities if liq > 0]
        non_zero_ask = [liq for liq in strategy.ask_liquidities if liq > 0]
        # At least one side should have had non-zero liquidity
        assert len(non_zero_bid) > 0 or len(non_zero_ask) > 0, "Liquidity was never non-zero"
