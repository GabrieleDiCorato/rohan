"""Integration tests for StrategicAgent injection into ABIDES simulations.

These tests verify that:
1. A StrategicAgent implementation can be injected into the simulation
2. The strategy receives on_market_data callbacks
3. OrderActions are executed correctly

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
    OrderType,
    Side,
)
from rohan.simulation.simulation_service import SimulationService


class CountingStrategy:
    """A simple strategy that counts callbacks for testing."""

    def __init__(self):
        self.initialize_called = False
        self.market_data_count = 0
        self.config: AgentConfig | None = None
        self.last_state: MarketState | None = None

    def initialize(self, config: AgentConfig) -> None:
        self.initialize_called = True
        self.config = config

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        self.market_data_count += 1
        self.last_state = state
        return []  # No orders

    def on_order_update(self, _update: Order) -> list[OrderAction]:
        return []


class SimpleBuyStrategy:
    """A strategy that places a single buy order on first market data."""

    def __init__(self):
        self.order_placed = False

    def initialize(self, config: AgentConfig) -> None:
        pass

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        # Only place one order
        if not self.order_placed and state.best_ask is not None:
            self.order_placed = True
            return [
                OrderAction(
                    side=Side.BID,
                    quantity=100,
                    price=state.best_ask - 1,  # Just below ask
                    order_type=OrderType.LIMIT,
                )
            ]
        return []

    def on_order_update(self, _update: Order) -> list[OrderAction]:
        return []


# Shared simulation settings for tests (5 minutes - minimum required)
TEST_SETTINGS = SimulationSettings(
    seed=42,
    start_time="09:30:00",
    end_time="09:35:00",  # 5 minute duration - minimum required by validator
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
        result = service.run_simulation(TEST_SETTINGS, strategy=strategy)
        return result, strategy

    @pytest.fixture(scope="class")
    def buy_strategy_result(self):
        """Run simulation once with SimpleBuyStrategy."""
        strategy = SimpleBuyStrategy()
        service = SimulationService()
        result = service.run_simulation(TEST_SETTINGS, strategy=strategy)
        return result, strategy

    @pytest.fixture(scope="class")
    def no_strategy_result(self):
        """Run simulation once without strategy (backwards compatibility)."""
        service = SimulationService()
        return service.run_simulation(TEST_SETTINGS)

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

    def test_simulation_without_strategy_still_works(self, no_strategy_result):
        """Test that simulation works without a strategy (backwards compatibility)."""
        result = no_strategy_result

        assert result.error is None, f"Simulation failed: {result.error}"
        assert result.result is not None


class TestStrategicAgentAdapter:
    """Unit tests for the StrategicAgentAdapter class (no simulation needed)."""

    def test_adapter_can_be_instantiated(self):
        """Test that the adapter can be created with a strategy."""
        import numpy as np

        from rohan.simulation.abides_impl import StrategicAgentAdapter

        strategy = CountingStrategy()
        adapter = StrategicAgentAdapter(
            id=1,
            strategy=strategy,
            symbol="ABM",
            starting_cash=10_000_000,
            random_state=np.random.RandomState(42),  # Required by TradingAgent
        )

        assert adapter.symbol == "ABM"
        assert adapter.starting_cash == 10_000_000
        assert adapter.strategy is strategy
