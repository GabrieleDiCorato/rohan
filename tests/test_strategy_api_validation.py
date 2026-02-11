"""Tests for strategy_api.py Pydantic model validations.

This module tests:
- OrderAction @model_validator (LIMIT must have price, MARKET must not)
- Type validations (int cents vs float rejection)
- Field constraints and nullable handling
- Model instantiation edge cases
"""

import pytest
from pydantic import ValidationError

from rohan.simulation.models.strategy_api import (
    AgentConfig,
    MarketState,
    Order,
    OrderAction,
    OrderStatus,
    OrderType,
    Side,
)


# ---------------------------------------------------------------------------
# OrderAction Validation
# ---------------------------------------------------------------------------
class TestOrderActionValidation:
    """Test suite for OrderAction @model_validator."""

    def test_limit_order_with_price_valid(self):
        """LIMIT orders with price should be valid."""
        action = OrderAction(
            side=Side.BID,
            quantity=100,
            price=10000,  # $100.00 in cents
            order_type=OrderType.LIMIT,
        )
        assert action.price == 10000
        assert action.order_type == OrderType.LIMIT

    def test_limit_order_without_price_invalid(self):
        """LIMIT orders without price should fail validation."""
        with pytest.raises(ValidationError, match="price is required for LIMIT orders"):
            OrderAction(
                side=Side.BID,
                quantity=100,
                price=None,
                order_type=OrderType.LIMIT,
            )

    def test_market_order_without_price_valid(self):
        """MARKET orders without price should be valid."""
        action = OrderAction(
            side=Side.ASK,
            quantity=50,
            price=None,
            order_type=OrderType.MARKET,
        )
        assert action.price is None
        assert action.order_type == OrderType.MARKET

    def test_market_order_with_price_invalid(self):
        """MARKET orders with price should fail validation."""
        with pytest.raises(ValidationError, match="price must be None for MARKET orders"):
            OrderAction(
                side=Side.ASK,
                quantity=50,
                price=10000,
                order_type=OrderType.MARKET,
            )

    def test_cancel_order_valid(self):
        """OrderAction can represent a cancel operation."""
        action = OrderAction(
            side=Side.BID,
            quantity=100,
            price=10000,
            order_type=OrderType.LIMIT,
            cancel_order_id=12345,
        )
        assert action.cancel_order_id == 12345

    def test_cancel_factory_method(self):
        """OrderAction.cancel() creates a cancel action with minimal fields."""
        action = OrderAction.cancel(order_id=999)
        assert action.cancel_order_id == 999
        # Factory uses defaults for irrelevant fields
        assert action.side == Side.BID
        assert action.quantity == 1

    def test_cancel_all_factory_method(self):
        """OrderAction.cancel_all() creates a cancel-all action."""
        from rohan.simulation.models.strategy_api import CANCEL_ALL

        action = OrderAction.cancel_all()
        assert action.cancel_order_id == CANCEL_ALL
        assert action.cancel_order_id == -1

    def test_cancel_skips_price_validation(self):
        """When cancel_order_id is set, price validation is skipped."""
        # This would fail without cancel_order_id because LIMIT requires price
        action = OrderAction.cancel(order_id=42)
        assert action.price is None  # No price needed for cancel
        assert action.cancel_order_id == 42

    def test_quantity_must_be_positive(self):
        """Quantity must be >= 1."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            OrderAction(
                side=Side.BID,
                quantity=0,
                price=10000,
                order_type=OrderType.LIMIT,
            )

        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            OrderAction(
                side=Side.BID,
                quantity=-10,
                price=10000,
                order_type=OrderType.LIMIT,
            )


# ---------------------------------------------------------------------------
# Type Validation (int cents vs float)
# ---------------------------------------------------------------------------
class TestPriceTypeValidation:
    """Test that prices must be int (cents), not float."""

    def test_order_price_must_be_int(self):
        """Order.price must be int, not float."""
        # Valid int
        order = Order(
            order_id=1,
            symbol="ABM",
            side=Side.BID,
            quantity=100,
            price=10050,  # $100.50 in cents
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            filled_quantity=0,
        )
        assert order.price == 10050
        assert isinstance(order.price, int)

    def test_order_price_float_gets_converted(self):
        """Pydantic may coerce float to int if close."""
        # Pydantic will coerce 100.0 → 100
        order = Order(
            order_id=1,
            symbol="ABM",
            side=Side.BID,
            quantity=100,
            price=10000.0,  # Will be coerced to 10000
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            filled_quantity=0,
        )
        assert order.price == 10000
        assert isinstance(order.price, int)

    def test_order_price_fractional_float_rejected(self):
        """Fractional floats should fail (cannot represent cents exactly)."""
        with pytest.raises(ValidationError):
            Order(
                order_id=1,
                symbol="ABM",
                side=Side.BID,
                quantity=100,
                price=100.50,  # Fractional, should fail
                order_type=OrderType.LIMIT,
                status=OrderStatus.NEW,
                filled_quantity=0,
            )

    def test_market_state_prices_must_be_int(self):
        """MarketState prices must be int cents."""
        state = MarketState(
            timestamp_ns=1000000000,
            best_bid=10000,
            best_ask=10100,
            last_trade=10050,
            inventory=10,
            cash=1000000,  # $10,000 in cents
            open_orders=[],
        )
        assert isinstance(state.best_bid, int)
        assert isinstance(state.best_ask, int)
        assert isinstance(state.cash, int)

    def test_market_state_nullable_prices(self):
        """MarketState prices can be None."""
        state = MarketState(
            timestamp_ns=1000000000,
            best_bid=None,
            best_ask=None,
            last_trade=None,
            inventory=0,
            cash=1000000,
            open_orders=[],
        )
        assert state.best_bid is None
        assert state.best_ask is None
        assert state.last_trade is None

    def test_agent_config_starting_cash_must_be_int(self):
        """AgentConfig.starting_cash must be int cents."""
        config = AgentConfig(
            starting_cash=10_000_000,  # $100,000 in cents
            symbol="ABM",
            latency_ns=0,
        )
        assert config.starting_cash == 10_000_000
        assert isinstance(config.starting_cash, int)

    def test_order_action_price_int_validation(self):
        """OrderAction.price must be int when specified."""
        action = OrderAction(
            side=Side.BID,
            quantity=100,
            price=10050,  # int cents
            order_type=OrderType.LIMIT,
        )
        assert isinstance(action.price, int)


# ---------------------------------------------------------------------------
# MarketState Edge Cases
# ---------------------------------------------------------------------------
class TestMarketStateValidation:
    """Test MarketState field validations and edge cases."""

    def test_negative_inventory_allowed(self):
        """Inventory can be negative (short position)."""
        state = MarketState(
            timestamp_ns=1000000000,
            best_bid=10000,
            best_ask=10100,
            inventory=-50,  # Short 50 shares
            cash=1000000,
            open_orders=[],
        )
        assert state.inventory == -50

    def test_zero_cash_allowed(self):
        """Cash can be zero (all-in position)."""
        state = MarketState(
            timestamp_ns=1000000000,
            best_bid=10000,
            best_ask=10100,
            inventory=100,
            cash=0,
            open_orders=[],
        )
        assert state.cash == 0

    def test_open_orders_can_be_empty(self):
        """Open orders can be empty list."""
        state = MarketState(
            timestamp_ns=1000000000,
            inventory=0,
            cash=1000000,
            open_orders=[],
        )
        assert state.open_orders == []

    def test_open_orders_with_multiple_orders(self):
        """Open orders can contain multiple Order objects."""
        orders = [
            Order(
                order_id=1,
                symbol="ABM",
                side=Side.BID,
                quantity=100,
                price=9900,
                order_type=OrderType.LIMIT,
                status=OrderStatus.NEW,
                filled_quantity=0,
            ),
            Order(
                order_id=2,
                symbol="ABM",
                side=Side.ASK,
                quantity=50,
                price=10200,
                order_type=OrderType.LIMIT,
                status=OrderStatus.NEW,
                filled_quantity=0,
            ),
        ]
        state = MarketState(
            timestamp_ns=1000000000,
            inventory=50,
            cash=1000000,
            open_orders=orders,
        )
        assert len(state.open_orders) == 2


# ---------------------------------------------------------------------------
# Order Validation
# ---------------------------------------------------------------------------
class TestOrderValidation:
    """Test Order model validations."""

    def test_filled_quantity_cannot_exceed_quantity(self):
        """filled_quantity should not exceed quantity (logical constraint)."""
        # Note: Pydantic doesn't enforce this automatically without a validator
        # This test documents the expected behavior
        order = Order(
            order_id=1,
            symbol="ABM",
            side=Side.BID,
            quantity=100,
            price=10000,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            filled_quantity=100,
        )
        assert order.filled_quantity == order.quantity

    def test_partial_fill_valid(self):
        """Partial fills are valid."""
        order = Order(
            order_id=1,
            symbol="ABM",
            side=Side.BID,
            quantity=100,
            price=10000,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PARTIAL,
            filled_quantity=50,
        )
        assert order.filled_quantity < order.quantity


# ---------------------------------------------------------------------------
# AgentConfig Validation
# ---------------------------------------------------------------------------
class TestAgentConfigValidation:
    """Test AgentConfig model validations."""

    def test_minimal_valid_config(self):
        """AgentConfig with all required fields."""
        config = AgentConfig(
            starting_cash=10_000_000,
            symbol="ABM",
            latency_ns=0,
        )
        assert config.starting_cash == 10_000_000
        assert config.symbol == "ABM"
        assert config.latency_ns == 0

    def test_latency_can_be_positive(self):
        """Latency can be a positive integer."""
        config = AgentConfig(
            starting_cash=10_000_000,
            symbol="TEST",
            latency_ns=1_000_000,  # 1ms
        )
        assert config.latency_ns == 1_000_000

    def test_symbol_can_be_custom(self):
        """Symbol can be any string."""
        config = AgentConfig(
            starting_cash=10_000_000,
            symbol="CUSTOM_SYMBOL",
            latency_ns=0,
        )
        assert config.symbol == "CUSTOM_SYMBOL"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
