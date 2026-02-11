import pytest

from rohan.simulation.strategy_validator import StrategyValidator


class TestStrategyValidator:
    def test_valid_strategy_code(self):
        validator = StrategyValidator()
        code = """
from rohan.simulation.models.strategy_api import (
    StrategicAgent, AgentConfig, MarketState, Order, OrderAction, OrderStatus, Side
)
import math

class MyStrategy:
    def initialize(self, config: AgentConfig) -> None:
        self.config = config

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        # Cancel all stale orders before placing new ones
        actions = [OrderAction.cancel(o.order_id) for o in state.open_orders]
        return actions

    def on_order_update(self, update: Order) -> list[OrderAction]:
        if update.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            pass  # track inventory
        return []
"""
        result = validator.validate(code)
        assert result.is_valid
        assert len(result.errors) == 0

        strategy_class = validator.execute_strategy(code, "MyStrategy")
        strategy = strategy_class()
        assert hasattr(strategy, "initialize")
        assert hasattr(strategy, "on_market_data")

    def test_forbidden_import(self):
        validator = StrategyValidator()
        code = """
import os
import sys

class BadStrategy:
    pass
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert len(result.errors) == 2
        assert "Import forbidden: 'os'" in result.errors
        assert "Import forbidden: 'sys'" in result.errors

    def test_forbidden_attribute(self):
        validator = StrategyValidator()
        code = """
class BadStrategy:
    def hack(self):
        return [].__class__.__base__.__subclasses__()
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert any("Access to private attribute '__class__'" in v for v in result.errors)

    def test_syntax_error(self):
        validator = StrategyValidator()
        code = "class BrokenStrategy: def init("  # Syntax error
        result = validator.validate(code)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "SyntaxError" in result.errors[0]

    def test_missing_class(self):
        validator = StrategyValidator()
        code = "x = 1"
        with pytest.raises(ValueError, match="Strategy class 'MyStrategy' not found"):
            validator.execute_strategy(code, "MyStrategy")

    def test_missing_methods(self):
        validator = StrategyValidator()
        code = """
class IncompleteStrategy:
    def initialize(self, config): pass
"""
        with pytest.raises(TypeError, match="missing required method"):
            validator.execute_strategy(code, "IncompleteStrategy")

    def test_runtime_error_in_strategy(self):
        validator = StrategyValidator()
        code = """
class CrashingStrategy:
    x = 1 / 0
"""
        with pytest.raises(RuntimeError, match="Strategy execution failed"):
            validator.execute_strategy(code, "CrashingStrategy")

    def test_safe_builtins_allowed(self):
        validator = StrategyValidator()
        code = """
class MathStrategy:
    def initialize(self, config):
        pass
    def on_market_data(self, state):
        return []
    def on_order_update(self, update):
        return []
    def calc(self):
        return max(1, 2) + min(3, 4)
"""
        strategy_class = validator.execute_strategy(code, "MathStrategy")
        strategy = strategy_class()
        assert strategy.calc() == 5

    def test_unsafe_builtins_blocked(self):
        validator = StrategyValidator()
        code = """
class HackerStrategy:
    # This runs at definition time
    open('hack.txt', 'w')

    def initialize(self, config): pass
    def on_market_data(self, state): return []
    def on_order_update(self, update): return []
"""
        # It passes validation (no forbidden imports/attributes),
        # but fails at runtime because 'open' is not in SAFE_BUILTINS
        with pytest.raises(RuntimeError, match="name 'open' is not defined"):
            validator.execute_strategy(code, "HackerStrategy")
