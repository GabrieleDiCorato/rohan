import pytest

from rohan.exceptions import StrategyExecutionError, StrategyValidationError
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

    def on_tick(self, state: MarketState) -> list[OrderAction]:
        return []

    def on_market_data(self, state: MarketState) -> list[OrderAction]:
        # Cancel all stale orders before placing new ones
        actions = [OrderAction.cancel(o.order_id) for o in state.open_orders]
        return actions

    def on_order_update(self, update: Order) -> list[OrderAction]:
        if update.status in (OrderStatus.FILLED, OrderStatus.PARTIAL):
            pass  # track inventory
        return []

    def on_simulation_end(self, final_state: MarketState) -> None:
        pass
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
        return [].__subclasses__()
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert any("Access to dangerous attribute '__subclasses__'" in v for v in result.errors)

    def test_dangerous_dunders_blocked(self):
        """DANGEROUS_DUNDERS like __globals__, __builtins__, __code__ are blocked."""
        validator = StrategyValidator()
        for attr in ("__globals__", "__builtins__", "__code__", "__subclasses__", "__bases__", "__import__"):
            code = f"""
class BadStrategy:
    def hack(self):
        return self.{attr}
"""
            result = validator.validate(code)
            assert not result.is_valid, f"Expected {attr} to be blocked"

    def test_safe_dunders_allowed(self):
        """Legitimate dunders like __init__, __class__, __name__ are allowed."""
        validator = StrategyValidator()
        code = """
class MyStrategy:
    def initialize(self, config):
        super().__init__()
        self._name = type(self).__name__

    def on_tick(self, state):
        return []

    def on_market_data(self, state):
        return []

    def on_order_update(self, update):
        return []

    def on_simulation_end(self, final_state):
        pass
"""
        result = validator.validate(code)
        assert result.is_valid, f"Unexpected errors: {result.errors}"

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
        with pytest.raises(StrategyValidationError, match="Strategy class 'MyStrategy' not found"):
            validator.execute_strategy(code, "MyStrategy")

    def test_missing_methods(self):
        validator = StrategyValidator()
        code = """
class IncompleteStrategy:
    def initialize(self, config): pass
"""
        with pytest.raises(StrategyValidationError, match="missing required method"):
            validator.execute_strategy(code, "IncompleteStrategy")

    def test_runtime_error_in_strategy(self):
        validator = StrategyValidator()
        code = """
class CrashingStrategy:
    x = 1 / 0
"""
        with pytest.raises(StrategyExecutionError, match="Strategy execution failed"):
            validator.execute_strategy(code, "CrashingStrategy")

    def test_safe_builtins_allowed(self):
        validator = StrategyValidator()
        code = """
class MathStrategy:
    def initialize(self, config):
        pass
    def on_tick(self, state):
        return []
    def on_market_data(self, state):
        return []
    def on_order_update(self, update):
        return []
    def on_simulation_end(self, final_state):
        pass
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
    def on_tick(self, state): return []
    def on_market_data(self, state): return []
    def on_order_update(self, update): return []
    def on_simulation_end(self, final_state): pass
"""
        # It passes validation (no forbidden imports/attributes),
        # but fails at runtime because 'open' is not in SAFE_BUILTINS
        with pytest.raises(StrategyExecutionError, match="Strategy execution failed"):
            validator.execute_strategy(code, "HackerStrategy")

    # ── SEC-1: __import__ bypass tests ──────────────────────────────────

    def test_import_bypass_via_dunder_import_blocked(self):
        """__import__('os') function call should be blocked at runtime."""
        validator = StrategyValidator()
        code = """
class HackerStrategy:
    def initialize(self, config): pass
    def on_tick(self, state): return []
    def on_market_data(self, state): return []
    def on_order_update(self, update): return []
    def on_simulation_end(self, final_state): pass
    def hack(self):
        os = __import__('os')
        return os.listdir('.')
"""
        # AST validation passes (no ast.Import node for __import__ calls)
        result = validator.validate(code)
        assert result.is_valid
        # But exec should work, and the __import__('os') call should raise
        strategy_class = validator.execute_strategy(code, "HackerStrategy")
        instance = strategy_class()
        with pytest.raises(ImportError, match="Import of 'os' is not allowed"):
            instance.hack()

    def test_safe_import_allows_whitelisted_modules(self):
        """__import__('numpy') should work since it's whitelisted."""
        validator = StrategyValidator()
        code = """
class SafeStrategy:
    def initialize(self, config): pass
    def on_tick(self, state): return []
    def on_market_data(self, state):
        np = __import__('numpy')
        return []
    def on_order_update(self, update): return []
    def on_simulation_end(self, final_state): pass
"""
        strategy_class = validator.execute_strategy(code, "SafeStrategy")
        instance = strategy_class()
        # Should not raise
        from rohan.simulation.models.strategy_api import MarketState

        mock_state = MarketState(
            timestamp_ns=1_000_000_000,
            best_bid=10000,
            best_ask=10010,
            inventory=0,
            cash=10_000_000,
            open_orders=[],
        )
        instance.on_market_data(mock_state)

    # ── SEC-2: config import blocked ────────────────────────────────────

    def test_config_import_blocked(self):
        """from rohan.config import LLMSettings should be blocked."""
        validator = StrategyValidator()
        code = """
from rohan.config import LLMSettings

class HackerStrategy:
    pass
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert any("Import forbidden" in e for e in result.errors)

    # ── SEC-4: forbidden calls ──────────────────────────────────────────

    def test_eval_call_blocked(self):
        """eval() calls should be blocked by the AST checker."""
        validator = StrategyValidator()
        code = """
class HackerStrategy:
    def hack(self):
        return eval("1+1")
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert any("Call to 'eval()' is forbidden" in e for e in result.errors)

    def test_exec_call_blocked(self):
        """exec() calls should be blocked by the AST checker."""
        validator = StrategyValidator()
        code = """
class HackerStrategy:
    def hack(self):
        exec("import os")
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert any("Call to 'exec()' is forbidden" in e for e in result.errors)

    def test_getattr_call_blocked(self):
        """getattr() calls should be blocked by the AST checker."""
        validator = StrategyValidator()
        code = """
class HackerStrategy:
    def hack(self):
        return getattr(self, '__globals__')
"""
        result = validator.validate(code)
        assert not result.is_valid
        assert any("Call to 'getattr()' is forbidden" in e for e in result.errors)
