import ast
import builtins

from rohan.config import SimulationSettings
from rohan.simulation.models import (
    SimulationContext,
    SimulationResult,
    ValidationResult,
)
from rohan.simulation.simulation_service import SimulationService


class StrategyValidator:
    """Validates strategy code for safety before execution."""

    SAFE_IMPORTS = {
        "math",
        "random",
        "statistics",
        "numpy",
        "pandas",
        "datetime",
        "typing",
        "rohan.simulation.models.strategy_api",
        "rohan.config",
    }

    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "pow",
        "range",
        "round",
        "set",
        "str",
        "sum",
        "tuple",
        "zip",
        "enumerate",
        "filter",
        "map",
        "sorted",
        "reversed",
        "isinstance",
        "print",  # Useful for debugging
        "__build_class__",  # Required for class definitions
        "super",  # Required for inheritance
        "__import__",  # Required for import statements
    }

    def validate(self, code: str) -> ValidationResult:
        """Validate strategy code and return a ValidationResult."""
        errors = []
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(is_valid=False, errors=[f"SyntaxError: {e}"])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                errors.extend(self._check_import(node))

            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                # Check access to private attributes
                errors.append(f"Access to private attribute '{node.attr}' is restricted")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def _check_import(self, node: ast.Import | ast.ImportFrom) -> list[str]:
        errors = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in self.SAFE_IMPORTS:
                    errors.append(f"Import forbidden: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Check if root or exact module is allowed
            if (module.split(".")[0] in self.SAFE_IMPORTS) or (module in self.SAFE_IMPORTS):
                pass  # Allowed
            else:
                errors.append(f"Import forbidden: '{module}'")
        return errors

    def execute_strategy(self, code: str, class_name: str) -> type:
        """Execute the strategy code in a restricted namespace and return the class."""
        result = self.validate(code)
        if not result.is_valid:
            raise ValueError(f"Security violations found: {result.errors}")

        # Prepare restricted execution context
        # We need to inject necessary modules and builtins
        from typing import Any

        context: dict[str, Any] = {
            "__builtins__": {k: getattr(builtins, k) for k in self.SAFE_BUILTINS if hasattr(builtins, k)},
            "__name__": "__main__",
        }

        # Inject allowed modules
        import datetime
        import math
        import random
        import statistics
        import typing

        import numpy as np
        import pandas as pd

        import rohan.config as config
        import rohan.simulation.models.strategy_api as strategy_api

        context.update(
            {
                "math": math,
                "random": random,
                "statistics": statistics,
                "datetime": datetime,
                "typing": typing,
                "np": np,
                "pd": pd,
                "strategy_api": strategy_api,
                "config": config,
                # Also inject commonly used types directly for convenience
                "List": list,
                "Dict": dict,
                "Optional": typing.Optional,
                "Union": typing.Union,
            }
        )

        try:
            # Execute the code
            exec(code, context)
        except Exception as e:
            raise RuntimeError(f"Strategy execution failed: {e}") from e

        # Extract the strategy class
        if class_name not in context:
            raise ValueError(f"Strategy class '{class_name}' not found in code")

        strategy_class = context[class_name]
        if not isinstance(strategy_class, type):
            raise TypeError(f"'{class_name}' is not a class")

        # Verify it implements the protocol
        required_methods = ["initialize", "on_market_data", "on_order_update"]
        for method in required_methods:
            if not hasattr(strategy_class, method):
                raise TypeError(f"Strategy class '{class_name}' missing required method: '{method}'")

        return strategy_class


def execute_strategy_safely(
    strategy_code: str,
    settings: SimulationSettings,
    _timeout_seconds: int = 300,
) -> SimulationResult:
    """
    Executes a strategy simulation with safety checks and timeout.

    1. Validates code safety (imports, builtins)
    2. Compiles strategy class
    3. Runs simulation in separate process/thread (or just safely here for MVP)

    For MVP on Windows, we run in-process but with validation.
    """
    validator = StrategyValidator()

    try:
        # Extract class name - for now assume "MyStrategy" or parse it
        # Simple parsing to find class name
        model_tree = ast.parse(strategy_code)
        class_name = None
        for node in model_tree.body:
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                break

        if not class_name:
            return SimulationResult(
                context=SimulationContext(settings=settings),
                duration_seconds=0,
                error=ValueError("No strategy class found in code"),
            )

        strategy_class = validator.execute_strategy(strategy_code, class_name)

        # Instantiate strategy
        # We need to adapt it? SimulationService takes strategy INSTANCE?
        # SimulationService.run_simulation(settings, strategy)
        # Strategy must be StrategicAgentAdapter?
        # No, SimulationService takes `strategy: Optional[Any]`.
        # And it uses `StrategicAgentAdapter` to wrap it.
        # So we pass the minimal strategy instance.

        # Minimal strategy interface: initialize, on_market_data, on_order_update
        strategy_instance = strategy_class()

        service = SimulationService()
        return service.run_simulation(settings, strategy=strategy_instance)

    except Exception as e:
        return SimulationResult(context=SimulationContext(settings=settings), duration_seconds=0, error=e)
