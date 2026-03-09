import ast
import builtins
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from rohan.config import SimulationSettings
from rohan.exceptions import SimulationTimeoutError, StrategyExecutionError, StrategyValidationError
from rohan.simulation.models import (
    SimulationContext,
    SimulationResult,
    ValidationResult,
)
from rohan.simulation.simulation_service import SimulationService


def _make_safe_import(allowed: set[str]):
    """Return a ``__import__`` replacement that enforces the module whitelist.

    Only modules whose root package or exact dotted name appears in *allowed*
    are permitted.  All other imports raise ``ImportError``.
    """
    _real_import = builtins.__import__

    def _safe_import(name: str, *args: Any, **kwargs: Any):
        root = name.split(".")[0]
        if root not in allowed and name not in allowed:
            raise ImportError(f"Import of {name!r} is not allowed. Permitted modules: {sorted(allowed)}")
        return _real_import(name, *args, **kwargs)

    return _safe_import


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
        # NOTE: "rohan.config" deliberately excluded — it re-exports
        # LLMSettings / SecretSettings which contain API keys.
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
        # NOTE: __import__ deliberately excluded from this set.
        # A safe wrapper (_make_safe_import) is injected in execute_strategy()
        # so that `import foo` statements still work for whitelisted modules
        # but `__import__('os')` function calls are blocked.
    }

    # Dangerous dunder attributes — accessing these can escape the sandbox.
    # Legitimate dunders like __init__, __class__, __name__ are allowed.
    DANGEROUS_DUNDERS = {
        "__builtins__",
        "__globals__",
        "__code__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__import__",
        "__loader__",
        "__spec__",
        "__qualname__",
    }

    # Built-in function names that must never appear as direct calls.
    FORBIDDEN_CALLS = {
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "breakpoint",
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

            elif isinstance(node, ast.Attribute) and node.attr in self.DANGEROUS_DUNDERS:
                errors.append(f"Access to dangerous attribute '{node.attr}' is restricted")

            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in self.FORBIDDEN_CALLS:
                # Block calls to dangerous built-in functions
                errors.append(f"Call to '{node.func.id}()' is forbidden")

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
            raise StrategyValidationError(f"Security violations found: {result.errors}")

        # Prepare restricted execution context
        # We need to inject necessary modules and builtins

        safe_builtins_dict = {k: getattr(builtins, k) for k in self.SAFE_BUILTINS if hasattr(builtins, k)}
        # Inject a safe __import__ that enforces the whitelist
        safe_builtins_dict["__import__"] = _make_safe_import(self.SAFE_IMPORTS)

        context: dict[str, Any] = {
            "__builtins__": safe_builtins_dict,
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
            raise StrategyExecutionError(f"Strategy execution failed: {e}") from e

        # Extract the strategy class
        if class_name not in context:
            raise StrategyValidationError(f"Strategy class '{class_name}' not found in code")

        strategy_class = context[class_name]
        if not isinstance(strategy_class, type):
            raise StrategyValidationError(f"'{class_name}' is not a class")

        # Verify it implements the protocol
        required_methods = [
            "initialize",
            "on_tick",
            "on_market_data",
            "on_order_update",
            "on_simulation_end",
        ]
        for method in required_methods:
            if not hasattr(strategy_class, method):
                raise StrategyValidationError(f"Strategy class '{class_name}' missing required method: '{method}'")

        return strategy_class

    def smoke_test(self, strategy_class: type) -> None:
        """Instantiate the strategy and call its core methods with mock data.

        This catches runtime errors (NameError, TypeError, AttributeError)
        that static AST validation and ``hasattr`` checks miss.  It runs
        in milliseconds, far cheaper than a full ABIDES simulation.

        Raises
        ------
        StrategyExecutionError
            If ``__init__``, ``initialize``, ``on_tick``, or
            ``on_market_data`` raise any exception.
        """
        from rohan.simulation.models.strategy_api import (
            AgentConfig,
            MarketState,
        )

        # 1. Instantiate
        try:
            instance = strategy_class()
        except Exception as e:
            raise StrategyExecutionError(f"Strategy __init__ failed: {e}") from e

        # 2. Call initialize() with a minimal config
        mock_config = AgentConfig(
            starting_cash=10_000_000,  # $100k in cents
            symbol="TEST",
            latency_ns=1_000_000,  # 1ms
        )
        try:
            instance.initialize(mock_config)
        except Exception as e:
            raise StrategyExecutionError(f"Strategy initialize() failed: {e}") from e

        # 3. Dry-run on_market_data and on_tick with a bare-minimum state
        mock_state = MarketState(
            timestamp_ns=1_000_000_000,
            best_bid=10000,  # $100.00
            best_ask=10010,  # $100.10
            inventory=0,
            cash=10_000_000,
            open_orders=[],
        )
        try:
            instance.on_market_data(mock_state)
        except Exception as e:
            raise StrategyExecutionError(f"Strategy on_market_data() failed on smoke test: {e}") from e

        try:
            instance.on_tick(mock_state)
        except Exception as e:
            raise StrategyExecutionError(f"Strategy on_tick() failed on smoke test: {e}") from e


def _run_simulation_in_thread(
    strategy_code: str,
    settings: SimulationSettings,
) -> SimulationResult:
    """Run strategy simulation in the current thread.  Used by execute_strategy_safely."""
    validator = StrategyValidator()

    tree = ast.parse(strategy_code)
    class_name: str | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            break

    if not class_name:
        return SimulationResult(
            context=SimulationContext(settings=settings),
            duration_seconds=0,
            error=StrategyValidationError("No strategy class found in code"),
        )

    strategy_class = validator.execute_strategy(strategy_code, class_name)
    strategy_instance = strategy_class()
    service = SimulationService()
    return service.run_simulation(settings, strategy=strategy_instance)


def execute_strategy_safely(
    strategy_code: str,
    settings: SimulationSettings,
    timeout_seconds: int = 300,
) -> SimulationResult:
    """Execute a strategy simulation with a wall-clock timeout.

    # SECURITY
    # ---------
    # Threat model: LLM-generated code may contain:
    #   - Infinite loops in __init__, on_market_data, or on_order_update
    #   - Excessive memory allocation (e.g. allocating huge arrays)
    #   - Attempts to access the filesystem or network
    #
    # Current mitigations:
    #   - AST-level import whitelist (StrategyValidator.validate)
    #   - Restricted builtins in exec() namespace (StrategyValidator.execute_strategy)
    #   - Execution in a daemon thread; parent aborts after ``timeout_seconds``
    #   - SimulationTimeoutError is raised on timeout, propagating to LangGraph nodes
    #
    # Timeout implementation note:
    #   We use a ThreadPoolExecutor rather than multiprocessing because the ABIDES
    #   simulation output (AbidesOutput / end_state) contains C-extension thread
    #   locks that are not picklable and cannot cross process boundaries.
    #   Thread-based timeout cannot forcibly kill a runaway thread (Python has no
    #   such primitive), but the AST-level import whitelist prevents user code from
    #   importing ``time``, ``asyncio``, or C extensions that could truly spin
    #   forever.  For true OS-level isolation, see Phase 3 (Docker sandbox).
    #
    # Known gaps (acceptable for thesis MVP):
    #   - No CPU cgroup limit
    #   - No memory limit
    #   - No filesystem or network isolation
    #
    # Planned mitigation: Docker sandbox (Phase 3)

    Steps:
        1. Submit _run_simulation_in_thread to a single-thread executor.
        2. Wait up to ``timeout_seconds`` for the future to complete.
        3. If the timeout is exceeded, raise SimulationTimeoutError (the
           background thread continues but is a daemon and will be reaped
           when the process exits).
        4. Otherwise, return the SimulationResult.
    """
    start_time = time.monotonic()
    # NOTE: We intentionally do NOT use `with ThreadPoolExecutor` as a
    # context manager.  If the future times out, __exit__ calls
    # shutdown(wait=True) which blocks forever when the worker thread is
    # stuck in an infinite loop — defeating the timeout entirely.
    executor = ThreadPoolExecutor(max_workers=1)
    future: Future[SimulationResult] = executor.submit(_run_simulation_in_thread, strategy_code, settings)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        # Abandon the stuck thread — don't wait for it
        executor.shutdown(wait=False, cancel_futures=True)
        elapsed = time.monotonic() - start_time
        raise SimulationTimeoutError(f"Strategy execution timed out after {elapsed:.1f}s (limit: {timeout_seconds}s). The strategy may contain an infinite loop.") from None
    except Exception as exc:
        executor.shutdown(wait=False)
        raise StrategyExecutionError(f"Strategy execution failed: {exc}") from exc
