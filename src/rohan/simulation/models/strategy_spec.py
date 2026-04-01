"""Serializable representation of an LLM-generated trading strategy.

A ``StrategySpec`` carries the source code and class name as plain strings,
deferring compilation (``exec()``) to the moment the agent is actually
created inside hasufel's ``compile()`` pipeline.  This makes the entire
simulation config JSON-serializable and picklable — enabling
``run_batch()`` parallelization, config hashing, and DB persistence.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class StrategySpec(BaseModel):
    """Frozen, hashable, picklable specification of a trading strategy.

    Fields
    ------
    source_code : str
        Complete Python source implementing a ``StrategicAgent`` class.
    class_name : str
        Name of the class inside *source_code* that implements the protocol.
    params : dict[str, Any]
        JSON-serializable constructor kwargs (currently empty, future-proofed).
    """

    model_config = ConfigDict(frozen=True)

    source_code: str
    class_name: str
    params: dict[str, Any] = {}

    def compile(self) -> type:
        """Execute the source code in a sandboxed namespace and return the strategy class.

        Reuses the same restricted sandbox as
        :meth:`StrategyValidator.execute_strategy` — safe builtins,
        import whitelist, and forbidden-call checks.

        Returns
        -------
        type
            The strategy class extracted from the executed source code.

        Raises
        ------
        StrategyValidationError
            If the code fails AST validation or the class is not found.
        StrategyExecutionError
            If ``exec()`` raises an exception.
        """
        from rohan.simulation.strategy_validator import StrategyValidator

        validator = StrategyValidator()
        return validator.execute_strategy(self.source_code, self.class_name)

    def __hash__(self) -> int:
        return hash((self.source_code, self.class_name, tuple(sorted(self.params.items()))))
