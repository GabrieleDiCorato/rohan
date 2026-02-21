class RohanError(Exception):
    """Base exception for all ROHAN errors."""

    pass


class StrategyValidationError(RohanError):
    """Raised when strategy code fails AST or sandbox validation."""

    pass


class SimulationTimeoutError(RohanError):
    """Raised when a simulation exceeds the configured timeout."""

    pass


class BaselineComparisonError(RohanError):
    """Raised when baseline simulation fails, preventing comparison."""

    pass


class StrategyExecutionError(RohanError):
    """Raised when strategy code fails during simulation execution."""

    pass
