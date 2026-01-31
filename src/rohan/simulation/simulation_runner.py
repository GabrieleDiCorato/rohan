"""Interface for market simulation implementations."""

from abc import ABC, abstractmethod

from rohan.config import SimulationSettings
from rohan.simulation.models import SimulationOutput


class SimulationRunner(ABC):
    """Abstract base class for market simulation implementations."""

    def __init__(self, settings: SimulationSettings):
        self.settings: SimulationSettings = settings

    @abstractmethod
    def run(self) -> SimulationOutput:
        """Runs the market simulation.

        Returns:
            SimulationOutput: The output of the simulation.
        """
        pass

    def validate(self) -> list[str]:
        """Validate configuration before running.

        Returns:
            List of validation error messages. Empty list if valid.

        Example:
            >>> runner = SimulationRunnerAbides(settings)
            >>> errors = runner.validate()
            >>> if errors:
            >>>     print(f"Validation failed: {errors}")
        """
        errors: list[str] = []

        # Basic validation that all runners should perform
        if self.settings is None:
            errors.append("Settings cannot be None")

        return errors

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources after simulation.

        This method is called after simulation completes (success or failure).
        Override this method to clean up any resources allocated during simulation.
        """
        pass
