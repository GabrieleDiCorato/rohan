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
