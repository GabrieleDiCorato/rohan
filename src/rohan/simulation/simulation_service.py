"""Service module for simulation-related operations. Injects dependencies and manages simulation execution."""

from rohan.config import SimulationEngine, SimulationSettings
from rohan.simulation.models import SimulationOutput
from rohan.simulation.simulation_runner import SimulationRunner


class SimulationService:
    """Service class for managing simulations."""

    def __init__(self, settings: SimulationSettings):
        self.settings = settings
        self.simulation_runner = SimulationService._get_runner(settings)

    def run_simulation(self) -> SimulationOutput:
        """Runs the simulation using the provided simulation runner."""
        return self.simulation_runner.run()

    @staticmethod
    def _get_runner(settings: SimulationSettings) -> SimulationRunner:
        """Returns the simulation engine based on settings."""
        if settings.engine == SimulationEngine.ABIDES:
            from rohan.simulation.abides_impl import SimulationRunnerAbides

            return SimulationRunnerAbides(settings)

        raise ValueError(f"Unsupported simulation engine: {settings.engine}")
