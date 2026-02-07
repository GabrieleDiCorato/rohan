"""Implementation of the SimulationRunner interface using the ABIDES framework."""

from typing import override

from abides_core import abides

from rohan.config import SimulationSettings
from rohan.simulation.abides_impl import AbidesConfigMapper, AbidesOutput
from rohan.simulation.models import SimulationOutput
from rohan.simulation.models.strategy_api import StrategicAgent
from rohan.simulation.simulation_runner import SimulationRunner


class SimulationRunnerAbides(SimulationRunner):
    """ABIDES implementation of the SimulationRunner interface."""

    def __init__(self, settings: SimulationSettings, strategy: StrategicAgent | None = None):
        super().__init__(settings)
        self.abides_wrapper = AbidesConfigMapper(settings, strategy=strategy)
        self.settings_abides = self.abides_wrapper.build_configuration()

    @override
    def run(self) -> SimulationOutput:
        """Runs the ABIDES market simulation.

        Returns:
            SimulationOutput: The output of the simulation.
        """

        random_state_handler = self.abides_wrapper.random_state_handler
        end_state = abides.run(
            self.settings_abides,
            kernel_random_state=random_state_handler.random_state_kernel,
        )

        return AbidesOutput(end_state)

    @override
    def cleanup(self) -> None:
        """Cleanup resources after simulation.

        For ABIDES simulations, no specific cleanup is needed.
        """
        pass
