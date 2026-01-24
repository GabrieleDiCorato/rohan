import numpy as np
from abides_core import LatencyModel
from abides_markets.utils import generate_uniform_random_pairwise_dist_on_line, meters_to_light_ns

from rohan.abides_wrapper.random_state_handler import RandomStateHandler
from rohan.config.simulation_settings import LatencyType, SimulationSettings


class AbidesWrapper:
    """Wrapper for ABIDES simulation components.
    Handles the conversion of configuration settings to ABIDES-compatible objects.
    """

    def __init__(self, simulation_settings: SimulationSettings):
        self.simulation_settings = simulation_settings
        self.random_state_handler = RandomStateHandler(simulation_settings.seed)

    def build_configuration(self):
        pass

    def _to_latency_model(self, agent_count: int) -> LatencyModel:
        """Uses LatencyModelSettings to create an instance of LatencyModel from abides_core.

        Arguments:
        latency_settings: LatencyModelSettings instance containing configuration.
        """
        settings = self.simulation_settings.latency

        random_state = self.random_state_handler.latency_state
        pairwise = (agent_count, agent_count)

        if settings.type == LatencyType.NO_LATENCY:
            pairwise_latencies = np.zeros(pairwise, dtype=int)
        else:
            # All agents sit on line from Milan to Lisbon
            milan_to_lisbon_meters = 1687000
            pairwise_distances = generate_uniform_random_pairwise_dist_on_line(0.0, milan_to_lisbon_meters, agent_count, random_state=random_state)
            pairwise_latencies = meters_to_light_ns(pairwise_distances)

        return LatencyModel(
            random_state=random_state,
            min_latency=pairwise_latencies,
            latency_model=settings.type,
            # Args for cubic latency model:
            connected=True,  # hardcoded like in abides_markets
            jitter=settings.jitter,
            jitter_clip=settings.jitter_clip,
            jitter_unit=settings.jitter_unit,
        )
