import numpy as np
from abides_core import LatencyModel
from abides_markets.utils import generate_uniform_random_pairwise_dist_on_line, meters_to_light_ns

from rohan.config.simulation_settings import LatencyType, SimulationSettings


class RandomStateHandler:
    """Handles the creation of numpy RandomState instances with a fixed seed for reproducibility.
    This allows different components of the simulation to have their own independent random states.
    Independent random states help avoid unintended correlations between components.
    """

    def __init__(self, seed: int):
        """We initialize all random states here to ensure reproducibility across different components."""
        self.seed = seed

        # Create a master random state from the seed WITHOUT affecting global state (different from ABIDES)
        master_random_state = np.random.RandomState(seed=self.seed)

        self.oracle_state = self._get_random_state(master_random_state)
        self.exchange_agent_state = self._get_random_state(master_random_state)
        self.noise_agent_state = self._get_random_state(master_random_state)
        self.value_agent_state = self._get_random_state(master_random_state)
        self.adaptive_market_maker_agent_state = self._get_random_state(master_random_state)
        self.momentum_agent_state = self._get_random_state(master_random_state)
        self.latency_state = self._get_random_state(master_random_state)
        self.random_state_kernel = self._get_random_state(master_random_state)

    @staticmethod
    def _get_random_state(master_random_state: np.random.RandomState) -> np.random.RandomState:
        """Creates a numpy RandomState from a master random state.
        Returns:
            np.random.RandomState: The created RandomState instance.
        """
        return np.random.RandomState(seed=master_random_state.randint(low=0, high=2**32, dtype="uint64"))


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
