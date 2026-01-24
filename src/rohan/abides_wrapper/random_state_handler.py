import numpy as np


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
    def _get_random_state(
        master_random_state: np.random.RandomState,
    ) -> np.random.RandomState:
        """Creates a numpy RandomState from a master random state.
        Returns:
            np.random.RandomState: The created RandomState instance.
        """
        return np.random.RandomState(seed=master_random_state.randint(low=0, high=2**32, dtype="uint64"))
