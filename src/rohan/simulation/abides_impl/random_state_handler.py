from numpy.random import RandomState


class RandomStateHandler:
    """Handles the creation of numpy RandomState instances with a fixed seed for reproducibility.
    This allows different components of the simulation to have their own independent random states.
    Independent random states help avoid unintended correlations between components.
    """

    def __init__(self, seed: int):
        """We initialize all random states here to ensure reproducibility across different components."""
        self.seed = seed

        # Create a master random state from the seed
        self.master_random_state: RandomState = RandomState(seed=self.seed)

        self.oracle_state: RandomState = self._get_random_state(self.master_random_state)
        self.exchange_agent_state: RandomState = self._get_random_state(self.master_random_state)
        self.latency_state: RandomState = self._get_random_state(self.master_random_state)
        self.random_state_kernel: RandomState = self._get_random_state(self.master_random_state)

    def get_random_state(self) -> RandomState:
        """Returns a new RandomState instance derived from the master random state.
        This can be used for components that need their own independent random state.
        Returns:
            RandomState: The created RandomState instance.
        """
        return self._get_random_state(self.master_random_state)

    @staticmethod
    def _get_random_state(
        master_random_state: RandomState,
    ) -> RandomState:
        """Creates a numpy RandomState from a master random state.
        Returns:
            RandomState: The created RandomState instance.
        """
        return RandomState(seed=master_random_state.randint(low=0, high=2**32, dtype="uint64"))
