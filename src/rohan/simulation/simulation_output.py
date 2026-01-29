"""Abstract base class for simulation output handlers."""

from abc import ABC, abstractmethod

import pandas as pd


class SimulationOutput(ABC):
    """Abstract base class for simulation output handlers."""

    @abstractmethod
    def get_order_book_l1(self) -> pd.DataFrame:
        """Returns the order book data from the simulation output."""
        pass

    @abstractmethod
    def get_order_book_l2(self, n_levels: int) -> pd.DataFrame:
        """Returns the Level 2 order book data from the simulation output."""
        pass

    @abstractmethod
    def get_logs_df(self) -> pd.DataFrame:
        """Returns the logs from the simulation output."""
        pass

    @abstractmethod
    def get_logs_by_agent(self) -> dict:
        """Returns the logs for a specific agent from the simulation output."""
        pass
