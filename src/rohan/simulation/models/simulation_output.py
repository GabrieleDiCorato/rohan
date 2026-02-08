"""Abstract base class for simulation output handlers.

Every concrete implementation must return DataFrames that conform to the
Pandera schemas defined in :mod:`rohan.simulation.models.schemas`.  The
schemas are the **single source of truth** for column names, types, and
constraints — see that module's docstring for design rationale.
"""

from abc import ABC, abstractmethod
from typing import Any

from pandera.typing.pandas import DataFrame

from rohan.simulation.models.schemas import (
    AgentLogsSchema,
    OrderBookL1Schema,
    OrderBookL2Schema,
)


class SimulationOutput(ABC):
    """Abstract base class for simulation output handlers.

    Subclasses must validate their DataFrames against the corresponding
    Pandera schema before returning.  See :class:`AbidesOutput` for the
    reference implementation.
    """

    @abstractmethod
    def get_order_book_l1(self) -> DataFrame[OrderBookL1Schema]:
        """Return Level 1 order-book snapshots.

        Returns
        -------
        DataFrame[OrderBookL1Schema]
            Columns: time, bid_price, bid_qty, ask_price, ask_qty, timestamp.
        """

    @abstractmethod
    def get_order_book_l2(self, n_levels: int) -> DataFrame[OrderBookL2Schema]:
        """Return Level 2 (depth) order-book snapshots in tidy format.

        Returns
        -------
        DataFrame[OrderBookL2Schema]
            Columns: time, level, side, price, qty, timestamp.
        """

    @abstractmethod
    def get_logs_df(self) -> DataFrame[AgentLogsSchema]:
        """Return parsed agent event logs.

        Returns
        -------
        DataFrame[AgentLogsSchema]
            Guaranteed columns: agent_id, agent_type, EventType.
        """

    @abstractmethod
    def get_logs_by_agent(self) -> dict[int, Any]:
        """Return raw per-agent logs keyed by agent ID."""
