"""Abstract base class for simulation output handlers.

Every concrete implementation must return DataFrames that conform to the
Pandera schemas defined in :mod:`rohan.simulation.models.schemas`.  The
schemas are the **single source of truth** for column names, types, and
constraints — see that module's docstring for design rationale.

Subclasses automatically get ``@pandera.check_types`` applied to their
``get_order_book_l1``, ``get_order_book_l2``, and ``get_logs_df``
implementations via ``__init_subclass__``, ensuring return values are
validated against the declared Pandera schemas without the implementor
having to call ``.validate()`` manually.
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pandera import check_types
from pandera.typing.pandas import DataFrame

from rohan.simulation.models.schemas import (
    AgentLogsSchema,
    OrderBookL1Schema,
    OrderBookL2Schema,
)


class SimulationOutput(ABC):
    """Abstract base class for simulation output handlers.

    Subclasses must validate their DataFrames against the corresponding
    Pandera schema before returning.  See :class:`HasufelOutput` for the
    reference implementation.

    ``__init_subclass__`` automatically wraps ``get_order_book_l1``,
    ``get_order_book_l2``, and ``get_logs_df`` with
    ``@pandera.check_types`` so that every concrete implementation gets
    return-value validation for free.
    """

    _CHECKED_METHODS: ClassVar[frozenset[str]] = frozenset(
        {
            "get_order_book_l1",
            "get_order_book_l2",
            "get_logs_df",
        }
    )

    strategic_agent_id: int | None = None
    """Numeric ID of the user-injected strategic agent, or ``None`` for
    baseline runs (no strategy).  Set at construction time by the
    simulation runner so callers never need to search for it."""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in SimulationOutput._CHECKED_METHODS:
            if name in cls.__dict__:
                method = cls.__dict__[name]
                if callable(method) and not getattr(method, "_pa_checked", False):
                    wrapped = check_types(method)
                    wrapped._pa_checked = True  # type: ignore[attr-defined]
                    setattr(cls, name, wrapped)

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
