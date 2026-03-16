"""Protocol for pluggable historical fundamental data providers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, override

import numpy as np
import pandas as pd
from abides_core import NanosecondTime


class FundamentalDataProvider(Protocol):
    """Protocol implemented by all providers consumed by ExternalDataOracle."""

    def get_fundamental_series(self, symbol: str, start: NanosecondTime, end: NanosecondTime) -> pd.Series:
        """Return a normalized fundamental series slice in integer cents."""
        ...

    @staticmethod
    def list_available(source: Path | str | None = None) -> list[str]:
        """Return discoverable dataset identifiers for UI selection."""
        ...


class PointDataProvider(Protocol):
    """Protocol for providers that answer single-timestamp lookups lazily."""

    def get_fundamental_at(self, symbol: str, timestamp: NanosecondTime) -> int:
        """Return the normalized fundamental value in cents for one timestamp."""
        ...


class LazyLinearPointAdapter(PointDataProvider):
    """Wraps a BatchDataProvider as a PointDataProvider for on-demand linear interpolation.

    ExternalDataOracle's batch mode pre-expands LINEAR to a full nanosecond-
    resolution grid (e.g. ~170 TiB for a 7-hour session).  This adapter loads
    the sparse series once and evaluates linear interpolation per query using
    numpy.interp, keeping memory proportional to the number of CSV rows.

    Because this class only exposes ``get_fundamental_at`` (not
    ``get_fundamental_series``), the oracle detects it as a PointDataProvider
    and uses its lazy LRU-cached lookup path instead.
    """

    def __init__(self, provider: FundamentalDataProvider, symbol: str, start: NanosecondTime, end: NanosecondTime) -> None:
        series = provider.get_fundamental_series(symbol, start, end)
        if series.empty:
            raise ValueError(f"Provider returned empty series for symbol '{symbol}'")
        self._xs = np.asarray(series.index.view("int64"), dtype=np.float64)
        self._ys = np.asarray(series, dtype=np.float64)

    @override
    def get_fundamental_at(self, symbol: str, timestamp: NanosecondTime) -> int:
        """Return linearly interpolated price in cents at the given nanosecond timestamp."""
        return int(round(float(np.interp(float(timestamp), self._xs, self._ys))))
