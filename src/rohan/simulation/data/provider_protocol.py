"""Protocol for pluggable historical fundamental data providers."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

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
