"""Data transfer objects for historical market data.

This module defines the core data models used to transport fundamental
value data throughout the pipeline from providers to the simulation oracle.
"""

from datetime import datetime
from enum import Enum
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class DataProviderType(str, Enum):
    """Supported types of historical data providers."""

    CSV = "CSV"
    DATABASE = "DATABASE"
    API = "API"


class FundamentalDataset(BaseModel):
    """Metadata for a fundamental data series.

    This class wraps the actual pd.Series data with rich metadata used
    for UI display and validation.
    """

    symbol: str = Field(description="The ticker symbol this data represents, e.g., 'ABM'")
    source: str = Field(description="Provenance of the data, e.g., 'ABIDES Generator' or 'Historical CSV'")
    row_count: int = Field(description="Total number of data points in the series")
    start_time: datetime = Field(description="The timestamp of the first data point")
    end_time: datetime = Field(description="The timestamp of the last data point")
    sampling_freq: str | None = Field(default=None, description="Inferred sampling frequency, e.g., '1S' for 1 second")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_series(cls, series: pd.Series, symbol: str, source: str) -> "FundamentalDataset":
        """Construct metadata from an actual pandas Series.

        Args:
            series: A pandas Series with a nanosecond-precision DatetimeIndex
                    and integer values representing prices in cents.
            symbol: The ticker symbol.
            source: The provenance string.
        """
        if series.empty:
            raise ValueError("Series cannot be empty")

        freq = None
        if len(series) > 1:
            try:
                # Try to infer frequency from index (just the first few rows for speed)
                inferred = pd.infer_freq(cast(Any, series.index[:100]))
                if inferred:
                    freq = str(inferred)
            except Exception:
                pass

        # Extract min and max values from the index in a way pyright understands
        timestamps = cast("list[Any]", list(series.index))
        if not timestamps:
            raise ValueError("Series index cannot be empty")

        raw_min = cast(Any, min(timestamps))
        raw_max = cast(Any, max(timestamps))

        # In pandas, index.min() usually returns a Timestamp if it's a DatetimeIndex
        # but Pyright infers Any or object. We safely downcast by string-ifying and parsing.
        # It's less efficient but 100% type-safe and this function only runs once per dataset.
        start_dt = pd.to_datetime(cast(Any, str(raw_min))).to_pydatetime()
        end_dt = pd.to_datetime(cast(Any, str(raw_max))).to_pydatetime()

        return cls(
            symbol=symbol,
            source=source,
            row_count=len(series),
            start_time=start_dt,
            end_time=end_dt,
            sampling_freq=freq,
        )
