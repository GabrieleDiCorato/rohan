"""Data normalization functions for fundamental value series.

Ensures that raw data from any provider conforms to the strict conventions
expected by the ABIDES simulation engine: nanosecond DatetimeIndex and
integer price values in cents.
"""

import pandas as pd
from pandera.errors import SchemaError

from rohan.config import PriceUnit
from rohan.simulation.models.schemas import FundamentalSeriesSchema


def normalize_fundamental_series(
    raw_series: pd.Series,
    *,
    r_bar: int | None = None,
    price_unit: PriceUnit = PriceUnit.CENTS,
    source_timezone: str = "America/New_York",
    validate: bool = True,
) -> pd.Series:
    """Normalize a raw price series to ABIDES conventions.

    Args:
        raw_series: A pandas Series to normalize.
        r_bar: Optional target mean price to re-center the series around.
               If provided, the series will be shifted so its mean matches r_bar.
         price_unit: Unit used by incoming prices before normalization.
         source_timezone: IANA timezone used for tz-aware timestamps.
        validate: Whether to run sanity checks after normalization.

    Returns:
        A new pd.Series with a nanosecond-precision DatetimeIndex and integer
        prices in cents, sorted chronologically.

    Raises:
        ValueError: If validation fails (e.g., negative prices, empty series).
    """
    if raw_series.empty:
        raise ValueError("Cannot normalize an empty series.")

    series = raw_series.copy()

    # 1. Convert index to DatetimeIndex (ns precision)
    if not isinstance(series.index, pd.DatetimeIndex):
        try:
            series.index = pd.to_datetime(series.index)
        except Exception as e:
            raise ValueError(f"Could not parse index as datetime: {e}") from e

    # ABIDES expects naive timestamps. For tz-aware inputs we preserve market
    # wall-clock semantics by converting to the configured exchange timezone,
    # then dropping timezone info.
    if series.index.tz is not None:
        try:
            series.index = series.index.tz_convert(source_timezone).tz_localize(None)
        except Exception as e:
            raise ValueError(f"Could not convert timezone-aware index using source_timezone={source_timezone}: {e}") from e

    # 2. Sort chronologically
    series.sort_index(inplace=True)

    # 3. Handle NaNs
    if series.hasnans:
        series.ffill(inplace=True)
        series.bfill(inplace=True)

    # 4. Convert prices to integer cents using explicit unit metadata.
    if price_unit == PriceUnit.CENTS:
        series = series.round().astype(int)
    elif price_unit == PriceUnit.DOLLARS:
        series = (series * 100).round().astype(int)
    else:
        raise ValueError(f"Unsupported price unit: {price_unit}")

    # 5. Optional re-centering
    if r_bar is not None:
        mean_price = series.mean()
        shift = int(r_bar - mean_price)
        series = series + shift

    # 6. Validate
    if validate:
        try:
            series = FundamentalSeriesSchema.validate(series)
        except SchemaError as e:
            raise ValueError(str(e)) from e
        if not series.index.is_monotonic_increasing:
            raise ValueError("Series index is not monotonically increasing.")

    return series
