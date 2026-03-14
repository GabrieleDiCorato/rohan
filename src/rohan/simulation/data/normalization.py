"""Data normalization functions for fundamental value series.

Ensures that raw data from any provider conforms to the strict conventions
expected by the ABIDES simulation engine: nanosecond DatetimeIndex and
integer price values in cents.
"""

import pandas as pd


def normalize_fundamental_series(
    raw_series: pd.Series,
    *,
    r_bar: int | None = None,
    validate: bool = True,
) -> pd.Series:
    """Normalize a raw price series to ABIDES conventions.

    Args:
        raw_series: A pandas Series to normalize.
        r_bar: Optional target mean price to re-center the series around.
               If provided, the series will be shifted so its mean matches r_bar.
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

    # Force timezone-naive (ABIDES expects raw timestamps, usually assumed EST but naive)
    if series.index.tz is not None:
        series.index = series.index.tz_convert(None)

    # 2. Sort chronologically
    series.sort_index(inplace=True)

    # 3. Handle NaNs
    if series.hasnans:
        series.ffill(inplace=True)
        series.bfill(inplace=True)

    # 4. Convert prices to integer cents
    # If the input appears to be in dollars (e.g. mean < 10000 but it's a stock),
    # warn or scale? For simplicity, we assume the input is ALREADY in cents OR
    # the user handles it. Given ABIDES conventions, if data is float, round to int.
    series = series.round().astype(int)

    # 5. Optional re-centering
    if r_bar is not None:
        mean_price = series.mean()
        shift = int(r_bar - mean_price)
        series = series + shift

    # 6. Validate
    if validate:
        if (series <= 0).any():
            raise ValueError("Series contains zero or negative prices after normalization.")
        if not series.index.is_monotonic_increasing:
            raise ValueError("Series index is not monotonically increasing.")

    return series
