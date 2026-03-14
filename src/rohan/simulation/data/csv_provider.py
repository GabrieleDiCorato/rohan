"""CSV-backed fundamental data provider.

Implements the BatchDataProvider protocol from abides-rohan to feed
historical or generated CSV data into the ExternalDataOracle.
"""

from pathlib import Path

import pandas as pd
from abides_core import NanosecondTime

from rohan.simulation.data.normalization import normalize_fundamental_series


class CsvDataProvider:
    """BatchDataProvider that loads a canonical CSV file.

    Expected CSV format:
        timestamp,price_cents
        2021-02-05T09:30:00.000000000,100000
        2021-02-05T09:30:01.000000000,100050

    The data is normalized upon load using standard ABIDES rules.
    """

    def __init__(self, path: Path | str, symbol: str = "ABM", r_bar: int | None = None) -> None:
        """Initialize the CSV provider.

        Args:
            path: Path to the CSV file.
            symbol: Ticker symbol this data represents.
            r_bar: Optional target mean price to re-center the series around.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV format is invalid or validation fails.
        """
        self._path = Path(path)
        self._symbol = symbol

        if not self._path.exists():
            raise FileNotFoundError(f"Historical data file not found: {self._path}")

        try:
            raw_df = pd.read_csv(self._path)
        except Exception as e:
            raise ValueError(f"Failed to read CSV at {self._path}: {e}") from e

        if "timestamp" not in raw_df.columns or "price_cents" not in raw_df.columns:
            raise ValueError(f"CSV missing required columns ('timestamp', 'price_cents'): {raw_df.columns.tolist()}")

        from typing import cast

        raw_series = cast("pd.Series", raw_df.set_index("timestamp")["price_cents"])

        # Normalize and validate the series
        self._data = normalize_fundamental_series(raw_series, r_bar=r_bar, validate=True)

    def get_fundamental_series(self, symbol: str, start: NanosecondTime, end: NanosecondTime) -> pd.Series:
        """Return the slice of data for *symbol* between *start* and *end*.

        This implements the abides-rohan `BatchDataProvider` protocol.

        Args:
            symbol: The requested ticker symbol.
            start: The start time in nanoseconds since epoch.
            end: The end time in nanoseconds since epoch.

        Returns:
            A pandas Series with a DatetimeIndex and integer prices in cents.

        Raises:
            KeyError: If the requested symbol does not match the provider's symbol.
        """
        if symbol != self._symbol:
            raise KeyError(f"CsvDataProvider configured for {self._symbol}, but {symbol} was requested")

        start_ts = pd.to_datetime(start, unit="ns")
        end_ts = pd.to_datetime(end, unit="ns")

        # If the requested range falls outside our dataset, return empty or what we have.
        # The ExternalDataOracle handles empty/partial slices via its interpolation proxy.
        # Note: loc slicing is inclusive.
        return self._data.loc[start_ts:end_ts]  # type: ignore

    @staticmethod
    def list_available(directory: Path | str) -> list[str]:
        """List all available CSV dataset names in a directory.

        Args:
            directory: The directory to search for .csv files.

        Returns:
            A list of filenames (with or without extension, depending on preference).
            Here we return the stem (filename without extension) for cleaner UI.
        """
        dir_path = Path(directory)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        return [f.stem for f in dir_path.glob("*.csv")]
