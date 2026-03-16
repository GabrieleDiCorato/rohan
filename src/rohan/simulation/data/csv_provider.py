"""CSV-backed fundamental data provider.

Implements the BatchDataProvider protocol from abides-rohan to feed
historical or generated CSV data into the ExternalDataOracle.
"""

from pathlib import Path
from typing import cast, override

import pandas as pd
from abides_core import NanosecondTime

from rohan.config import PriceUnit
from rohan.simulation.data.normalization import normalize_fundamental_series
from rohan.simulation.data.provider_protocol import FundamentalDataProvider


class CsvDataProvider(FundamentalDataProvider):
    """BatchDataProvider that loads a canonical CSV file.

    Expected CSV format:
        timestamp,price_cents
        2021-02-05T09:30:00.000000000,100000
        2021-02-05T09:30:01.000000000,100050

    The data is normalized upon load using standard ABIDES rules.
    """

    def __init__(
        self,
        path: Path | str,
        symbol: str = "ABM",
        r_bar: int | None = None,
        price_unit: PriceUnit = PriceUnit.CENTS,
        source_timezone: str = "America/New_York",
    ) -> None:
        """Initialize the CSV provider.

        Args:
            path: Path to the CSV file.
            symbol: Ticker symbol this data represents.
            r_bar: Optional target mean price to re-center the series around.
            price_unit: Unit used by the CSV price column.
            source_timezone: IANA timezone to preserve market clock semantics.

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

        if "timestamp" not in raw_df.columns:
            raise ValueError(f"CSV missing required column 'timestamp': {raw_df.columns.tolist()}")

        value_column = "price_cents" if price_unit == PriceUnit.CENTS else "price"
        if value_column not in raw_df.columns:
            raise ValueError(f"CSV missing required price column '{value_column}' for unit={price_unit.value}: {raw_df.columns.tolist()}")

        raw_series = cast("pd.Series", raw_df.set_index("timestamp")[value_column])

        # Normalize and validate the series
        self._data = normalize_fundamental_series(
            raw_series,
            r_bar=r_bar,
            price_unit=price_unit,
            source_timezone=source_timezone,
            validate=True,
        )

    @override
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

        data = self._data

        # If the CSV was generated for a different calendar date than the
        # simulation date, shift the entire index so that the intraday time
        # pattern is preserved but the date matches the simulation.  This lets
        # a single CSV be reused across any simulation date without manual
        # regeneration.
        if not data.empty:
            data_index = cast(pd.DatetimeIndex, data.index)
            first_csv_timestamp = data_index.min()
            if not isinstance(first_csv_timestamp, pd.Timestamp):
                raise ValueError("Historical CSV index must contain valid timestamps")
            csv_date = first_csv_timestamp.normalize()
            req_date = start_ts.normalize()
            if csv_date != req_date:
                data = data.copy()
                data.index = data.index + (req_date - csv_date)

        # Note: loc slicing is inclusive.
        return data.loc[start_ts:end_ts]  # type: ignore

    @staticmethod
    @override
    def list_available(_source: Path | str | None = None) -> list[str]:
        """List all available CSV dataset names in a directory.

        Args:
            source: The directory to search for .csv files.

        Returns:
            A list of filenames (with or without extension, depending on preference).
            Here we return the stem (filename without extension) for cleaner UI.
        """
        if _source is None:
            return []

        dir_path = Path(_source)
        if not dir_path.exists() or not dir_path.is_dir():
            return []

        return [f.stem for f in dir_path.glob("*.csv")]
