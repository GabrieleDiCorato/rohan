"""Database-backed fundamental data provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast, override

import pandas as pd
from abides_core import NanosecondTime
from sqlalchemy import text

from rohan.config import PriceUnit
from rohan.framework.database.database_connector import DatabaseConnector, get_database_connector
from rohan.simulation.data.normalization import normalize_fundamental_series
from rohan.simulation.data.provider_protocol import FundamentalDataProvider


class DatabaseDataProvider(FundamentalDataProvider):
    """Provider that loads curated historical data from SQL storage."""

    def __init__(
        self,
        dataset_id: str,
        *,
        symbol: str,
        db: DatabaseConnector | None = None,
        r_bar: int | None = None,
        source_timezone: str = "America/New_York",
    ) -> None:
        self._dataset_id = dataset_id
        self._symbol = symbol
        self._db = db or get_database_connector()

        session = self._db.get_session()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT timestamp, price_cents
                    FROM fundamental_data_points
                    WHERE dataset_id = :dataset_id
                    ORDER BY timestamp ASC
                    """
                ),
                {"dataset_id": dataset_id},
            ).fetchall()
        except Exception as e:
            raise ValueError("Failed to query fundamental_data_points. Ensure historical dataset tables exist and are initialized.") from e
        finally:
            self._db.remove_session()

        if not rows:
            raise ValueError(f"No data points found for dataset_id={dataset_id}")

        records: list[tuple[Any, Any]] = []
        for row in rows:
            records.append((cast(Any, row[0]), cast(Any, row[1])))
        df = pd.DataFrame.from_records(records, columns=["timestamp", "price_cents"])
        if df.shape[1] < 2:
            raise ValueError(f"Dataset {dataset_id} did not include timestamp and price columns")
        raw_series = cast(pd.Series, df.set_index("timestamp")["price_cents"])
        self._data = normalize_fundamental_series(
            raw_series,
            r_bar=r_bar,
            price_unit=PriceUnit.CENTS,
            source_timezone=source_timezone,
            validate=True,
        )

    @override
    def get_fundamental_series(self, symbol: str, start: NanosecondTime, end: NanosecondTime) -> pd.Series:
        if symbol != self._symbol:
            raise KeyError(f"DatabaseDataProvider configured for {self._symbol}, but {symbol} was requested")

        start_ts = pd.to_datetime(start, unit="ns")
        end_ts = pd.to_datetime(end, unit="ns")
        return self._data.loc[start_ts:end_ts]  # type: ignore

    @staticmethod
    @override
    def list_available(_source: Path | str | None = None) -> list[str]:
        _ = _source
        db = get_database_connector()
        session = db.get_session()
        try:
            rows = session.execute(
                text(
                    """
                    SELECT CAST(dataset_id AS TEXT)
                    FROM fundamental_datasets
                    ORDER BY created_at DESC
                    """
                )
            ).fetchall()
            return [cast(Any, row[0]) for row in rows]
        except Exception:
            return []
        finally:
            db.remove_session()
