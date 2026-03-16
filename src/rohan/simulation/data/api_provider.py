"""HTTP/API-backed fundamental data provider."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast, override

import pandas as pd
from abides_core import NanosecondTime

from rohan.config import PriceUnit
from rohan.simulation.data.normalization import normalize_fundamental_series
from rohan.simulation.data.provider_protocol import FundamentalDataProvider

ApiFetcher = Callable[[str, str, NanosecondTime, NanosecondTime, str], pd.DataFrame | pd.Series]


class ApiDataProvider(FundamentalDataProvider):
    """Provider that fetches historical fundamentals from external HTTP APIs."""

    def __init__(
        self,
        *,
        provider_name: str,
        symbol: str,
        api_key: str,
        price_unit: PriceUnit = PriceUnit.DOLLARS,
        source_timezone: str = "America/New_York",
        r_bar: int | None = None,
        fetcher: ApiFetcher | None = None,
    ) -> None:
        self._provider_name = provider_name
        self._symbol = symbol
        self._api_key = api_key
        self._price_unit = price_unit
        self._source_timezone = source_timezone
        self._r_bar = r_bar
        self._fetcher = fetcher

    @override
    def get_fundamental_series(self, symbol: str, start: NanosecondTime, end: NanosecondTime) -> pd.Series:
        if symbol != self._symbol:
            raise KeyError(f"ApiDataProvider configured for {self._symbol}, but {symbol} was requested")

        if self._fetcher is None:
            raise NotImplementedError("ApiDataProvider requires a fetcher adapter. Inject a fetcher callable that returns canonical timestamp/price payload.")

        payload = self._fetcher(self._provider_name, self._symbol, start, end, self._api_key)
        if isinstance(payload, pd.Series):
            raw_series = payload
        else:
            if "timestamp" not in payload.columns or "price" not in payload.columns:
                raise ValueError("API fetcher output must include columns: timestamp, price")
            raw_series = cast(pd.Series, payload.set_index("timestamp")["price"])

        return normalize_fundamental_series(
            raw_series,
            r_bar=self._r_bar,
            price_unit=self._price_unit,
            source_timezone=self._source_timezone,
            validate=True,
        )

    @staticmethod
    @override
    def list_available(_source: Path | str | None = None) -> list[str]:
        _ = _source
        return ["alpaca", "polygon"]
