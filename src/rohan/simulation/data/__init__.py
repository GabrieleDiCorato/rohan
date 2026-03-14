"""Data pipeline for historical market data integration.

Provides the data transfer objects, normalization utilities, and providers
for loading external fundamental value series into ABIDES simulations.
"""

from .api_provider import ApiDataProvider
from .csv_provider import CsvDataProvider
from .database_provider import DatabaseDataProvider
from .generator import generate_fundamental_csv
from .models import DataProviderType, FundamentalDataset
from .normalization import normalize_fundamental_series
from .provider_protocol import FundamentalDataProvider

__all__ = [
    "ApiDataProvider",
    "CsvDataProvider",
    "DatabaseDataProvider",
    "generate_fundamental_csv",
    "DataProviderType",
    "FundamentalDataset",
    "FundamentalDataProvider",
    "normalize_fundamental_series",
]
