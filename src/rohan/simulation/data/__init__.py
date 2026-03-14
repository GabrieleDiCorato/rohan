"""Data pipeline for historical market data integration.

Provides the data transfer objects, normalization utilities, and providers
for loading external fundamental value series into ABIDES simulations.
"""

from .csv_provider import CsvDataProvider
from .generator import generate_fundamental_csv
from .models import DataProviderType, FundamentalDataset
from .normalization import normalize_fundamental_series

__all__ = [
    "CsvDataProvider",
    "generate_fundamental_csv",
    "DataProviderType",
    "FundamentalDataset",
    "normalize_fundamental_series",
]
