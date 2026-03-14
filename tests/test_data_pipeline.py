"""Tests for the historical data pipeline (Component 1)."""

import pandas as pd
import pytest

from rohan.simulation.data.csv_provider import CsvDataProvider
from rohan.simulation.data.models import FundamentalDataset
from rohan.simulation.data.normalization import normalize_fundamental_series


class TestNormalization:
    def test_timestamp_conversion(self):
        # Even if index is strings, it should convert to DatetimeIndex
        raw = pd.Series([100000, 100050], index=["2026-01-30T09:30:00", "2026-01-30T09:30:01"])
        norm = normalize_fundamental_series(raw)
        assert isinstance(norm.index, pd.DatetimeIndex)
        assert norm.index[0] == pd.Timestamp("2026-01-30T09:30:00")

    def test_cents_conversion(self):
        # Floats should round to ints
        raw = pd.Series([100000.1, 100050.6], index=pd.date_range("2026-01-30", periods=2, freq="S"))
        norm = normalize_fundamental_series(raw)
        assert norm.iloc[0] == 100000
        assert norm.iloc[1] == 100051
        assert norm.dtype == int

    def test_monotonicity_check(self):
        # Should sort chronologically
        idx = [pd.Timestamp("2026-01-30T09:30:01"), pd.Timestamp("2026-01-30T09:30:00")]
        raw = pd.Series([100050, 100000], index=idx)
        norm = normalize_fundamental_series(raw)
        assert norm.index.is_monotonic_increasing
        assert norm.iloc[0] == 100000

    def test_recenter_r_bar(self):
        raw = pd.Series([50000, 50100, 49900], index=pd.date_range("2026-01-30", periods=3, freq="S"))
        # Mean is 50000. Recenter to 100000.
        norm = normalize_fundamental_series(raw, r_bar=100000)
        assert norm.mean() == 100000
        assert norm.iloc[0] == 100000

    def test_negative_prices_validation(self):
        raw = pd.Series([-100, 50], index=pd.date_range("2026-01-30", periods=2, freq="S"))
        with pytest.raises(ValueError, match="zero or negative"):
            normalize_fundamental_series(raw, validate=True)


class TestCsvDataProvider:
    @pytest.fixture
    def mock_csv(self, tmp_path):
        csv_path = tmp_path / "test_data.csv"
        df = pd.DataFrame({"timestamp": pd.date_range("2026-01-30T09:00:00", periods=10, freq="S"), "price_cents": range(100000, 100010)})
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_load_valid_csv(self, mock_csv):
        provider = CsvDataProvider(mock_csv, symbol="ABM")
        end_ns = int(pd.Timestamp("2200-01-01").value)
        series = provider.get_fundamental_series("ABM", 0, end_ns)  # large end time
        assert len(series) == 10
        assert series.iloc[0] == 100000

    def test_wrong_symbol(self, mock_csv):
        provider = CsvDataProvider(mock_csv, symbol="ABM")
        with pytest.raises(KeyError):
            end_ns = int(pd.Timestamp("2200-01-01").value)
            provider.get_fundamental_series("AAPL", 0, end_ns)

    def test_slice_range(self, mock_csv):
        provider = CsvDataProvider(mock_csv, symbol="ABM")
        start_ns = int(pd.Timestamp("2026-01-30T09:00:02").value)
        end_ns = int(pd.Timestamp("2026-01-30T09:00:05").value)

        sliced = provider.get_fundamental_series("ABM", start_ns, end_ns)
        assert len(sliced) == 4
        assert sliced.iloc[0] == 100002
        assert sliced.iloc[-1] == 100005


class TestDataModels:
    def test_fundamental_dataset_from_series(self):
        series = pd.Series([1, 2, 3], index=pd.date_range("2026-01-01", periods=3, freq="S"))
        dataset = FundamentalDataset.from_series(series, symbol="TEST", source="Mock")

        assert dataset.symbol == "TEST"
        assert dataset.row_count == 3
        assert dataset.start_time == pd.Timestamp("2026-01-01")
        assert dataset.end_time == pd.Timestamp("2026-01-01 00:00:02")
        assert dataset.sampling_freq in ["S", "s", "np.timedelta64(1,'s')"]
