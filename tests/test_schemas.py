"""Tests for Pandera DataFrame schemas.

Validates that:
1. Well-formed DataFrames pass schema validation.
2. Malformed DataFrames (missing columns, wrong types, violated constraints)
   are rejected.
3. The ``strict=False`` policy allows extra columns without error.
"""

import pandas as pd
import pandera.pandas as pa
import pytest

from rohan.simulation.models.schemas import (
    AgentLogsSchema,
    OrderBookL1Schema,
    OrderBookL2Schema,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def valid_l1_df() -> pd.DataFrame:
    """Minimal valid L1 DataFrame."""
    return pd.DataFrame(
        {
            "time": [1_000_000, 2_000_000],
            "bid_price": [100.0, 100.5],
            "bid_qty": [10.0, 12.0],
            "ask_price": [101.0, 101.5],
            "ask_qty": [8.0, 9.0],
            "timestamp": pd.to_datetime(["2026-01-30 09:30:00", "2026-01-30 09:30:01"]),
        }
    )


@pytest.fixture
def valid_l2_df() -> pd.DataFrame:
    """Minimal valid L2 DataFrame (2 timestamps × 2 levels × 2 sides)."""
    return pd.DataFrame(
        {
            "time": [1_000_000] * 4 + [2_000_000] * 4,
            "level": [1, 2, 1, 2] * 2,
            "side": ["bid", "bid", "ask", "ask"] * 2,
            "price": [100.0, 99.5, 101.0, 101.5, 100.1, 99.6, 101.1, 101.6],
            "qty": [10.0, 5.0, 8.0, 4.0, 11.0, 6.0, 9.0, 3.0],
            "timestamp": pd.to_datetime(["2026-01-30 09:30:00"] * 4 + ["2026-01-30 09:30:01"] * 4),
        }
    )


@pytest.fixture
def valid_logs_df() -> pd.DataFrame:
    """Minimal valid agent logs DataFrame."""
    return pd.DataFrame(
        {
            "agent_id": [0, 1, 1],
            "agent_type": ["ExchangeAgent", "NoiseAgent", "NoiseAgent"],
            "EventType": ["ORDER_SUBMITTED", "ORDER_SUBMITTED", "ORDER_EXECUTED"],
        }
    )


# ---------------------------------------------------------------------------
# OrderBookL1Schema
# ---------------------------------------------------------------------------
class TestOrderBookL1Schema:
    def test_valid_dataframe_passes(self, valid_l1_df):
        result = OrderBookL1Schema.validate(valid_l1_df)
        assert len(result) == 2

    def test_nullable_prices(self, valid_l1_df):
        """bid/ask prices may be NaN (crossed book)."""
        valid_l1_df.loc[0, "bid_price"] = float("nan")
        result = OrderBookL1Schema.validate(valid_l1_df)
        assert len(result) == 2

    def test_extra_columns_allowed(self, valid_l1_df):
        """strict=False means downstream-computed columns (e.g. mid_price) are fine."""
        valid_l1_df["mid_price"] = (valid_l1_df["bid_price"] + valid_l1_df["ask_price"]) / 2
        result = OrderBookL1Schema.validate(valid_l1_df)
        assert "mid_price" in result.columns

    def test_missing_column_fails(self, valid_l1_df):
        bad_df = valid_l1_df.drop(columns=["bid_price"])
        with pytest.raises(pa.errors.SchemaError):
            OrderBookL1Schema.validate(bad_df)

    def test_empty_dataframe_passes(self):
        """An empty DataFrame with correct columns should be valid."""
        empty = pd.DataFrame(
            {
                "time": pd.array([], dtype="int64"),
                "bid_price": pd.array([], dtype="float64"),
                "bid_qty": pd.array([], dtype="float64"),
                "ask_price": pd.array([], dtype="float64"),
                "ask_qty": pd.array([], dtype="float64"),
                "timestamp": pd.to_datetime([]),
            }
        )
        result = OrderBookL1Schema.validate(empty)
        assert result.empty


# ---------------------------------------------------------------------------
# OrderBookL2Schema
# ---------------------------------------------------------------------------
class TestOrderBookL2Schema:
    def test_valid_dataframe_passes(self, valid_l2_df):
        result = OrderBookL2Schema.validate(valid_l2_df)
        assert len(result) == 8

    def test_invalid_side_fails(self, valid_l2_df):
        valid_l2_df.loc[0, "side"] = "INVALID"
        with pytest.raises(pa.errors.SchemaError):
            OrderBookL2Schema.validate(valid_l2_df)

    def test_level_below_one_fails(self, valid_l2_df):
        valid_l2_df.loc[0, "level"] = 0
        with pytest.raises(pa.errors.SchemaError):
            OrderBookL2Schema.validate(valid_l2_df)

    def test_missing_column_fails(self, valid_l2_df):
        bad_df = valid_l2_df.drop(columns=["side"])
        with pytest.raises(pa.errors.SchemaError):
            OrderBookL2Schema.validate(bad_df)


# ---------------------------------------------------------------------------
# AgentLogsSchema
# ---------------------------------------------------------------------------
class TestAgentLogsSchema:
    def test_valid_dataframe_passes(self, valid_logs_df):
        result = AgentLogsSchema.validate(valid_logs_df)
        assert len(result) == 3

    def test_extra_columns_allowed(self, valid_logs_df):
        """Upstream parse_logs_df may add more columns."""
        valid_logs_df["time_placed"] = [100, 200, 300]
        valid_logs_df["extra_column"] = ["a", "b", "c"]
        result = AgentLogsSchema.validate(valid_logs_df)
        assert "extra_column" in result.columns

    def test_missing_required_column_fails(self, valid_logs_df):
        bad_df = valid_logs_df.drop(columns=["agent_id"])
        with pytest.raises(pa.errors.SchemaError):
            AgentLogsSchema.validate(bad_df)
