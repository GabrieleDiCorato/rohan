"""Tests for Pandera DataFrame schemas.

Validates that:
1. Well-formed DataFrames pass schema validation.
2. Malformed DataFrames (missing columns, wrong types, violated constraints)
   are rejected.
3. The ``strict=False`` policy allows extra columns without error.
"""

import numpy as np
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
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
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
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
            OrderBookL2Schema.validate(valid_l2_df)

    def test_level_below_one_fails(self, valid_l2_df):
        valid_l2_df.loc[0, "level"] = 0
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
            OrderBookL2Schema.validate(valid_l2_df)

    def test_missing_column_fails(self, valid_l2_df):
        bad_df = valid_l2_df.drop(columns=["side"])
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
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
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
            AgentLogsSchema.validate(bad_df)


# ---------------------------------------------------------------------------
# Type Coercion and Edge Cases
# ---------------------------------------------------------------------------
class TestSchemaTypeCoercion:
    """Test Pandera type coercion and validation behavior."""

    def test_l1_time_coerces_to_int64(self):
        """time column should coerce to int64."""
        df = pd.DataFrame(
            {
                "time": [1000, 2000],  # Python int
                "bid_price": [100.0, 100.5],
                "bid_qty": [10.0, 12.0],
                "ask_price": [101.0, 101.5],
                "ask_qty": [8.0, 9.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00", "2026-01-30 09:30:01"]),
            }
        )
        result = OrderBookL1Schema.validate(df)
        assert result["time"].dtype == "int64"

    def test_l1_prices_as_int_coerced_to_float(self):
        """Integer prices should be coerced to float64."""
        df = pd.DataFrame(
            {
                "time": [1_000_000, 2_000_000],
                "bid_price": [10000, 10050],  # int (cents)
                "bid_qty": [10, 12],
                "ask_price": [10100, 10150],
                "ask_qty": [8, 9],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00", "2026-01-30 09:30:01"]),
            }
        )
        result = OrderBookL1Schema.validate(df)
        assert result["bid_price"].dtype == "float64"
        assert result["bid_price"].iloc[0] == 10000.0

    def test_l1_with_inf_prices_allowed(self):
        """Infinity values in prices should be allowed (edge case)."""
        df = pd.DataFrame(
            {
                "time": [1_000_000],
                "bid_price": [float("inf")],
                "bid_qty": [10.0],
                "ask_price": [101.0],
                "ask_qty": [8.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00"]),
            }
        )
        # Pandera allows inf by default unless explicitly constrained
        result = OrderBookL1Schema.validate(df)
        assert np.isinf(result["bid_price"].iloc[0])

    def test_l2_level_negative_fails(self):
        """L2 level must be >= 1."""
        df = pd.DataFrame(
            {
                "time": [1_000_000],
                "level": [-1],  # Invalid
                "side": ["bid"],
                "price": [100.0],
                "qty": [10.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00"]),
            }
        )
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
            OrderBookL2Schema.validate(df)

    def test_l2_side_case_sensitive(self):
        """L2 side validation is case-sensitive."""
        df = pd.DataFrame(
            {
                "time": [1_000_000],
                "level": [1],
                "side": ["BID"],  # Should be lowercase 'bid'
                "price": [100.0],
                "qty": [10.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00"]),
            }
        )
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
            OrderBookL2Schema.validate(df)

    def test_l2_side_accepts_only_bid_or_ask(self):
        """L2 side must be exactly 'bid' or 'ask'."""
        df = pd.DataFrame(
            {
                "time": [1_000_000],
                "level": [1],
                "side": ["offer"],  # Invalid
                "price": [100.0],
                "qty": [10.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00"]),
            }
        )
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
            OrderBookL2Schema.validate(df)

    def test_logs_agent_id_must_be_numeric(self):
        """agent_id must be int."""
        df = pd.DataFrame(
            {
                "agent_id": ["not_a_number"],  # Wrong type
                "agent_type": ["ExchangeAgent"],
                "EventType": ["ORDER_SUBMITTED"],
            }
        )
        with pytest.raises(pa.errors.SchemaError):  # pyright: ignore[reportPrivateImportUsage]
            AgentLogsSchema.validate(df)

    def test_logs_agent_id_coerces_float_to_int(self):
        """agent_id can coerce from float to int."""
        df = pd.DataFrame(
            {
                "agent_id": [1.0, 2.0],  # float → int
                "agent_type": ["ExchangeAgent", "NoiseAgent"],
                "EventType": ["ORDER_SUBMITTED", "ORDER_EXECUTED"],
            }
        )
        result = AgentLogsSchema.validate(df)
        assert result["agent_id"].dtype in ["int64", "int32"]


# ---------------------------------------------------------------------------
# Real-world Edge Cases
# ---------------------------------------------------------------------------
class TestSchemaRealWorldEdgeCases:
    """Test edge cases from actual simulation data."""

    def test_l1_with_all_nan_prices(self):
        """L1 with completely empty market (all NaN)."""
        df = pd.DataFrame(
            {
                "time": [1_000_000, 2_000_000],
                "bid_price": [float("nan"), float("nan")],
                "bid_qty": [float("nan"), float("nan")],
                "ask_price": [float("nan"), float("nan")],
                "ask_qty": [float("nan"), float("nan")],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00", "2026-01-30 09:30:01"]),
            }
        )
        result = OrderBookL1Schema.validate(df)
        assert bool(result["bid_price"].isna().all())  # pyright: ignore[reportGeneralTypeIssues]

    def test_l1_with_zero_quantities(self):
        """L1 with zero quantities (empty book level)."""
        df = pd.DataFrame(
            {
                "time": [1_000_000],
                "bid_price": [100.0],
                "bid_qty": [0.0],  # Zero quantity
                "ask_price": [101.0],
                "ask_qty": [0.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00"]),
            }
        )
        result = OrderBookL1Schema.validate(df)
        assert result["bid_qty"].iloc[0] == 0.0

    def test_l2_with_missing_levels(self):
        """L2 with gaps in levels (e.g., level 1 and 3 but not 2)."""
        df = pd.DataFrame(
            {
                "time": [1_000_000] * 2,
                "level": [1, 3],  # Missing level 2
                "side": ["bid", "bid"],
                "price": [100.0, 99.0],
                "qty": [10.0, 5.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00"] * 2),
            }
        )
        # Pandera doesn't enforce contiguity, just that level >= 1
        result = OrderBookL2Schema.validate(df)
        assert len(result) == 2

    def test_l1_very_large_prices(self):
        """L1 with very large price values."""
        df = pd.DataFrame(
            {
                "time": [1_000_000],
                "bid_price": [1e12],  # 1 trillion cents
                "bid_qty": [10.0],
                "ask_price": [1e12 + 100],
                "ask_qty": [8.0],
                "timestamp": pd.to_datetime(["2026-01-30 09:30:00"]),
            }
        )
        result = OrderBookL1Schema.validate(df)
        assert result["bid_price"].iloc[0] == 1e12

    def test_logs_with_many_event_types(self):
        """Logs with various event types."""
        df = pd.DataFrame(
            {
                "agent_id": [1, 1, 1, 1, 2],
                "agent_type": ["NoiseAgent"] * 5,
                "EventType": [
                    "ORDER_SUBMITTED",
                    "ORDER_ACCEPTED",
                    "ORDER_EXECUTED",
                    "ORDER_CANCELLED",
                    "WAKEUP",
                ],
            }
        )
        result = AgentLogsSchema.validate(df)
        assert len(result["EventType"].unique()) == 5
