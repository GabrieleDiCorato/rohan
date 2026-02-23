"""Edge-case tests for AbidesOutput (§2.7.8b).

Validates:
    - Empty order book (no snapshots) → empty DataFrame with correct schema.
    - Single snapshot → L1/L2 computation handles degenerate inputs.
    - Missing columns in raw ABIDES output → SchemaError at validation boundary.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pandera.errors import SchemaError

from rohan.simulation.abides_impl.abides_output import AbidesOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output(
    best_bids: list | None = None,
    best_asks: list | None = None,
    l2_times: list | None = None,
    l2_bids: list | None = None,
    l2_asks: list | None = None,
) -> AbidesOutput:
    """Build an AbidesOutput backed by mocked ABIDES objects."""
    order_book = MagicMock()
    order_book.get_L1_snapshots.return_value = {
        "best_bids": best_bids or [],
        "best_asks": best_asks or [],
    }
    order_book.get_L2_snapshots.return_value = {
        "times": l2_times or [],
        "bids": l2_bids or [],
        "asks": l2_asks or [],
    }

    exchange_agent = MagicMock()
    exchange_agent.order_books = {"ABM": order_book}
    exchange_agent.id = 0
    exchange_agent.log = []

    end_state: dict = {"agents": [exchange_agent]}
    return AbidesOutput(end_state)


# ---------------------------------------------------------------------------
# 2.7.8b — Empty order book
# ---------------------------------------------------------------------------


class TestEmptyOrderBook:
    """Empty order book (no snapshots) should produce empty DataFrames, not exceptions."""

    def test_l1_empty(self):
        output = _make_output()
        l1 = output.get_order_book_l1()

        assert l1.empty
        expected_cols = {"time", "bid_price", "bid_qty", "ask_price", "ask_qty", "timestamp"}
        assert expected_cols.issubset(set(l1.columns))

    def test_l2_empty(self):
        output = _make_output()
        l2 = output.get_order_book_l2(n_levels=5)

        assert l2.empty
        expected_cols = {"time", "level", "side", "price", "qty", "timestamp"}
        assert expected_cols == set(l2.columns)

    def test_l1_only_bids(self):
        """One-sided book: only bids, no asks."""
        time_ns = 34_200_000_000_000  # 09:30:00 in ns from midnight
        output = _make_output(best_bids=[(time_ns, 10000, 100)], best_asks=[])
        l1 = output.get_order_book_l1()

        assert len(l1) == 1
        row = l1.iloc[0]
        assert row["bid_price"] == 10000
        assert pd.isna(row["ask_price"])

    def test_l1_only_asks(self):
        """One-sided book: only asks, no bids."""
        time_ns = 34_200_000_000_000
        output = _make_output(best_bids=[], best_asks=[(time_ns, 10010, 200)])
        l1 = output.get_order_book_l1()

        assert len(l1) == 1
        row = l1.iloc[0]
        assert pd.isna(row["bid_price"])
        assert row["ask_price"] == 10010


# ---------------------------------------------------------------------------
# 2.7.8b — Single snapshot
# ---------------------------------------------------------------------------


class TestSingleSnapshot:
    """Single snapshot should be handled without errors."""

    def test_l1_single_snapshot(self):
        time_ns = 34_200_000_000_000
        output = _make_output(
            best_bids=[(time_ns, 10000, 100)],
            best_asks=[(time_ns, 10010, 200)],
        )
        l1 = output.get_order_book_l1()

        assert len(l1) == 1
        row = l1.iloc[0]
        assert row["bid_price"] == 10000
        assert row["ask_price"] == 10010
        assert row["bid_qty"] == 100
        assert row["ask_qty"] == 200

    def test_l2_single_snapshot(self):
        time_ns = 34_200_000_000_000
        output = _make_output(
            l2_times=[time_ns],
            l2_bids=[[[10000, 100], [9990, 50]]],
            l2_asks=[[[10010, 200], [10020, 300]]],
        )
        l2 = output.get_order_book_l2(n_levels=2)

        # 1 time * 2 levels * 2 sides = 4 rows
        assert len(l2) == 4

        bids = pd.DataFrame(l2[l2["side"] == "bid"]).sort_values(by=["level"])
        assert bids.iloc[0]["price"] == 10000.0
        assert bids.iloc[0]["qty"] == 100.0
        assert bids.iloc[1]["price"] == 9990.0
        assert bids.iloc[1]["qty"] == 50.0

        asks = pd.DataFrame(l2[l2["side"] == "ask"]).sort_values(by=["level"])
        assert asks.iloc[0]["price"] == 10010.0
        assert asks.iloc[1]["price"] == 10020.0

    def test_l2_caching(self):
        """Calling get_order_book_l2 twice with the same n_levels should return the cached result."""
        time_ns = 34_200_000_000_000
        output = _make_output(
            l2_times=[time_ns],
            l2_bids=[[[10000, 100]]],
            l2_asks=[[[10010, 200]]],
        )
        l2_a = output.get_order_book_l2(n_levels=1)
        l2_b = output.get_order_book_l2(n_levels=1)
        assert l2_a is l2_b  # same object — cached


# ---------------------------------------------------------------------------
# 2.7.8b — Missing columns → SchemaError
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# L1 positional concat — duplicate timestamps must not inflate row count
# ---------------------------------------------------------------------------


class TestDuplicateTimestamps:
    """get_L1_snapshots returns parallel arrays: positional concat must not
    create cross-product rows at duplicate timestamps."""

    def test_no_cross_product(self):
        """5 book_log2 entries (3 at same time) → exactly 5 L1 rows."""
        t0 = 34_200_000_000_000  # 09:30:00 in ns
        bids = [
            (t0, 10000, 100),
            (t0, 10001, 110),
            (t0, 10002, 120),
            (t0 + 100, 10010, 130),
            (t0 + 200, 10020, 140),
        ]
        asks = [
            (t0, 10050, 200),
            (t0, 10051, 210),
            (t0, 10052, 220),
            (t0 + 100, 10060, 230),
            (t0 + 200, 10070, 240),
        ]
        output = _make_output(best_bids=bids, best_asks=asks)
        l1 = output.get_order_book_l1()

        # With correct positional concat: 5 rows.
        # With the old pd.merge(on="time"): 3×3 + 1×1 + 1×1 = 11 rows.
        assert len(l1) == 5

    def test_positional_pairing(self):
        """Row i of bids pairs with row i of asks (not by timestamp key)."""
        t0 = 34_200_000_000_000
        bids = [
            (t0, 10000, 100),  # row 0
            (t0, 10001, 110),  # row 1
        ]
        asks = [
            (t0, 10050, 200),  # row 0
            (t0, 10051, 210),  # row 1
        ]
        output = _make_output(best_bids=bids, best_asks=asks)
        l1 = output.get_order_book_l1()

        assert len(l1) == 2
        # Row 0: first bid + first ask
        assert l1.iloc[0]["bid_price"] == 10000
        assert l1.iloc[0]["ask_price"] == 10050
        # Row 1: second bid + second ask
        assert l1.iloc[1]["bid_price"] == 10001
        assert l1.iloc[1]["ask_price"] == 10051

    def test_many_duplicates_linear(self):
        """20 entries with 5 at each of 4 timestamps → exactly 20 rows."""
        t0 = 34_200_000_000_000
        bids = [(t0 + (i // 5) * 1000, 10000 + i, 100) for i in range(20)]
        asks = [(t0 + (i // 5) * 1000, 10050 + i, 200) for i in range(20)]
        output = _make_output(best_bids=bids, best_asks=asks)
        l1 = output.get_order_book_l1()

        assert len(l1) == 20


class TestSchemaValidationBoundary:
    """Missing or malformed columns should raise SchemaError at validation."""

    def test_l1_missing_column_raises_schema_error(self):
        """If _compute_order_book_l1 returns a DataFrame missing a required column,
        Pandera should raise SchemaError."""
        output = _make_output()

        bad_df = pd.DataFrame(
            {
                "time": [1],
                # bid_price deliberately missing
                "bid_qty": [100],
                "ask_price": [10010],
                "ask_qty": [200],
                "timestamp": pd.to_datetime([1], unit="ns"),
            }
        )

        with patch.object(AbidesOutput, "_compute_order_book_l1", staticmethod(lambda _: bad_df)):
            # Clear any cached property
            vars(output).pop("_order_book_l1", None)
            with pytest.raises(SchemaError):
                output.get_order_book_l1()

    def test_logs_missing_column_raises_schema_error(self):
        """If parse_logs_df returns a DataFrame missing AgentID, SchemaError should be raised."""
        output = _make_output()

        bad_df = pd.DataFrame(
            {
                # agent_id deliberately missing
                "agent_type": ["ExchangeAgent"],
                "EventType": ["ORDER_SUBMITTED"],
            }
        )

        with patch("rohan.simulation.abides_impl.abides_output.parse_logs_df", return_value=bad_df):
            vars(output).pop("_logs_df", None)
            with pytest.raises(SchemaError):
                output.get_logs_df()

    def test_l2_invalid_side_raises_schema_error(self):
        """L2 schema has strict=True and side ∈ {'bid', 'ask'}.  Invalid side should fail."""
        output = _make_output()

        bad_df = pd.DataFrame(
            {
                "time": [1, 1],
                "level": [1, 1],
                "side": ["bid", "INVALID"],
                "price": [100.0, 200.0],
                "qty": [10.0, 20.0],
                "timestamp": pd.to_datetime([1, 1], unit="ns"),
            }
        )

        with patch.object(AbidesOutput, "_compute_order_book_l2", staticmethod(lambda _, __: bad_df)):
            output._order_book_l2_cache.clear()
            with pytest.raises(SchemaError):
                output.get_order_book_l2(n_levels=1)
