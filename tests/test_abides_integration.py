"""Integration tests using a real ABIDES OrderBook.

Unlike other tests that mock `get_L1_snapshots()` / `get_L2_snapshots()`,
these tests instantiate a real `OrderBook`, feed it limit orders, and verify
that `AbidesOutput` produces schema-valid DataFrames from the real return
format.  This catches upstream changes in ABIDES (tuple ordering, dtype,
ragged arrays) that mock-based tests would miss.

Marked ``@pytest.mark.slow`` — excluded from default ``-m 'not slow'`` runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from abides_markets.order_book import OrderBook
from abides_markets.orders import LimitOrder, Side

from rohan.simulation.abides_impl.abides_output import AbidesOutput

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0 = 34_200_000_000_000  # 09:30:00 in ns since epoch


def _make_real_order_book(n_updates: int = 20, spread: int = 10) -> OrderBook:
    """Create a real OrderBook with ``n_updates`` bid/ask pairs.

    Each update advances time by 1 second and shifts the price by ±1 cent
    randomly (clamped positive).
    """
    owner = MagicMock()
    owner.mkt_open = _T0

    ob = OrderBook(owner, "ABM")

    bid_price = 10_000
    rng = np.random.default_rng(42)

    for i in range(n_updates):
        t = _T0 + i * 1_000_000_000  # 1 second apart
        owner.current_time = t

        bid_price = max(100, bid_price + int(rng.choice([-1, 0, 1])))
        ask_price = bid_price + spread

        bid = LimitOrder(
            agent_id=1,
            time_placed=t,
            symbol="ABM",
            quantity=100 + int(rng.integers(0, 50)),
            side=Side.BID,
            limit_price=bid_price,
        )
        ob.handle_limit_order(bid, quiet=False)

        ask = LimitOrder(
            agent_id=2,
            time_placed=t,
            symbol="ABM",
            quantity=100 + int(rng.integers(0, 50)),
            side=Side.ASK,
            limit_price=ask_price,
        )
        ob.handle_limit_order(ask, quiet=False)

    return ob


def _wrap_in_abides_output(ob: OrderBook) -> AbidesOutput:
    """Wrap a real OrderBook in an AbidesOutput (minimal mock of end_state)."""
    exchange_agent = MagicMock()
    exchange_agent.order_books = {"ABM": ob}
    exchange_agent.id = 0
    exchange_agent.log = []

    end_state: dict = {"agents": [exchange_agent]}
    return AbidesOutput(end_state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRealOrderBookL1:
    """L1 output from a real OrderBook must be schema-valid."""

    def test_l1_schema_valid(self):
        """get_order_book_l1() produces a schema-valid DataFrame."""
        ob = _make_real_order_book(n_updates=20)
        output = _wrap_in_abides_output(ob)

        l1 = output.get_order_book_l1()

        # Basic shape checks
        assert not l1.empty
        expected_cols = {"time", "bid_price", "bid_qty", "ask_price", "ask_qty", "timestamp"}
        assert expected_cols.issubset(set(l1.columns))

        # Types
        assert l1["time"].dtype == np.int64 or l1["time"].dtype == int
        assert l1["bid_price"].dtype == np.float64
        assert l1["ask_price"].dtype == np.float64

    def test_l1_parallel_arrays_same_length(self):
        """Real OrderBook produces equal-length bids and asks."""
        ob = _make_real_order_book(n_updates=30)
        l1_raw = ob.get_L1_snapshots()
        assert len(l1_raw["best_bids"]) == len(l1_raw["best_asks"])

    def test_l1_times_are_ns_since_midnight(self):
        """After ns_date subtraction, all times should be within [0, 24h)."""
        ob = _make_real_order_book(n_updates=15)
        output = _wrap_in_abides_output(ob)
        l1 = output.get_order_book_l1()

        max_ns_in_day = 24 * 3600 * 1e9
        assert (l1["time"] >= 0).all()
        assert (l1["time"] < max_ns_in_day).all()

    def test_l1_monotonic_time(self):
        """L1 time column should be monotonically non-decreasing."""
        ob = _make_real_order_book(n_updates=20)
        output = _wrap_in_abides_output(ob)
        l1 = output.get_order_book_l1()

        # Non-strict monotonic (duplicates allowed)
        diffs = l1["time"].diff().dropna()
        assert (diffs >= 0).all()


@pytest.mark.slow
class TestRealOrderBookL2:
    """L2 output from a real OrderBook must be schema-valid."""

    def test_l2_schema_valid(self):
        """get_order_book_l2() produces a schema-valid DataFrame."""
        ob = _make_real_order_book(n_updates=20)
        output = _wrap_in_abides_output(ob)

        l2 = output.get_order_book_l2(n_levels=3)

        assert not l2.empty
        expected_cols = {"time", "level", "side", "price", "qty", "timestamp"}
        assert expected_cols == set(l2.columns)

        # Side values
        assert set(l2["side"].unique()).issubset({"bid", "ask"})

        # Level values (1-indexed)
        assert l2["level"].min() >= 1

    def test_l2_shape_consistent(self):
        """Within each (time, side) group, every group should have the same
        number of rows — i.e. the requested n_levels (ABIDES zero-pads
        sparse levels).  Multiple book_log2 entries at the same timestamp
        produce duplicate (time, side, level) rows, so we just verify
        uniformity across groups rather than a hardcoded count."""
        ob = _make_real_order_book(n_updates=10)
        output = _wrap_in_abides_output(ob)
        n_levels = 5
        l2 = output.get_order_book_l2(n_levels=n_levels)

        if not l2.empty:
            group_sizes = l2.groupby(["time", "side"]).size()
            # All groups should have the same number of rows
            assert group_sizes.nunique() == 1, f"Non-uniform L2 group sizes: {group_sizes.unique()}"
            # Each group size should be a multiple of n_levels
            assert group_sizes.iloc[0] % n_levels == 0


@pytest.mark.slow
class TestLengthMismatchGuard:
    """Verify that a length mismatch between bids and asks raises ValueError."""

    def test_length_mismatch_raises(self):
        """Artificially mismatched arrays should trigger the length guard."""
        ob = _make_real_order_book(n_updates=10)
        output = _wrap_in_abides_output(ob)

        # Tamper with the order book to create a length mismatch
        original_get = ob.get_L1_snapshots

        def tampered_get():
            result = original_get()
            # Remove one bid entry to create mismatch
            result["best_bids"] = result["best_bids"][:-1]
            return result

        ob.get_L1_snapshots = tampered_get  # type: ignore[assignment]

        # Clear cache
        vars(output).pop("_order_book_l1", None)

        with pytest.raises(ValueError, match="parallel-array invariant"):
            output.get_order_book_l1()
