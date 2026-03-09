"""Unit tests for RandomStateHandler (TEST-2).

Tests deterministic seed derivation, independence of derived random states,
and edge-case seed values.
"""

from __future__ import annotations

import numpy as np
import pytest

from rohan.simulation.abides_impl.random_state_handler import RandomStateHandler


class TestDeterminism:
    """Same seed must produce identical random-state sequences."""

    def test_same_seed_produces_same_oracle_state(self):
        h1 = RandomStateHandler(42)
        h2 = RandomStateHandler(42)
        assert np.array_equal(
            h1.oracle_state.get_state()[1],
            h2.oracle_state.get_state()[1],
        )

    def test_same_seed_produces_same_kernel_state(self):
        h1 = RandomStateHandler(42)
        h2 = RandomStateHandler(42)
        assert np.array_equal(
            h1.random_state_kernel.get_state()[1],
            h2.random_state_kernel.get_state()[1],
        )

    def test_same_seed_produces_same_exchange_state(self):
        h1 = RandomStateHandler(42)
        h2 = RandomStateHandler(42)
        assert np.array_equal(
            h1.exchange_agent_state.get_state()[1],
            h2.exchange_agent_state.get_state()[1],
        )

    def test_same_seed_produces_same_latency_state(self):
        h1 = RandomStateHandler(42)
        h2 = RandomStateHandler(42)
        assert np.array_equal(
            h1.latency_state.get_state()[1],
            h2.latency_state.get_state()[1],
        )

    def test_get_random_state_is_deterministic(self):
        """Sequential calls to get_random_state produce the same sequence for same seed."""
        h1 = RandomStateHandler(99)
        h2 = RandomStateHandler(99)
        s1 = h1.get_random_state()
        s2 = h2.get_random_state()
        assert np.array_equal(s1.get_state()[1], s2.get_state()[1])


class TestDifferentSeeds:
    """Different seeds must produce different sequences."""

    def test_different_seeds_different_oracle_state(self):
        h1 = RandomStateHandler(42)
        h2 = RandomStateHandler(43)
        assert not np.array_equal(
            h1.oracle_state.get_state()[1],
            h2.oracle_state.get_state()[1],
        )

    def test_different_seeds_different_kernel_state(self):
        h1 = RandomStateHandler(42)
        h2 = RandomStateHandler(43)
        assert not np.array_equal(
            h1.random_state_kernel.get_state()[1],
            h2.random_state_kernel.get_state()[1],
        )


class TestIndependence:
    """Derived random states must be independent of each other."""

    def test_oracle_and_exchange_differ(self):
        h = RandomStateHandler(42)
        assert not np.array_equal(
            h.oracle_state.get_state()[1],
            h.exchange_agent_state.get_state()[1],
        )

    def test_oracle_and_latency_differ(self):
        h = RandomStateHandler(42)
        assert not np.array_equal(
            h.oracle_state.get_state()[1],
            h.latency_state.get_state()[1],
        )

    def test_exchange_and_kernel_differ(self):
        h = RandomStateHandler(42)
        assert not np.array_equal(
            h.exchange_agent_state.get_state()[1],
            h.random_state_kernel.get_state()[1],
        )

    def test_get_random_state_returns_new_state(self):
        """Successive calls return different states."""
        h = RandomStateHandler(42)
        s1 = h.get_random_state()
        s2 = h.get_random_state()
        assert not np.array_equal(s1.get_state()[1], s2.get_state()[1])


class TestEdgeCaseSeeds:
    """Edge case seed values must not crash."""

    @pytest.mark.parametrize("seed", [0, 1, 2**32 - 1])
    def test_boundary_seeds(self, seed: int):
        h = RandomStateHandler(seed)
        assert h.seed == seed
        # All states should be valid RandomState instances
        assert isinstance(h.oracle_state, np.random.RandomState)
        assert isinstance(h.exchange_agent_state, np.random.RandomState)
        assert isinstance(h.latency_state, np.random.RandomState)
        assert isinstance(h.random_state_kernel, np.random.RandomState)

    def test_seed_stored(self):
        h = RandomStateHandler(12345)
        assert h.seed == 12345
