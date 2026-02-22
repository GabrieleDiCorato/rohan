"""Property-based tests using Hypothesis (§2.7.8c).

Fuzzes Pydantic validators on agent settings, simulation settings, and the
OrderAction model to ensure boundary conditions are handled correctly.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from rohan.config.agent_settings import (
    MomentumAgentSettings,
    NoiseAgentSettings,
    OracleSettings,
    ValueAgentSettings,
)
from rohan.config.simulation_settings import SimulationSettings
from rohan.simulation.models.strategy_api import OrderAction, OrderType, Side

# Hypothesis profile: keep tests fast (default max_examples=100).
SETTINGS = settings(max_examples=200, deadline=None)


# ---------------------------------------------------------------------------
# MomentumAgentSettings
# ---------------------------------------------------------------------------


@SETTINGS
@given(max_size=st.integers(min_value=-1000, max_value=10_000))
def test_momentum_max_size_bounds(max_size: int):
    """max_size must be in (0, 1000]."""
    if max_size < 1 or max_size > 1000:
        with pytest.raises(ValidationError):
            MomentumAgentSettings(max_size=max_size)
    else:
        s = MomentumAgentSettings(max_size=max_size)
        assert s.max_size == max_size


@SETTINGS
@given(min_size=st.integers(min_value=-10, max_value=1500), max_size=st.integers(min_value=-10, max_value=1500))
def test_momentum_size_relationship(min_size: int, max_size: int):
    """min_size must be in (0, ..] and min_size <= max_size, max_size <= 1000."""
    is_valid = min_size >= 1 and max_size >= 1 and max_size <= 1000 and min_size <= max_size
    if not is_valid:
        with pytest.raises(ValidationError):
            MomentumAgentSettings(min_size=min_size, max_size=max_size)
    else:
        s = MomentumAgentSettings(min_size=min_size, max_size=max_size)
        assert s.min_size == min_size
        assert s.max_size == max_size


# ---------------------------------------------------------------------------
# ValueAgentSettings
# ---------------------------------------------------------------------------


@SETTINGS
@given(r_bar=st.integers(min_value=-100, max_value=20_000_000))
def test_value_agent_r_bar(r_bar: int):
    """r_bar must be in (0, 10_000_000]."""
    if r_bar <= 0 or r_bar > 10_000_000:
        with pytest.raises(ValidationError):
            ValueAgentSettings(r_bar=r_bar)
    else:
        s = ValueAgentSettings(r_bar=r_bar)
        assert s.r_bar == r_bar


@SETTINGS
@given(kappa=st.floats(min_value=-1e-9, max_value=1e-9, allow_nan=False, allow_infinity=False))
def test_value_agent_kappa(kappa: float):
    """kappa must be in [0, 1e-10]."""
    if kappa < 0 or kappa > 1e-10:
        with pytest.raises(ValidationError):
            ValueAgentSettings(kappa=kappa)
    else:
        s = ValueAgentSettings(kappa=kappa)
        assert s.kappa == kappa


@SETTINGS
@given(lambda_a=st.floats(min_value=-1e-3, max_value=1e-2, allow_nan=False, allow_infinity=False))
def test_value_agent_lambda_a(lambda_a: float):
    """lambda_a must be in (0, 1e-3]."""
    if lambda_a <= 0 or lambda_a > 1e-3:
        with pytest.raises(ValidationError):
            ValueAgentSettings(lambda_a=lambda_a)
    else:
        s = ValueAgentSettings(lambda_a=lambda_a)
        assert s.lambda_a == lambda_a


# ---------------------------------------------------------------------------
# OracleSettings
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    megashock_mean=st.integers(min_value=-200_000, max_value=200_000),
    megashock_var=st.integers(min_value=-100, max_value=20_000_000),
    fund_vol=st.floats(min_value=-0.1, max_value=0.1, allow_nan=False, allow_infinity=False),
)
def test_oracle_settings_validators(megashock_mean: int, megashock_var: int, fund_vol: float):
    is_valid = abs(megashock_mean) <= 100_000 and 0 <= megashock_var <= 10_000_000 and 0 <= fund_vol <= 1e-2
    if not is_valid:
        with pytest.raises(ValidationError):
            OracleSettings(megashock_mean=megashock_mean, megashock_var=megashock_var, fund_vol=fund_vol)
    else:
        s = OracleSettings(megashock_mean=megashock_mean, megashock_var=megashock_var, fund_vol=fund_vol)
        assert s.megashock_mean == megashock_mean
        assert s.megashock_var == megashock_var
        assert s.fund_vol == fund_vol


# ---------------------------------------------------------------------------
# OrderAction — LIMIT must have price, MARKET must not
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    price=st.one_of(st.none(), st.integers(min_value=1, max_value=1_000_000)),
    order_type=st.sampled_from(OrderType),
    side=st.sampled_from(Side),
    quantity=st.integers(min_value=1, max_value=10_000),
)
def test_order_action_price_validation(price: int | None, order_type: OrderType, side: Side, quantity: int):
    """LIMIT orders require a price; MARKET orders must not have one."""
    if order_type == OrderType.LIMIT and price is None:
        with pytest.raises(ValidationError, match="price is required for LIMIT orders"):
            OrderAction(order_type=order_type, price=price, side=side, quantity=quantity)
    elif order_type == OrderType.MARKET and price is not None:
        with pytest.raises(ValidationError, match="price must be None for MARKET orders"):
            OrderAction(order_type=order_type, price=price, side=side, quantity=quantity)
    else:
        action = OrderAction(order_type=order_type, price=price, side=side, quantity=quantity)
        assert action.order_type == order_type
        assert action.price == price
        assert action.side == side
        assert action.quantity == quantity


# ---------------------------------------------------------------------------
# SimulationSettings — minimum duration and seed bounds
# ---------------------------------------------------------------------------


@SETTINGS
@given(
    start_hour=st.integers(min_value=0, max_value=22),
    start_minute=st.integers(min_value=0, max_value=59),
    duration_minutes=st.integers(min_value=-10, max_value=120),
)
def test_simulation_duration_validator(start_hour: int, start_minute: int, duration_minutes: int):
    """Simulation duration must be >= 5 minutes (300 s)."""
    start_time = f"{start_hour:02d}:{start_minute:02d}:00"
    total_end_minutes = start_hour * 60 + start_minute + duration_minutes
    # Skip invalid wall-clock times
    if total_end_minutes < 0 or total_end_minutes >= 24 * 60:
        return
    end_hour = total_end_minutes // 60
    end_minute = total_end_minutes % 60
    end_time = f"{end_hour:02d}:{end_minute:02d}:00"

    if duration_minutes < 5:
        with pytest.raises(ValidationError, match="Simulation duration must be at least 5 minutes"):
            SimulationSettings(start_time=start_time, end_time=end_time)
    else:
        s = SimulationSettings(start_time=start_time, end_time=end_time)
        assert s.start_time == start_time
        assert s.end_time == end_time


@SETTINGS
@given(seed=st.integers(min_value=-(2**16), max_value=2**32 + 1000))
def test_simulation_seed_bounds(seed: int):
    """Seed must be within [0, 2^32 - 1]."""
    if seed < 0 or seed > 2**32 - 1:
        with pytest.raises(ValidationError):
            SimulationSettings(seed=seed)
    else:
        s = SimulationSettings(seed=seed)
        assert s.seed == seed


# ---------------------------------------------------------------------------
# BaseAgentSettings — num_agents bounds
# ---------------------------------------------------------------------------


@SETTINGS
@given(num_agents=st.integers(min_value=-100, max_value=20_000))
def test_agent_count_bounds(num_agents: int):
    """num_agents must be in [0, 10_000]."""
    if num_agents < 0 or num_agents > 10_000:
        with pytest.raises(ValidationError):
            NoiseAgentSettings(num_agents=num_agents)
    else:
        s = NoiseAgentSettings(num_agents=num_agents)
        assert s.num_agents == num_agents
