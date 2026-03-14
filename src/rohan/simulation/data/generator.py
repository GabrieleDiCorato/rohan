"""Generator for baseline fundamental value scenarios.

Creates canonical CSV datasets using ABIDES' built-in synthetic oracles
for testing and fallback when external data is not provided.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from abides_core import NanosecondTime
from abides_core.utils import str_to_ns
from abides_markets.oracles import SparseMeanRevertingOracle

from rohan.simulation.abides_impl.random_state_handler import RandomStateHandler


def generate_fundamental_csv(
    output_path: Path,
    *,
    symbol: str = "ABM",
    r_bar: int = 100_000,
    kappa: float = 1.67e-16,
    fund_vol: float = 5e-5,
    megashock_lambda_a: float = 2.77778e-18,
    megashock_mean: int = 1000,
    megashock_var: int = 50_000,
    date: str = "20260130",
    start_time: str = "09:30:00",
    end_time: str = "16:00:00",
    seed: int = 42,
) -> Path:
    """Generate a fundamental value CSV using SparseMeanRevertingOracle.

    This runs a standalone instance of the oracle to generate a full day
    of prices, which is then serialized to a canonical CSV format.

    Args:
        output_path: Destination file path.
        symbol: The ticker symbol.
        r_bar: Starting mean price (cents).
        kappa: Mean reversion rate.
        fund_vol: Baseline volatility.
        megashock_lambda_a: Poisson rate for megashocks.
        megashock_mean: Mean jump size for megashocks.
        megashock_var: Variance of megashocks.
        date: YYYYMMDD date string.
        start_time: Market open time (HH:MM:SS).
        end_time: Market close time (HH:MM:SS).
        seed: Random seed for determinism.

    Returns:
        The resolved output path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Setup ABIDES time logic
    day_ns = int(pd.to_datetime(date).value)
    mkt_open: NanosecondTime = day_ns + str_to_ns(start_time)
    mkt_close: NanosecondTime = day_ns + str_to_ns(end_time)

    handler = RandomStateHandler(seed)

    # Instantiate the Oracle
    oracle = SparseMeanRevertingOracle(
        mkt_open=mkt_open,
        mkt_close=mkt_close,
        symbols={
            symbol: {
                "r_bar": r_bar,
                "kappa": kappa,
                "fund_vol": fund_vol,
                "megashock_lambda_a": megashock_lambda_a,
                "megashock_mean": megashock_mean,
                "megashock_var": megashock_var,
                "random_state": handler.oracle_state,
            }
        },
    )

    # We step through the day and sample the oracle at 1-second intervals
    timestamps = []
    prices = []

    current_time = mkt_open
    step_ns = int(1e9)  # 1 second step

    # Use a dummy random state for observation, though sigma_n=0 ignores it
    obs_random_state = np.random.RandomState(seed)

    while current_time <= mkt_close:
        # sigma_n=0 means return the exact fundamental value without observation noise
        price = oracle.observe_price(
            symbol=symbol,
            current_time=current_time,
            random_state=obs_random_state,
            sigma_n=0,
        )
        timestamps.append(current_time)
        prices.append(price)
        current_time += step_ns

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, unit="ns"),
            "price_cents": prices,
        }
    )

    df.to_csv(output_path, index=False)
    return output_path
