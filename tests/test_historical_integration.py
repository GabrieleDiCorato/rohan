"""Integration tests for the Historical Data Provider pipeline."""

import pytest

from rohan.config import (
    AdaptiveMarketMakerSettings,
    AgentSettings,
    CsvHistoricalProviderSettings,
    ExchangeAgentSettings,
    HistoricalOracleSettings,
    InterpolationMode,
    MomentumAgentSettings,
    NoiseAgentSettings,
    OracleSettings,
    OracleType,
    ProviderType,
    SimulationSettings,
    ValueAgentSettings,
)
from rohan.simulation.data.generator import generate_fundamental_csv
from rohan.simulation.simulation_service import SimulationService


@pytest.fixture(scope="session")
def historical_csv_path(tmp_path_factory):
    """Generate a valid historical dataset once per test session."""
    out_path = tmp_path_factory.mktemp("data") / "test_dataset.csv"
    generate_fundamental_csv(out_path, seed=1234, r_bar=50_000, fund_vol=1e-5)
    return out_path


@pytest.mark.integration
def test_historical_simulation_runs_successfully(historical_csv_path):
    """Verify a simulation using HistoricalOracleSettings completes without error."""

    # 1. Setup config using historical oracle
    settings = SimulationSettings(
        date="20260130",
        start_time="09:30:00",
        end_time="10:00:00",
        starting_cash=10_000_000,
        agents=AgentSettings(
            exchange=ExchangeAgentSettings(),
            # Fast/light simulation parameters
            noise=NoiseAgentSettings(num_agents=50),
            value=ValueAgentSettings(num_agents=10, r_bar=50_000),
            adaptive_market_maker=AdaptiveMarketMakerSettings(num_agents=1),
            momentum=MomentumAgentSettings(num_agents=5),
            oracle=OracleSettings(
                oracle_type=OracleType.HISTORICAL,
                historical=HistoricalOracleSettings(
                    provider_type=ProviderType.CSV,
                    interpolation=InterpolationMode.FORWARD_FILL,
                    csv=CsvHistoricalProviderSettings(csv_path=str(historical_csv_path)),
                    recenter_r_bar=False,
                ),
            ),
        ),
    )

    # 2. Run simulation
    service = SimulationService()
    result = service.run_simulation(settings)

    # 3. Verify
    assert result.error is None, f"Simulation failed: {result.error}"
    assert result.result is not None

    output = result.result

    # Verify order book data was produced
    l1 = output.get_order_book_l1()
    assert not l1.empty, "L1 order book should not be empty"

    # Verify agent logs were produced
    logs = output.get_logs_df()
    assert not logs.empty, "Agent logs should not be empty"

    # 4. Verify the fundamental value didn't explode to NaN
    # Check that L1 prices are roughly near our 50,000 r_bar
    import pandas as pd

    bid_prices = l1["bid_price"].dropna()
    ask_prices = l1["ask_price"].dropna()
    all_prices = pd.concat([bid_prices, ask_prices])
    if not all_prices.empty:
        avg_price = all_prices.mean()
        assert 40_000 < avg_price < 60_000, f"Average price {avg_price} outside expected range"
