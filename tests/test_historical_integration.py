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
    assert result.error is None
    assert result.result is not None
    assert len(result.result.exchange_messages) > 0
    assert len(result.result.agent_logs) > 0

    # 4. Verify the oracle fundamental value actually matches our expectations
    # The true fundamental value is logged by the exchange in its own logs, or we can check the prices
    # For a quick smoke test, ensure the market didn't explode to NaN
    prices = [msg.price for msg in result.result.exchange_messages if hasattr(msg, "price") and msg.price is not None]
    if prices:
        # Should be roughly near our 50,000 r_bar
        avg_price = sum(prices) / len(prices)
        assert 40_000 < avg_price < 60_000
