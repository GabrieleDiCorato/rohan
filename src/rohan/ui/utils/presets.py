"""Preset simulation configurations for quick testing."""

from rohan.config import SimulationSettings
from rohan.config.agent_settings import (
    AdaptiveMarketMakerSettings,
    AgentSettings,
    MomentumAgentSettings,
    NoiseAgentSettings,
    OracleSettings,
    ValueAgentSettings,
)


def get_preset_names() -> list[str]:
    """Get list of available preset names."""
    return [
        "Default (Balanced Market)",
        "High Volatility",
        "Low Liquidity",
        "Market Maker Stress Test",
        "Momentum Dominated",
    ]


def get_preset_config(preset_name: str) -> SimulationSettings:
    """Get a preset simulation configuration by name."""
    presets = {
        "Default (Balanced Market)": _default_config(),
        "High Volatility": _high_volatility_config(),
        "Low Liquidity": _low_liquidity_config(),
        "Market Maker Stress Test": _market_maker_stress_config(),
        "Momentum Dominated": _momentum_dominated_config(),
    }
    return presets.get(preset_name, _default_config())


def _default_config() -> SimulationSettings:
    """Default balanced market configuration."""
    return SimulationSettings(
        date="20260130",
        start_time="09:30:00",
        end_time="10:00:00",
        starting_cash=10_000_000,
        log_orders=True,
    )


def _high_volatility_config() -> SimulationSettings:
    """High volatility market configuration."""
    oracle = OracleSettings(
        kappa=1.67e-16,
        fund_vol=1e-4,  # Double the default volatility (but still within safe range)
        megashock_lambda_a=5.55556e-18,  # More frequent megashocks
        megashock_mean=2000,  # Larger megashocks (in cents, so $20)
        megashock_var=100_000,  # Higher variance but within safe limits
    )

    agents = AgentSettings(
        oracle=oracle,
        noise=NoiseAgentSettings(num_agents=1500),  # More noise traders
        value=ValueAgentSettings(
            num_agents=102,
            r_bar=100_000,  # $1000 stock price (safe value)
            kappa=1.67e-15,
            lambda_a=5.7e-12,
        ),
        momentum=MomentumAgentSettings(
            num_agents=25,  # More momentum traders
            min_size=1,
            max_size=15,  # Larger orders but still reasonable
            wake_up_freq="20s",  # More frequent trading
        ),
    )

    return SimulationSettings(
        date="20260130",
        start_time="09:30:00",
        end_time="10:00:00",
        starting_cash=10_000_000,
        log_orders=True,
        agents=agents,
    )


def _low_liquidity_config() -> SimulationSettings:
    """Low liquidity market configuration."""
    agents = AgentSettings(
        noise=NoiseAgentSettings(num_agents=200),  # Fewer noise traders
        value=ValueAgentSettings(
            num_agents=50,  # Fewer value traders
            r_bar=100_000,  # $1000 stock price
            kappa=1.67e-15,
            lambda_a=5.7e-12,
        ),
        adaptive_market_maker=AdaptiveMarketMakerSettings(
            num_agents=1,  # Only one market maker
            pov=0.01,  # Lower participation
        ),
    )

    return SimulationSettings(
        date="20260130",
        start_time="09:30:00",
        end_time="10:00:00",
        starting_cash=10_000_000,
        log_orders=True,
        agents=agents,
    )


def _market_maker_stress_config() -> SimulationSettings:
    """Market maker stress test configuration."""
    agents = AgentSettings(
        adaptive_market_maker=AdaptiveMarketMakerSettings(
            num_agents=5,  # Multiple market makers competing
            pov=0.05,
            num_ticks=20,  # Deeper order book
            wake_up_freq="30s",
        ),
        noise=NoiseAgentSettings(num_agents=2000),  # High noise
        value=ValueAgentSettings(
            num_agents=102,
            r_bar=100_000,  # $1000 stock price
            kappa=1.67e-15,
            lambda_a=5.7e-12,
        ),
        momentum=MomentumAgentSettings(
            num_agents=30,
            min_size=1,
            max_size=20,  # Larger orders but within safe limits
            wake_up_freq="15s",
        ),
    )

    return SimulationSettings(
        date="20260130",
        start_time="09:30:00",
        end_time="10:00:00",
        starting_cash=10_000_000,
        log_orders=True,
        agents=agents,
    )


def _momentum_dominated_config() -> SimulationSettings:
    """Momentum-dominated market configuration."""
    agents = AgentSettings(
        momentum=MomentumAgentSettings(
            num_agents=50,  # Many momentum traders
            min_size=5,
            max_size=25,  # Larger but safe order sizes
            wake_up_freq="10s",  # Very frequent
        ),
        noise=NoiseAgentSettings(num_agents=500),
        value=ValueAgentSettings(
            num_agents=50,
            r_bar=100_000,  # $1000 stock price
            kappa=1.67e-15,
            lambda_a=5.7e-12,
        ),
    )

    return SimulationSettings(
        date="20260130",
        start_time="09:30:00",
        end_time="10:00:00",
        starting_cash=10_000_000,
        log_orders=True,
        agents=agents,
    )
