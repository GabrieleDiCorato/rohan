"""Tests for UI preset configurations.

This module tests:
- Preset name retrieval
- Preset configuration loading
- Validity of preset configurations
- Preset parameter values
"""

import pytest

from rohan.config import SimulationSettings
from rohan.ui.utils.presets import get_preset_config, get_preset_names


class TestUIPresets:
    """Test suite for UI preset functionality."""

    def test_get_preset_names_returns_list(self):
        """Test that get_preset_names returns a list."""
        names = get_preset_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_preset_names_contains_expected_presets(self):
        """Test that preset names contain expected configurations."""
        names = get_preset_names()

        expected_presets = [
            "Default (Balanced Market)",
            "High Volatility",
            "Low Liquidity",
            "Market Maker Stress Test",
            "Momentum Dominated",
        ]

        for expected in expected_presets:
            assert expected in names, f"Missing preset: {expected}"

    def test_get_preset_config_returns_simulation_settings(self):
        """Test that get_preset_config returns SimulationSettings instance."""
        preset_name = "Default (Balanced Market)"
        config = get_preset_config(preset_name)

        assert isinstance(config, SimulationSettings)

    def test_get_preset_config_invalid_name_returns_default(self):
        """Test that invalid preset name returns default configuration."""
        invalid_name = "NonExistent Preset"
        config = get_preset_config(invalid_name)

        assert isinstance(config, SimulationSettings)
        # Should return default config
        assert config.date == "20260130"
        assert config.start_time == "09:30:00"

    def test_default_preset_configuration(self):
        """Test Default (Balanced Market) preset configuration."""
        config = get_preset_config("Default (Balanced Market)")

        assert config.date == "20260130"
        assert config.start_time == "09:30:00"
        assert config.end_time == "10:00:00"
        assert config.starting_cash == 10_000_000
        assert config.log_orders is True

    def test_high_volatility_preset_configuration(self):
        """Test High Volatility preset configuration."""
        config = get_preset_config("High Volatility")

        # Should have increased noise agents
        assert config.agents.noise.num_agents == 1500

        # Should have more momentum traders
        assert config.agents.momentum.num_agents == 25

        # Oracle should have higher volatility settings
        assert config.agents.oracle.fund_vol > 5e-05  # Higher than default

        # Should have increased megashock frequency
        assert config.agents.oracle.megashock_lambda_a > 2.77778e-18  # Higher than default

    def test_low_liquidity_preset_configuration(self):
        """Test Low Liquidity preset configuration."""
        config = get_preset_config("Low Liquidity")

        # Should have fewer noise agents
        assert config.agents.noise.num_agents == 200

        # Should have fewer value traders
        assert config.agents.value.num_agents == 50

        # Should have minimal market makers
        assert config.agents.adaptive_market_maker.num_agents == 1

        # Should have lower POV
        assert config.agents.adaptive_market_maker.pov == 0.01

    def test_market_maker_stress_preset_configuration(self):
        """Test Market Maker Stress Test preset configuration."""
        config = get_preset_config("Market Maker Stress Test")

        # Should have multiple market makers
        assert config.agents.adaptive_market_maker.num_agents == 5

        # Should have deeper order book
        assert config.agents.adaptive_market_maker.num_ticks == 20

        # Should have high noise
        assert config.agents.noise.num_agents == 2000

        # Should have momentum traders
        assert config.agents.momentum.num_agents == 30

    def test_momentum_dominated_preset_configuration(self):
        """Test Momentum Dominated preset configuration."""
        config = get_preset_config("Momentum Dominated")

        # Should have many momentum traders
        assert config.agents.momentum.num_agents == 50

        # Should have larger order sizes
        assert config.agents.momentum.min_size == 5
        assert config.agents.momentum.max_size == 25

        # Should have frequent wake-ups
        assert config.agents.momentum.wake_up_freq == "10s"

    def test_all_presets_are_valid(self):
        """Test that all presets return valid SimulationSettings."""
        for preset_name in get_preset_names():
            config = get_preset_config(preset_name)

            # Should be valid SimulationSettings
            assert isinstance(config, SimulationSettings)

            # Should have basic required fields
            assert config.date is not None
            assert config.start_time is not None
            assert config.end_time is not None
            assert config.starting_cash > 0

            # Agent counts should be non-negative
            assert config.agents.exchange.num_agents >= 0
            assert config.agents.noise.num_agents >= 0
            assert config.agents.value.num_agents >= 0
            assert config.agents.adaptive_market_maker.num_agents >= 0
            assert config.agents.momentum.num_agents >= 0

    def test_all_presets_have_consistent_time_settings(self):
        """Test that all presets have consistent time settings."""
        for preset_name in get_preset_names():
            config = get_preset_config(preset_name)

            # All should use the same time window for comparison
            assert config.start_time == "09:30:00"
            assert config.end_time == "10:00:00"

    def test_presets_have_safe_parameter_values(self):
        """Test that presets don't have extreme/unsafe values."""
        for preset_name in get_preset_names():
            config = get_preset_config(preset_name)

            # Check oracle parameters are reasonable
            assert 0 < config.agents.oracle.fund_vol < 1e-2
            assert 0 < config.agents.oracle.kappa < 1e-10
            assert config.agents.oracle.sigma_s >= 0  # Can be 0 or positive
            assert config.agents.oracle.sigma_s < 1e6

            # Check agent counts are reasonable (not billions)
            assert config.agents.noise.num_agents < 10000
            assert config.agents.value.num_agents < 1000
            assert config.agents.adaptive_market_maker.num_agents < 100
            assert config.agents.momentum.num_agents < 200

            # Check starting cash is reasonable
            assert 1_000_000 <= config.starting_cash <= 100_000_000

    def test_preset_names_are_unique(self):
        """Test that all preset names are unique."""
        names = get_preset_names()
        assert len(names) == len(set(names)), "Preset names must be unique"

    def test_preset_configurations_are_independent(self):
        """Test that modifying one preset doesn't affect others."""
        # Get two different presets
        config1 = get_preset_config("Default (Balanced Market)")
        get_preset_config("High Volatility")

        # Modify config1
        config1.agents.noise.num_agents = 99999

        # Get the presets again
        config1_fresh = get_preset_config("Default (Balanced Market)")
        config2_fresh = get_preset_config("High Volatility")

        # Changes should not persist
        assert config1_fresh.agents.noise.num_agents != 99999
        assert config2_fresh.agents.noise.num_agents == 1500  # Original value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
