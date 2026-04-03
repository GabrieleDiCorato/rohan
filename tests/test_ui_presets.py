"""Tests for UI preset configurations.

This module tests:
- Preset name retrieval
- Preset configuration loading
- Template-based presets build valid hasufel configs
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
        # Should return default config (rmsc04 template)
        assert config.date == "20260130"
        assert config.start_time == "09:30:00"
        assert config.template == "rmsc04"

    def test_default_preset_configuration(self):
        """Test Default (Balanced Market) preset configuration."""
        config = get_preset_config("Default (Balanced Market)")

        assert config.template == "rmsc04"
        assert config.date == "20260130"
        assert config.start_time == "09:30:00"
        assert config.end_time == "11:30:00"
        assert config.starting_cash == 10_000_000
        assert config.log_orders is True

    def test_high_volatility_preset_uses_template(self):
        """Test High Volatility preset uses volatile_day template."""
        config = get_preset_config("High Volatility")
        assert config.template == "volatile_day"

    def test_low_liquidity_preset_uses_template(self):
        """Test Low Liquidity preset uses low_liquidity template."""
        config = get_preset_config("Low Liquidity")
        assert config.template == "low_liquidity"

    def test_market_maker_stress_preset_uses_template(self):
        """Test Market Maker Stress Test preset uses stress_test template."""
        config = get_preset_config("Market Maker Stress Test")
        assert config.template == "stress_test"

    def test_momentum_dominated_preset_uses_template(self):
        """Test Momentum Dominated preset uses trending_day template."""
        config = get_preset_config("Momentum Dominated")
        assert config.template == "trending_day"

    def test_all_presets_are_valid(self):
        """Test that all presets return valid SimulationSettings."""
        for preset_name in get_preset_names():
            config = get_preset_config(preset_name)

            assert isinstance(config, SimulationSettings)
            assert config.date is not None
            assert config.start_time is not None
            assert config.end_time is not None
            assert config.starting_cash > 0
            assert config.template is not None

    def test_all_presets_have_consistent_time_settings(self):
        """Test that all presets have consistent time settings."""
        for preset_name in get_preset_names():
            config = get_preset_config(preset_name)

            assert config.start_time == "09:30:00"
            assert config.end_time == "11:30:00"

    def test_all_presets_build_valid_hasufel_config(self):
        """Test that all template-based presets produce valid hasufel configs."""
        from rohan.simulation.abides_impl.config_builder import create_simulation_builder

        for preset_name in get_preset_names():
            config = get_preset_config(preset_name)
            builder = create_simulation_builder(config)
            built = builder.build()
            assert built is not None, f"Preset '{preset_name}' failed to build"

    def test_preset_names_are_unique(self):
        """Test that all preset names are unique."""
        names = get_preset_names()
        assert len(names) == len(set(names)), "Preset names must be unique"

    def test_preset_configurations_are_independent(self):
        """Test that getting presets twice returns independent objects."""
        config1 = get_preset_config("Default (Balanced Market)")
        config2 = get_preset_config("Default (Balanced Market)")

        # Should be equal but not the same object
        assert config1.template == config2.template
        assert config1 is not config2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
