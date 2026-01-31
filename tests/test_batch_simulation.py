"""
Tests for batch simulation functionality.

This module tests:
- Running multiple simulations with different settings
- SimulationResult structure and metadata
- Error handling in batch mode
- Builder pattern for creating setting variations
"""

import pytest

from rohan.config import SimulationSettings
from rohan.simulation import SimulationService


class TestBatchSimulation:
    """Test suite for batch simulation functionality."""

    def test_run_batch_with_different_seeds(self):
        """Test running multiple simulations with different seeds."""
        service = SimulationService()

        # Create settings with different seeds
        base_settings = SimulationSettings(
            start_time="09:30:00",
            end_time="09:35:00",
            stdout_log_level="OFF",
        )
        base_settings.agents.noise.num_agents = 5
        base_settings.agents.value.num_agents = 2
        base_settings.agents.momentum.num_agents = 0
        base_settings.agents.adaptive_market_maker.num_agents = 0

        # Create variations with different seeds
        settings_list = []
        for seed in [42, 43, 44]:
            settings = base_settings.model_copy(deep=True)
            settings.seed = seed
            settings_list.append(settings)

        # Run batch
        results = service.run_batch(settings_list)

        # Verify all simulations completed
        assert len(results) == 3
        assert all(r.error is None for r in results)
        assert all(r.result is not None for r in results)
        assert all(r.error is None for r in results)

        # Verify each has unique run_id
        run_ids = [r.context.run_id for r in results]
        assert len(set(run_ids)) == 3

        # Verify duration is tracked
        assert all(r.duration_seconds > 0 for r in results)

    def test_simulation_result_structure(self):
        """Test that SimulationResult contains expected fields."""
        service = SimulationService()

        settings = SimulationSettings()
        settings.start_time = "09:30:00"
        settings.end_time = "09:35:00"
        settings.stdout_log_level = "OFF"
        settings.agents.noise.num_agents = 3
        settings.agents.value.num_agents = 1
        settings.agents.momentum.num_agents = 0
        settings.agents.adaptive_market_maker.num_agents = 0

        results = service.run_batch([settings])

        assert len(results) == 1
        result = results[0]

        # Verify structure
        assert result.context.run_id is not None
        assert result.context.settings == settings
        assert result.result is not None
        assert result.duration_seconds > 0
        assert result.error is None
        assert result.context is not None

    def test_settings_builder_variations(self):
        """Test creating setting variations."""
        base_settings = SimulationSettings(
            start_time="09:30:00",
            end_time="09:35:00",
            date="20260131",
            stdout_log_level="OFF",
        )

        # Create variations with different agent counts
        variations = []
        for num_agents in [5, 10, 15]:
            settings = base_settings.model_copy(deep=True)
            settings.agents.noise.num_agents = num_agents
            variations.append(settings)

        assert len(variations) == 3
        assert variations[0].agents.noise.num_agents == 5
        assert variations[1].agents.noise.num_agents == 10
        assert variations[2].agents.noise.num_agents == 15

        # All should have the same base settings
        assert all(v.start_time == "09:30:00" for v in variations)
        assert all(v.end_time == "09:35:00" for v in variations)
        assert all(v.date == "20260131" for v in variations)

    def test_settings_builder_chaining(self):
        """Test creating settings with multiple parameters."""
        settings = SimulationSettings(
            seed=42,
            start_time="09:30:00",
            end_time="10:00:00",
            date="20260201",
            stdout_log_level="INFO",
        )
        settings.agents.noise.num_agents = 10
        settings.agents.value.num_agents = 5
        settings.agents.momentum.num_agents = 2
        settings.agents.adaptive_market_maker.num_agents = 1

        assert settings.seed == 42
        assert settings.start_time == "09:30:00"
        assert settings.end_time == "10:00:00"
        assert settings.date == "20260201"
        assert settings.stdout_log_level == "INFO"
        assert settings.agents.noise.num_agents == 10
        assert settings.agents.value.num_agents == 5
        assert settings.agents.momentum.num_agents == 2
        assert settings.agents.adaptive_market_maker.num_agents == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
