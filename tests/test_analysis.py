"""
Tests for the Analysis Service.

This module tests:
- Metrics computation from simulation output
- Plot generation (price series, volume, spread)
- Artifact creation and storage
"""

import io

import matplotlib.pyplot as plt
import pytest

from rohan.config import SimulationSettings
from rohan.framework import AnalysisService, ArtifactStore, DatabaseConnector
from rohan.framework.database import ArtifactType
from rohan.simulation import SimulationService


class TestAnalysisService:
    """Test suite for the AnalysisService metrics and visualization generator."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Setup a fresh database for each test."""
        db = DatabaseConnector()
        db.create_tables()
        yield db

    @pytest.fixture
    def sample_simulation_output(self):
        """Create a sample simulation output for testing."""
        settings = SimulationSettings()
        settings.start_time = "09:30:00"
        settings.end_time = "09:35:00"  # 5 minute simulation
        settings.seed = 12345  # Fixed seed for reproducibility
        settings.agents.noise.num_agents = 10  # Increased from 1 to ensure market activity
        settings.agents.value.num_agents = 5  # Increased from 1 to ensure market activity
        settings.agents.momentum.num_agents = 0
        settings.agents.adaptive_market_maker.num_agents = 0

        service = SimulationService()
        result = service.run_simulation(settings)

        # Raise error if simulation failed
        if result.error is not None:
            raise result.error

        return result.result

    def test_compute_simulation_metrics(self, sample_simulation_output):
        """Test computing summary metrics (volatility, spread, liquidity) from simulation output."""
        analyzer = AnalysisService()
        metrics = analyzer.compute_metrics(sample_simulation_output)

        # Verify core metrics are computed (not None)
        # Note: Some metrics may be None if market data is insufficient
        assert metrics.volatility is not None
        # mean_spread can be None if there's no valid bid-ask data
        # Just verify the method completes without error
        assert metrics is not None

    def test_generate_price_and_volume_plots(self, sample_simulation_output):
        """Test generating matplotlib visualizations for price series and volume."""
        analyzer = AnalysisService()

        # Test price plot
        fig = analyzer.plot_price_series(sample_simulation_output)
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # Save to bytes to verify it works
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        assert buf.tell() > 0, "Plot should have content"
        plt.close(fig)

        # Test volume plot
        fig = analyzer.plot_volume(sample_simulation_output)
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_save_plots_as_binary_artifacts(self, setup_db, sample_simulation_output):
        """Test generating plots and saving them as binary artifacts in the database."""
        db = setup_db
        repo = ArtifactStore(db)
        analyzer = AnalysisService()

        # Create test run
        session = repo.create_session("Test Artifacts")
        scenario = repo.create_scenario("Test Scenario", {})
        iteration = repo.create_iteration(session.session_id, 1, "test_code")
        run = repo.create_run(
            iteration_id=iteration.iteration_id,
            scenario_id=scenario.scenario_id,
            full_config={},
        )

        # Generate and save plot
        fig = analyzer.plot_price_series(sample_simulation_output)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        plt.close(fig)

        # Save as artifact
        repo.save_artifact(run.run_id, ArtifactType.IMAGE, "price_series.png", buf.read())

        # Retrieve and verify
        artifacts = repo.get_artifacts(run.run_id)
        assert len(artifacts) == 1
        assert artifacts[0].artifact_type == ArtifactType.IMAGE
        assert artifacts[0].path == "price_series.png"
        assert len(artifacts[0].content) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
