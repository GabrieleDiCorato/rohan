"""
Tests for the Simulation Engine.

This module tests:
- End-to-end simulation execution
- Result persistence to database
- Metrics computation and storage
- Error handling and status updates
"""

import pytest

from rohan.config import SimulationSettings
from rohan.framework import ArtifactStore, DatabaseConnector, SimulationEngine


class TestSimulationEngine:
    """Test suite for the SimulationEngine orchestrator."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Setup a fresh database for each test."""
        db = DatabaseConnector()
        db.create_tables()
        yield db

    def test_run_simulation_and_persist_results(self, setup_db):
        """Test running a complete simulation and persisting all results to database."""
        db = setup_db
        repo = ArtifactStore(db)
        engine = SimulationEngine(db)

        # Create test data
        session = repo.create_session("Test Engine Run")
        scenario = repo.create_scenario("Test Scenario", {})
        iteration = repo.create_iteration(session.session_id, 1, "test_code")

        # Create minimal simulation settings
        settings = SimulationSettings()
        settings.start_time = "09:30:00"
        settings.end_time = "09:30:30"  # 30 second simulation
        settings.agents.noise.num_agents = 2
        settings.agents.value.num_agents = 1
        settings.agents.momentum.num_agents = 0
        settings.agents.adaptive_market_maker.num_agents = 0

        # Create a run
        run = repo.create_run(
            iteration_id=iteration.iteration_id,
            scenario_id=scenario.scenario_id,
            full_config=settings.model_dump(),
            status="PENDING",
        )

        # Execute the simulation
        engine.run_local(run.run_id, settings)

        # Verify results were saved
        updated_run = repo.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.status == "COMPLETED"
        assert updated_run.metrics_summary is not None

        # Verify market data was saved (may be empty for very short simulations)
        market_data = repo.get_market_data(run.run_id)
        assert market_data is not None
        # For very short simulations, market data might be empty - that's OK

        # Verify logs were saved
        logs = repo.get_logs(run.run_id)
        assert len(logs) > 0, "Agent logs should be saved"

        # Verify metrics contain expected fields
        metrics = updated_run.metrics_summary
        assert "volatility" in metrics
        assert "custom_metrics" in metrics

    def test_simulation_failure_handling(self, setup_db):
        """Test that the engine properly handles and records simulation failures."""
        db = setup_db
        repo = ArtifactStore(db)
        engine = SimulationEngine(db)

        # Create test data
        session = repo.create_session("Test Engine Failure")
        scenario = repo.create_scenario("Bad Scenario", {})
        iteration = repo.create_iteration(session.session_id, 1, "test_code")

        # Create settings that will cause an exception during analysis
        settings = SimulationSettings()
        settings.start_time = "09:30:00"
        settings.end_time = "09:30:01"

        run = repo.create_run(
            iteration_id=iteration.iteration_id,
            scenario_id=scenario.scenario_id,
            full_config=settings.model_dump(),
            status="PENDING",
        )

        # Mock a failure by passing None - this should raise an exception
        raised = False
        try:
            engine.run_local(run.run_id, None)  # type: ignore
        except Exception:
            raised = True
        if not raised:
            pytest.fail("Expected exception was not raised during simulation run.")

        # Verify run status was updated to FAILED
        updated_run = repo.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.status == "FAILED"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
