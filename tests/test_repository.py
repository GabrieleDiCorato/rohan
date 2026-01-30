"""
Tests for database models, schemas, and repository layer.

This module tests:
- Pydantic schema validation
- SQLAlchemy model operations
- Database connection and initialization
- Repository CRUD operations
- Data persistence and retrieval
"""

from datetime import datetime

import pandas as pd
import pytest

from rohan.framework import ArtifactStore, MarketState, OrderAction, SimulationMetrics
from rohan.framework.database import (
    DatabaseConnector,
)


class TestDatabaseAndRepository:
    """Test suite for database models, schemas, and repository operations."""

    @pytest.fixture(autouse=True)
    def setup_db(self):
        """Setup a fresh database for each test."""
        db = DatabaseConnector()
        db.create_tables()
        yield db
        # Cleanup would go here if needed

    def test_schema_validation(self):
        """Test Pydantic schema validation for market state, orders, and metrics."""
        # Test MarketState
        market_state = MarketState(
            timestamp=1000000000,
            best_bid=100.0,
            best_ask=101.0,
            last_trade_price=100.5,
            inventory=10,
            cash=10000.0,
        )
        assert market_state.timestamp == 1000000000
        assert market_state.best_bid == 100.0

        # Test OrderAction
        order = OrderAction(side="BUY", type="LIMIT", price=100.0, quantity=10)
        assert order.side == "BUY"
        assert order.type == "LIMIT"

        # Test SimulationMetrics
        metrics = SimulationMetrics(
            total_pnl=1000.0,
            sharpe_ratio=1.5,
            max_drawdown=0.1,
            win_rate=0.6,
            volatility=0.2,
            traded_volume=500,
        )
        assert metrics.total_pnl == 1000.0
        assert metrics.sharpe_ratio == 1.5

    def test_create_session_and_scenarios(self, setup_db):
        """Test creating strategy sessions, scenarios, iterations, and simulation runs."""
        repo = ArtifactStore(setup_db)

        # Create a strategy session
        session = repo.create_session("Test market making strategy")
        assert session.session_id is not None
        assert session.goal_description == "Test market making strategy"

        # Create a scenario
        scenario = repo.create_scenario(name="High Volatility", config_override={"noise_agents": 1000})
        assert scenario.scenario_id is not None
        assert scenario.name == "High Volatility"

        # Create an iteration
        iteration = repo.create_iteration(
            session_id=session.session_id,
            generation_number=1,
            code="class MyAgent: pass",
            reasoning="Initial implementation",
        )
        assert iteration.iteration_id is not None
        assert iteration.generation_number == 1

        # Create a run
        run = repo.create_run(
            iteration_id=iteration.iteration_id,
            scenario_id=scenario.scenario_id,
            full_config={"test": "config"},
            status="PENDING",
        )
        assert run.run_id is not None
        assert run.status == "PENDING"

    def test_save_and_retrieve_market_data(self, setup_db):
        """Test bulk insert and retrieval of L1 market data."""
        repo = ArtifactStore(setup_db)

        # Create necessary objects
        session = repo.create_session("Test")
        scenario = repo.create_scenario("Test Scenario", {})
        iteration = repo.create_iteration(session.session_id, 1, "code")
        run = repo.create_run(iteration.iteration_id, scenario.scenario_id, {})

        # Create sample market data
        market_data = pd.DataFrame(
            {
                "time": [1000000000, 1000000001, 1000000002],
                "bid_price": [100.0, 100.1, 100.2],
                "bid_qty": [10, 15, 20],
                "ask_price": [101.0, 101.1, 101.2],
                "ask_qty": [10, 15, 20],
                "timestamp": [datetime.now(), datetime.now(), datetime.now()],
            }
        )

        # Save market data
        repo.save_market_data(run.run_id, market_data)

        # Retrieve and verify
        retrieved_data = repo.get_market_data(run.run_id)
        assert len(retrieved_data) == 3
        assert retrieved_data["bid_price"].tolist() == [100.0, 100.1, 100.2]

    def test_save_and_retrieve_agent_logs(self, setup_db):
        """Test bulk insert and retrieval of agent event logs."""
        repo = ArtifactStore(setup_db)

        # Create necessary objects
        session = repo.create_session("Test")
        scenario = repo.create_scenario("Test Scenario", {})
        iteration = repo.create_iteration(session.session_id, 1, "code")
        run = repo.create_run(iteration.iteration_id, scenario.scenario_id, {})

        # Create sample logs
        logs_df = pd.DataFrame(
            {
                "AgentID": [1, 2, 3],
                "AgentType": ["ValueAgent", "NoiseAgent", "ValueAgent"],
                "EventType": ["ORDER_SUBMITTED", "ORDER_EXECUTED", "ORDER_CANCELLED"],
                "time_placed": [datetime.now(), datetime.now(), datetime.now()],
                "extra_field": ["data1", "data2", "data3"],
            }
        )

        # Save logs
        repo.save_logs(run.run_id, logs_df)

        # Retrieve and verify
        retrieved_logs = repo.get_logs(run.run_id)
        assert len(retrieved_logs) == 3
        assert retrieved_logs["agent_type"].tolist() == [
            "ValueAgent",
            "NoiseAgent",
            "ValueAgent",
        ]

    def test_save_and_retrieve_artifacts(self, setup_db):
        """Test storing and retrieving binary artifacts like plots and files."""
        repo = ArtifactStore(setup_db)

        # Create necessary objects
        session = repo.create_session("Test")
        scenario = repo.create_scenario("Test Scenario", {})
        iteration = repo.create_iteration(session.session_id, 1, "code")
        run = repo.create_run(iteration.iteration_id, scenario.scenario_id, {})

        # Save an artifact (e.g., a plot image)
        test_content = b"fake image data"
        repo.save_artifact(run.run_id, "IMAGE", "pnl_plot.png", test_content)

        # Retrieve and verify
        artifacts = repo.get_artifacts(run.run_id)
        assert len(artifacts) == 1
        assert artifacts[0].type == "IMAGE"
        assert artifacts[0].path == "pnl_plot.png"
        assert artifacts[0].content == test_content

    def test_update_simulation_run_status(self, setup_db):
        """Test updating simulation run status and storing computed metrics."""
        repo = ArtifactStore(setup_db)

        # Create necessary objects
        session = repo.create_session("Test")
        scenario = repo.create_scenario("Test Scenario", {})
        iteration = repo.create_iteration(session.session_id, 1, "code")
        run = repo.create_run(iteration.iteration_id, scenario.scenario_id, {})

        # Update status
        metrics = {"sharpe": 1.5, "pnl": 5000.0}
        repo.update_run_status(run.run_id, "COMPLETED", metrics)

        # Retrieve and verify
        updated_run = repo.get_run(run.run_id)
        assert updated_run is not None
        assert updated_run.status == "COMPLETED"
        assert updated_run.metrics_summary["sharpe"] == 1.5
        assert updated_run.metrics_summary["pnl"] == 5000.0

    def test_database_connection(self, setup_db):
        """Test that database connection works correctly."""
        db = setup_db
        assert db.engine is not None

        # Test session creation
        session = db.get_session()
        assert session is not None
        session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
