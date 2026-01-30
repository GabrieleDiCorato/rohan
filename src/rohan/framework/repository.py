from typing import Any
from uuid import UUID

import pandas as pd
from sqlalchemy import insert, select

from rohan.framework.database import (
    AgentLog,
    Artifact,
    DatabaseConnector,
    MarketDataL1,
    SimulationRun,
    SimulationScenario,
    StrategyIteration,
    StrategySession,
)


class ArtifactStore:
    """
    Repository for storing simulation artifacts (Runs, Market Data, Logs).
    """

    def __init__(self, db: DatabaseConnector | None = None):
        self.db = db or DatabaseConnector()

    def save_run(self, run: SimulationRun) -> None:
        """
        Save a SimulationRun to the database.
        """
        with self.db.get_session() as session:
            session.add(run)
            session.commit()
            session.refresh(run)

    def save_market_data(self, run_id: UUID, data: pd.DataFrame) -> None:
        """
        Bulk insert Level 1 Market Data.
        Expects DataFrame columns matching MarketDataL1 model:
        ['time', 'bid_price', 'bid_qty', 'ask_price', 'ask_qty', 'timestamp']
        """
        if data.empty:
            return

        # Handle NaN values that can occur from outer merge of bids/asks
        # Forward-fill prices and quantities
        data = data.copy()
        data["bid_price"] = data["bid_price"].ffill().bfill()
        data["bid_qty"] = data["bid_qty"].ffill().bfill().fillna(0).astype(int)
        data["ask_price"] = data["ask_price"].ffill().bfill()
        data["ask_qty"] = data["ask_qty"].ffill().bfill().fillna(0).astype(int)

        # Drop any remaining rows with NaN (shouldn't happen after ffill/bfill)
        data = data.dropna(subset=["bid_price", "ask_price"])

        if data.empty:
            return

        # Prepare data for bulk insert
        records = data.to_dict(orient="records")
        # Add run_id to each record
        for record in records:
            record["run_id"] = run_id

        # Use Core Insert for performance
        with self.db.get_session() as session:
            session.execute(insert(MarketDataL1), records)
            session.commit()

    def _sanitize_for_json(self, data: Any) -> Any:
        """Recursively convert non-JSON-serializable objects to strings."""
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_for_json(v) for v in data]
        if isinstance(data, str | int | float | bool | type(None)):
            return data
        return str(data)

    def save_logs(self, run_id: UUID, logs: pd.DataFrame) -> None:
        """
        Bulk insert Agent Logs.
        Expects DataFrame with columns:
        ['AgentID', 'AgentType', 'EventType', 'time_placed', ...] (and other log fields)
        We map these to the AgentLog model.
        """
        if logs.empty:
            return

        # Map DataFrame columns to AgentLog model fields
        # Provided logs_df from AbidesOutput usually has:
        # AgentID, AgentType, EventType, time_placed, ...
        # and other columns that go into log_json.

        # We need to extract the specific columns we extract as structured data
        # and dump the rest into 'log_json'.

        structured_cols = ["AgentID", "AgentType", "EventType", "time_placed"]

        records = []
        for _, row in logs.iterrows():
            # Extract agent_id - handle various cases
            agent_id = row.get("AgentID")
            if pd.isna(agent_id) and "agent_id" in row:
                agent_id = row.get("agent_id")
            if pd.isna(agent_id):
                # Skip rows without agent ID
                continue

            # Extract agent_type
            agent_type = row.get("AgentType", "Unknown")
            if pd.isna(agent_type) and "agent_type" in row:
                agent_type = row.get("agent_type", "Unknown")

            # Extract event_type
            event_type = row.get("EventType", "Unknown")

            # Extract time_placed
            time_placed = row.get("time_placed")
            if pd.isna(time_placed):
                time_placed = None

            log_item = row.drop(labels=structured_cols, errors="ignore").to_dict()
            log_json = self._sanitize_for_json(log_item)

            record = {
                "run_id": run_id,
                "agent_id": int(agent_id),
                "agent_type": str(agent_type),
                "event_type": str(event_type),
                "time_placed": time_placed,
                "log_json": log_json,
            }
            records.append(record)

        if not records:
            return

        with self.db.get_session() as session:
            session.execute(insert(AgentLog), records)
            session.commit()

    def save_artifact(self, run_id: UUID, artifact_type: str, path: str, content: bytes) -> None:
        """
        Save an artifact (plot image, log file, etc.) for a simulation run.

        Args:
            run_id: The simulation run ID
            artifact_type: Type of artifact ('IMAGE', 'LOG_FILE', etc.)
            path: Virtual path/name for the artifact
            content: Binary content of the artifact
        """
        artifact = Artifact(run_id=run_id, type=artifact_type, path=path, content=content)
        with self.db.get_session() as session:
            session.add(artifact)
            session.commit()

    def get_run(self, run_id: UUID) -> SimulationRun | None:
        """Retrieve a simulation run by ID."""
        with self.db.get_session() as session:
            result = session.execute(select(SimulationRun).where(SimulationRun.run_id == run_id))
            run: SimulationRun | None = result.scalar_one_or_none()
            return run

    def get_market_data(self, run_id: UUID) -> pd.DataFrame:
        """Retrieve market data for a simulation run as a DataFrame."""
        with self.db.get_session() as session:
            result = session.execute(select(MarketDataL1).where(MarketDataL1.run_id == run_id))
            data = result.scalars().all()
            if not data:
                return pd.DataFrame()
            return pd.DataFrame([vars(d) for d in data])

    def get_logs(self, run_id: UUID) -> pd.DataFrame:
        """Retrieve agent logs for a simulation run as a DataFrame."""
        with self.db.get_session() as session:
            result = session.execute(select(AgentLog).where(AgentLog.run_id == run_id))
            logs = result.scalars().all()
            if not logs:
                return pd.DataFrame()
            return pd.DataFrame([vars(log) for log in logs])

    def get_artifacts(self, run_id: UUID) -> list[Artifact]:
        """Retrieve all artifacts for a simulation run."""
        with self.db.get_session() as session:
            result = session.execute(select(Artifact).where(Artifact.run_id == run_id))
            return list(result.scalars().all())

    def get_session_runs(self, session_id: UUID) -> list[SimulationRun]:
        """Retrieve all simulation runs for a strategy session."""
        with self.db.get_session() as session:
            result = session.execute(select(SimulationRun).join(StrategyIteration).where(StrategyIteration.session_id == session_id))
            return list(result.scalars().all())

    def update_run_status(self, run_id: UUID, status: str, metrics: dict | None = None) -> None:
        """Update the status and optionally metrics of a simulation run."""
        with self.db.get_session() as session:
            run = session.get(SimulationRun, run_id)
            if run:
                run.status = status
                if metrics is not None:
                    run.metrics_summary = metrics
                session.commit()

    def create_session(self, goal_description: str) -> StrategySession:
        """Create a new strategy session."""
        session_obj = StrategySession(goal_description=goal_description)
        with self.db.get_session() as session:
            session.add(session_obj)
            session.commit()
            session.refresh(session_obj)
        return session_obj

    def create_scenario(self, name: str, config_override: dict, session_id: UUID | None = None) -> SimulationScenario:
        """Create a new simulation scenario."""
        scenario = SimulationScenario(name=name, config_override=config_override, session_id=session_id)
        with self.db.get_session() as session:
            session.add(scenario)
            session.commit()
            session.refresh(scenario)
        return scenario

    def create_iteration(self, session_id: UUID, generation_number: int, code: str, reasoning: str = "") -> StrategyIteration:
        """Create a new strategy iteration."""
        iteration = StrategyIteration(
            session_id=session_id,
            generation_number=generation_number,
            code=code,
            reasoning=reasoning,
        )
        with self.db.get_session() as session:
            session.add(iteration)
            session.commit()
            session.refresh(iteration)
        return iteration

    def create_run(
        self,
        iteration_id: UUID,
        scenario_id: UUID,
        full_config: dict,
        status: str = "PENDING",
    ) -> SimulationRun:
        """Create a new simulation run."""
        run = SimulationRun(
            iteration_id=iteration_id,
            scenario_id=scenario_id,
            full_config=full_config,
            status=status,
        )
        with self.db.get_session() as session:
            session.add(run)
            session.commit()
            session.refresh(run)
        return run
