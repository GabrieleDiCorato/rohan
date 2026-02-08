"""Repository layer for persisting simulation artifacts.

Handles CRUD operations for sessions, scenarios, iterations, runs,
market data, agent logs, and artifacts (DB or filesystem backed).
"""

import logging
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pandas as pd
from sqlalchemy import insert, select

from rohan.framework.database import (
    AgentLog,
    Artifact,
    ArtifactType,
    DatabaseConnector,
    MarketDataL1,
    RunStatus,
    SimulationRun,
    SimulationScenario,
    StrategyIteration,
    StrategySession,
)

logger = logging.getLogger(__name__)


class ArtifactStore:
    """
    Repository for storing simulation artifacts (Runs, Market Data, Logs).

    Supports two artifact storage backends:
    - ``"database"``: content stored in the DB ``LargeBinary`` column (default).
    - ``"filesystem"``: content written to *artifact_root/<run_id>/<path>*.
    """

    def __init__(
        self,
        db: DatabaseConnector | None = None,
        artifact_root: str | Path | None = None,
    ):
        self.db = db or DatabaseConnector()
        self._artifact_root = Path(artifact_root) if artifact_root else None

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def save_run(self, run: SimulationRun) -> None:
        """Save a SimulationRun to the database."""
        session = self.db.get_session()
        try:
            session.add(run)
            session.commit()
            session.refresh(run)
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def save_market_data(self, run_id: UUID, data: pd.DataFrame) -> None:
        """Bulk insert Level 1 Market Data."""
        if data.empty:
            return

        data = data.copy()
        data["bid_price"] = data["bid_price"].ffill().bfill()
        data["bid_qty"] = data["bid_qty"].ffill().bfill().fillna(0).astype(int)
        data["ask_price"] = data["ask_price"].ffill().bfill()
        data["ask_qty"] = data["ask_qty"].ffill().bfill().fillna(0).astype(int)
        data = data.dropna(subset=["bid_price", "ask_price"])

        if data.empty:
            return

        records = data.to_dict(orient="records")
        for record in records:
            record["run_id"] = run_id

        session = self.db.get_session()
        try:
            session.execute(insert(MarketDataL1), records)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Agent logs
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_for_json(data: Any) -> Any:
        """Recursively convert non-JSON-serializable objects to strings."""
        if isinstance(data, dict):
            return {k: ArtifactStore._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [ArtifactStore._sanitize_for_json(v) for v in data]
        if isinstance(data, str | int | float | bool | type(None)):
            return data
        return str(data)

    def save_logs(self, run_id: UUID, logs: pd.DataFrame) -> None:
        """Bulk insert Agent Logs."""
        if logs.empty:
            return

        structured_cols = ["AgentID", "AgentType", "EventType", "time_placed"]

        records = []
        for _, row in logs.iterrows():
            agent_id = row.get("AgentID")
            if pd.isna(agent_id) and "agent_id" in row:
                agent_id = row.get("agent_id")
            if pd.isna(agent_id):
                continue

            agent_type = row.get("AgentType", "Unknown")
            if pd.isna(agent_type) and "agent_type" in row:
                agent_type = row.get("agent_type", "Unknown")

            event_type = row.get("EventType", "Unknown")

            time_placed = row.get("time_placed")
            if pd.isna(time_placed):
                time_placed = None
            elif isinstance(time_placed, str):
                try:
                    time_placed = pd.to_datetime(time_placed).to_pydatetime()
                except Exception:
                    time_placed = None
            elif isinstance(time_placed, pd.Timestamp):
                time_placed = time_placed.to_pydatetime()

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

        session = self.db.get_session()
        try:
            session.execute(insert(AgentLog), records)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Artifacts (dual backend)
    # ------------------------------------------------------------------

    def save_artifact(
        self,
        run_id: UUID,
        artifact_type: ArtifactType | str,
        path: str,
        content: bytes,
    ) -> None:
        """Save an artifact for a simulation run.

        If ``artifact_root`` was provided at construction time, content is
        written to a file under ``<artifact_root>/<run_id>/<path>`` and only
        the metadata row is persisted in the DB.  Otherwise the bytes are
        stored directly in the ``content`` column.
        """
        if isinstance(artifact_type, str):
            artifact_type = ArtifactType(artifact_type)

        if self._artifact_root is not None:
            # Filesystem-backed storage
            dest = self._artifact_root / str(run_id) / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
            logger.debug("Artifact written to %s", dest)
            db_content = None
            backend = "filesystem"
        else:
            db_content = content
            backend = "database"

        artifact = Artifact(
            run_id=run_id,
            artifact_type=artifact_type,
            path=path,
            content=db_content,
            storage_backend=backend,
        )
        session = self.db.get_session()
        try:
            session.add(artifact)
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()

    def load_artifact_content(self, artifact: Artifact) -> bytes | None:
        """Load the binary content of an artifact, regardless of backend."""
        if artifact.storage_backend == "filesystem" and self._artifact_root is not None:
            fs_path = self._artifact_root / str(artifact.run_id) / artifact.path
            if fs_path.exists():
                return cast(bytes, fs_path.read_bytes())
            logger.warning("Artifact file not found: %s", fs_path)
            return None
        return cast(bytes | None, artifact.content)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_run(self, run_id: UUID) -> SimulationRun | None:
        """Retrieve a simulation run by ID."""
        session = self.db.get_session()
        try:
            result = session.execute(select(SimulationRun).where(SimulationRun.run_id == run_id))
            return cast(SimulationRun | None, result.scalar_one_or_none())
        finally:
            self.db.remove_session()

    def get_market_data(self, run_id: UUID) -> pd.DataFrame:
        """Retrieve market data for a simulation run as a DataFrame."""
        session = self.db.get_session()
        try:
            result = session.execute(select(MarketDataL1).where(MarketDataL1.run_id == run_id))
            data = result.scalars().all()
            if not data:
                return pd.DataFrame()
            return pd.DataFrame([vars(d) for d in data])
        finally:
            self.db.remove_session()

    def get_logs(self, run_id: UUID) -> pd.DataFrame:
        """Retrieve agent logs for a simulation run as a DataFrame."""
        session = self.db.get_session()
        try:
            result = session.execute(select(AgentLog).where(AgentLog.run_id == run_id))
            logs = result.scalars().all()
            if not logs:
                return pd.DataFrame()
            return pd.DataFrame([vars(log) for log in logs])
        finally:
            self.db.remove_session()

    def get_artifacts(self, run_id: UUID) -> list[Artifact]:
        """Retrieve all artifacts for a simulation run."""
        session = self.db.get_session()
        try:
            result = session.execute(select(Artifact).where(Artifact.run_id == run_id))
            return list(result.scalars().all())
        finally:
            self.db.remove_session()

    def get_session_runs(self, session_id: UUID) -> list[SimulationRun]:
        """Retrieve all simulation runs for a strategy session."""
        session = self.db.get_session()
        try:
            result = session.execute(select(SimulationRun).join(StrategyIteration).where(StrategyIteration.session_id == session_id))
            return list(result.scalars().all())
        finally:
            self.db.remove_session()

    def update_run_status(
        self,
        run_id: UUID,
        status: RunStatus | str,
        metrics: dict | None = None,
        error_message: str | None = None,
        error_traceback: str | None = None,
    ) -> None:
        """Update the status (and optionally metrics/error info) of a simulation run."""
        if isinstance(status, str):
            status = RunStatus(status)

        session = self.db.get_session()
        try:
            run = session.get(SimulationRun, run_id)
            if run:
                run.status = status
                if metrics is not None:
                    run.metrics_summary = metrics
                if error_message is not None:
                    run.error_message = error_message
                if error_traceback is not None:
                    run.error_traceback = error_traceback
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    def create_session(self, goal_description: str) -> StrategySession:
        """Create a new strategy session."""
        session_obj = StrategySession(goal_description=goal_description)
        session = self.db.get_session()
        try:
            session.add(session_obj)
            session.commit()
            session.refresh(session_obj)
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()
        return session_obj

    def create_scenario(self, name: str, config_override: dict, session_id: UUID | None = None) -> SimulationScenario:
        """Create a new simulation scenario."""
        scenario = SimulationScenario(name=name, config_override=config_override, session_id=session_id)
        session = self.db.get_session()
        try:
            session.add(scenario)
            session.commit()
            session.refresh(scenario)
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()
        return scenario

    def create_iteration(self, session_id: UUID, generation_number: int, code: str, reasoning: str = "") -> StrategyIteration:
        """Create a new strategy iteration."""
        iteration = StrategyIteration(
            session_id=session_id,
            generation_number=generation_number,
            code=code,
            reasoning=reasoning,
        )
        session = self.db.get_session()
        try:
            session.add(iteration)
            session.commit()
            session.refresh(iteration)
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()
        return iteration

    def create_run(
        self,
        iteration_id: UUID,
        scenario_id: UUID,
        full_config: dict,
        status: RunStatus | str = RunStatus.PENDING,
    ) -> SimulationRun:
        """Create a new simulation run."""
        if isinstance(status, str):
            status = RunStatus(status)

        run = SimulationRun(
            iteration_id=iteration_id,
            scenario_id=scenario_id,
            full_config=full_config,
            status=status,
        )
        session = self.db.get_session()
        try:
            session.add(run)
            session.commit()
            session.refresh(run)
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()
        return run
