from uuid import UUID

from rohan.config import SimulationSettings
from rohan.simulation import SimulationService

from .analysis_service import AnalysisService
from .database import DatabaseConnector, SimulationRun
from .repository import ArtifactStore


class SimulationEngine:
    """
    Orchestrates the execution of a simulation run.
    1. Loads Config
    2. Runs Simulation
    3. Saves Results (L1, Logs, Metrics) to DB
    """

    def __init__(self, db: DatabaseConnector | None = None):
        self.db = db or DatabaseConnector()
        self.repo = ArtifactStore(self.db)
        self.analyzer = AnalysisService()

    def run_local(self, run_id: UUID, settings: SimulationSettings) -> None:
        """
        Executes a simulation locally (blocking) and saves results.
        Updates the SimulationRun status in DB.
        """
        print(f"Starting Local Run: {run_id}")

        # 1. Update Status to RUNNING
        with self.db.get_session() as session:
            run = session.get(SimulationRun, run_id)
            if run:
                run.status = "RUNNING"
                session.commit()

        try:
            # 2. Run Simulation
            service = SimulationService()
            sim_result = service.run_simulation(settings)

            # Check if simulation succeeded
            if sim_result.error is not None:
                raise sim_result.error

            result = sim_result.result
            assert result is not None, "Result must not be None when error is None"

            # 3. Extract Data
            l1_df = result.get_order_book_l1()
            logs_df = result.get_logs_df()
            metrics = self.analyzer.compute_metrics(result)

            # 4. Save to DB
            # 4. Save to DB
            self.repo.save_market_data(run_id, l1_df)
            self.repo.save_logs(run_id, logs_df)

            # 5. Update Run with Metrics and Completion Status
            with self.db.get_session() as session:
                run = session.get(SimulationRun, run_id)
                if run:
                    run.status = "COMPLETED"
                    run.metrics_summary = metrics.model_dump()
                    session.commit()

            print(f"Run {run_id} Completed successfully.")

        except Exception as e:
            print(f"Run {run_id} Failed: {e}")
            with self.db.get_session() as session:
                run = session.get(SimulationRun, run_id)
                if run:
                    run.status = "FAILED"
                    # run.error_log = str(e) # If we had an error column
                    session.commit()
            raise e
