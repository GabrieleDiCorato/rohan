"""Orchestrates end-to-end simulation execution and persistence."""

import logging
import traceback
from uuid import UUID

from rohan.config import SimulationSettings
from rohan.simulation import SimulationService

from .analysis_service import AnalysisService
from .database import ArtifactType, DatabaseConnector, RunStatus
from .repository import ArtifactStore

logger = logging.getLogger(__name__)


class SimulationEngine:
    """
    Orchestrates the execution of a simulation run.
    1. Loads Config
    2. Runs Simulation
    3. Saves Results (L1, Logs, Metrics, Plot artifacts) to DB
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
        logger.info("Starting local run: %s", run_id)

        # 1. Update Status to RUNNING
        self.repo.update_run_status(run_id, RunStatus.RUNNING)

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
            self.repo.save_market_data(run_id, l1_df)
            self.repo.save_logs(run_id, logs_df)

            # 5. Save plot artifacts
            self._save_plot_artifacts(run_id, result)

            # 6. Update Run with Metrics and Completion Status
            self.repo.update_run_status(
                run_id,
                RunStatus.COMPLETED,
                metrics=metrics.model_dump(),
            )

            logger.info("Run %s completed successfully.", run_id)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error("Run %s failed: %s", run_id, e)
            self.repo.update_run_status(
                run_id,
                RunStatus.FAILED,
                error_message=str(e),
                error_traceback=tb,
            )
            raise

    def _save_plot_artifacts(self, run_id: UUID, result) -> None:
        """Generate and persist standard plot artifacts for a run."""
        plot_methods = [
            ("price_series.png", self.analyzer.plot_price_series),
            ("volume.png", self.analyzer.plot_volume),
            ("spread.png", self.analyzer.plot_spread),
        ]
        for filename, plot_fn in plot_methods:
            try:
                fig = plot_fn(result)
                content = AnalysisService.figure_to_bytes(fig)
                self.repo.save_artifact(
                    run_id=run_id,
                    artifact_type=ArtifactType.IMAGE,
                    path=filename,
                    content=content,
                )
            except Exception:
                logger.warning("Failed to save plot artifact %s for run %s", filename, run_id, exc_info=True)
