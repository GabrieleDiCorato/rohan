"""Single Iteration Pipeline — Phase 1.5.6.

Wires together the complete cycle:

    1. **Validate** – AST safety checks on strategy code.
    2. **Execute**  – Run strategy + baseline simulation and compare.
    3. **Analyse**  – Build ``RunSummary`` with metrics and chart images.
    4. **Persist**  – Store the iteration, runs, and artifacts in the DB.
    5. **Interpret** – Format a prompt for LLM interpretation.

This is the entry‐point for the autonomous "game loop" that will later
be orchestrated by LangGraph (Phase 2).
"""

from __future__ import annotations

import logging
import time
import traceback
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from rohan.config import SimulationSettings
from rohan.framework.analysis_service import AnalysisService
from rohan.framework.database import ArtifactType, DatabaseConnector, RunStatus
from rohan.framework.prompts import format_interpreter_prompt
from rohan.framework.repository import ArtifactStore
from rohan.simulation.models import ValidationResult
from rohan.simulation.models.simulation_metrics import (
    ComparisonResult,
    MarketImpact,
    MarketMetrics,
    RunSummary,
    SimulationMetrics,
)
from rohan.simulation.simulation_service import SimulationService
from rohan.simulation.strategy_validator import StrategyValidator, execute_strategy_safely

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline result model
# ---------------------------------------------------------------------------


class IterationResult(BaseModel):
    """Outcome of a single iteration through the pipeline."""

    iteration_id: UUID | None = Field(default=None, description="DB iteration ID (set when persistence is enabled)")
    generation_number: int = Field(default=0)
    validation: ValidationResult = Field(...)
    comparison: ComparisonResult | None = Field(default=None)
    summary: RunSummary | None = Field(default=None)
    interpreter_prompt: str | None = Field(default=None)
    duration_seconds: float = 0.0
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def success(self) -> bool:
        return self.validation.is_valid and self.error is None and self.comparison is not None


# ---------------------------------------------------------------------------
# Pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Knobs for a single iteration."""

    settings: SimulationSettings = field(default_factory=SimulationSettings)
    baseline_override: dict[str, Any] | None = None
    goal_description: str = ""
    timeout_seconds: int = 300
    # When True the pipeline persists iterations & artifacts to the DB.
    persist: bool = False
    # DB ids (required when persist=True)
    session_id: UUID | None = None
    scenario_id: UUID | None = None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IterationPipeline:
    """Runs one generate→validate→execute→interpret cycle.

    Usage::

        pipeline = IterationPipeline()
        result = pipeline.run(strategy_code, config)
        if result.success:
            print(result.interpreter_prompt)
    """

    def __init__(self, db: DatabaseConnector | None = None):
        self._validator = StrategyValidator()
        self._analyzer = AnalysisService()
        self._service = SimulationService()
        self._db = db
        self._repo = ArtifactStore(db) if db else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_code: str,
        config: PipelineConfig,
        generation_number: int = 1,
    ) -> IterationResult:
        """Execute the full pipeline for *strategy_code*.

        Parameters
        ----------
        strategy_code:
            Python source implementing ``StrategicAgent``.
        config:
            Simulation & pipeline settings.
        generation_number:
            Iteration counter (for DB bookkeeping).

        Returns
        -------
        IterationResult
            Contains validation, comparison, summary, and the generated
            LLM prompt.  Check ``.success`` before using metrics.
        """
        start = time.time()

        # 1. Validate -------------------------------------------------------
        validation = self._validator.validate(strategy_code)
        if not validation.is_valid:
            logger.warning("Validation failed: %s", validation.errors)
            return IterationResult(
                generation_number=generation_number,
                validation=validation,
                duration_seconds=time.time() - start,
                error=f"Validation errors: {validation.errors}",
            )

        # 2. Execute (strategy + baseline) ----------------------------------
        try:
            comparison, strategy_output = self._run_comparison(strategy_code, config)
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("Execution failed: %s", exc)
            return IterationResult(
                generation_number=generation_number,
                validation=validation,
                duration_seconds=time.time() - start,
                error=f"Execution error: {exc}\n{tb}",
            )

        # 3. Analyse (summary + prompt) -------------------------------------
        summary = self._analyzer.generate_summary(
            comparison,
            strategy_output=strategy_output,
            duration_seconds=time.time() - start,
        )

        prompt = format_interpreter_prompt(summary, goal=config.goal_description)

        # 4. Persist (optional) ---------------------------------------------
        iteration_id: UUID | None = None
        if config.persist and self._repo is not None:
            try:
                iteration_id = self._persist(
                    strategy_code,
                    generation_number,
                    config,
                    comparison,
                    strategy_output,
                )
            except Exception:
                logger.warning("Persistence failed", exc_info=True)

        duration = time.time() - start
        logger.info(
            "Iteration %d completed in %.1fs (PnL=%s)",
            generation_number,
            duration,
            comparison.strategy_metrics.total_pnl,
        )

        return IterationResult(
            iteration_id=iteration_id,
            generation_number=generation_number,
            validation=validation,
            comparison=comparison,
            summary=summary,
            interpreter_prompt=prompt,
            duration_seconds=duration,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_comparison(
        self,
        strategy_code: str,
        config: PipelineConfig,
    ) -> tuple[ComparisonResult, Any]:
        """Run strategy + baseline and return (ComparisonResult, strategy_output)."""

        # --- Run strategy ---
        strategy_result = execute_strategy_safely(strategy_code, config.settings, config.timeout_seconds)
        if strategy_result.error:
            raise RuntimeError(f"Strategy run failed: {strategy_result.error}")
        if not strategy_result.result:
            raise RuntimeError("Strategy run returned no result")

        strategy_output = strategy_result.result
        strategy_sim_metrics = self._analyzer.compute_metrics(strategy_output)

        if strategy_output.strategic_agent_id is None:
            raise RuntimeError("No strategic agent found in simulation output")
        strategy_agent_metrics = self._analyzer.compute_agent_metrics(
            strategy_output,
            strategy_output.strategic_agent_id,
            initial_cash=config.settings.starting_cash,
        )

        # --- Run baseline ---
        baseline_settings = config.settings.model_copy(deep=True)
        if config.baseline_override:
            for k, v in config.baseline_override.items():
                if hasattr(baseline_settings, k):
                    setattr(baseline_settings, k, v)

        baseline_result = self._service.run_simulation(baseline_settings, strategy=None)
        if baseline_result.error:
            raise RuntimeError(f"Baseline run failed: {baseline_result.error}")
        if not baseline_result.result:
            raise RuntimeError("Baseline run returned no result")

        baseline_sim_metrics = self._analyzer.compute_metrics(baseline_result.result)

        # --- Build ComparisonResult ---
        def _to_market(m: SimulationMetrics) -> MarketMetrics:
            return MarketMetrics(
                volatility=m.volatility,
                mean_spread=m.mean_spread,
                effective_spread=m.effective_spread,
                avg_bid_liquidity=m.avg_bid_liquidity,
                avg_ask_liquidity=m.avg_ask_liquidity,
                traded_volume=m.traded_volume,
            )

        def _pct(a: float | None, b: float | None) -> float | None:
            if a is None or b is None:
                return None
            if b == 0:
                return 0.0 if a == 0 else float("inf")
            return (a - b) / b

        strat_market = _to_market(strategy_sim_metrics)
        base_market = _to_market(baseline_sim_metrics)

        impact = MarketImpact(
            spread_delta_pct=_pct(strat_market.mean_spread, base_market.mean_spread),
            volatility_delta_pct=_pct(strat_market.volatility, base_market.volatility),
            bid_liquidity_delta_pct=_pct(strat_market.avg_bid_liquidity, base_market.avg_bid_liquidity),
            ask_liquidity_delta_pct=_pct(strat_market.avg_ask_liquidity, base_market.avg_ask_liquidity),
        )

        comparison = ComparisonResult(
            strategy_metrics=strategy_agent_metrics,
            strategy_market_metrics=strat_market,
            baseline_metrics=base_market,
            market_impact=impact,
        )

        return comparison, strategy_output

    def _persist(
        self,
        strategy_code: str,
        generation_number: int,
        config: PipelineConfig,
        comparison: ComparisonResult,
        strategy_output: Any,
    ) -> UUID:
        """Persist iteration + artifacts to the database.  Returns iteration_id."""
        assert self._repo is not None
        assert config.session_id is not None

        iteration = self._repo.create_iteration(
            session_id=config.session_id,
            generation_number=generation_number,
            code=strategy_code,
            reasoning="",
        )

        # Create a run record linked to the iteration
        scenario_id = config.scenario_id
        if scenario_id is None:
            scenario = self._repo.create_scenario(
                name=f"iteration-{generation_number}",
                config_override=config.baseline_override or {},
                session_id=config.session_id,
            )
            scenario_id = scenario.scenario_id

        run = self._repo.create_run(
            iteration_id=iteration.iteration_id,
            scenario_id=scenario_id,
            full_config=config.settings.model_dump(),
            status=RunStatus.COMPLETED,
        )

        # Save metrics
        self._repo.update_run_status(
            run.run_id,
            RunStatus.COMPLETED,
            metrics=comparison.strategy_metrics.model_dump(),
        )

        # Save plot artifacts
        if strategy_output is not None:
            for name, plot_fn in [
                ("price_series.png", self._analyzer.plot_price_series),
                ("spread.png", self._analyzer.plot_spread),
            ]:
                try:
                    fig = plot_fn(strategy_output)
                    content = AnalysisService.figure_to_bytes(fig)
                    self._repo.save_artifact(
                        run_id=run.run_id,
                        artifact_type=ArtifactType.IMAGE,
                        path=name,
                        content=content,
                    )
                except Exception:
                    logger.warning("Failed to save artifact %s", name, exc_info=True)

        return iteration.iteration_id
