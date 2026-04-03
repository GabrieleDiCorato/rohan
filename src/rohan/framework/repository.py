"""Unified repository for session persistence.

Replaces the old ArtifactStore and RefinementRepository with a single
SessionRepository that saves/loads complete refinement sessions using
hasufel-native JSON documents (compiled_config, hasufel_summary,
domain_metrics) stored in JSON columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from rohan.framework.database import (
    Artifact,
    DatabaseConnector,
    Iteration,
    ScenarioRun,
    Session,
    get_database_connector,
)
from rohan.llm.models import IterationSummary, ScenarioMetrics

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data-transfer objects (lightweight containers for the save path)
# ------------------------------------------------------------------


@dataclass
class ScenarioRunData:
    """Data needed to persist one per-scenario result row."""

    scenario_name: str
    compiled_config: dict[str, Any] | None = None
    hasufel_summary: dict[str, Any] | None = None
    domain_metrics: dict[str, Any] | None = None
    rich_analysis: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float | None = None
    # Chart base64 strings → stored as Artifact rows
    price_chart_b64: str | None = None
    spread_chart_b64: str | None = None
    volume_chart_b64: str | None = None
    pnl_chart_b64: str | None = None
    inventory_chart_b64: str | None = None
    fill_scatter_b64: str | None = None


@dataclass
class IterationData:
    """Data needed to persist one iteration row."""

    iteration_number: int
    strategy_code: str
    class_name: str | None = None
    reasoning: str | None = None
    judge_score: float | None = None
    judge_reasoning: str | None = None
    aggregated_explanation: str | None = None
    rolled_back: bool = False
    axis_scores: dict[str, Any] | None = None
    scenario_runs: list[ScenarioRunData] = field(default_factory=list)


@dataclass
class SessionSummary:
    """Lightweight projection returned by ``list_sessions``."""

    session_id: UUID
    name: str
    goal: str
    final_score: float | None
    iteration_count: int
    status: str
    created_at: datetime


# ------------------------------------------------------------------
# Repository
# ------------------------------------------------------------------


class SessionRepository:
    """Persist and reload complete refinement sessions."""

    def __init__(self, db: DatabaseConnector | None = None) -> None:
        self.db = db or get_database_connector()

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_session(
        self,
        *,
        name: str,
        goal: str,
        max_iterations: int,
        scenario_configs: list[dict[str, Any]],
        status: str,
        final_score: float | None,
        total_duration: float | None,
        progress_log: list[str],
        final_code: str | None,
        final_class_name: str | None,
        final_reasoning: str | None,
        iterations: list[IterationData],
    ) -> Session:
        """Persist a full refinement run in a single transaction."""
        session_obj = Session(
            name=name,
            goal=goal,
            max_iterations=max_iterations,
            scenario_configs=scenario_configs,
            status=status,
            final_score=final_score,
            total_duration=total_duration,
            progress_log=progress_log,
            final_code=final_code,
            final_class_name=final_class_name,
            final_reasoning=final_reasoning,
        )

        for it_data in iterations:
            it_orm = Iteration(
                iteration_number=it_data.iteration_number,
                strategy_code=it_data.strategy_code,
                class_name=it_data.class_name,
                reasoning=it_data.reasoning,
                judge_score=it_data.judge_score,
                judge_reasoning=it_data.judge_reasoning,
                aggregated_explanation=it_data.aggregated_explanation,
                rolled_back=it_data.rolled_back,
                axis_scores=it_data.axis_scores,
            )
            for sr in it_data.scenario_runs:
                run_orm = ScenarioRun(
                    scenario_name=sr.scenario_name,
                    compiled_config=sr.compiled_config,
                    hasufel_summary=sr.hasufel_summary,
                    domain_metrics=sr.domain_metrics,
                    rich_analysis=sr.rich_analysis,
                    error=sr.error,
                    duration_seconds=sr.duration_seconds,
                )
                # Store chart base64 strings as Artifact rows
                chart_fields = [
                    "price_chart_b64",
                    "spread_chart_b64",
                    "volume_chart_b64",
                    "pnl_chart_b64",
                    "inventory_chart_b64",
                    "fill_scatter_b64",
                ]
                for field_name in chart_fields:
                    val = getattr(sr, field_name, None)
                    if val is not None:
                        run_orm.artifacts.append(Artifact(artifact_type=field_name, content=val))
                it_orm.scenario_runs.append(run_orm)
            session_obj.iterations.append(it_orm)

        db_session = self.db.get_session()
        try:
            db_session.add(session_obj)
            db_session.commit()
            db_session.refresh(session_obj)
            for it_orm in session_obj.iterations:
                _ = it_orm.scenario_runs
        except Exception:
            db_session.rollback()
            raise
        finally:
            self.db.remove_session()

        return session_obj

    # ------------------------------------------------------------------
    # List (lightweight)
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[SessionSummary]:
        """Return lightweight summaries of all saved runs (newest first)."""
        db_session = self.db.get_session()
        try:
            result = db_session.execute(select(Session).options(selectinload(Session.iterations)).order_by(Session.created_at.desc()))
            rows = result.scalars().all()
            return [
                SessionSummary(
                    session_id=r.session_id,
                    name=r.name,
                    goal=r.goal[:120],
                    final_score=r.final_score,
                    iteration_count=len(r.iterations),
                    status=r.status,
                    created_at=r.created_at,
                )
                for r in rows
            ]
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Load (full reconstruction)
    # ------------------------------------------------------------------

    def load_session(self, session_id: UUID) -> dict[str, Any] | None:
        """Reconstruct the full session-state dict from the DB.

        Returns a dict compatible with the Streamlit session keys
        or ``None`` if the session doesn't exist.
        """
        db_session = self.db.get_session()
        try:
            session_obj = db_session.execute(
                select(Session).options(selectinload(Session.iterations).selectinload(Iteration.scenario_runs)).where(Session.session_id == session_id)
            ).scalar_one_or_none()
            if session_obj is None:
                return None

            iteration_summaries: list[IterationSummary] = []
            for it_orm in session_obj.iterations:
                sc_metrics: dict[str, ScenarioMetrics] = {}
                for sr in it_orm.scenario_runs:
                    # Reconstruct ScenarioMetrics from domain_metrics JSON blob
                    dm = sr.domain_metrics or {}
                    market = dm.get("market", {})
                    agent = dm.get("agent", {})

                    sc_metrics[sr.scenario_name] = ScenarioMetrics(
                        scenario_name=sr.scenario_name,
                        total_pnl=agent.get("total_pnl"),
                        sharpe_ratio=agent.get("sharpe_ratio"),
                        max_drawdown=agent.get("max_drawdown"),
                        trade_count=agent.get("trade_count", 0),
                        fill_rate=agent.get("fill_rate"),
                        order_to_trade_ratio=agent.get("order_to_trade_ratio"),
                        inventory_std=agent.get("inventory_std"),
                        end_inventory=agent.get("end_inventory", 0),
                        volatility_delta_pct=dm.get("impact", {}).get("volatility_delta_pct"),
                        spread_delta_pct=dm.get("impact", {}).get("spread_delta_pct"),
                        vpin=market.get("vpin"),
                        lob_imbalance_mean=market.get("lob_imbalance_mean"),
                        resilience_mean_ns=market.get("resilience_mean_ns"),
                        market_ott_ratio=market.get("market_ott_ratio"),
                        pct_time_two_sided=market.get("pct_time_two_sided"),
                        compiled_config=sr.compiled_config,
                        hasufel_summary=sr.hasufel_summary,
                        # Charts lazy-loaded via load_scenario_artifacts()
                        price_chart_b64=None,
                        spread_chart_b64=None,
                        volume_chart_b64=None,
                        pnl_chart_b64=None,
                        inventory_chart_b64=None,
                        fill_scatter_b64=None,
                    )

                axis = it_orm.axis_scores or {}
                iteration_summaries.append(
                    IterationSummary(
                        iteration_number=it_orm.iteration_number,
                        strategy_code=it_orm.strategy_code,
                        scenario_metrics=sc_metrics,
                        aggregated_explanation=it_orm.aggregated_explanation or "",
                        judge_score=it_orm.judge_score,
                        judge_reasoning=it_orm.judge_reasoning,
                        timestamp=it_orm.created_at,
                        rolled_back=it_orm.rolled_back,
                        profitability_score=axis.get("profitability"),
                        risk_score=axis.get("risk"),
                        volatility_impact_score=axis.get("volatility_impact"),
                        spread_impact_score=axis.get("spread_impact"),
                        liquidity_impact_score=axis.get("liquidity_impact"),
                        execution_score=axis.get("execution"),
                        scoring_profile=axis.get("profile"),
                    )
                )

            final_state: dict[str, Any] = {
                "goal": session_obj.goal,
                "max_iterations": session_obj.max_iterations,
                "scenarios": session_obj.scenario_configs,
                "current_code": session_obj.final_code,
                "current_class_name": session_obj.final_class_name,
                "current_reasoning": session_obj.final_reasoning,
                "validation_errors": [],
                "validation_attempts": 0,
                "scenario_results": [],
                "explanations": [],
                "aggregated_feedback": None,
                "iterations": iteration_summaries,
                "iteration_number": len(iteration_summaries) + 1,
                "status": session_obj.status,
                "messages": [],
            }

            return {
                "refine_final_state": final_state,
                "refine_goal": session_obj.goal,
                "refine_max_iterations": session_obj.max_iterations,
                "refine_duration": session_obj.total_duration,
                "refine_progress": session_obj.progress_log or [],
                "refine_timestamp": session_obj.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "refine_saved": True,
                "refine_saved_id": session_obj.session_id,
                "refine_is_dirty": False,
                "refine_error": None,
            }
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Load Artifacts
    # ------------------------------------------------------------------

    def load_scenario_artifacts(
        self,
        session_id: UUID,
        iteration_number: int,
        scenario_name: str,
    ) -> dict[str, str]:
        """Lazy-load the heavy artifacts (charts) for a single scenario run."""
        db_session = self.db.get_session()
        try:
            stmt = (
                select(Artifact)
                .join(ScenarioRun)
                .join(Iteration)
                .where(
                    Iteration.session_id == session_id,
                    Iteration.iteration_number == iteration_number,
                    ScenarioRun.scenario_name == scenario_name,
                )
            )
            artifacts = db_session.execute(stmt).scalars().all()
            return {a.artifact_type: a.content for a in artifacts if a.content}
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_session(self, session_id: UUID) -> bool:
        """Delete a session and all its children. Returns True if deleted."""
        db_session = self.db.get_session()
        try:
            obj = db_session.get(Session, session_id)
            if obj is None:
                return False
            db_session.delete(obj)
            db_session.commit()
            return True
        except Exception:
            db_session.rollback()
            raise
        finally:
            self.db.remove_session()
