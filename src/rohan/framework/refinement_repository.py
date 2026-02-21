"""Repository for Refinement Lab run persistence.

Saves and loads complete refinement sessions — including all iterations,
per-scenario metrics, charts, and progress logs — so users can reload
any past run and see exactly what happened.

The schema is intentionally decoupled from the agentic-workflow internals
(``StrategySession``, ``StrategyIteration``, etc.) to remain modular and
reusable for future non-agentic flows (e.g. user-authored strategies
evaluated by AI-generated scenarios).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import select

from rohan.framework.database import (
    DatabaseConnector,
    RefinementIteration,
    RefinementScenarioResult,
    RefinementSession,
    get_database_connector,
)
from rohan.llm.models import IterationSummary, ScenarioMetrics

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data-transfer objects (lightweight containers for the save path)
# ------------------------------------------------------------------


@dataclass
class ScenarioResultData:
    """Data needed to persist one per-scenario result row."""

    scenario_name: str
    total_pnl: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    trade_count: int = 0
    volatility_delta_pct: float | None = None
    spread_delta_pct: float | None = None
    price_chart_b64: str | None = None
    spread_chart_b64: str | None = None
    volume_chart_b64: str | None = None


@dataclass
class IterationData:
    """Data needed to persist one refinement iteration row."""

    iteration_number: int
    strategy_code: str
    class_name: str | None = None
    reasoning: str | None = None
    judge_score: float | None = None
    judge_reasoning: str | None = None
    aggregated_explanation: str | None = None
    scenario_results: list[ScenarioResultData] = field(default_factory=list)


@dataclass
class RefinementSessionSummary:
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


class RefinementRepository:
    """Persist and reload complete Refinement Lab runs."""

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
    ) -> RefinementSession:
        """Persist a full refinement run in a single transaction."""
        session_obj = RefinementSession(
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

        # Build ORM children
        for it_data in iterations:
            it_orm = RefinementIteration(
                iteration_number=it_data.iteration_number,
                strategy_code=it_data.strategy_code,
                class_name=it_data.class_name,
                reasoning=it_data.reasoning,
                judge_score=it_data.judge_score,
                judge_reasoning=it_data.judge_reasoning,
                aggregated_explanation=it_data.aggregated_explanation,
            )
            for sr in it_data.scenario_results:
                it_orm.scenario_results.append(
                    RefinementScenarioResult(
                        scenario_name=sr.scenario_name,
                        total_pnl=sr.total_pnl,
                        sharpe_ratio=sr.sharpe_ratio,
                        max_drawdown=sr.max_drawdown,
                        trade_count=sr.trade_count,
                        volatility_delta_pct=sr.volatility_delta_pct,
                        spread_delta_pct=sr.spread_delta_pct,
                        price_chart_b64=sr.price_chart_b64,
                        spread_chart_b64=sr.spread_chart_b64,
                        volume_chart_b64=sr.volume_chart_b64,
                    )
                )
            session_obj.iterations.append(it_orm)

        db_session = self.db.get_session()
        try:
            db_session.add(session_obj)
            db_session.commit()
            db_session.refresh(session_obj)
            # Eagerly load relationships before detaching from session
            for it_orm in session_obj.iterations:
                _ = it_orm.scenario_results
        except Exception:
            db_session.rollback()
            raise
        finally:
            self.db.remove_session()

        return session_obj

    # ------------------------------------------------------------------
    # List (lightweight)
    # ------------------------------------------------------------------

    def list_sessions(self) -> list[RefinementSessionSummary]:
        """Return lightweight summaries of all saved runs (newest first)."""
        db_session = self.db.get_session()
        try:
            result = db_session.execute(select(RefinementSession).order_by(RefinementSession.created_at.desc()))
            rows = result.scalars().all()
            return [
                RefinementSessionSummary(
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
        (``refine_final_state``, ``refine_duration``, etc.) or ``None``
        if the session doesn't exist.
        """
        db_session = self.db.get_session()
        try:
            session_obj = db_session.get(RefinementSession, session_id)
            if session_obj is None:
                return None

            # Rebuild IterationSummary pydantic objects
            iteration_summaries: list[IterationSummary] = []
            for it_orm in session_obj.iterations:
                sc_metrics: dict[str, ScenarioMetrics] = {}
                for sr in it_orm.scenario_results:
                    sc_metrics[sr.scenario_name] = ScenarioMetrics(
                        scenario_name=sr.scenario_name,
                        total_pnl=sr.total_pnl,
                        sharpe_ratio=sr.sharpe_ratio,
                        max_drawdown=sr.max_drawdown,
                        trade_count=sr.trade_count,
                        volatility_delta_pct=sr.volatility_delta_pct,
                        spread_delta_pct=sr.spread_delta_pct,
                        price_chart_b64=sr.price_chart_b64,
                        spread_chart_b64=sr.spread_chart_b64,
                        volume_chart_b64=sr.volume_chart_b64,
                    )
                iteration_summaries.append(
                    IterationSummary(
                        iteration_number=it_orm.iteration_number,
                        strategy_code=it_orm.strategy_code,
                        scenario_metrics=sc_metrics,
                        aggregated_explanation=it_orm.aggregated_explanation or "",
                        judge_score=it_orm.judge_score,
                        judge_reasoning=it_orm.judge_reasoning,
                        timestamp=it_orm.created_at,
                    )
                )

            # Build the accumulated state dict (mirrors what _run_refinement stores)
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
    # Delete
    # ------------------------------------------------------------------

    def delete_session(self, session_id: UUID) -> bool:
        """Delete a refinement session and all its children. Returns True if deleted."""
        db_session = self.db.get_session()
        try:
            obj = db_session.get(RefinementSession, session_id)
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
