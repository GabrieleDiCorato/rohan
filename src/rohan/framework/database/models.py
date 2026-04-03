"""SQLAlchemy ORM models for the persistence layer.

Unified 5-table schema:
    Session → Iteration → ScenarioRun → Artifact
    + SavedScenario (standalone)

Stores hasufel-native JSON documents (compiled_config, hasufel_summary,
domain_metrics) rather than decomposing into per-column or row-per-tick
tables.  Deterministic replay from compiled_config replaces raw L1/log
storage.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Session hierarchy
# ═══════════════════════════════════════════════════════════════════════════


class Session(Base):
    """A complete refinement run — goal, config, and outcome metadata."""

    __tablename__ = "sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False)
    goal: Mapped[str] = mapped_column(Text, nullable=False)
    max_iterations: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario_configs: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="done")
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_class_name: Mapped[str | None] = mapped_column(String, nullable=True)
    final_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    progress_log: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    total_duration: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    iterations: Mapped[list["Iteration"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Iteration.iteration_number",
    )


class Iteration(Base):
    """One iteration of a refinement loop — code, score, reasoning."""

    __tablename__ = "iterations"

    iteration_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    strategy_code: Mapped[str] = mapped_column(Text, nullable=False)
    class_name: Mapped[str | None] = mapped_column(String, nullable=True)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    judge_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    judge_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    aggregated_explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    rolled_back: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    axis_scores: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["Session"] = relationship(back_populates="iterations")
    scenario_runs: Mapped[list["ScenarioRun"]] = relationship(
        back_populates="iteration",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    __table_args__ = (Index("ix_iterations_session_num", "session_id", "iteration_number"),)


class ScenarioRun(Base):
    """Per-scenario execution result with hasufel-native JSON documents."""

    __tablename__ = "scenario_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    iteration_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("iterations.iteration_id", ondelete="CASCADE"), nullable=False)
    scenario_name: Mapped[str] = mapped_column(String, nullable=False)
    compiled_config: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    hasufel_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    domain_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    rich_analysis: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    iteration: Mapped["Iteration"] = relationship(back_populates="scenario_runs")
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="scenario_run",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class Artifact(Base):
    """Heavy blobs (base64 chart PNGs), lazily loaded."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("scenario_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    artifact_type: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    scenario_run: Mapped["ScenarioRun"] = relationship(back_populates="artifacts")


# ═══════════════════════════════════════════════════════════════════════════
# Saved Scenarios  (standalone, reusable across Terminal & Refinement Lab)
# ═══════════════════════════════════════════════════════════════════════════


class SavedScenario(Base):
    """A reusable simulation configuration saved by the user."""

    __tablename__ = "saved_scenarios"

    scenario_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    full_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
