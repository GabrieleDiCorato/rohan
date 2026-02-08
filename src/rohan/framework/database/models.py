"""SQLAlchemy ORM models for the agentic simulation framework.

Hierarchy:
    StrategySession (User goal)
    -> StrategyIteration (One code version)
    -> SimulationScenario (Conditions)
    -> SimulationRun (Execution)
"""

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.sql import func


class Base(DeclarativeBase):
    pass


class RunStatus(str, enum.Enum):
    """Status of a simulation run."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class ArtifactType(str, enum.Enum):
    """Type of stored artifact."""

    IMAGE = "IMAGE"
    LOG_FILE = "LOG_FILE"
    CODE = "CODE"
    REPORT = "REPORT"


class StrategySession(Base):
    __tablename__ = "strategy_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    goal_description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    iterations: Mapped[list["StrategyIteration"]] = relationship(back_populates="session", cascade="all, delete-orphan", passive_deletes=True)
    scenarios: Mapped[list["SimulationScenario"]] = relationship(back_populates="session", cascade="all, delete-orphan", passive_deletes=True)


class SimulationScenario(Base):
    __tablename__ = "simulation_scenarios"

    scenario_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[StrategySession | None] = mapped_column(ForeignKey("strategy_sessions.session_id", ondelete="CASCADE"), nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    config_override: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default={})
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped[StrategySession | None] = relationship(back_populates="scenarios")
    runs: Mapped[list["SimulationRun"]] = relationship(back_populates="scenario", cascade="all, delete-orphan", passive_deletes=True)


class StrategyIteration(Base):
    __tablename__ = "strategy_iterations"

    iteration_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("strategy_sessions.session_id", ondelete="CASCADE"), nullable=False)
    generation_number: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["StrategySession"] = relationship(back_populates="iterations")
    runs: Mapped[list["SimulationRun"]] = relationship(back_populates="iteration", cascade="all, delete-orphan", passive_deletes=True)


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    iteration_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("strategy_iterations.iteration_id", ondelete="CASCADE"), nullable=False)
    scenario_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_scenarios.scenario_id", ondelete="CASCADE"), nullable=False)
    full_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[RunStatus] = mapped_column(Enum(RunStatus), nullable=False, default=RunStatus.PENDING)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_traceback: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    iteration: Mapped["StrategyIteration"] = relationship(back_populates="runs")
    scenario: Mapped["SimulationScenario"] = relationship(back_populates="runs")
    market_data: Mapped[list["MarketDataL1"]] = relationship(back_populates="run", cascade="all, delete-orphan", passive_deletes=True)
    logs: Mapped[list["AgentLog"]] = relationship(back_populates="run", cascade="all, delete-orphan", passive_deletes=True)
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run", cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (Index("ix_simulation_runs_status", "status"),)


class MarketDataL1(Base):
    __tablename__ = "market_data_l1"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    time: Mapped[int] = mapped_column(BigInteger, nullable=False)  # Nanoseconds from midnight
    bid_price: Mapped[float] = mapped_column(Float, nullable=False)
    bid_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    ask_price: Mapped[float] = mapped_column(Float, nullable=False)
    ask_qty: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    run: Mapped["SimulationRun"] = relationship(back_populates="market_data")


class AgentLog(Base):
    __tablename__ = "agent_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_type: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    time_placed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    log_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    run: Mapped["SimulationRun"] = relationship(back_populates="logs")

    __table_args__ = (
        Index("ix_agent_logs_agent_type", "agent_type"),
        Index("ix_agent_logs_event_type", "event_type"),
    )


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_runs.run_id", ondelete="CASCADE"), nullable=False, index=True)
    artifact_type: Mapped[ArtifactType] = mapped_column(Enum(ArtifactType), nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    storage_backend: Mapped[str] = mapped_column(String, nullable=False, default="database")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    run: Mapped["SimulationRun"] = relationship(back_populates="artifacts")
