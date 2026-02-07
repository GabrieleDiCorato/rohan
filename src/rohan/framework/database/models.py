import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
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


class StrategySession(Base):
    __tablename__ = "strategy_sessions"

    session_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    goal_description: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    iterations: Mapped[list["StrategyIteration"]] = relationship(back_populates="session")
    scenarios: Mapped[list["SimulationScenario"]] = relationship(back_populates="session")


class SimulationScenario(Base):
    __tablename__ = "simulation_scenarios"

    scenario_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[StrategySession | None] = mapped_column(ForeignKey("strategy_sessions.session_id"), nullable=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    config_override: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default={})

    session: Mapped[StrategySession | None] = relationship(back_populates="scenarios")
    runs: Mapped[list["SimulationRun"]] = relationship(back_populates="scenario")


class StrategyIteration(Base):
    __tablename__ = "strategy_iterations"

    iteration_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("strategy_sessions.session_id"), nullable=False)
    generation_number: Mapped[int] = mapped_column(Integer, nullable=False)
    code: Mapped[str] = mapped_column(Text, nullable=False)
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    session: Mapped["StrategySession"] = relationship(back_populates="iterations")
    runs: Mapped[list["SimulationRun"]] = relationship(back_populates="iteration")


class SimulationRun(Base):
    __tablename__ = "simulation_runs"

    run_id: Mapped[uuid.UUID] = mapped_column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    iteration_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("strategy_iterations.iteration_id"), nullable=False)
    scenario_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_scenarios.scenario_id"), nullable=False)
    full_config: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    metrics_summary: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="PENDING")

    iteration: Mapped["StrategyIteration"] = relationship(back_populates="runs")
    scenario: Mapped["SimulationScenario"] = relationship(back_populates="runs")
    market_data: Mapped[list["MarketDataL1"]] = relationship(back_populates="run")
    logs: Mapped[list["AgentLog"]] = relationship(back_populates="run")
    artifacts: Mapped[list["Artifact"]] = relationship(back_populates="run")


class MarketDataL1(Base):
    __tablename__ = "market_data_l1"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_runs.run_id"), nullable=False, index=True)
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
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_runs.run_id"), nullable=False, index=True)
    agent_id: Mapped[int] = mapped_column(Integer, nullable=False)
    agent_type: Mapped[str] = mapped_column(String, nullable=False)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    time_placed: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    log_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    run: Mapped["SimulationRun"] = relationship(back_populates="logs")


class Artifact(Base):
    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("simulation_runs.run_id"), nullable=False, index=True)
    type: Mapped[str] = mapped_column(String, nullable=False)  # 'IMAGE', 'LOG_FILE'
    path: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    run: Mapped["SimulationRun"] = relationship(back_populates="artifacts")
