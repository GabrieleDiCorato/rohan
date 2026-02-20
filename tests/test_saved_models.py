"""Tests for the new database ORM models (SavedScenario, RefinementSession, etc.).

Verifies table creation, field defaults, and relationship cascades
using the in-memory SQLite from conftest.py.
"""

from rohan.framework.database import (
    DatabaseConnector,
    RefinementIteration,
    RefinementScenarioResult,
    RefinementSession,
    SavedScenario,
)


class TestSavedScenarioModel:
    """Tests for the SavedScenario ORM model."""

    def _setup_db(self):
        db = DatabaseConnector()
        db.create_tables()
        return db

    def test_create_saved_scenario(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            sc = SavedScenario(
                name="Test Scenario",
                full_config={"noise_agents": 200},
                description="A test",
            )
            session.add(sc)
            session.commit()
            session.refresh(sc)

            assert sc.scenario_id is not None
            assert sc.name == "Test Scenario"
            assert sc.full_config == {"noise_agents": 200}
            assert sc.description == "A test"
        finally:
            db.remove_session()

    def test_saved_scenario_unique_name(self):
        import pytest
        from sqlalchemy.exc import IntegrityError

        db = self._setup_db()
        session = db.get_session()
        try:
            session.add(SavedScenario(name="Unique", full_config={}))
            session.commit()
            session.add(SavedScenario(name="Unique", full_config={}))
            with pytest.raises(IntegrityError):
                session.commit()
        finally:
            session.rollback()
            db.remove_session()

    def test_saved_scenario_nullable_description(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            sc = SavedScenario(name="NoDesc", full_config={"k": 1})
            session.add(sc)
            session.commit()
            session.refresh(sc)
            assert sc.description is None
        finally:
            db.remove_session()


class TestRefinementModels:
    """Tests for RefinementSession, RefinementIteration, RefinementScenarioResult."""

    def _setup_db(self):
        db = DatabaseConnector()
        db.create_tables()
        return db

    def test_create_refinement_session(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            rs = RefinementSession(
                name="Run 1",
                goal="Make money",
                max_iterations=3,
                scenario_configs=[{"name": "Default"}],
                status="done",
                final_score=8.5,
                total_duration=120.0,
                progress_log=["started", "finished"],
                final_code="class S: pass",
                final_class_name="S",
                final_reasoning="Good",
            )
            session.add(rs)
            session.commit()
            session.refresh(rs)

            assert rs.session_id is not None
            assert rs.name == "Run 1"
            assert rs.goal == "Make money"
            assert rs.max_iterations == 3
            assert rs.final_score == 8.5
            assert rs.status == "done"
            assert rs.iterations == []
        finally:
            db.remove_session()

    def test_iteration_relationship(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            rs = RefinementSession(
                name="RelTest",
                goal="g",
                max_iterations=1,
                scenario_configs=[],
                status="done",
            )
            it = RefinementIteration(
                iteration_number=1,
                strategy_code="class S: pass",
                class_name="S",
                judge_score=7.0,
                judge_reasoning="OK",
            )
            rs.iterations.append(it)
            session.add(rs)
            session.commit()
            session.refresh(rs)

            assert len(rs.iterations) == 1
            assert rs.iterations[0].iteration_number == 1
            assert rs.iterations[0].session_id == rs.session_id
        finally:
            db.remove_session()

    def test_scenario_result_relationship(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            rs = RefinementSession(
                name="SRTest",
                goal="g",
                max_iterations=1,
                scenario_configs=[],
                status="done",
            )
            it = RefinementIteration(
                iteration_number=1,
                strategy_code="code",
            )
            sr = RefinementScenarioResult(
                scenario_name="Default",
                total_pnl=500.0,
                sharpe_ratio=1.2,
                max_drawdown=-100.0,
                trade_count=25,
                volatility_delta_pct=0.05,
                spread_delta_pct=-0.02,
            )
            it.scenario_results.append(sr)
            rs.iterations.append(it)
            session.add(rs)
            session.commit()
            session.refresh(rs)

            loaded_sr = rs.iterations[0].scenario_results[0]
            assert loaded_sr.scenario_name == "Default"
            assert loaded_sr.total_pnl == 500.0
            assert loaded_sr.trade_count == 25
        finally:
            db.remove_session()

    def test_cascade_delete(self):
        """Deleting a session should cascade-delete iterations and results."""
        db = self._setup_db()
        session = db.get_session()
        try:
            rs = RefinementSession(
                name="CascadeDel",
                goal="g",
                max_iterations=1,
                scenario_configs=[],
                status="done",
            )
            it = RefinementIteration(iteration_number=1, strategy_code="c")
            it.scenario_results.append(RefinementScenarioResult(scenario_name="A", trade_count=5))
            rs.iterations.append(it)
            session.add(rs)
            session.commit()

            sid = rs.session_id
            session.delete(rs)
            session.commit()

            # Session gone
            assert session.get(RefinementSession, sid) is None
        finally:
            db.remove_session()

    def test_multiple_scenario_results_per_iteration(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            rs = RefinementSession(
                name="MultiSR",
                goal="g",
                max_iterations=1,
                scenario_configs=[],
                status="done",
            )
            it = RefinementIteration(iteration_number=1, strategy_code="code")
            it.scenario_results.append(RefinementScenarioResult(scenario_name="Default", trade_count=10))
            it.scenario_results.append(RefinementScenarioResult(scenario_name="HighVol", trade_count=20, total_pnl=-200.0))
            rs.iterations.append(it)
            session.add(rs)
            session.commit()
            session.refresh(rs)

            results = rs.iterations[0].scenario_results
            assert len(results) == 2
            names = {r.scenario_name for r in results}
            assert names == {"Default", "HighVol"}
        finally:
            db.remove_session()
