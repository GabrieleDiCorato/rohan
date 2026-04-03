"""Tests for the database ORM models (Session, Iteration, ScenarioRun, Artifact, SavedScenario).

Verifies table creation, field defaults, and relationship cascades
using the in-memory SQLite from conftest.py.
"""

from rohan.framework.database import (
    Artifact,
    DatabaseConnector,
    Iteration,
    SavedScenario,
    ScenarioRun,
    Session,
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


class TestSessionModels:
    """Tests for Session, Iteration, ScenarioRun, Artifact ORM models."""

    def _setup_db(self):
        db = DatabaseConnector()
        db.create_tables()
        return db

    def test_create_session(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            s = Session(
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
            session.add(s)
            session.commit()
            session.refresh(s)

            assert s.session_id is not None
            assert s.name == "Run 1"
            assert s.goal == "Make money"
            assert s.max_iterations == 3
            assert s.final_score == 8.5
            assert s.status == "done"
            assert s.iterations == []
        finally:
            db.remove_session()

    def test_iteration_relationship(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            s = Session(
                name="RelTest",
                goal="g",
                max_iterations=1,
                scenario_configs=[],
                status="done",
            )
            it = Iteration(
                iteration_number=1,
                strategy_code="class S: pass",
                class_name="S",
                judge_score=7.0,
                judge_reasoning="OK",
            )
            s.iterations.append(it)
            session.add(s)
            session.commit()
            session.refresh(s)

            assert len(s.iterations) == 1
            assert s.iterations[0].iteration_number == 1
            assert s.iterations[0].session_id == s.session_id
        finally:
            db.remove_session()

    def test_scenario_run_relationship(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            s = Session(
                name="SRTest",
                goal="g",
                max_iterations=1,
                scenario_configs=[],
                status="done",
            )
            it = Iteration(
                iteration_number=1,
                strategy_code="code",
            )
            sr = ScenarioRun(
                scenario_name="Default",
                domain_metrics={
                    "agent": {"total_pnl": 500.0, "trade_count": 25},
                    "impact": {"volatility_delta_pct": 0.05, "spread_delta_pct": -0.02},
                },
                compiled_config={"agents": [{"type": "NoiseAgent"}]},
                hasufel_summary={"total_volume": 5000},
            )
            it.scenario_runs.append(sr)
            s.iterations.append(it)
            session.add(s)
            session.commit()
            session.refresh(s)

            loaded_sr = s.iterations[0].scenario_runs[0]
            assert loaded_sr.scenario_name == "Default"
            assert loaded_sr.domain_metrics["agent"]["total_pnl"] == 500.0
            assert loaded_sr.compiled_config["agents"][0]["type"] == "NoiseAgent"
            assert loaded_sr.hasufel_summary["total_volume"] == 5000
        finally:
            db.remove_session()

    def test_artifact_relationship(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            s = Session(name="ArtTest", goal="g", max_iterations=1, scenario_configs=[], status="done")
            it = Iteration(iteration_number=1, strategy_code="code")
            sr = ScenarioRun(scenario_name="Default")
            art = Artifact(artifact_type="price_chart_b64", content="cHJpY2U=")
            sr.artifacts.append(art)
            it.scenario_runs.append(sr)
            s.iterations.append(it)
            session.add(s)
            session.commit()
            session.refresh(s)

            loaded_art = s.iterations[0].scenario_runs[0].artifacts[0]
            assert loaded_art.artifact_type == "price_chart_b64"
            assert loaded_art.content == "cHJpY2U="
        finally:
            db.remove_session()

    def test_cascade_delete(self):
        """Deleting a session should cascade-delete iterations, runs, and artifacts."""
        db = self._setup_db()
        session = db.get_session()
        try:
            s = Session(name="CascadeDel", goal="g", max_iterations=1, scenario_configs=[], status="done")
            it = Iteration(iteration_number=1, strategy_code="c")
            sr = ScenarioRun(scenario_name="A")
            sr.artifacts.append(Artifact(artifact_type="chart", content="data"))
            it.scenario_runs.append(sr)
            s.iterations.append(it)
            session.add(s)
            session.commit()

            sid = s.session_id
            session.delete(s)
            session.commit()

            assert session.get(Session, sid) is None
        finally:
            db.remove_session()

    def test_multiple_scenario_runs_per_iteration(self):
        db = self._setup_db()
        session = db.get_session()
        try:
            s = Session(name="MultiSR", goal="g", max_iterations=1, scenario_configs=[], status="done")
            it = Iteration(iteration_number=1, strategy_code="code")
            it.scenario_runs.append(ScenarioRun(scenario_name="Default", domain_metrics={"agent": {"trade_count": 10}}))
            it.scenario_runs.append(ScenarioRun(scenario_name="HighVol", domain_metrics={"agent": {"total_pnl": -200.0, "trade_count": 20}}))
            s.iterations.append(it)
            session.add(s)
            session.commit()
            session.refresh(s)

            runs = s.iterations[0].scenario_runs
            assert len(runs) == 2
            names = {r.scenario_name for r in runs}
            assert names == {"Default", "HighVol"}
        finally:
            db.remove_session()
