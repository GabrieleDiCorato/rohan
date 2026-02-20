"""Tests for ScenarioRepository — CRUD on saved_scenarios table.

Uses the in-memory SQLite database wired by conftest.py.
"""

from rohan.framework.database import DatabaseConnector
from rohan.framework.scenario_repository import ScenarioRepository


class TestScenarioRepository:
    """Test suite for ScenarioRepository CRUD operations."""

    def _make_repo(self) -> ScenarioRepository:
        db = DatabaseConnector()
        db.create_tables()
        return ScenarioRepository(db)

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def test_save_scenario(self):
        repo = self._make_repo()
        sc = repo.save_scenario(
            name="High Volatility",
            full_config={"noise_agents": 500, "starting_cash": 100_000_00},
            description="Stress test with noisy market",
        )
        assert sc.scenario_id is not None
        assert sc.name == "High Volatility"
        assert sc.full_config["noise_agents"] == 500
        assert sc.description == "Stress test with noisy market"
        assert sc.created_at is not None

    def test_save_duplicate_name_raises(self):
        import pytest
        from sqlalchemy.exc import IntegrityError

        repo = self._make_repo()
        repo.save_scenario(name="dup", full_config={"a": 1})
        with pytest.raises(IntegrityError):
            repo.save_scenario(name="dup", full_config={"b": 2})

    def test_save_scenario_no_description(self):
        repo = self._make_repo()
        sc = repo.save_scenario(name="Minimal", full_config={"x": 42})
        assert sc.description is None

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def test_list_scenarios_empty(self):
        repo = self._make_repo()
        assert repo.list_scenarios() == []

    def test_list_scenarios_returns_all(self):
        repo = self._make_repo()
        repo.save_scenario(name="First", full_config={"a": 1})
        repo.save_scenario(name="Second", full_config={"b": 2})
        names = {s.name for s in repo.list_scenarios()}
        assert names == {"First", "Second"}

    def test_get_scenario_by_id(self):
        repo = self._make_repo()
        sc = repo.save_scenario(name="Lookup", full_config={"z": 0})
        result = repo.get_scenario(sc.scenario_id)
        assert result is not None
        assert result.name == "Lookup"

    def test_get_scenario_missing_returns_none(self):
        import uuid

        repo = self._make_repo()
        assert repo.get_scenario(uuid.uuid4()) is None

    def test_get_scenario_by_name(self):
        repo = self._make_repo()
        repo.save_scenario(name="ByName", full_config={"k": 1})
        result = repo.get_scenario_by_name("ByName")
        assert result is not None
        assert result.full_config["k"] == 1

    def test_get_scenario_by_name_missing_returns_none(self):
        repo = self._make_repo()
        assert repo.get_scenario_by_name("ghost") is None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def test_update_scenario_name(self):
        repo = self._make_repo()
        sc = repo.save_scenario(name="Old", full_config={"a": 1})
        updated = repo.update_scenario(sc.scenario_id, name="New")
        assert updated is not None
        assert updated.name == "New"
        assert updated.full_config == {"a": 1}

    def test_update_scenario_config(self):
        repo = self._make_repo()
        sc = repo.save_scenario(name="Cfg", full_config={"a": 1})
        updated = repo.update_scenario(sc.scenario_id, full_config={"a": 2, "b": 3})
        assert updated is not None
        assert updated.full_config == {"a": 2, "b": 3}

    def test_update_nonexistent_returns_none(self):
        import uuid

        repo = self._make_repo()
        assert repo.update_scenario(uuid.uuid4(), name="nope") is None

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def test_delete_scenario(self):
        repo = self._make_repo()
        sc = repo.save_scenario(name="ToDelete", full_config={"x": 1})
        assert repo.delete_scenario(sc.scenario_id) is True
        assert repo.get_scenario(sc.scenario_id) is None

    def test_delete_nonexistent_returns_false(self):
        import uuid

        repo = self._make_repo()
        assert repo.delete_scenario(uuid.uuid4()) is False
