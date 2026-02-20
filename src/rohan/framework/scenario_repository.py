"""Repository for saved simulation scenarios (Terminal ↔ Refinement Lab).

CRUD operations on the ``saved_scenarios`` table.  Each saved scenario
stores a full ``SimulationSettings`` JSON blob so it is self-contained
and reusable across any context.
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID

from sqlalchemy import select

from rohan.framework.database import DatabaseConnector, SavedScenario

logger = logging.getLogger(__name__)


class ScenarioRepository:
    """Manage saved simulation scenarios in the database."""

    def __init__(self, db: DatabaseConnector | None = None) -> None:
        self.db = db or DatabaseConnector()

    # ------------------------------------------------------------------
    # Create
    # ------------------------------------------------------------------

    def save_scenario(
        self,
        name: str,
        full_config: dict[str, Any],
        description: str | None = None,
    ) -> SavedScenario:
        """Persist a new named scenario. Raises on duplicate name."""
        scenario = SavedScenario(
            name=name,
            description=description,
            full_config=full_config,
        )
        session = self.db.get_session()
        try:
            session.add(scenario)
            session.commit()
            session.refresh(scenario)
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()
        return scenario

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def list_scenarios(self) -> list[SavedScenario]:
        """Return all saved scenarios ordered by creation date (newest first)."""
        session = self.db.get_session()
        try:
            result = session.execute(select(SavedScenario).order_by(SavedScenario.created_at.desc()))
            return list(result.scalars().all())
        finally:
            self.db.remove_session()

    def get_scenario(self, scenario_id: UUID) -> SavedScenario | None:
        """Retrieve a single scenario by its ID."""
        session = self.db.get_session()
        try:
            result: SavedScenario | None = session.get(SavedScenario, scenario_id)
            return result
        finally:
            self.db.remove_session()

    def get_scenario_by_name(self, name: str) -> SavedScenario | None:
        """Retrieve a single scenario by its unique name."""
        session = self.db.get_session()
        try:
            result = session.execute(select(SavedScenario).where(SavedScenario.name == name))
            row: SavedScenario | None = result.scalar_one_or_none()
            return row
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update_scenario(
        self,
        scenario_id: UUID,
        *,
        name: str | None = None,
        description: str | None = None,
        full_config: dict[str, Any] | None = None,
    ) -> SavedScenario | None:
        """Update fields of an existing saved scenario."""
        session = self.db.get_session()
        try:
            scenario: SavedScenario | None = session.get(SavedScenario, scenario_id)
            if scenario is None:
                return None
            if name is not None:
                scenario.name = name
            if description is not None:
                scenario.description = description
            if full_config is not None:
                scenario.full_config = full_config
            session.commit()
            session.refresh(scenario)
            return scenario
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_scenario(self, scenario_id: UUID) -> bool:
        """Delete a saved scenario. Returns True if something was deleted."""
        session = self.db.get_session()
        try:
            obj = session.get(SavedScenario, scenario_id)
            if obj is None:
                return False
            session.delete(obj)
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            self.db.remove_session()
