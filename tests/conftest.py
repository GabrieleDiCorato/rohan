"""Shared test fixtures.

Provides a fresh in-memory SQLite database for every test, ensuring each
test starts with clean tables matching the latest schema.
"""

import os

import pytest

# Force an in-memory SQLite URL for the whole test run, before any
# DatabaseConnector is instantiated.
os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")


from rohan.framework.database.database_connector import DatabaseConnector  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    """Reset the DB singleton before each test so every test gets fresh tables.

    This avoids leaking state between tests (stale engine/session pointing
    at a DB with the old schema, or leftover rows).
    """
    # Tear down any previous singleton
    DatabaseConnector.reset_singleton()

    # Configure a new in-memory DB
    os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"

    yield

    # Cleanup after test
    DatabaseConnector.reset_singleton()
