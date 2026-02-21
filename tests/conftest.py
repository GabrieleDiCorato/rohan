"""Shared test fixtures.

Provides a fresh in-memory SQLite database for every test, ensuring each
test starts with clean tables matching the latest schema.
"""

import contextlib
import os

import pytest

# Force an in-memory SQLite URL for the whole test run, before any
# DatabaseConnector is instantiated.
os.environ.setdefault("DB_CONNECTION_STRING", "sqlite:///:memory:")


from rohan.framework.database.database_connector import get_database_connector  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_db_singleton():
    """Reset the DB lru_cache before each test so every test gets fresh tables.

    This avoids leaking state between tests (stale engine/session pointing
    at a DB with the old schema, or leftover rows).
    """
    # Tear down any cached connector
    get_database_connector.cache_clear()

    # Configure a new in-memory DB
    os.environ["DB_CONNECTION_STRING"] = "sqlite:///:memory:"

    yield

    # Cleanup after test
    try:
        get_database_connector().dispose()
    except Exception:
        contextlib.suppress(Exception)  # Ignore errors during cleanup
    get_database_connector.cache_clear()
