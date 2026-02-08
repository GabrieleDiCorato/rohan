"""
Database initialization script.

This module provides utilities to initialize the database schema
for the agentic simulation framework.
"""

import logging

from .database_connector import DatabaseConnector

logger = logging.getLogger(__name__)


def initialize_database() -> None:
    """
    Create all tables in the database.

    This function uses SQLAlchemy's create_all() to create all tables
    defined in models.py. It's idempotent - running it multiple times
    won't cause errors.
    """
    db = DatabaseConnector()
    db.create_tables()
    logger.info("Database tables created successfully")


def drop_all_tables() -> None:
    """
    Drop all tables from the database.

    WARNING: This will delete all data! Use with caution.
    """
    from rohan.framework.database.models import Base

    db = DatabaseConnector()
    Base.metadata.drop_all(bind=db.engine)
    logger.info("All tables dropped")


def reset_database() -> None:
    """
    Drop all tables and recreate them.

    WARNING: This will delete all data! Use with caution.
    """
    drop_all_tables()
    initialize_database()
    logger.info("Database reset complete")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "init":
            initialize_database()
        elif command == "drop":
            drop_all_tables()
        elif command == "reset":
            reset_database()
        else:
            logger.error("Unknown command: %s", command)
            print("Usage: python -m rohan.framework.init_db [init|drop|reset]")
    else:
        # Default action
        initialize_database()
