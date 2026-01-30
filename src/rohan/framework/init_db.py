"""
Database initialization script.

This module provides utilities to initialize the database schema
for the agentic simulation framework.
"""

from rohan.framework.database import Database


def initialize_database() -> None:
    """
    Create all tables in the database.

    This function uses SQLAlchemy's create_all() to create all tables
    defined in models.py. It's idempotent - running it multiple times
    won't cause errors.
    """
    db = Database()
    db.create_tables()
    print("✓ Database tables created successfully")


def drop_all_tables() -> None:
    """
    Drop all tables from the database.

    WARNING: This will delete all data! Use with caution.
    """
    from rohan.framework.models import Base

    db = Database()
    Base.metadata.drop_all(bind=db.engine)
    print("✓ All tables dropped")


def reset_database() -> None:
    """
    Drop all tables and recreate them.

    WARNING: This will delete all data! Use with caution.
    """
    drop_all_tables()
    initialize_database()
    print("✓ Database reset complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "init":
            initialize_database()
        elif command == "drop":
            drop_all_tables()
        elif command == "reset":
            reset_database()
        else:
            print(f"Unknown command: {command}")
            print("Usage: python -m rohan.framework.init_db [init|drop|reset]")
    else:
        # Default action
        initialize_database()
