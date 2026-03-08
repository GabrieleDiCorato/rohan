"""Shared Streamlit startup helpers."""

import logging

import streamlit as st

from rohan.framework.database import initialize_database

logger = logging.getLogger(__name__)


def ensure_db_initialized() -> None:
    """Initialize the database once per Streamlit session.

    Idempotent: repeated calls are no-ops after the first successful
    (or failed-but-logged) attempt.
    """
    if st.session_state.get("_db_initialised"):
        return
    try:
        initialize_database()
    except Exception:
        logger.warning("Database initialization failed — persistence disabled", exc_info=True)
    st.session_state["_db_initialised"] = True
