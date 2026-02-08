"""LLM integration layer for the Rohan framework.

Provides:
- :mod:`~rohan.llm.factory` — Model factory using LangChain chat models.
- :mod:`~rohan.llm.models`  — Pydantic response/state models.
- :mod:`~rohan.llm.prompts` — Prompt templates for each agent role.
- :mod:`~rohan.llm.tools`   — LangChain tools wrapping analysis functions.
"""

from __future__ import annotations
