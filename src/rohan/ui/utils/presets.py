"""Preset simulation configurations backed by hasufel templates.

Discovers available templates dynamically via hasufel's ``list_templates()``
so that new upstream templates are automatically surfaced.  Only market timing
and infrastructure settings are overridden.
"""

from __future__ import annotations

import functools
import logging

from rohan.config import SimulationSettings

logger = logging.getLogger(__name__)

# Shared timing/infra overrides applied to every preset.
_COMMON_OVERRIDES: dict[str, object] = {
    "date": "20260130",
    "start_time": "09:30:00",
    "end_time": "11:30:00",
    "starting_cash": 10_000_000,
    "log_orders": True,
}

# Friendly display names for well-known templates.
# Any template NOT in this mapping gets a title-cased name derived
# from its hasufel identifier (e.g. "thin_market" → "Thin Market").
_DISPLAY_NAMES: dict[str, str] = {
    "rmsc04": "Default (Balanced Market)",
    "volatile_day": "High Volatility",
    "low_liquidity": "Low Liquidity",
    "stress_test": "Market Maker Stress Test",
    "trending_day": "Momentum Dominated",
    "liquid_market": "High Liquidity",
    "thin_market": "Thin Market",
    "stable_day": "Stable Day",
}


@functools.lru_cache(maxsize=1)
def _discover_templates() -> list[dict[str, str]]:
    """Return ``[{display, template, description}, ...]`` for non-overlay templates."""
    from abides_markets.config_system import list_templates

    entries: list[dict[str, str]] = []
    for t in list_templates():
        if t.get("is_overlay"):
            continue
        name: str = t["name"]
        display = _DISPLAY_NAMES.get(name, name.replace("_", " ").title())
        entries.append(
            {
                "display": display,
                "template": name,
                "description": t.get("description", ""),
            }
        )

    # Put the default first, then alphabetical by display name.
    entries.sort(key=lambda e: (e["template"] != "rmsc04", e["display"]))
    return entries


def get_preset_names() -> list[str]:
    """Get list of available preset display names."""
    return [e["display"] for e in _discover_templates()]


def get_preset_description(preset_name: str) -> str:
    """Get the hasufel description for a preset, or empty string."""
    for e in _discover_templates():
        if e["display"] == preset_name:
            return e["description"]
    return ""


def get_preset_config(preset_name: str) -> SimulationSettings:
    """Get a preset simulation configuration by display name.

    Returns a ``SimulationSettings`` with the ``template`` field set so that
    ``create_simulation_builder()`` loads agent composition from hasufel.
    """
    template = "rmsc04"  # fallback
    for e in _discover_templates():
        if e["display"] == preset_name:
            template = e["template"]
            break
    return SimulationSettings(template=template, **_COMMON_OVERRIDES)  # type: ignore[arg-type]
