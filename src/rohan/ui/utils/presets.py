"""Preset simulation configurations backed by hasufel templates.

Each preset maps to a registered hasufel template so that agent compositions
and oracle parameters stay in sync with upstream tuning.  Only market timing
and infrastructure settings are overridden.
"""

from rohan.config import SimulationSettings

# Mapping: UI display name → hasufel template name.
_PRESET_TEMPLATES: dict[str, str] = {
    "Default (Balanced Market)": "rmsc04",
    "High Volatility": "volatile_day",
    "Low Liquidity": "low_liquidity",
    "Market Maker Stress Test": "stress_test",
    "Momentum Dominated": "trending_day",
}

# Shared timing/infra overrides applied to every preset.
_COMMON_OVERRIDES: dict[str, object] = {
    "date": "20260130",
    "start_time": "09:30:00",
    "end_time": "11:30:00",
    "starting_cash": 10_000_000,
    "log_orders": True,
}


def get_preset_names() -> list[str]:
    """Get list of available preset names."""
    return list(_PRESET_TEMPLATES.keys())


def get_preset_config(preset_name: str) -> SimulationSettings:
    """Get a preset simulation configuration by name.

    Returns a ``SimulationSettings`` with the ``template`` field set so that
    ``create_simulation_builder()`` loads agent composition from hasufel.
    """
    template = _PRESET_TEMPLATES.get(preset_name, "rmsc04")
    return SimulationSettings(template=template, **_COMMON_OVERRIDES)  # type: ignore[arg-type]
