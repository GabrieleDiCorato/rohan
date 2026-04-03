"""Reusable HTML/CSS UI components for the Rohan financial cockpit.

Every function returns a raw HTML string suitable for
``st.markdown(html, unsafe_allow_html=True)``.
"""

from __future__ import annotations

import html as _html
from typing import Any

# ── Glassmorphism metric card ─────────────────────────────────────────────────


def glassmorphism_card(label: str, value: str, subtitle: str = "", delta: str = "", delta_color: str = "", description: str = "") -> str:
    """Single KPI card with backdrop blur and glass border.

    Parameters
    ----------
    description:
        If provided, an **ⓘ** icon is appended to the label.  Hovering over the
        icon reveals the full description via the native browser ``title`` tooltip.
    """
    esc = _html.escape
    delta_html = ""
    if delta:
        color = delta_color or "#8A919B"
        delta_html = f"<div style=\"font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:{esc(color)};margin-top:2px\">{esc(delta)}</div>"
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div style="font-family:\'Inter\',sans-serif;font-size:0.65rem;color:#6B7280;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="{esc(subtitle)}">{esc(subtitle)}</div>'

    info_html = ""
    if description:
        info_html = f' <span title="{esc(description, quote=True)}" style="cursor:help;font-size:0.62rem;color:#6B7280;vertical-align:middle">ⓘ</span>'

    card_title = f' title="{esc(description, quote=True)}"' if description else ""

    return (
        f'<div{card_title} style="'
        "background:rgba(11,14,20,0.65);"
        "backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);"
        "border:1px solid rgba(255,255,255,0.08);"
        "border-radius:8px;"
        "padding:12px 16px;"
        "min-width:0;"
        "position:relative;"
        '">'
        f"<div style=\"font-family:'Inter',sans-serif;font-size:0.68rem;font-weight:500;color:#8A919B;text-transform:uppercase;letter-spacing:0.04em;white-space:nowrap;overflow:hidden;text-overflow:ellipsis\">{esc(label)}{info_html}</div>"
        f"<div style=\"font-family:'JetBrains Mono',monospace;font-size:1.35rem;font-weight:700;color:#E0E0E0;margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis\">{esc(value)}</div>"
        f"{delta_html}"
        f"{subtitle_html}"
        "</div>"
    )


def metric_row(cards: list[dict[str, Any]]) -> str:
    """Wrap N glassmorphism cards in a responsive CSS-grid row.

    Each entry in *cards* is a dict of kwargs for :func:`glassmorphism_card`.
    If a card dict has no ``description`` key, the matching entry from
    :data:`~rohan.ui.utils.descriptions.METRIC_DESCRIPTIONS` is injected
    automatically so callers don't need to pass descriptions manually.
    """
    from rohan.ui.utils.descriptions import get_description

    enriched: list[dict[str, Any]] = []
    for c in cards:
        if "description" not in c:
            c = {**c, "description": get_description(c.get("label", ""))}
        enriched.append(c)

    inner = "".join(glassmorphism_card(**c) for c in enriched)
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(140px, 1fr));gap:10px;margin-bottom:12px;">{inner}</div>'


# ── Agent composition "Market Recipe" bar ─────────────────────────────────────

_RECIPE_COLORS: list[str] = [
    "#0070FF",  # blue
    "#00C805",  # green
    "#FF3B3F",  # red
    "#FFA500",  # orange
    "#9467bd",  # purple
    "#17becf",  # cyan
    "#e377c2",  # pink
    "#8c564b",  # brown
]


def agent_recipe_bar(agent_configs: dict[str, Any]) -> str:
    """Horizontal stacked bar showing agent composition by type.

    *agent_configs* maps agent-type name to an object with a ``.count`` attribute
    (or a dict with a ``"count"`` key).
    """
    segments: list[tuple[str, int]] = []
    for name, cfg in agent_configs.items():
        cnt = cfg.count if hasattr(cfg, "count") else cfg.get("count", 0)
        if cnt > 0:
            segments.append((name, cnt))

    if not segments:
        return ""

    total = sum(c for _, c in segments)
    bar_parts = ""
    legend_parts = ""
    for i, (name, cnt) in enumerate(segments):
        pct = cnt / total * 100
        color = _RECIPE_COLORS[i % len(_RECIPE_COLORS)]
        esc_name = _html.escape(name)
        bar_parts += f'<div style="width:{pct:.1f}%;background:{color};height:100%;display:inline-block" title="{esc_name}: {cnt} ({pct:.1f}%)"></div>'
        legend_parts += (
            f'<span style="font-size:0.65rem;color:#8A919B;white-space:nowrap">'
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:2px;background:{color};margin-right:3px;vertical-align:middle"></span>'
            f"{esc_name} {pct:.0f}%</span> "
        )

    return (
        '<div style="margin:8px 0">'
        '<div style="'
        "height:14px;"
        "border-radius:4px;"
        "overflow:hidden;"
        "background:rgba(255,255,255,0.04);"
        "display:flex;"
        f'">{bar_parts}</div>'
        f'<div style="margin-top:4px;line-height:1.5">{legend_parts}</div>'
        "</div>"
    )


# ── Execution console (CRT-style log viewer) ─────────────────────────────────


def execution_console(log_lines: list[str], wall_time: float | None = None) -> str:
    """Terminal-style execution log with green CRT glow."""
    lines_html = "".join(
        f'<div style="padding:1px 0;opacity:0.92"><span style="color:#4a5568;margin-right:8px;user-select:none">{i + 1:>4}</span>{_html.escape(line)}</div>' for i, line in enumerate(log_lines)
    )

    footer = ""
    if wall_time is not None:
        footer = f'<div style="border-top:1px solid rgba(0,200,5,0.15);margin-top:8px;padding-top:6px;font-size:0.72rem;color:#4a5568">Simulation completed in {wall_time:.2f}s</div>'

    return (
        '<div style="'
        "background:rgba(0,20,0,0.35);"
        "border:1px solid rgba(0,200,5,0.12);"
        "border-radius:6px;"
        "box-shadow:0 0 20px rgba(0,255,65,0.06),inset 0 0 60px rgba(0,20,0,0.3);"
        "padding:12px 16px;"
        "font-family:'JetBrains Mono',monospace;"
        "font-size:0.72rem;"
        "line-height:1.55;"
        "color:#00C805;"
        "max-height:400px;"
        "overflow-y:auto;"
        "opacity:0.85;"
        '">'
        '<div style="color:#4a5568;margin-bottom:6px;font-size:0.68rem;letter-spacing:0.05em">'
        "─── EXECUTION LOG ───</div>"
        f"{lines_html}"
        f"{footer}"
        "</div>"
    )
