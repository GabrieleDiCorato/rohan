"""Scenario planner — selects and augments simulation scenarios before the refinement loop.

The planner runs ONCE before the LangGraph refinement graph starts.  It takes
the user-selected scenarios and a goal description, then optionally adds
adversarial / stress scenarios using a ReAct agent equipped with hasufel's
template-discovery tools.

Architecture
------------
Three-tier fallback to guarantee scenarios are always returned:

1. **ReAct agent** — full agentic planning with ``list_scenarios`` and
   ``build_scenario`` tools.
2. **Structured LLM** — single-shot structured output if the agent fails.
3. **Keyword heuristic** — deterministic keyword matching (no LLM needed).

The planner never *removes* user-selected scenarios — it only *adds* adversarial
ones (up to ``max_adversarial_scenarios``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from rohan.llm.models import PlannedScenario, ScenarioPlan

if TYPE_CHECKING:
    from rohan.config.llm_settings import LLMSettings
    from rohan.llm.state import ScenarioConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for the planner agent
# ---------------------------------------------------------------------------

PLANNER_SYSTEM = """\
You are the Scenario Planner for the ROHAN strategy-refinement framework.

Given a user's strategy goal and the scenarios they have already selected,
your job is to propose UP TO {max_adversarial} additional *adversarial* or
*stress-test* scenarios that will challenge the strategy from angles the user
may not have considered.

## Guidelines
- Use ``list_scenarios`` to discover available templates and their regime tags.
- Use ``build_scenario`` to assemble a concrete config for each proposed scenario.
- **Overlay composition**: ``build_scenario`` accepts an ``overlays`` parameter
  (list of overlay template names) that stack additional agent populations on top
  of the base template.  Use overlays like ``with_momentum`` or ``with_execution``
  to create richer adversarial environments without replacing the base config.
- Each adversarial scenario should target a different weakness axis
  (e.g. volatility, thin liquidity, wide spreads, informed flow).
- Give each scenario a short descriptive ``name`` slug and a ``rationale``
  explaining why it challenges the goal.
- Do NOT duplicate scenarios the user already selected.
- If the user's selection already covers the main risk axes, propose 0 extras.

## Output
Return a JSON object with:
- ``scenarios``: list of objects with ``name``, ``template_name``, ``regime_tags``,
  ``config_override``, and ``rationale``
- ``reasoning``: brief overall planning rationale
"""


# ---------------------------------------------------------------------------
# Keyword-based fallback (tier 3)
# ---------------------------------------------------------------------------

# Maps goal keywords → adversarial scenario descriptors
_GOAL_ADVERSARIAL_MAP: dict[str, PlannedScenario] = {
    "market-mak": PlannedScenario(
        name="volatile_stress",
        template_name="rmsc04",
        regime_tags=["volatile"],
        config_override={},
        rationale="Market-making strategies must survive volatile regimes",
    ),
    "spread": PlannedScenario(
        name="wide_spread",
        template_name="rmsc04",
        regime_tags=["wide_spread"],
        config_override={},
        rationale="Spread-focused strategies need testing under wide-spread conditions",
    ),
    "profit": PlannedScenario(
        name="thin_liquidity",
        template_name="rmsc04",
        regime_tags=["thin"],
        config_override={},
        rationale="Profit-seeking strategies should be tested with thin order books",
    ),
    "risk": PlannedScenario(
        name="volatile_stress",
        template_name="rmsc04",
        regime_tags=["volatile", "informed"],
        config_override={},
        rationale="Risk-focused strategies must handle volatility + informed flow",
    ),
    "vpin": PlannedScenario(
        name="informed_flow",
        template_name="rmsc04",
        regime_tags=["informed"],
        config_override={},
        rationale="VPIN-sensitive goals need testing under high informed-trading pressure",
    ),
    "execution": PlannedScenario(
        name="fast_moving",
        template_name="rmsc04",
        regime_tags=["volatile", "fast"],
        config_override={"overlays": ["with_execution"]},
        rationale="Execution quality degrades in fast-moving markets with competing execution agents",
    ),
}


def _keyword_fallback(goal: str, max_adversarial: int) -> list[PlannedScenario]:
    """Deterministic keyword matching — always succeeds."""
    goal_lower = goal.lower()
    seen_names: set[str] = set()
    results: list[PlannedScenario] = []

    for keyword, scenario in _GOAL_ADVERSARIAL_MAP.items():
        if keyword in goal_lower and scenario.name not in seen_names:
            results.append(scenario)
            seen_names.add(scenario.name)
            if len(results) >= max_adversarial:
                break

    # If nothing matched, add a generic volatile stress scenario
    if not results and max_adversarial > 0:
        results.append(
            PlannedScenario(
                name="volatile_stress",
                template_name="rmsc04",
                regime_tags=["volatile"],
                config_override={},
                rationale="Default adversarial scenario — volatile market conditions",
            )
        )

    return results[:max_adversarial]


# ---------------------------------------------------------------------------
# Structured LLM fallback (tier 2)
# ---------------------------------------------------------------------------


def _structured_fallback(
    goal: str,
    user_scenario_names: list[str],
    max_adversarial: int,
    settings: LLMSettings,
) -> list[PlannedScenario]:
    """Single-shot structured output — no tools, just scenario selection."""
    from rohan.llm.factory import get_planner_model, get_structured_model

    model = get_planner_model(settings)
    structured = get_structured_model(model, ScenarioPlan)

    prompt = (
        f"Goal: {goal}\n"
        f"User-selected scenarios: {user_scenario_names}\n"
        f"Propose up to {max_adversarial} adversarial scenarios that would "
        f"stress-test a strategy designed for this goal. "
        f"Do NOT duplicate user-selected scenarios."
    )

    try:
        result = structured.invoke(prompt)
        # get_structured_model with include_raw=True returns a dict
        parsed = result.get("parsed") if isinstance(result, dict) else result
        if parsed is None:
            logger.warning("Structured planner returned None — falling back to keywords")
            return _keyword_fallback(goal, max_adversarial)
        return parsed.scenarios[:max_adversarial]
    except Exception:
        logger.exception("Structured planner failed")
        return _keyword_fallback(goal, max_adversarial)


# ---------------------------------------------------------------------------
# ReAct agent planner (tier 1)
# ---------------------------------------------------------------------------


def _react_planner(
    goal: str,
    user_scenario_names: list[str],
    max_adversarial: int,
    settings: LLMSettings,
) -> ScenarioPlan:
    """Full ReAct agent with scenario tools — the richest planning mode."""
    from langgraph.prebuilt import create_react_agent

    from rohan.llm.factory import get_planner_model, get_structured_model
    from rohan.llm.scenario_tools import make_scenario_tools

    model = get_planner_model(settings)
    tools = make_scenario_tools()

    agent = create_react_agent(
        model,
        tools,
        prompt=PLANNER_SYSTEM.format(max_adversarial=max_adversarial),
    )

    user_msg = f"Goal: {goal}\nUser-selected scenarios: {user_scenario_names}\nPropose up to {max_adversarial} additional adversarial scenarios."

    result = agent.invoke({"messages": [("user", user_msg)]})

    # Extract the final AI message content
    messages = result.get("messages", [])
    if not messages:
        raise ValueError("ReAct agent returned no messages")

    last_content = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # Parse the agent output as ScenarioPlan using structured extraction
    structured = get_structured_model(get_planner_model(settings), ScenarioPlan)
    parsed_result = structured.invoke(f"Extract the scenario plan from this agent output:\n\n{last_content}")
    parsed = parsed_result.get("parsed") if isinstance(parsed_result, dict) else parsed_result
    if parsed is None:
        raise ValueError("Failed to parse agent output into ScenarioPlan")

    return parsed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _to_scenario_config(planned: PlannedScenario) -> ScenarioConfig:
    """Convert a PlannedScenario to a ScenarioConfig for the graph."""
    from rohan.llm.state import ScenarioConfig

    override = dict(planned.config_override)
    if planned.template_name:
        override["template"] = planned.template_name
    if planned.regime_tags:
        override["regime_tags"] = planned.regime_tags

    return ScenarioConfig(name=planned.name, config_override=override)


def plan_scenarios(
    goal: str,
    user_scenarios: list[ScenarioConfig],
    settings: LLMSettings | None = None,
) -> tuple[list[ScenarioConfig], str]:
    """Plan the full scenario set: user-selected + adversarial.

    Parameters
    ----------
    goal:
        Natural language strategy description.
    user_scenarios:
        Scenarios explicitly chosen by the user (never removed).
    settings:
        LLM configuration.  Loaded from env if not provided.

    Returns
    -------
    tuple[list[ScenarioConfig], str]
        (merged scenario list, planning reasoning text)
    """
    from rohan.config.llm_settings import LLMSettings

    if settings is None:
        settings = LLMSettings()

    max_adversarial = settings.max_adversarial_scenarios
    if max_adversarial <= 0:
        logger.info("Adversarial planning disabled (max_adversarial_scenarios=0)")
        return list(user_scenarios), "Adversarial planning disabled by configuration."

    user_names = [s.name for s in user_scenarios]
    reasoning = ""

    # Tier 1: ReAct agent
    try:
        logger.info("Planner tier 1: ReAct agent (max_adversarial=%d)", max_adversarial)
        plan = _react_planner(goal, user_names, max_adversarial, settings)
        # Filter out duplicates of user-selected scenarios
        adversarial = [s for s in plan.scenarios if s.name not in user_names][:max_adversarial]
        reasoning = plan.reasoning
        logger.info("ReAct planner proposed %d adversarial scenario(s)", len(adversarial))
    except Exception:
        logger.exception("Planner tier 1 (ReAct) failed — trying structured fallback")

        # Tier 2: Structured LLM
        try:
            adversarial = _structured_fallback(goal, user_names, max_adversarial, settings)
            reasoning = "Planned via structured LLM fallback (ReAct agent unavailable)."
        except Exception:
            logger.exception("Planner tier 2 (structured) failed — using keyword heuristic")

            # Tier 3: Keywords
            adversarial = _keyword_fallback(goal, max_adversarial)
            reasoning = "Planned via keyword heuristic (LLM unavailable)."

    # Merge: user scenarios first, then adversarial
    merged = list(user_scenarios) + [_to_scenario_config(s) for s in adversarial]

    logger.info(
        "Final scenario plan: %d user + %d adversarial = %d total",
        len(user_scenarios),
        len(adversarial),
        len(merged),
    )
    return merged, reasoning
