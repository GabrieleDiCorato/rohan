"""CLI entry-point for the strategy refinement loop.

Usage::

    python -m rohan.llm.cli --goal "Create a simple market-making strategy" --max-iterations 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rohan.llm.graph import DEFAULT_MAX_ITERATIONS, run_refinement
from rohan.llm.state import ScenarioConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the autonomous strategy refinement loop",
        prog="rohan-refine",
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="Natural language description of the desired trading strategy",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum refinement iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help='JSON array of scenario configs, e.g. \'[{"name":"volatile","config_override":{"seed":42}}]\'',
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable DEBUG logging",
    )
    parser.add_argument(
        "--adversarial-scenarios",
        type=int,
        default=None,
        help="Number of adversarial scenarios to add (overrides config; 0 disables planner)",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Parse scenarios
    scenarios = None
    if args.scenarios:
        raw = json.loads(args.scenarios)
        scenarios = [ScenarioConfig(**s) for s in raw]

    # Run scenario planner
    from rohan.config.llm_settings import LLMSettings
    from rohan.llm.planner import plan_scenarios

    settings = LLMSettings()
    if args.adversarial_scenarios is not None:
        settings.max_adversarial_scenarios = args.adversarial_scenarios

    user_scenarios = scenarios or [ScenarioConfig(name="default")]
    planned_scenarios, plan_reasoning = plan_scenarios(
        goal=args.goal,
        user_scenarios=user_scenarios,
        settings=settings,
    )

    if plan_reasoning:
        print(f"\n[Planner] {plan_reasoning}")
    print(f"[Planner] Running with {len(planned_scenarios)} scenario(s): {[s.name for s in planned_scenarios]}")

    # Run
    final_state = run_refinement(
        goal=args.goal,
        max_iterations=args.max_iterations,
        scenarios=planned_scenarios,
        scenario_plan_reasoning=plan_reasoning,
    )

    # Print results
    iterations = final_state.get("iterations", [])
    print(f"\n{'=' * 60}")
    print(f"Refinement complete — {len(iterations)} iteration(s)")
    print(f"{'=' * 60}")

    if iterations:
        last = iterations[-1]
        print(f"Final score: {last.judge_score}/10")
        print(f"Judge reasoning: {last.judge_reasoning}")
        print(f"\nFinal strategy code:\n{'─' * 60}")
        print(final_state.get("current_code", "(no code)"))
    else:
        print("No iterations completed.")
        errors = final_state.get("validation_errors", [])
        if errors:
            print(f"Validation errors: {errors}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
