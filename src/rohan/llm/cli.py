"""CLI entry-point for the strategy refinement loop.

Usage::

    python -m rohan.llm.cli --goal "Create a simple market-making strategy" --max-iterations 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from rohan.llm.graph import run_refinement
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
        default=3,
        help="Maximum refinement iterations (default: 3)",
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

    # Run
    final_state = run_refinement(
        goal=args.goal,
        max_iterations=args.max_iterations,
        scenarios=scenarios,
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
