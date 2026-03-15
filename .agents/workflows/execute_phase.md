---
description: Execute a scenario customizability implementation phase
---
# Execute Scenario Customizability Phase

Use this workflow to kick off a new session for executing a specific phase of the Scenario Customizability plan.

**Instructions:**
1. Determine which phase you are ready to execute (1 through 5).
2. Copy the prompt template below.
3. Replace the `[INSERT PHASE NUMBER]` placeholder with the target phase.
4. Start a new chat session with your AI assistant and paste the prompt.

## The Kickoff Prompt Template

```markdown
I want to execute **[INSERT PHASE NUMBER: e.g., Phase 1]** from `docs/technical/scenario_customizability_plan.md`.

**Context:**
- The authoritative plan is in `docs/technical/scenario_customizability_plan.md`. Read the section for this phase carefully.
- If this phase references detailed design (e.g., Regimes or Adversary), read the relevant section in `docs/technical/adversarial_scenario_system.md`.
- Do not worry about backward compatibility or migration scripts. We can drop the database and break old scenarios.

**Workflow:**
1. **Detailed Design First:** Do not write code yet. Read the plan, explore the current codebase, and write a detailed technical design for this specific phase in a temporary artifact. Ask me to review it.
2. **Execution:** Once I approve the design, implement the code. Follow the rule of Incremental Complexity (interfaces first, then implementation, then UI).
3. **Verification:**
   - Write tests covering standard and edge cases (use parameterized tests and `hypothesis` when necessary).
   - Run a subset of impacted test cases iteratively to fix errors (the full parallel suite takes >5 mins). Include mocks if external APIs or DBs are used.
   - Run all `pre-commit` checks and fix failures. Ensure type safety for `pyright`.
   - If there are UI changes, guide me on how to test them locally.
4. **Commit & Document:** Update the appropriate `docs/` markdown files (do not document in the plan file, just check off the completed items there). Commit the changes using past-tense descriptions (e.g., "Added...", "Fixed..."). Explain the *what* and *why*, but do *not* reference the plan file in the commit body.

Please begin step 1: read the plan and prepare the detailed design for my review.
```
