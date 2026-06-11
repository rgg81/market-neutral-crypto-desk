from pathlib import Path

import pytest

# Task 4.1 analyst roster for the market-neutral desk: the seven signal/analysis agents.
# Each emits a JSON output validated against its pydantic contract (see test_agent_conformance).
# The two deterministic doc-only roles (Neutrality Constructor, Risk Gate) arrive in Task 4.3 and
# are EXEMPT from the Output-JSON requirement.
ANALYST_ROLES = [
    "universe_scout",
    "funding_carry",
    "pair_analyst",
    "factor_analyst",
    "sentiment",
    "technical",
    "derivatives",
]

# Roles exempt from the `## Output (return ONLY this JSON ...)` requirement (deterministic,
# code-enforced; no LLM JSON). Added in later Phase-4 tasks; listed here so the structural test
# stays correct as the roster grows.
DOC_ONLY_ROLES = {"neutrality_constructor", "risk_gate"}

# Mandatory section structure every analyst/decision prompt must carry (plan Task 4.1 Step 1).
REQUIRED_SECTIONS = ["## Mission", "## Inputs", "## How you think", "## Example"]


@pytest.mark.parametrize("role", ANALYST_ROLES)
def test_analyst_role_file_exists_and_has_mandatory_sections(role):
    p = Path("agents") / f"{role}.md"
    assert p.exists(), f"missing role file: {p}"
    text = p.read_text()

    # `# Title` — a top-level H1 header on the first non-blank line.
    first = next(line for line in text.splitlines() if line.strip())
    assert first.startswith("# "), f"{role}: missing `# Title` H1 header"

    for section in REQUIRED_SECTIONS:
        assert section in text, f"{role}: missing `{section}` section"

    if role not in DOC_ONLY_ROLES:
        # The Output contract must be returned as JSON ONLY (no prose) and carry a fenced
        # ```json``` example block — that block is a literal example of the agent's contract.
        assert "## Output (return ONLY this JSON, no prose)" in text, (
            f"{role}: missing the canonical `## Output (return ONLY this JSON, no prose)` header"
        )
        assert "```json" in text, f"{role}: missing a fenced ```json``` Output block"
