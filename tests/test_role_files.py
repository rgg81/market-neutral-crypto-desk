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

# Task 4.2 debate + decision + execution + learning roles. Each emits a JSON output validated
# against its pydantic contract (see test_agent_conformance): Bull/Bear -> AnalystReport (stance
# bullish/bearish), Research Manager -> ResearchPlan (5-tier rating), Trader -> TraderOutput,
# Reflector -> Lesson. They carry the SAME mandatory section structure as the analysts.
DECISION_ROLES = [
    "bull",
    "bear",
    "research_manager",
    "trader",
    "reflector",
]

# Roles exempt from the `## Output (return ONLY this JSON ...)` requirement (deterministic,
# code-enforced; no LLM JSON). Added in later Phase-4 tasks; listed here so the structural test
# stays correct as the roster grows.
DOC_ONLY_ROLES = {"neutrality_constructor", "risk_gate"}

# Mandatory section structure every analyst/decision prompt must carry (plan Task 4.1 Step 1).
REQUIRED_SECTIONS = ["## Mission", "## Inputs", "## How you think", "## Example"]


@pytest.mark.parametrize("role", ANALYST_ROLES + DECISION_ROLES)
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


# Task 4.3: the two deterministic doc-only roles. Their final numbers are computed by code
# (`neutrality.py` / `risk_gate.py`), so the LLM emits NO JSON — they are EXEMPT from the
# `## Output (return ONLY this JSON ...)` requirement (like `risk_manager`/`portfolio_manager`
# in the weekly desk's API map). They DO carry the `# Title`/`## Mission`/`## Inputs`/
# `## How you think` structure so the orchestrator and team can read the rule they cannot
# argue past.
DOC_ONLY_REQUIRED_SECTIONS = ["## Mission", "## Inputs", "## How you think"]


@pytest.mark.parametrize("role", sorted(DOC_ONLY_ROLES))
def test_doc_only_role_file_exists_and_is_output_exempt(role):
    p = Path("agents") / f"{role}.md"
    assert p.exists(), f"missing doc-only role file: {p}"
    text = p.read_text()

    # `# Title` — a top-level H1 header on the first non-blank line.
    first = next(line for line in text.splitlines() if line.strip())
    assert first.startswith("# "), f"{role}: missing `# Title` H1 header"

    for section in DOC_ONLY_REQUIRED_SECTIONS:
        assert section in text, f"{role}: missing `{section}` section"

    # EXEMPT from the Output-JSON requirement: a deterministic, code-enforced role emits no
    # LLM JSON, so it must NOT carry the canonical Output-JSON header.
    assert "## Output (return ONLY this JSON, no prose)" not in text, (
        f"{role}: doc-only role must be EXEMPT from the Output-JSON requirement"
    )

    # The doc must say explicitly that the final numbers are computed by code, not the LLM.
    assert ".py" in text, (
        f"{role}: doc-only role must name the code module that computes the final numbers"
    )
