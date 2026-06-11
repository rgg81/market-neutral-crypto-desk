# Risk Gate (Deterministic, Non-Overridable — Documentation)

> Adapted from the weekly desk's `agents/risk_manager.md`, re-cast for OPERATION MARKET-NEUTRAL.
> This role is **not an LLM** — it is deterministic Python: `futures_fund/risk_gate.py`, invoked at
> the execute boundary by `scripts/gate_execute_cli.py`. This file documents the survival mechanism
> so the orchestrator and the team understand the rule they cannot argue past: **the LLM team
> proposes; the code gate disposes.** The final sizing/approval numbers are computed by code, never
> by the LLM, so there is no JSON to author here.

## Mission
You serve OPERATION MARKET-NEUTRAL (the charter is injected above). The Risk Gate is the desk's
survival mechanism and it is **final**. It takes the Trader's per-leg proposals + the optimizer's
`TargetWeights` and decides — deterministically — whether and how large each leg trades. Leverage is
the **output** of liq-distance geometry, never an input; no agent sets leverage or size.

## Inputs
- The Trader's `AgentProposal` order envelopes (`symbol`, `direction`, `entry`, `stop`,
  `take_profit`, `trigger_type`) — advisory stops/levels the gate verifies, not trusts.
- The optimizer's `TargetWeights` legs (the per-leg notional the gate sizes against; the Trader
  does no sizing).
- Per-symbol marks, liquidation/MMR geometry, costs (fees + funding + depth-aware slippage), and the
  portfolio-health / regime state (drawdown, loss streak, heat).
- The charter (`MISSION.md`) injected above.

## How you think
- **Adaptive sizing, leverage as output.** Position size is computed from regime × portfolio-health
  caps — risk-per-trade shrinks in high-vol regimes and as drawdown deepens. Leverage falls out of
  the liq-distance geometry (`choose_leverage`), never set by an agent.
- **Liquidation distance.** The liquidation price must sit at least ~**2.5×** the stop distance
  beyond entry, so a normal stop-out can never be a liquidation. Trades that cannot satisfy this are
  rejected or down-sized.
- **Reward-to-risk floor.** Proposals must clear **RR ≥ 2** after costs (fees + signed, unclamped
  funding so a real carry credit is visible) — but the carry credit un-hides only what is real and
  **never weakens the RR ≥ 2 floor**. Thinner trades are rejected.
- **Heat cap + circuit breakers.** Aggregate open risk ("heat") is capped; a breaching new trade is
  trimmed or rejected. Drawdown / loss-streak thresholds can **HALT** the desk entirely — no new
  risk until cleared.
- **Neutrality is a survival input too.** A book that is not dollar- and beta-neutral within the
  §5 bands, or that fails the reviewer gate, does not reach execution.

## How the team should treat it
- The gate's verdict is **final and cannot be overridden** by the orchestrator or any subagent.
  There is no prompt that talks past it.
- Any agent "risk" reasoning is **advisory only**: the Trader anchors stops to liq-distance/structure
  and reports the values the gate verifies, but the gate — not the agent — decides whether and how
  large the trade is.
- The orchestrator must **never weaken a risk limit** to make a trade fit or an error disappear. If
  something cannot pass safely, it does not trade — that is the system working, not failing.
- Survival-first is the whole point: you cannot compound from zero. A rejected marginal trade is a
  win for the mandate.

## Note on output
This is a deterministic, non-overridable gate, so there is **no JSON output contract and no
`## Output` section** — the final sizing/approval and the cycle report are computed and emitted by
`futures_fund/risk_gate.py` via `scripts/gate_execute_cli.py`, not by this prompt.
