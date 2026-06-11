# Neutrality & Portfolio Constructor (Deterministic — Documentation)

> This role is **not an LLM**. It is deterministic Python: `futures_fund/neutrality.py`
> (`optimize_book`). This file documents the optimizer so the orchestrator and the team
> understand the constraint set they cannot argue past — **the analysts tilt; the optimizer
> disposes.** There is no JSON to author here; the final `TargetWeights` numbers are computed by
> code, never by the LLM.

## Mission
You serve OPERATION MARKET-NEUTRAL (the charter is injected above). You are the heart of the desk
(spec §8): you turn the four sleeves' tilts + per-coin geometry (incl. sentiment) into the week's
**`TargetWeights`** — a dollar- and beta-neutral book scaled to the deployment target — subject to
**hard constraints** that no agent and no prompt can relax. You run on **both cadences**: a full
re-projection weekly, a turnover-aware delta toward target daily.

## Inputs
- The four sleeves' `SleeveSignal` tilts (funding-carry, pairs/cointegration, factor, technical) and
  any sentiment conviction tilts already applied (§7.3: tilts run BEFORE projection).
- `CoinGeometry` per coin: `beta_btc`, notional/structure, liq-distance geometry.
- The prior book (`prior_legs`) for the L1 turnover penalty + no-trade band on the daily rebalance.
- `NeutralityConfig` bands (dollar `±0.03`, beta `±0.05`, drift `0.20`), per-name (`0.25`) and
  per-cluster (`0.40`) caps, deployment floor (`≥0.90`/side) + dry-powder reserve (`0.10`), gross
  `≈ $20k`, and the optional `RegimeState` (stress-tightens the bands under a correlation spike).
- The charter (`MISSION.md`) injected above.

## How you think
- **Hard constraints, computed in code — not negotiated.** `optimize_book` merges sleeves → applies
  sentiment tilts → shapes weights via Ledoit-Wolf **shrunk covariance** → **HRP** (cluster →
  quasi-diagonalize → recursive bisection) or risk-parity → enforces per-name & per-cluster
  (correlated-as-one) caps → applies the L1 turnover / no-trade band vs the prior book → sizes the
  **BTC hedge leg** on the alpha legs' residual beta → **projects** alpha+hedge back onto the
  dollar- and beta-neutral constraint set → scales the neutral book to the per-side deployment
  target with a single positive scalar (which preserves neutrality).
- **Re-projection is mandatory and ordered.** Sentiment conviction tilts are applied FIRST; the
  weight vector is then projected back onto the neutrality set so a tilt can never push the book
  out of its dollar/beta bands. The reviewer re-derives the residuals against the bands and the
  spine recomputes them — the numbers here are never trusted blindly.
- **Fail loud, never silently un-neutral.** If the bands or the deployment floor cannot be met,
  `optimize_book` sets `feasible=False` rather than emit an out-of-band or under-deployed book.
  The desk stands down; it does not pretend.
- **The Trader does no sizing.** Notional lives in every `TargetWeights` leg you emit; the Trader
  only maps each leg to a per-leg order. Inventing or scaling notional downstream would break
  neutrality, so it is forbidden by contract.

## Note on output
This is a deterministic optimizer, so there is **no JSON output contract and no `## Output`
section** — the final `TargetWeights` (legs + residuals + per-side deployment + `feasible`) are
computed and serialized by `futures_fund/neutrality.py` (`optimize_book`), not by this prompt. Any
LLM "neutrality" reasoning is advisory only; the code is the source of truth.
