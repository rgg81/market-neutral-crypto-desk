from __future__ import annotations

from pydantic import BaseModel


class NeutralityConfig(BaseModel):
    capital_usdt: float = 20000.0
    target_gross_usdt: float = 20000.0
    side_budget_usdt: float = 10000.0
    deployment_floor: float = 0.90
    dry_powder_frac: float = 0.10
    per_name_cap: float = 0.25
    cluster_cap: float = 0.40
    dollar_band: float = 0.03
    beta_band: float = 0.05
    drift_band: float = 0.20
    turnover_penalty: float = 0.001
    corr_threshold: float = 0.7
    stress_band_mult: float = 0.5

    @property
    def deploy_target_frac(self) -> float:
        """Per-side deployment target the optimizer scales each side up to: the midpoint of
        the [deployment_floor, 1 - dry_powder_frac] band. With defaults: (0.90 + 0.90)/2 =
        0.90 — i.e. deploy at the floor while still holding the full dry-powder reserve.
        Always lands in [floor, 1 - dry_powder] so both spec-§4 constraints hold by
        construction."""
        lo = self.deployment_floor
        hi = 1.0 - self.dry_powder_frac
        return (lo + hi) / 2.0


def dollar_residual(weights: dict[str, float], notionals: dict[str, float]) -> float:
    """Sum(long$) - Sum(short$) in USDT, using signed per-symbol notionals."""
    longs = sum(n for n in notionals.values() if n > 0.0)
    shorts = sum(-n for n in notionals.values() if n < 0.0)
    return longs - shorts


def beta_residual(weights: dict[str, float], betas: dict[str, float]) -> float:
    """Sum_i w_i * beta_i (equity-normalized beta-dollar exposure)."""
    return sum(w * betas.get(sym, 1.0) for sym, w in weights.items())
