"""Task 3.2 — per-cadence cycle artifacts.

`cycle_dir(state_dir, n, cadence="weekly")` MUST resolve to `state/weekly/cycle/<n>` — the SAME
root `scheduling.cycle_due(loop="weekly")` reads (CADENCE-ROOT INVARIANT, §14). Pinning the path
against the due-gate root here means the writer and the gate can never drift onto two directories.
The no-cadence path stays `state/cycle/<n>` for back-compat.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from futures_fund.contracts import TargetWeights
from futures_fund.control_loop import cadence_cycle_root
from futures_fund.cycle_io import cycle_dir, load_output, save_output


def test_cycle_dir_cadence_matches_due_gate_root(tmp_path):
    d = cycle_dir(tmp_path, 3, cadence="weekly")
    # MUST equal scheduling.cycle_due(loop="weekly")'s root state/weekly/cycle/3
    assert d == Path(tmp_path) / "weekly" / "cycle" / "3"
    # ... which is exactly the canonical root the due-gate scans for cycle 3.
    assert d == cadence_cycle_root(tmp_path, "weekly") / "3"


def test_cycle_dir_no_cadence_is_back_compat(tmp_path):
    assert cycle_dir(tmp_path, 3) == Path(tmp_path) / "cycle" / "3"


def test_save_load_target_weights_round_trip_under_cadence(tmp_path):
    tw = TargetWeights(
        dollar_residual=0.0,
        dollar_residual_frac=0.0,
        beta_residual=0.0,
        gross_long=100.0,
        gross_short=100.0,
        deploy_long_frac=0.5,
        deploy_short_frac=0.5,
        gross_notional=200.0,
        as_of_ts=datetime(2026, 6, 11, tzinfo=UTC),
    )
    p = save_output(tmp_path, 3, "target_weights", tw, cadence="weekly")
    # Persisted under the cadence root, where the due-gate reader expects it.
    assert p == cycle_dir(tmp_path, 3, cadence="weekly") / "target_weights.json"
    loaded = load_output(tmp_path, 3, "target_weights", cadence="weekly")
    assert TargetWeights.model_validate(loaded) == tw
