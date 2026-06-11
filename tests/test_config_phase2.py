from __future__ import annotations

from pathlib import Path

import yaml

_CFG = yaml.safe_load(Path("config.yaml").read_text())


def test_sleeves_block_lists_four_sleeves():
    assert _CFG["sleeves"]["enabled"] == ["carry", "pairs", "factor", "sentiment"]
    assert _CFG["sleeves"]["risk_parity"] is True


def test_pairs_block_thresholds():
    p = _CFG["sleeves"]["pairs"]
    assert p["adf_pvalue_max"] == 0.05
    assert p["fdr_method"] == "bh"
    assert p["entry_z"] == 2.0
    assert p["exit_z"] == 0.0
    assert p["stop_z"] == 3.0


def test_sentiment_block_defaults():
    s = _CFG["sentiment"]
    assert s["kappa"] == 0.5
    assert s["cap"] == 0.25
    assert s["halflife_days"] == 3


def test_factor_block_defaults():
    f = _CFG["sleeves"]["factor"]
    assert f["factors"] == ["momentum", "carry", "low_vol"]
    assert f["weighting"] == "inverse_vol"
