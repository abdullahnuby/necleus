from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_ema_tau_v0_39_confirmation import (  # noqa: E402
    OFFICIAL_SEEDS, TAU_CANDIDATES, protocol_config, paired_ci,
)
from experiments import run_activity_adaptive_inhibition_v0_35 as v35  # noqa: E402


def test_official_seeds_are_reserved_and_exact():
    stage2 = __import__("json").loads((ROOT / "results/ema_tau_v0_38/development_tuning/preregistration.json").read_text())
    assert OFFICIAL_SEEDS == list(range(12000, 12040))
    assert stage2["config"]["official_seed_range_reserved"] == [12000, 12039]


def test_only_tau4_and_tau8_are_compared():
    assert TAU_CANDIDATES == (4.0, 8.0)
    assert protocol_config()["fixed_local_inhibition_slope"] == 2.0


def test_official_splits_are_balanced_and_disjoint():
    trials, bounds, labels, diag = v35.make_data(12000, "sync")
    assert len(trials) == 8000
    assert (len(labels["train"]), len(labels["validation"]), len(labels["test"])) == (6000, 1000, 1000)
    assert diag["train_validation_overlap"] == diag["train_test_overlap"] == diag["validation_test_overlap"] == 0
    for split in ("train", "validation", "test"):
        assert int((labels[split] == 0).sum()) == int((labels[split] == 1).sum())
    assert bounds["test"] == (7000, 8000)


def test_paired_ci_preserves_zero_effect():
    mean, low, high = paired_ci([0.0] * 40, 123)
    assert mean == 0.0
    assert low == 0.0 and high == 0.0


def test_report_protocol_has_only_primary_epoch_10():
    cfg = protocol_config()
    assert cfg["checkpoints"] == [1, 3, 5, 10]
    assert "at epoch 10" in cfg["primary_endpoint"]
    assert cfg["tuning_after_official_test"] is False
