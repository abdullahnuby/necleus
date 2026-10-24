from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.diagnose_input_gain_v0_40 import (  # noqa: E402
    INPUT_SCALES,
    LOCAL_SLOPE,
    LOCAL_TAU,
    N_PER_CLASS,
    SEED,
    balanced_train_slice,
    scale_input_weights,
)
from pulse_core.sequential_stdp import DynamicPulseReservoir  # noqa: E402
from experiments import run_activity_adaptive_inhibition_v0_35 as v35  # noqa: E402


def test_scales_only_input_weights_and_never_mutates_source():
    reservoir = DynamicPulseReservoir(seed=SEED)
    original = reservoir.w_in.copy()
    scaled = scale_input_weights(reservoir, 0.75)
    np.testing.assert_array_equal(reservoir.w_in, original)
    np.testing.assert_array_equal(scaled.w_in, original * np.float32(0.75))
    np.testing.assert_array_equal(scaled.w_rec, reservoir.w_rec)
    np.testing.assert_array_equal(scaled.threshold, reservoir.threshold)


def test_invalid_input_scales_are_rejected():
    reservoir = DynamicPulseReservoir(seed=SEED)
    for scale in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            scale_input_weights(reservoir, float(scale))


def test_diagnostic_config_is_small_and_fixed():
    assert SEED == 14040
    assert INPUT_SCALES == (0.75, 1.0, 1.25)
    assert LOCAL_SLOPE == 2.0
    assert LOCAL_TAU == 4.0
    assert N_PER_CLASS == 16


def test_balanced_slice_uses_training_labels_only():
    trials, bounds, labels, _ = v35.make_data(SEED, "sync")
    chosen, y = balanced_train_slice(trials, labels)
    assert len(chosen) == 2 * N_PER_CLASS
    assert int(np.sum(y == 0)) == N_PER_CLASS
    assert int(np.sum(y == 1)) == N_PER_CLASS
    assert max(trials.index(t) for t in chosen) < bounds["train"][1]


def test_data_splits_remain_balanced_and_disjoint():
    _, _, _, diag = v35.make_data(SEED, "sync")
    assert diag["train_validation_overlap"] == 0
    assert diag["train_test_overlap"] == 0
    assert diag["validation_test_overlap"] == 0


def test_fixed_baseline_and_local_adaptive_are_distinct_conditions():
    from experiments.diagnose_input_gain_v0_40 import CONDITIONS
    assert CONDITIONS == ("fixed_inhibition", "local_adaptive_inhibition")


def test_input_scales_are_monotonic_and_centered_on_nominal():
    assert sorted(INPUT_SCALES) == list(INPUT_SCALES)
    assert INPUT_SCALES[1] == 1.0


def test_component_smoke_artifact_has_12_finite_rows_and_no_accuracy_claims():
    import json
    result_path = ROOT / "results/input_gain_v0_40/component_smoke/results.json"
    assert result_path.exists()
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 12
    assert payload["integrity"]["readout_training_performed"] is False
    assert payload["integrity"]["official_test_used"] is False
    for row in payload["rows"]:
        assert row["condition"] in ("fixed_inhibition", "local_adaptive_inhibition")
        assert np.isfinite(row["mean_spikes_per_trial"])
        assert np.isfinite(row["feature_relative_l2_delta_vs_nominal"])


def test_local_condition_initial_gain_is_exactly_one_for_every_scale():
    import csv
    csv_path = ROOT / "results/input_gain_v0_40/component_smoke/results.csv"
    assert csv_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    local_rows = [r for r in rows if r["condition"] == "local_adaptive_inhibition"]
    assert len(local_rows) == 6
    assert all(abs(float(r["first_tick_gain_mean"]) - 1.0) <= 1e-7 for r in local_rows)


def test_component_plots_are_written():
    outdir = ROOT / "results/input_gain_v0_40/component_smoke"
    for filename in ("sync_input_gain_response.png", "order_input_gain_response.png"):
        path = outdir / filename
        assert path.exists() and path.stat().st_size > 1000


def test_input_raster_hash_is_same_across_scale_diagnostic():
    import json
    payload = json.loads((ROOT / "results/input_gain_v0_40/component_smoke/results.json").read_text(encoding="utf-8"))
    for task in ("sync", "order"):
        assert len(payload["integrity"]["tasks"][task]["input_raster_sha256"]) == 64
        assert payload["integrity"]["tasks"][task]["same_event_raster_across_scales"] is True
