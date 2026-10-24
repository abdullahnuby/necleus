from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_input_gain_development_v0_41 import (  # noqa: E402
    DEV_SEEDS, TASKS, INPUT_SCALES, CONDITIONS, N_PER_CLASS, LOCAL_SLOPE,
    LOCAL_TAU, mean_ci, relative_delta, scale_input_weights, balanced_train_slice,
    verify_preregistration, OUTDIR, PREREG,
)
from pulse_core.sequential_stdp import DynamicPulseReservoir  # noqa: E402
from experiments.run_learning_curve_v0_30 import make_epoch_labels, make_nested_training_pool, N_TRAIN_POOL  # noqa: E402


def test_frozen_development_configuration():
    assert DEV_SEEDS == [14111, 14129, 14147, 14165, 14183, 14201, 14219, 14237]
    assert TASKS == ("sync", "order")
    assert INPUT_SCALES == (0.75, 1.0, 1.25)
    assert CONDITIONS == ("fixed_inhibition", "local_adaptive_inhibition")
    assert N_PER_CLASS == 16 and LOCAL_SLOPE == 2.0 and LOCAL_TAU == 4.0


def test_input_scaling_changes_only_w_in():
    reservoir = DynamicPulseReservoir(seed=14111)
    original_w_in = reservoir.w_in.copy()
    original_w_rec = reservoir.w_rec.copy()
    clone = scale_input_weights(reservoir, 1.25)
    np.testing.assert_array_equal(clone.w_in, original_w_in * np.float32(1.25))
    np.testing.assert_array_equal(clone.w_rec, original_w_rec)
    np.testing.assert_array_equal(reservoir.w_in, original_w_in)


def test_invalid_input_scale_rejected():
    reservoir = DynamicPulseReservoir(seed=14111)
    for scale in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            scale_input_weights(reservoir, scale)


def test_training_pool_slice_is_balanced_and_unique():
    pool = make_nested_training_pool(25222, "sync", N_TRAIN_POOL)
    labels = make_epoch_labels(pool)
    chosen, y = balanced_train_slice(pool, {"train": labels}, N_PER_CLASS)
    assert len(chosen) == 32
    assert int(np.sum(y == 0)) == 16 and int(np.sum(y == 1)) == 16
    keys = [(t.a, t.b) for t in chosen]
    assert len(keys) == len(set(keys))


def test_mean_ci_and_relative_delta_are_finite():
    mean, low, high = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert mean == pytest.approx(2.5)
    assert low < mean < high
    assert relative_delta(12.0, 10.0) == pytest.approx(0.2)


def test_preregistration_hashes_are_current():
    prereg = verify_preregistration()
    assert prereg["version"] == "0.41-stage2-input-gain-development"
    assert prereg["config"]["reader_training"] is False
    assert prereg["config"]["validation_or_test_sets_used"] is False


def test_full_development_artifact_schema_if_run():
    result = OUTDIR / "results.json"
    if not result.exists():
        pytest.skip("The full multi-seed development diagnostic has not run yet")
    payload = json.loads(result.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 8 * 2 * 3 * 2
    assert len(payload["rows"]) == payload["n_rows"]
    assert payload["readout_training_performed"] is False
    assert payload["validation_or_test_sets_generated_or_used"] is False
    assert len(payload["integrity"]["seeds"]) == 8
