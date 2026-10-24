import json
import numpy as np

from experiments.run_multi_epoch_v0_32 import (
    EPOCH_CHECKPOINTS,
    FEATURE_SETS,
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    TARGET_VALIDATION_ACCURACY,
    aggregate_records,
    atomic_json,
    make_eval_dataset,
    verify_preregistration,
)
from experiments.run_learning_curve_v0_30 import make_epoch_labels, make_nested_training_pool
from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments.run_learning_curve_v0_30 import make_features, simulate_batch_vectorized
from experiments.run_incremental_windows_v0_31 import transform_features, trial_key


def test_preregistration_frozen_and_test_not_used_for_selection():
    doc = verify_preregistration()
    assert doc["config"]["training_pass_checkpoints"] == EPOCH_CHECKPOINTS == [1, 3, 5, 10]
    assert doc["config"]["target_validation_accuracy"] == TARGET_VALIDATION_ACCURACY == 0.85
    assert doc["config"]["final_test_used_for_tuning_or_selection"] is False


def test_training_budget_is_6000_unique_examples_and_split_sets_are_disjoint():
    train = make_nested_training_pool(1234, "order", N_TRAIN)
    train_keys = {trial_key(row) for row in train}
    validation = make_eval_dataset(8881, "order", N_VALIDATION, set(train_keys))
    val_keys = {trial_key(row) for row in validation}
    test = make_eval_dataset(8882, "order", N_TEST, set(train_keys) | val_keys)
    test_keys = {trial_key(row) for row in test}
    assert len(train_keys) == N_TRAIN
    assert not (train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)
    for split in (train, validation, test):
        labels = [row.target_class for row in split]
        assert labels.count(0) == labels.count(1)


def test_features_match_expected_widths_and_are_finite():
    train = make_nested_training_pool(1235, "sync", 200)
    reservoir = DynamicPulseReservoir(seed=101, n_hidden=24)
    inputs, spikes = simulate_batch_vectorized(reservoir, train)
    features = transform_features(make_features(inputs, spikes))
    assert set(features) == set(FEATURE_SETS)
    assert features["phase4_96"].shape == (200, 96)
    assert features["spike_bins8_rate_192"].shape == (200, 192)
    assert features["spike_bins16_rate_384"].shape == (200, 384)
    assert all(np.isfinite(values).all() for values in features.values())


def test_aggregate_records_bootstraps_across_seeds():
    rows = []
    for seed, accuracy in ((5000, 0.70), (5001, 0.80), (5002, 0.90)):
        row = {"seed": seed, "task": "sync", "feature_set": FEATURE_SETS[0], "epochs": 1}
        for split in ("validation", "test"):
            for metric in ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate"):
                row[f"{split}_{metric}"] = accuracy
        rows.append(row)
    summary = aggregate_records(rows)
    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 3
    assert abs(summary[0]["test_accuracy_mean_percent"] - 80.0) < 1e-9
    assert summary[0]["test_accuracy_ci95_low_percent"] <= 80.0 <= summary[0]["test_accuracy_ci95_high_percent"]


def test_json_serialization_is_strict_and_encodes_unavailable_as_null(tmp_path):
    path = tmp_path / "strict.json"
    atomic_json(path, {"good": 1.0, "not_available": float("nan")})
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {"good": 1.0, "not_available": None}
