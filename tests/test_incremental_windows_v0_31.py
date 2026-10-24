import numpy as np

from experiments.run_incremental_windows_v0_31 import (
    FEATURE_SETS,
    N_TEST,
    N_VALIDATION,
    TARGET_VALIDATION_ACCURACY,
    TRAIN_SIZES,
    aggregate_records,
    atomic_json,
    make_eval_dataset,
    make_nested_training_pool,
    transform_features,
    verify_preregistration,
)
from experiments.run_learning_curve_v0_30 import make_features, make_epoch_labels, simulate_batch_vectorized
from pulse_core.sequential_stdp import DynamicPulseReservoir


def test_nested_training_prefixes_are_balanced_and_unique():
    pool = make_nested_training_pool(1337, "sync", max(TRAIN_SIZES))
    keys = {(t.a, t.b) for t in pool}
    labels = np.asarray([t.target_class for t in pool])
    assert len(pool) == 6000 and len(keys) == 6000
    for n in TRAIN_SIZES:
        assert int(np.sum(labels[:n] == 0)) == n // 2
        assert int(np.sum(labels[:n] == 1)) == n // 2


def test_eval_splits_are_disjoint_and_balanced():
    train = make_nested_training_pool(1338, "order", max(TRAIN_SIZES))
    train_keys = {(t.a, t.b) for t in train}
    validation = make_eval_dataset(9991, "order", N_VALIDATION, set(train_keys))
    val_keys = {(t.a, t.b) for t in validation}
    test = make_eval_dataset(9992, "order", N_TEST, set(train_keys) | val_keys)
    test_keys = {(t.a, t.b) for t in test}
    assert not train_keys & val_keys
    assert not train_keys & test_keys
    assert not val_keys & test_keys
    for split in (train, validation, test):
        y = [x.target_class for x in split]
        assert y.count(0) == y.count(1)


def test_online_feature_transforms_have_expected_shapes_and_finite_values():
    trials = make_nested_training_pool(1339, "sync", 200)
    reservoir = DynamicPulseReservoir(seed=73, n_hidden=24)
    inputs, spikes = simulate_batch_vectorized(reservoir, trials)
    features = transform_features(make_features(inputs, spikes))
    assert set(features) == set(FEATURE_SETS)
    assert features["phase4_96"].shape == (200, 96)
    assert features["spike_bins8_rate_192"].shape == (200, 192)
    assert features["spike_bins16_rate_384"].shape == (200, 384)
    for x in features.values():
        assert np.isfinite(x).all()
        assert np.all(x >= 0.0)
        assert np.max(x) <= 1.0 + 1e-6


def test_summary_retains_seed_level_bootstrap_unit():
    rows = []
    for seed, acc in ((4000, 0.7), (4001, 0.8), (4002, 0.9)):
        row = {"seed": seed, "task": "sync", "feature_set": FEATURE_SETS[0], "train_size": 200}
        for split in ("validation", "test"):
            for metric in ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall"):
                row[f"{split}_{metric}"] = acc
        rows.append(row)
    summary = aggregate_records(rows)
    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 3
    assert abs(summary[0]["test_accuracy_mean_percent"] - 80.0) < 1e-9
    assert summary[0]["test_accuracy_ci95_low_percent"] <= 80.0 <= summary[0]["test_accuracy_ci95_high_percent"]


def test_threshold_is_preregistered_and_test_is_not_used_for_selection():
    prereg = verify_preregistration()
    assert prereg["config"]["target_validation_accuracy"] == TARGET_VALIDATION_ACCURACY == 0.85
    assert prereg["config"]["test_used_for_tuning_or_checkpoint_selection"] is False
    assert prereg["config"]["one_pass_per_unique_example"] is True


def test_json_serialization_encodes_unavailable_values_as_null(tmp_path):
    import json

    path = tmp_path / "safe.json"
    atomic_json(path, {"available": 1.0, "unavailable": float("nan")})
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed == {"available": 1.0, "unavailable": None}
