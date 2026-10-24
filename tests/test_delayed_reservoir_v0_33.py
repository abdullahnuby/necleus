import numpy as np

from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments.run_learning_curve_v0_30 import (
    make_epoch_labels,
    make_features,
    make_nested_training_pool,
    simulate_batch_vectorized,
    trial_key,
)
from experiments.run_incremental_windows_v0_31 import make_eval_dataset, transform_features
from experiments.run_delayed_reservoir_v0_33 import (
    CONDITIONS,
    EPOCH_CHECKPOINTS,
    FEATURE_SET,
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    SEEDS,
    aggregate_records,
    extract_16window_rate_features,
    make_recurrent_delay_matrix,
    paired_delta_summary,
    simulate_batch_with_recurrent_delays,
    verify_preregistration,
    wilson_ci,
)


def test_frozen_preregistration_and_config():
    doc = verify_preregistration()
    assert doc["config"]["conditions"] == CONDITIONS
    assert doc["config"]["seeds"] == SEEDS
    assert doc["config"]["primary_epoch"] == 10
    assert doc["config"]["final_test_used_for_tuning_or_selection"] is False


def test_delay_matrix_changes_only_existing_edge_delay_labels():
    reservoir = DynamicPulseReservoir(seed=6002, n_hidden=24)
    weights_before = reservoir.w_rec.copy()
    input_before = reservoir.w_in.copy()
    baseline = make_recurrent_delay_matrix(reservoir, "baseline_delay1", seed=6002)
    varied = make_recurrent_delay_matrix(reservoir, "heterogeneous_delay1_4", seed=6002)
    edge_mask = reservoir.w_rec != 0.0
    assert np.all(baseline[edge_mask] == 1)
    assert np.all(baseline[~edge_mask] == 0)
    assert np.all((varied[edge_mask] >= 1) & (varied[edge_mask] <= 4))
    assert np.all(varied[~edge_mask] == 0)
    assert int(np.sum(varied > 0)) == int(np.sum(edge_mask))
    assert np.array_equal(reservoir.w_rec, weights_before)
    assert np.array_equal(reservoir.w_in, input_before)


def test_delay_one_simulator_matches_original_and_features_match_v032():
    seed = 6003
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=24)
    trials = make_nested_training_pool(seed + 9, "sync", 200)
    input_ref, spike_ref = simulate_batch_vectorized(reservoir, trials)
    delay_one = make_recurrent_delay_matrix(reservoir, "baseline_delay1", seed)
    input_new, spike_new = simulate_batch_with_recurrent_delays(reservoir, trials, delay_one)
    assert np.array_equal(input_ref, input_new)
    assert np.allclose(spike_ref, spike_new, rtol=0, atol=1e-7)
    direct = extract_16window_rate_features(spike_ref)
    expected = transform_features(make_features(input_ref, spike_ref))[FEATURE_SET]
    assert direct.shape == (len(trials), 384)
    assert np.allclose(direct, expected, rtol=0, atol=1e-7)


def test_trial_sets_are_balanced_and_disjoint():
    seed = 6004
    train = make_nested_training_pool(seed + 11111, "order", N_TRAIN)
    train_keys = {trial_key(t) for t in train}
    validation = make_eval_dataset(seed + 52222, "order", N_VALIDATION, set(train_keys))
    val_keys = {trial_key(t) for t in validation}
    test = make_eval_dataset(seed + 63333, "order", N_TEST, set(train_keys) | val_keys)
    test_keys = {trial_key(t) for t in test}
    assert len(train_keys) == N_TRAIN
    assert len(val_keys) == N_VALIDATION
    assert len(test_keys) == N_TEST
    assert not (train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)
    for split in (train, validation, test):
        labels = make_epoch_labels(split)
        assert int(np.sum(labels == 0)) == len(split) // 2
        assert int(np.sum(labels == 1)) == len(split) // 2


def test_aggregate_and_paired_deltas_keep_seed_pairing():
    records = []
    for seed, baseline_acc, varied_acc in ((6000, 0.60, 0.65), (6001, 0.70, 0.68), (6002, 0.80, 0.85)):
        for task in ("sync", "order"):
            for epoch in EPOCH_CHECKPOINTS:
                for condition, acc in (("baseline_delay1", baseline_acc), ("heterogeneous_delay1_4", varied_acc)):
                    row = {"seed": seed, "task": task, "condition": condition, "epochs": epoch}
                    for split in ("validation", "test"):
                        for metric in ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate"):
                            row[f"{split}_{metric}"] = acc
                    records.append(row)
    summary = aggregate_records(records)
    assert len(summary) == 2 * 2 * len(EPOCH_CHECKPOINTS)
    paired = paired_delta_summary(records)
    sync_primary = next(r for r in paired if r["task"] == "sync" and r["epochs"] == 10)
    assert sync_primary["n_paired_seeds"] == 3
    assert abs(sync_primary["mean_delta_percentage_points"] - ((5 - 2 + 5) / 3)) < 1e-9


def test_wilson_interval_is_not_degenerate_at_zero_hits():
    low, high = wilson_ci(0, 40)
    assert 0.0 <= low < high < 20.0
    low_all, high_all = wilson_ci(40, 40)
    assert 80.0 < low_all < high_all <= 100.0
