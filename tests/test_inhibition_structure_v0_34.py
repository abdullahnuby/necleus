import numpy as np

from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments.run_learning_curve_v0_30 import (
    make_epoch_labels,
    make_nested_training_pool,
    simulate_batch_vectorized,
    trial_key,
)
from experiments.run_incremental_windows_v0_31 import make_eval_dataset
from experiments.run_inhibition_structure_v0_34 import (
    CONDITIONS,
    EPOCH_CHECKPOINTS,
    N_TEST,
    N_TRAIN,
    N_VALIDATION,
    SEEDS,
    aggregate_records,
    make_recurrent_weight_matrices,
    paired_delta_summary,
    simulate_batch_with_recurrent_weights,
    spectral_radius,
    verify_preregistration,
    wilson_ci,
    TARGET_SPECTRAL_RADIUS,
    N_HIDDEN,
)


def test_frozen_preregistration_and_config():
    doc = verify_preregistration()
    assert doc["config"]["conditions"] == CONDITIONS
    assert doc["config"]["seeds"] == SEEDS
    assert doc["config"]["primary_epoch"] == 10
    assert doc["config"]["final_test_used_for_tuning_or_selection"] is False


def test_connectivity_polarity_and_spectral_radius_are_controlled():
    seed = 7001
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    original_input = reservoir.w_in.copy()
    original_rec = reservoir.w_rec.copy()
    mats, diag = make_recurrent_weight_matrices(reservoir, seed)
    edge_mask = original_rec != 0
    assert set(mats) == set(CONDITIONS)
    assert np.array_equal(mats[CONDITIONS[0]] != 0, edge_mask)
    assert np.array_equal(mats[CONDITIONS[1]] != 0, edge_mask)
    assert np.all(mats["all_excitatory"][edge_mask] > 0)
    assert np.all(mats["all_excitatory"][~edge_mask] == 0)
    assert np.isclose(spectral_radius(mats["all_excitatory"]), TARGET_SPECTRAL_RADIUS, atol=5e-4)
    assert np.isclose(spectral_radius(mats["dale_inhibition_25"]), TARGET_SPECTRAL_RADIUS, atol=5e-4)
    inhibitory = set(diag["inhibitory_source_indices"])
    assert len(inhibitory) == round(N_HIDDEN * 0.25)
    for source in range(N_HIDDEN):
        column = mats["dale_inhibition_25"][:, source]
        nonzero = column[column != 0]
        if source in inhibitory:
            assert len(nonzero) > 0 and np.all(nonzero < 0)
        else:
            assert np.all(nonzero > 0)
    assert np.array_equal(reservoir.w_rec, original_rec)
    assert np.array_equal(reservoir.w_in, original_input)
    assert diag["topology_equal"] and diag["pre_normalization_absolute_magnitudes_equal"]


def test_explicit_weight_simulator_matches_reference_when_weights_are_swapped():
    seed = 7002
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    mats, _ = make_recurrent_weight_matrices(reservoir, seed)
    trials = make_nested_training_pool(seed + 9, "sync", 200)
    inputs_ref, spikes_ref = None, None
    saved = reservoir.w_rec
    try:
        reservoir.w_rec = mats["dale_inhibition_25"].copy()
        inputs_ref, spikes_ref = simulate_batch_vectorized(reservoir, trials)
    finally:
        reservoir.w_rec = saved
    inputs_new, spikes_new = simulate_batch_with_recurrent_weights(reservoir, trials, mats["dale_inhibition_25"])
    assert np.array_equal(inputs_ref, inputs_new)
    assert np.allclose(spikes_ref, spikes_new, rtol=0, atol=1e-7)
    assert spikes_new.shape == (200, 32, N_HIDDEN)
    assert np.isfinite(spikes_new).all()


def test_data_splits_are_balanced_unique_and_disjoint():
    seed = 7003
    train = make_nested_training_pool(seed + 11111, "order", N_TRAIN)
    train_keys = {trial_key(t) for t in train}
    val = make_eval_dataset(seed + 52222, "order", N_VALIDATION, set(train_keys))
    val_keys = {trial_key(t) for t in val}
    test = make_eval_dataset(seed + 63333, "order", N_TEST, set(train_keys) | val_keys)
    test_keys = {trial_key(t) for t in test}
    assert len(train_keys) == N_TRAIN and len(val_keys) == N_VALIDATION and len(test_keys) == N_TEST
    assert not (train_keys & val_keys or train_keys & test_keys or val_keys & test_keys)
    for split in (train, val, test):
        y = make_epoch_labels(split)
        assert np.sum(y == 0) == len(split) // 2
        assert np.sum(y == 1) == len(split) // 2


def test_aggregation_pairs_the_correct_conditions_and_reports_epochs():
    records = []
    for seed, control, treatment in ((7000, .60, .64), (7001, .70, .68), (7002, .80, .85)):
        for task in ("sync", "order"):
            for epoch in EPOCH_CHECKPOINTS:
                for condition, acc in (("all_excitatory", control), ("dale_inhibition_25", treatment)):
                    row = {"seed": seed, "task": task, "condition": condition, "epochs": epoch}
                    for split in ("validation", "test"):
                        for metric in ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate"):
                            row[f"{split}_{metric}"] = acc
                    records.append(row)
    summary = aggregate_records(records)
    assert len(summary) == 2 * 2 * len(EPOCH_CHECKPOINTS)
    deltas = paired_delta_summary(records)
    row = next(r for r in deltas if r["task"] == "sync" and r["epochs"] == 10)
    assert row["n_paired_seeds"] == 3
    assert abs(row["mean_delta_percentage_points"] - ((4 - 2 + 5) / 3)) < 1e-8


def test_wilson_interval_is_non_degenerate_at_zero_hits():
    lo, hi = wilson_ci(0, 40)
    assert 0 <= lo < hi < 20
    lo_all, hi_all = wilson_ci(40, 40)
    assert 80 < lo_all < hi_all <= 100
