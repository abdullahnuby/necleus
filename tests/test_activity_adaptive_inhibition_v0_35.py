import numpy as np

from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments.run_activity_adaptive_inhibition_v0_35 import (
    CONDITIONS,
    N_HIDDEN,
    N_TRAIN,
    N_VALIDATION,
    N_TEST,
    SEEDS,
    TARGET_SPECTRAL_RADIUS,
    TARGET_ACTIVITY_RATE,
    INHIBITORY_GAIN_MIN,
    INHIBITORY_GAIN_MAX,
    make_shared_dale_matrix,
    simulate_batch_activity_adaptive_inhibition,
    make_data,
    run_smoke,
    verify_preregistration,
)
from experiments.run_inhibition_structure_v0_34 import (
    simulate_batch_with_recurrent_weights,
    spectral_radius,
)
from experiments.run_learning_curve_v0_30 import make_nested_training_pool, trial_key


def test_frozen_protocol_config_and_hashes():
    doc = verify_preregistration()
    assert doc["config"]["conditions"] == CONDITIONS
    assert doc["config"]["seeds"] == SEEDS
    assert doc["config"]["treatment"]["only_negative_recurrent_current_is_scaled"] is True
    assert doc["config"]["final_test_used_for_tuning_or_selection"] is False


def test_both_conditions_start_with_identical_dale_matrix_and_polarity():
    seed = 8001
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, diag = make_shared_dale_matrix(reservoir, seed)
    fixed = matrices["dale_fixed_25"]
    adaptive = matrices["dale_activity_adaptive_25"]
    assert np.array_equal(fixed, adaptive)
    assert np.isclose(spectral_radius(fixed), TARGET_SPECTRAL_RADIUS, atol=5e-4)
    assert np.array_equal(fixed != 0, reservoir.w_rec != 0)
    inhibitory = set(diag["inhibitory_source_indices"])
    assert len(inhibitory) == round(0.25 * N_HIDDEN)
    for source in range(N_HIDDEN):
        values = fixed[:, source]
        nonzero = values[values != 0]
        if source in inhibitory:
            assert nonzero.size > 0 and np.all(nonzero < 0)
        else:
            assert np.all(nonzero > 0)


def test_zero_adaptive_slope_matches_fixed_dale_simulation():
    seed = 8002
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, _ = make_shared_dale_matrix(reservoir, seed)
    trials = make_nested_training_pool(seed + 7, "sync", 160)
    fixed_inputs, fixed_spikes = simulate_batch_with_recurrent_weights(
        reservoir, trials, matrices["dale_fixed_25"]
    )
    adaptive_inputs, adaptive_spikes, diag = simulate_batch_activity_adaptive_inhibition(
        reservoir, trials, matrices["dale_activity_adaptive_25"], slope=0.0, return_diagnostics=True
    )
    assert np.array_equal(fixed_inputs, adaptive_inputs)
    assert np.allclose(fixed_spikes, adaptive_spikes, rtol=0.0, atol=1e-7)
    assert diag["first_tick_inhibitory_gain_mean"] == 1.0
    assert diag["mean_inhibitory_gain"] == 1.0
    assert diag["min_inhibitory_gain"] == 1.0
    assert diag["max_inhibitory_gain"] == 1.0


def test_activity_adaptive_gain_is_finite_and_bounded():
    seed = 8003
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, _ = make_shared_dale_matrix(reservoir, seed)
    trials = make_nested_training_pool(seed + 8, "order", 250)
    inputs, spikes, diag = simulate_batch_activity_adaptive_inhibition(
        reservoir, trials, matrices["dale_activity_adaptive_25"], return_diagnostics=True
    )
    assert inputs.shape == (250, 32, 4)
    assert spikes.shape == (250, 32, N_HIDDEN)
    assert np.isfinite(spikes).all()
    assert INHIBITORY_GAIN_MIN <= diag["min_inhibitory_gain"] <= diag["max_inhibitory_gain"] <= INHIBITORY_GAIN_MAX
    assert INHIBITORY_GAIN_MIN <= diag["mean_inhibitory_gain"] <= INHIBITORY_GAIN_MAX
    assert diag["first_tick_inhibitory_gain_mean"] == 1.0
    assert diag["target_activity_rate_per_neuron_per_tick"] == TARGET_ACTIVITY_RATE
    assert diag["fraction_gain_at_min"] >= 0 and diag["fraction_gain_at_max"] >= 0


def test_data_splits_are_balanced_unique_and_disjoint():
    seed = 8004
    for task in ("sync", "order"):
        combined, bounds, labels, integrity = make_data(seed, task)
        assert len(combined) == N_TRAIN + N_VALIDATION + N_TEST
        assert bounds["train"] == (0, N_TRAIN)
        assert bounds["validation"] == (N_TRAIN, N_TRAIN + N_VALIDATION)
        assert bounds["test"][1] == len(combined)
        assert integrity["n_train_unique"] == N_TRAIN
        assert integrity["n_validation_unique"] == N_VALIDATION
        assert integrity["n_test_unique"] == N_TEST
        assert integrity["train_validation_overlap"] == 0
        assert integrity["train_test_overlap"] == 0
        assert integrity["validation_test_overlap"] == 0
        for split, ys in labels.items():
            assert int(np.sum(ys == 0)) == len(ys) // 2
            assert int(np.sum(ys == 1)) == len(ys) // 2


def test_smoke_entrypoint_runs():
    # Small engineering smoke test only; not used for model selection.
    run_smoke()
