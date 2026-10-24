import numpy as np

from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments import run_activity_adaptive_inhibition_v0_35 as v35
from experiments.run_local_inhibition_v0_36 import (
    CONDITIONS, SEEDS, LOCAL_SLOPE, GAIN_MIN, GAIN_MAX, TARGET_SPECTRAL_RADIUS,
    make_shared_dale_matrix, simulate_local, run_smoke, verify_preregistration,
)
from experiments.run_inhibition_structure_v0_34 import simulate_batch_with_recurrent_weights, spectral_radius
from experiments.run_learning_curve_v0_30 import N_HIDDEN, make_nested_training_pool


def test_frozen_preregistration_and_source_hashes():
    doc = verify_preregistration()
    assert doc["config"]["official_seeds"] == SEEDS
    assert doc["config"]["selected_local_slope"] == LOCAL_SLOPE
    assert doc["config"]["final_test_used_for_tuning_or_selection"] is False


def test_all_conditions_share_exact_same_signed_matrix_and_topology():
    seed = 10009
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, _ = make_shared_dale_matrix(reservoir, seed)
    assert all(np.array_equal(matrices[CONDITIONS[0]], matrices[c]) for c in CONDITIONS[1:])
    assert all(np.array_equal(matrices[CONDITIONS[0]] != 0, matrices[c] != 0) for c in CONDITIONS[1:])
    assert np.isclose(spectral_radius(matrices[CONDITIONS[0]]), TARGET_SPECTRAL_RADIUS, atol=5e-4)


def test_local_gain_is_bounded_finite_and_starts_at_one():
    seed = 10010
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, _ = make_shared_dale_matrix(reservoir, seed)
    trials = make_nested_training_pool(seed + 2, "order", 200)
    inputs, spikes, diag = simulate_local(reservoir, trials, matrices[CONDITIONS[0]], LOCAL_SLOPE,
                                           return_diagnostics=True)
    assert inputs.shape == (200, 32, 4)
    assert spikes.shape == (200, 32, N_HIDDEN)
    assert np.isfinite(spikes).all()
    assert diag["first_tick_gain_mean"] == 1.0
    assert GAIN_MIN <= diag["min_gain_seen"] <= diag["max_gain_seen"] <= GAIN_MAX
    assert GAIN_MIN <= diag["mean_gain"] <= GAIN_MAX
    assert diag["mean_within_trial_gain_std"] >= 0


def test_zero_local_slope_matches_fixed_dale_dynamics():
    seed = 10011
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, _ = make_shared_dale_matrix(reservoir, seed)
    trials = make_nested_training_pool(seed + 3, "sync", 160)
    fixed_inputs, fixed_spikes = simulate_batch_with_recurrent_weights(reservoir, trials, matrices[CONDITIONS[0]])
    local_inputs, local_spikes, diag = simulate_local(reservoir, trials, matrices[CONDITIONS[2]], 0.0,
                                                       return_diagnostics=True)
    assert np.array_equal(fixed_inputs, local_inputs)
    assert np.allclose(fixed_spikes, local_spikes, atol=1e-7, rtol=0.0)
    assert diag["mean_gain"] == 1.0
    assert diag["min_gain_seen"] == 1.0
    assert diag["max_gain_seen"] == 1.0


def test_development_only_slope_selection_is_fixed_before_official_seeds():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    selected = json.loads((root / "results/local_inhibition_v0_36/development_tuning/selected_slope.json").read_text())
    assert selected["selected_slope"] == LOCAL_SLOPE
    assert selected["development_seeds"] == list(range(9100, 9106))
    assert not set(selected["development_seeds"]) & set(SEEDS)
    assert selected["selection_basis"] == "development validation only"


def test_smoke_entrypoint():
    run_smoke()
