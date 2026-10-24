import json
from pathlib import Path

import numpy as np

from experiments.run_sequential_learning_v0_29 import (
    ROOT,
    SEEDS,
    LinearSoftmaxReadout,
    bootstrap_ci,
    macro_metrics,
    make_epoch_orders,
    simulate_batch,
    features_from_spikes,
    verify_preregistration,
)
from pulse_core.sequential_stdp import DynamicPulseReservoir, RewardSTDPReadout, make_disjoint_dataset


def test_seed_list_has_40_fixed_unique_seeds():
    assert len(SEEDS) == 40
    assert len(set(SEEDS)) == 40
    assert SEEDS[:5] == [104, 887, 2027, 90210, 731]


def test_linear_readout_learns_separable_binary_data():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(128, 3)).astype(np.float32)
    y = (x[:, 0] + 0.25 * x[:, 1] > 0).astype(np.int64)
    model = LinearSoftmaxReadout(3, learning_rate=0.02, weight_decay=0.0)
    orders = make_epoch_orders(9, len(y), 20)
    model.fit_sequential(x, y, orders, batch_size=16)
    assert np.mean(model.predict(x) == y) > 0.94


def test_metrics_report_class_recall_and_positive_bias():
    trials = make_disjoint_dataset(2, "sync", 20, "test")
    # For a balanced dataset, predicting positive for every sample must be
    # surfaced as perfect positive recall and zero negative recall.
    metrics = macro_metrics(np.ones(20, dtype=np.int64), trials)
    assert metrics["class_0_recall"] == 0.0
    assert metrics["class_1_recall"] == 1.0
    assert metrics["predicted_positive_rate"] == 1.0
    assert metrics["balanced_accuracy"] == 0.5


def test_bootstrap_ci_is_deterministic_and_handles_empty():
    first = bootstrap_ci([1.0, 2.0, 3.0, 4.0], reps=1000, seed=77)
    second = bootstrap_ci([1.0, 2.0, 3.0, 4.0], reps=1000, seed=77)
    assert first == second
    assert first["n_seeds"] == 4
    assert first["ci95_low"] <= first["mean"] <= first["ci95_high"]
    assert bootstrap_ci([], reps=20)["n_seeds"] == 0


def test_same_fixed_reservoir_features_are_four_phase_readout_features():
    trials = make_disjoint_dataset(5, "sync", 4, "test")
    reservoir = DynamicPulseReservoir(seed=4, n_hidden=8)
    readout = RewardSTDPReadout(seed=6, n_hidden=8)
    spikes = simulate_batch(reservoir, trials)
    features = features_from_spikes(readout, spikes)
    assert spikes.shape == (4, 32, 8)
    assert features.shape == (4, 8 * 4)
    assert np.isfinite(features).all()


def test_preregistered_hashes_and_configuration_are_current():
    prereg = verify_preregistration()
    assert prereg["config"]["test_per_task"] == 500
    assert prereg["config"]["eligibility_threshold"] == 0.85
    assert len(prereg["sha256"]) >= 4
