import numpy as np

from experiments.run_learning_curve_v0_30 import (
    N_HIDDEN,
    PIPELINES,
    TRAIN_SIZES,
    aggregate_results,
    make_features,
    make_nested_training_pool,
    rasterize_batch,
    simulate_batch_vectorized,
    verify_preregistration,
)
from pulse_core.sequential_stdp import DynamicPulseReservoir, make_disjoint_dataset


def test_training_subsets_are_nested_and_balanced():
    pool = make_nested_training_pool(17, "sync", 6000)
    labels = np.asarray([t.target_class for t in pool])
    assert len(pool) == 6000
    assert len({(t.a, t.b) for t in pool}) == len(pool)
    for n in TRAIN_SIZES:
        assert np.sum(labels[:n] == 0) == n // 2
        assert np.sum(labels[:n] == 1) == n // 2
    assert set((t.a, t.b) for t in pool[:200]).issubset(set((t.a, t.b) for t in pool[:500]))


def test_vectorized_reservoir_matches_single_trial_simulation():
    trials = make_disjoint_dataset(13, "order", 8, "test")
    reservoir = DynamicPulseReservoir(seed=123, n_hidden=N_HIDDEN)
    inputs, batch_spikes = simulate_batch_vectorized(reservoir, trials)
    assert inputs.shape == (8, 32, 4)
    for index, trial in enumerate(trials):
        single = reservoir.simulate(trial)
        np.testing.assert_array_equal(batch_spikes[index], single)


def test_features_are_finite_and_have_expected_shapes():
    trials = make_disjoint_dataset(22, "sync", 12, "test")
    reservoir = DynamicPulseReservoir(seed=7, n_hidden=N_HIDDEN)
    inputs, spikes = simulate_batch_vectorized(reservoir, trials)
    x = make_features(inputs, spikes)
    expected = {
        "phase4_96": (12, 96),
        "spike_counts_24": (12, 24),
        "spike_bins8_192": (12, 192),
        "spike_bins16_384": (12, 384),
        "spike_raster_768": (12, 768),
        "raw_input_128": (12, 128),
    }
    assert set(x) == set(expected)
    for key, shape in expected.items():
        assert x[key].shape == shape
        assert np.isfinite(x[key]).all()


def test_pipeline_matrix_covers_both_readers_and_raw_input():
    assert {p["reader"] for p in PIPELINES} == {"linear", "gbm"}
    assert any(p["feature_set"] == "raw_input_128" for p in PIPELINES)
    assert any(p["feature_set"] == "spike_raster_768" for p in PIPELINES)


def test_aggregate_results_keeps_seed_level_n():
    rows = []
    for seed, acc in ((1, 0.6), (2, 0.7), (3, 0.8)):
        rows.append({"seed": seed, "task": "sync", "reader": "linear", "feature_set": "spike_counts_24",
                     "train_size": 200, "accuracy": acc, "balanced_accuracy": acc,
                     "class_0_recall": acc, "class_1_recall": acc,
                     "predicted_positive_rate": 0.5})
    summary = aggregate_results(rows)
    assert len(summary) == 1
    assert summary[0]["n_seeds"] == 3
    assert abs(summary[0]["accuracy_mean_percent"] - 70.0) < 1e-9
    assert summary[0]["accuracy_ci95_low_percent"] <= 70.0 <= summary[0]["accuracy_ci95_high_percent"]


def test_frozen_learning_curve_protocol_hashes_match():
    prereg = verify_preregistration()
    assert prereg["config"]["seeds"][:5] == [104, 887, 2027, 90210, 731]
    assert prereg["config"]["train_sizes"] == [200, 500, 1000, 2000, 6000]
    assert prereg["config"]["test_used_for_tuning"] is False
