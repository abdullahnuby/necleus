import numpy as np
import pytest

from experiments.run_stdp_vs_linear_v0_42 import (
    N_HIDDEN,
    N_WINDOWS,
    RewardSTDP16WindowReadout,
    extract_16window_rate_features,
    stdp_eligibility,
)


def test_extract_16window_features_shape_range_and_counts():
    spikes = np.zeros((2, 32, N_HIDDEN), dtype=np.float32)
    spikes[0, 0, 0] = 1.0
    spikes[0, 1, 0] = 1.0
    spikes[0, 30, 1] = 1.0
    features = extract_16window_rate_features(spikes)
    assert features.shape == (2, N_WINDOWS * N_HIDDEN)
    assert features[0, 0] == pytest.approx(1.0)
    assert features[0, 15 * N_HIDDEN + 1] == pytest.approx(0.5)
    assert float(features.min()) >= 0.0
    assert float(features.max()) <= 1.0


def test_stdp_eligibility_is_finite_and_unit_norm_for_nonzero_rows():
    x = np.zeros((3, N_WINDOWS * N_HIDDEN), dtype=np.float32)
    x[0, 0] = 1.0
    x[1, 15 * N_HIDDEN + 2] = 0.5
    e = stdp_eligibility(x)
    assert e.shape == x.shape
    assert np.isfinite(e).all()
    assert np.linalg.norm(e[0]) == pytest.approx(1.0)
    assert np.linalg.norm(e[1]) == pytest.approx(1.0)
    assert np.linalg.norm(e[2]) == pytest.approx(0.0)


def test_stdp_reader_trains_once_per_example_and_predicts_deterministically():
    x = np.zeros((6, N_WINDOWS * N_HIDDEN), dtype=np.float32)
    x[0::2, 0] = 1.0
    x[1::2, -1] = 1.0
    y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    reader = RewardSTDP16WindowReadout(seed=123)
    before = reader.w.copy()
    for xi, yi in zip(x, y):
        reader.train_one(xi, int(yi))
    assert len(reader.correct_actions) == len(y)
    assert not np.array_equal(before, reader.w)
    p1 = reader.predict(x)
    p2 = reader.predict(x)
    np.testing.assert_array_equal(p1, p2)
    assert p1.shape == y.shape


def test_wrong_feature_dimensions_are_rejected():
    with pytest.raises(ValueError):
        stdp_eligibility(np.zeros(10, dtype=np.float32))
