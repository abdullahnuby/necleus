import math
import numpy as np
import pytest
from experiments.diagnose_ema_tau_v0_38 import ema_decay_factor, run


def test_ema_tau_decay_factor_is_monotonic_and_expected():
    rhos = [ema_decay_factor(t) for t in (2.0, 4.0, 8.0)]
    assert np.allclose(rhos, [math.exp(-0.5), math.exp(-0.25), math.exp(-0.125)])
    assert rhos[0] < rhos[1] < rhos[2]


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf")])
def test_invalid_tau_is_rejected(tau):
    with pytest.raises(ValueError):
        ema_decay_factor(tau)


def test_smoke_checks_same_inputs_initial_gain_and_finite_outputs(tmp_path, monkeypatch):
    import experiments.diagnose_ema_tau_v0_38 as diag
    monkeypatch.setattr(diag, "OUTDIR", tmp_path)
    result = run()
    assert result["integrity"]["same_recurrent_matrix_for_all_taus"]
    assert set(result["integrity"]["tasks"]) == {"sync", "order"}
    assert len(result["rows"]) == 6
    assert all(r["first_tick_gain_mean"] == pytest.approx(1.0) for r in result["rows"])
    assert all(r["n_trials"] == 32 and r["class_0"] == 16 and r["class_1"] == 16 for r in result["rows"])
    assert all(np.isfinite(r["mean_gain"]) and np.isfinite(r["mean_spikes_per_trial"]) for r in result["rows"])
    assert all(result["integrity"]["tasks"][t]["same_inputs_across_tau"] for t in ("sync", "order"))
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "results.csv").exists()
    assert (tmp_path / "report_ar.md").exists()
