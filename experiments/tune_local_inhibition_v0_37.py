"""Development-only tuning for Nucleus Pulse v0.37 local inhibitory gain.

Uses separate development seeds and validation examples only. The final official
seed range and official test splits are not accessed by this script.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments import run_activity_adaptive_inhibition_v0_35 as v35
from experiments.run_inhibition_structure_v0_34 import (
    simulate_batch_with_recurrent_weights, extract_features, spectral_radius,
)
from experiments.run_learning_curve_v0_30 import N_HIDDEN

DEV_SEEDS = list(range(9200, 9208))
LOCAL_SLOPE_CANDIDATES = [2.0, 4.0]
OUTDIR = ROOT / "results" / "local_inhibition_v0_37" / "development_tuning"


def simulate_local(reservoir, trials, recurrent_weights, slope, target_rate=0.13, tau=4.0,
                   gain_min=0.5, gain_max=2.0, return_diagnostics=False):
    """Locally scale inhibitory input to each postsynaptic cell by its own EMA activity."""
    if recurrent_weights.shape != reservoir.w_rec.shape:
        raise ValueError("recurrent_weights shape mismatch")
    if not np.isfinite(recurrent_weights).all():
        raise ValueError("non-finite recurrent matrix")
    if tau <= 0 or slope < 0 or not (0 <= target_rate <= 1):
        raise ValueError("invalid local inhibition parameters")
    from experiments.run_learning_curve_v0_30 import rasterize_batch
    if len(trials) == 0:
        raise ValueError("trials must be nonempty")
    inputs = rasterize_batch(trials, reservoir.duration)
    count, duration, _ = inputs.shape
    hidden = reservoir.n_hidden
    voltage = np.zeros((count, hidden), dtype=np.float32)
    i_syn = np.zeros((count, hidden), dtype=np.float32)
    refractory = np.zeros((count, hidden), dtype=np.int8)
    previous_spikes = np.zeros((count, hidden), dtype=np.float32)
    spikes = np.zeros((count, duration, hidden), dtype=np.float32)
    w_in_t = np.asarray(reservoir.w_in, dtype=np.float32).T
    weights = np.asarray(recurrent_weights, dtype=np.float32)
    w_exc_t = np.maximum(weights, 0.0).T
    w_inh_t = np.minimum(weights, 0.0).T
    rho = float(np.exp(-1.0 / tau))
    # One activity trace and one gain per trial × postsynaptic neuron.
    trace = np.full((count, hidden), target_rate, dtype=np.float32)
    gain_sum = 0.0
    gain_values = 0
    gain_min_seen = float("inf")
    gain_max_seen = float("-inf")
    first_gain_mean = None
    within_trial_gain_std_sum = 0.0
    at_min = at_max = 0
    neuron_gain_spreads = []
    for t in range(duration):
        gain = np.clip(1.0 + slope * (trace - target_rate), gain_min, gain_max).astype(np.float32)
        gain_sum += float(gain.sum())
        gain_values += gain.size
        within_trial_gain_std_sum += float(gain.std(axis=1).mean()) * count
        gain_min_seen = min(gain_min_seen, float(gain.min()))
        gain_max_seen = max(gain_max_seen, float(gain.max()))
        at_min += int(np.sum(gain <= gain_min + 1e-7))
        at_max += int(np.sum(gain >= gain_max - 1e-7))
        if t == 0:
            first_gain_mean = float(gain.mean())
        # Recurrent matrix is [post, pre]; current arrays are [trial, post].
        exc_current = previous_spikes @ w_exc_t
        inh_current = previous_spikes @ w_inh_t
        recurrent_current = exc_current + inh_current * gain
        i_syn = (reservoir.syn_leak[None, :] * i_syn
                 + inputs[:, t, :] @ w_in_t + recurrent_current)
        voltage = reservoir.reset_leak[None, :] * voltage + (1.0 - reservoir.reset_leak[None, :]) * i_syn
        active = (voltage >= reservoir.threshold[None, :]) & (refractory == 0)
        spikes[:, t, :] = active.astype(np.float32)
        voltage[active] = 0.0
        refractory = np.maximum(refractory - 1, 0)
        refractory[active] = 1
        trace = rho * trace + (1.0 - rho) * active.astype(np.float32)
        previous_spikes = active.astype(np.float32)
    diag = {
        "gain_type": "local_per_postsynaptic_cell",
        "target_rate": float(target_rate), "trace_tau_ticks": float(tau), "slope": float(slope),
        "gain_bounds": [float(gain_min), float(gain_max)],
        "first_tick_gain_mean": float(first_gain_mean),
        "mean_gain": float(gain_sum / gain_values),
        "mean_within_trial_gain_std": float(within_trial_gain_std_sum / max(1, duration * count)),
        "min_gain_seen": float(gain_min_seen), "max_gain_seen": float(gain_max_seen),
        "fraction_at_min": float(at_min / gain_values), "fraction_at_max": float(at_max / gain_values),
    }
    if not np.isfinite(spikes).all():
        raise RuntimeError("local inhibition produced non-finite spikes")
    return (inputs, spikes, diag) if return_diagnostics else (inputs, spikes)


def run_validation_reader(features, labels, seed, task_i):
    """Fit only on train and report only validation; never evaluate test here."""
    from experiments.run_incremental_windows_v0_31 import classification_metrics
    x_train = features["train"]
    x_val = features["validation"]
    y_train = labels["train"]
    y_val = labels["validation"]
    model_seed = int(seed + 940001 + task_i * 1000)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          learning_rate="constant", eta0=0.01, average=True,
                          shuffle=True, random_state=model_seed)
    for epoch in range(1, 11):
        if epoch == 1:
            model.partial_fit(x_train, y_train, classes=np.asarray([0, 1], dtype=np.int64))
        else:
            model.partial_fit(x_train, y_train)
    pred = model.predict(x_val)
    return classification_metrics(y_val, pred)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    conditions = ["dale_fixed_25", "dale_global_adaptive_25"] + [f"dale_local_adaptive_slope_{s:g}" for s in LOCAL_SLOPE_CANDIDATES]
    rows = []
    activity_rows = []
    started = time.time()
    for seed in DEV_SEEDS:
        reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
        matrices, matrix_diag = v35.make_shared_dale_matrix(reservoir, seed)
        weights = matrices["dale_fixed_25"]
        if not np.isclose(spectral_radius(weights), v35.TARGET_SPECTRAL_RADIUS, atol=5e-4):
            raise RuntimeError("development reservoir spectral radius mismatch")
        for task_i, task in enumerate(v35.TASKS):
            concatenated, bounds, labels, data_integrity = v35.make_data(seed, task)
            cutoff = v35.N_TRAIN + v35.N_VALIDATION
            dev_trials = concatenated[:cutoff]
            dev_labels = {"train": labels["train"], "validation": labels["validation"]}
            dev_bounds = {"train": bounds["train"], "validation": bounds["validation"]}
            for condition in conditions:
                if condition == "dale_fixed_25":
                    inputs, spikes = simulate_batch_with_recurrent_weights(reservoir, dev_trials, weights)
                    diag = {"mean_gain": 1.0, "first_tick_gain_mean": 1.0}
                elif condition == "dale_global_adaptive_25":
                    inputs, spikes, diag = v35.simulate_batch_activity_adaptive_inhibition(
                        reservoir, dev_trials, weights, return_diagnostics=True
                    )
                else:
                    slope = float(condition.rsplit("_", 1)[-1])
                    inputs, spikes, diag = simulate_local(reservoir, dev_trials, weights, slope, return_diagnostics=True)
                feature_splits = {split: extract_features(spikes[left:right]) for split, (left, right) in dev_bounds.items()}
                metrics = run_validation_reader(feature_splits, dev_labels, seed, task_i)
                rows.append({"seed": seed, "task": task, "condition": condition,
                             "epochs": 10, **{f"validation_{k}": float(v) for k, v in metrics.items()}})
                spike_rate = float(np.mean(spikes))
                activity_rows.append({"seed": seed, "task": task, "condition": condition,
                                      "mean_spikes_per_trial": float(np.sum(spikes, axis=(1, 2)).mean()),
                                      "mean_spike_rate_per_hidden_tick": spike_rate,
                                      "mean_gain": float(diag.get("mean_gain", diag.get("mean_inhibitory_gain", 1.0)))})
                del inputs, spikes, feature_splits
        print(f"development seed {seed} complete ({len(set(r['seed'] for r in rows))}/{len(DEV_SEEDS)})", flush=True)
    # Choose slope only by mean validation balanced accuracy, tie-break toward smaller slope.
    slope_scores = []
    for slope in LOCAL_SLOPE_CANDIDATES:
        cond = f"dale_local_adaptive_slope_{slope:g}"
        subset = [r for r in rows if r["condition"] == cond]
        score = float(np.mean([r["validation_balanced_accuracy"] for r in subset]))
        acc = float(np.mean([r["validation_accuracy"] for r in subset]))
        slope_scores.append({"slope": slope, "mean_validation_balanced_accuracy": score,
                             "mean_validation_accuracy": acc, "n_seed_task_pairs": len(subset)})
    selected = sorted(slope_scores, key=lambda x: (-x["mean_validation_balanced_accuracy"], x["slope"]))[0]
    # Clear record that only validation metrics determined the selected slope.
    payload = {
        "stage": "development_only",
        "development_seeds": DEV_SEEDS,
        "candidate_local_gain_slopes": LOCAL_SLOPE_CANDIDATES,
        "selection_rule": "maximize mean validation balanced accuracy at epoch 10 across development seeds and tasks; ties choose smaller slope",
        "final_test_used_for_tuning_or_selection": False,
        "selected_slope": selected["slope"],
        "candidate_scores": slope_scores,
        "rows": rows,
        "activity_diagnostics": activity_rows,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (OUTDIR / "dev_tuning_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    with (OUTDIR / "dev_tuning_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (OUTDIR / "selected_slope.json").write_text(json.dumps({
        "selected_slope": selected["slope"],
        "selection_basis": "development validation only",
        "development_seeds": DEV_SEEDS,
        "candidate_scores": slope_scores,
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected_slope": selected["slope"], "scores": slope_scores,
                      "elapsed_seconds": round(time.time() - started, 2)}, indent=2), flush=True)

if __name__ == "__main__":
    main()
