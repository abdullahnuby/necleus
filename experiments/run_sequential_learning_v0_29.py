"""Expanded sequential-learning benchmark for Nucleus Pulse v0.29.

Keeps the v0.28.1 pulse learner and fixed reservoir unchanged. Adds a linear
softmax/logistic readout on the exact same reservoir-derived temporal features,
40 fixed seeds, 500 IID test examples per task, and validation-only eligibility
for forgetting analyses. Test outcomes for every seed are always reported.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import (  # noqa: E402
    DURATION,
    DynamicPulseReservoir,
    RewardSTDPReadout,
    TemporalTrial,
    make_disjoint_dataset,
)

# Preserve the original five, add 35 fixed seeds. This list is preregistered.
SEEDS = [104, 887, 2027, 90210, 731] + list(range(3000, 3035))
N_TRAIN_PER_TASK = 200
N_VALIDATION_TASK1 = 300
N_TEST_PER_TASK = 500
N_SHIFTED_PER_TASK = 500
N_HIDDEN = 24
TRAIN_PASSES = 8
PULSE_LR = 0.01
PULSE_TEMPERATURE = 0.70
ELIGIBILITY_THRESHOLD = 0.85
LINEAR_LR = 0.01
LINEAR_WEIGHT_DECAY = 0.001
LINEAR_BATCH_SIZE = 16
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20261024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_preregistration() -> dict:
    path = ROOT / "results" / "sequential_learning_v0_29" / "preregistration.json"
    if not path.exists():
        raise RuntimeError("Missing frozen v0.29 preregistration; refusing final run.")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("seeds") != SEEDS:
        raise RuntimeError("Seed list differs from the frozen v0.29 protocol.")
    for relative, expected in protocol["sha256"].items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Protocol hash mismatch for {relative}: {actual} != {expected}")
    expected_config = {
        "train_per_task": N_TRAIN_PER_TASK,
        "validation_task1": N_VALIDATION_TASK1,
        "test_per_task": N_TEST_PER_TASK,
        "shifted_per_task": N_SHIFTED_PER_TASK,
        "hidden_neurons": N_HIDDEN,
        "passes_per_task": TRAIN_PASSES,
        "eligibility_threshold": ELIGIBILITY_THRESHOLD,
    }
    if protocol.get("config") != expected_config:
        raise RuntimeError("Configuration differs from the frozen v0.29 protocol.")
    return protocol


class LinearSoftmaxReadout:
    """Supervised linear/logistic baseline trained on pulse-reservoir features.

    It receives exactly the same 4-phase temporal feature vector as the pulse
    readout. No hidden layer is used. Train order is shared with the pulse model.
    """

    def __init__(self, n_features: int, learning_rate: float = LINEAR_LR,
                 weight_decay: float = LINEAR_WEIGHT_DECAY):
        self.w = np.zeros((2, n_features), dtype=np.float64)
        self.b = np.zeros(2, dtype=np.float64)
        self.lr = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self._mw = np.zeros_like(self.w)
        self._vw = np.zeros_like(self.w)
        self._mb = np.zeros_like(self.b)
        self._vb = np.zeros_like(self.b)
        self._step = 0

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = np.asarray(features, dtype=np.float64) @ self.w.T + self.b
        logits -= logits.max(axis=1, keepdims=True)
        exps = np.exp(np.clip(logits, -40.0, 40.0))
        return exps / exps.sum(axis=1, keepdims=True)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1).astype(np.int64)

    def fit_sequential(self, features: np.ndarray, labels: np.ndarray,
                       epoch_orders: Sequence[np.ndarray], batch_size: int = LINEAR_BATCH_SIZE) -> None:
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2 or len(x) != len(y):
            raise ValueError("features and labels must be aligned 2D/1D arrays")
        for order in epoch_orders:
            if sorted(np.asarray(order, dtype=int).tolist()) != list(range(len(y))):
                raise ValueError("each epoch order must be a permutation of the training set")
            for start in range(0, len(order), batch_size):
                ix = np.asarray(order[start:start + batch_size], dtype=int)
                xb, yb = x[ix], y[ix]
                probs = self.predict_proba(xb)
                probs[np.arange(len(yb)), yb] -= 1.0
                grad_w = probs.T @ xb / len(yb) + self.weight_decay * self.w
                grad_b = probs.mean(axis=0)
                self._step += 1
                beta1, beta2, eps = 0.9, 0.999, 1e-8
                self._mw = beta1 * self._mw + (1.0 - beta1) * grad_w
                self._vw = beta2 * self._vw + (1.0 - beta2) * (grad_w * grad_w)
                self._mb = beta1 * self._mb + (1.0 - beta1) * grad_b
                self._vb = beta2 * self._vb + (1.0 - beta2) * (grad_b * grad_b)
                mw_hat = self._mw / (1.0 - beta1 ** self._step)
                vw_hat = self._vw / (1.0 - beta2 ** self._step)
                mb_hat = self._mb / (1.0 - beta1 ** self._step)
                vb_hat = self._vb / (1.0 - beta2 ** self._step)
                self.w -= self.lr * mw_hat / (np.sqrt(vw_hat) + eps)
                self.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + eps)


def simulate_batch(reservoir: DynamicPulseReservoir,
                   trials: Sequence[TemporalTrial]) -> np.ndarray:
    """Cache actual hidden-neuron spike rasters for a trial batch."""
    if not trials:
        return np.empty((0, reservoir.duration, reservoir.n_hidden), dtype=np.float32)
    return np.stack([reservoir.simulate(trial) for trial in trials]).astype(np.float32)


def features_from_spikes(readout: RewardSTDPReadout, spike_batch: np.ndarray) -> np.ndarray:
    """Map the cached spike rasters to the readout's shared temporal features."""
    if len(spike_batch) == 0:
        return np.empty((0, readout.n_synapses), dtype=np.float32)
    return np.stack([readout.temporal_eligibility(spikes) for spikes in spike_batch]).astype(np.float32)


def make_temporal_features(reservoir: DynamicPulseReservoir,
                           readout: RewardSTDPReadout,
                           trials: Sequence[TemporalTrial]) -> np.ndarray:
    """Convenience helper to encode trials with the fixed reservoir."""
    return features_from_spikes(readout, simulate_batch(reservoir, trials))


def make_epoch_orders(seed: int, n_examples: int, n_passes: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.permutation(n_examples) for _ in range(n_passes)]


def train_pulse(readout: RewardSTDPReadout, spike_rows: np.ndarray,
                labels: np.ndarray, epoch_orders: Sequence[np.ndarray],
                action_rng: np.random.Generator) -> None:
    """Train on the original hidden-neuron spike rasters, not precomputed features."""
    for order in epoch_orders:
        for index in order:
            readout.train_trial(spike_rows[int(index)], int(labels[int(index)]), action_rng)


def macro_metrics(predictions: np.ndarray, trials: Sequence[TemporalTrial]) -> dict:
    y_true = np.asarray([t.target_class for t in trials], dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if y_true.size == 0 or y_true.shape != y_pred.shape:
        raise ValueError("predictions must align with non-empty trial sequence")
    recalls = {}
    for cls in (0, 1):
        mask = y_true == cls
        recalls[f"class_{cls}_recall"] = float(np.mean(y_pred[mask] == cls)) if np.any(mask) else None
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": float(np.mean([recalls["class_0_recall"], recalls["class_1_recall"]])),
        "class_0_recall": recalls["class_0_recall"],
        "class_1_recall": recalls["class_1_recall"],
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
    }


def bootstrap_ci(values: Sequence[float], reps: int = BOOTSTRAP_REPS,
                 seed: int = BOOTSTRAP_SEED) -> dict:
    """Percentile bootstrap CI over independent seed-level summary values."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n_seeds": 0, "mean": None, "ci95_low": None, "ci95_high": None}
    mean = float(arr.mean())
    if arr.size == 1:
        return {"n_seeds": 1, "mean": mean, "ci95_low": None, "ci95_high": None}
    rng = np.random.default_rng(seed)
    # Batch the bootstrap so memory use remains bounded even with many replicates.
    draws = np.empty(reps, dtype=np.float64)
    chunk = 1000
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        index = rng.integers(0, arr.size, size=(count, arr.size))
        draws[start:start + count] = arr[index].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"n_seeds": int(arr.size), "mean": mean,
            "ci95_low": float(low), "ci95_high": float(high)}


def summarise(rows: Sequence[dict]) -> dict:
    summary = {}
    for model in ("pulse", "linear"):
        summary[model] = {}
        for name in ("task1_validation_before_task2", "task1_test_before_task2",
                     "task1_test_after_task2", "task1_shifted_phase_test",
                     "task2_test_after_training", "task2_shifted_phase_test"):
            for metric in ("accuracy", "balanced_accuracy", "class_0_recall",
                           "class_1_recall", "predicted_positive_rate"):
                values = [100.0 * row[model][name][metric] for row in rows]
                summary[model][name + ("_percent" if metric == "accuracy" else f"_{metric}_percent")] = bootstrap_ci(
                    values, seed=BOOTSTRAP_SEED + len(name) + len(metric))
        forget = [row[model]["forgetting_task1_pp"] for row in rows]
        summary[model]["forgetting_task1_pp_all_seeds"] = bootstrap_ci(forget, seed=BOOTSTRAP_SEED + 101)
        eligible_rows = [row for row in rows if row["eligibility"][model]]
        eligible_forgetting = [row[model]["forgetting_task1_pp"] for row in eligible_rows]
        summary[model]["eligible_seed_count"] = len(eligible_rows)
        summary[model]["forgetting_task1_pp_eligible_only"] = bootstrap_ci(eligible_forgetting, seed=BOOTSTRAP_SEED + 102)
    matched = [row for row in rows if row["eligibility"]["matched"]]
    pulse_forgetting = [row["pulse"]["forgetting_task1_pp"] for row in matched]
    linear_forgetting = [row["linear"]["forgetting_task1_pp"] for row in matched]
    # Positive delta means the linear readout forgets more than the pulse learner.
    paired_delta = [lin - pul for lin, pul in zip(linear_forgetting, pulse_forgetting)]
    summary["matched_eligible_seed_count"] = len(matched)
    summary["paired_forgetting_delta_linear_minus_pulse_pp"] = bootstrap_ci(paired_delta, seed=BOOTSTRAP_SEED + 103)
    summary["paired_task2_delta_pulse_minus_linear_pp"] = bootstrap_ci([
        100.0 * (row["pulse"]["task2_test_after_training"]["accuracy"] - row["linear"]["task2_test_after_training"]["accuracy"])
        for row in matched
    ], seed=BOOTSTRAP_SEED + 104)

    # Paired all-seed deltas are descriptive, not conditional evidence of less
    # forgetting, because the 85% validation criterion may leave no eligible seeds.
    summary["paired_all_seed_differences"] = {
        "task1_before_linear_minus_pulse_pp": bootstrap_ci([
            100.0 * (r["linear"]["task1_test_before_task2"]["accuracy"] - r["pulse"]["task1_test_before_task2"]["accuracy"]) for r in rows
        ], seed=BOOTSTRAP_SEED + 201),
        "task1_after_linear_minus_pulse_pp": bootstrap_ci([
            100.0 * (r["linear"]["task1_test_after_task2"]["accuracy"] - r["pulse"]["task1_test_after_task2"]["accuracy"]) for r in rows
        ], seed=BOOTSTRAP_SEED + 202),
        "task2_linear_minus_pulse_pp": bootstrap_ci([
            100.0 * (r["linear"]["task2_test_after_training"]["accuracy"] - r["pulse"]["task2_test_after_training"]["accuracy"]) for r in rows
        ], seed=BOOTSTRAP_SEED + 203),
        "task2_shifted_linear_minus_pulse_pp": bootstrap_ci([
            100.0 * (r["linear"]["task2_shifted_phase_test"]["accuracy"] - r["pulse"]["task2_shifted_phase_test"]["accuracy"]) for r in rows
        ], seed=BOOTSTRAP_SEED + 204),
        "forgetting_linear_minus_pulse_pp": bootstrap_ci([
            r["linear"]["forgetting_task1_pp"] - r["pulse"]["forgetting_task1_pp"] for r in rows
        ], seed=BOOTSTRAP_SEED + 205),
    }
    return summary


def dataset_key_set(trials: Sequence[TemporalTrial]) -> set[tuple[tuple[int, ...], tuple[int, ...]]]:
    return {(trial.a, trial.b) for trial in trials}


def run_seed(seed: int, outdir: Path) -> dict:
    start = time.time()
    # Independent train, validation, IID test, and phase-shifted test sets.
    train_sync = make_disjoint_dataset(seed + 11, "sync", N_TRAIN_PER_TASK, "train")
    train_order = make_disjoint_dataset(seed + 23, "order", N_TRAIN_PER_TASK, "train")
    train_sync_keys, train_order_keys = dataset_key_set(train_sync), dataset_key_set(train_order)

    val_sync = make_disjoint_dataset(seed + 3011, "sync", N_VALIDATION_TASK1, "test", forbidden=train_sync_keys)
    val_sync_keys = dataset_key_set(val_sync)
    test_sync = make_disjoint_dataset(seed + 1011, "sync", N_TEST_PER_TASK, "test",
                                      forbidden=train_sync_keys | val_sync_keys)
    test_sync_keys = dataset_key_set(test_sync)
    shifted_sync = make_disjoint_dataset(seed + 1511, "sync", N_SHIFTED_PER_TASK, "shifted_test",
                                         forbidden=train_sync_keys | val_sync_keys | test_sync_keys)

    val_order = make_disjoint_dataset(seed + 3023, "order", N_VALIDATION_TASK1, "test", forbidden=train_order_keys)
    val_order_keys = dataset_key_set(val_order)
    test_order = make_disjoint_dataset(seed + 1023, "order", N_TEST_PER_TASK, "test",
                                       forbidden=train_order_keys | val_order_keys)
    test_order_keys = dataset_key_set(test_order)
    shifted_order = make_disjoint_dataset(seed + 1523, "order", N_SHIFTED_PER_TASK, "shifted_test",
                                          forbidden=train_order_keys | val_order_keys | test_order_keys)

    # The main outcome is task 1 before and after task 2; task 2 validation/test
    # are independent and are never used for eligibility or parameter tuning.
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    pulse = RewardSTDPReadout(seed=seed + 30001, n_hidden=N_HIDDEN,
                              lr=PULSE_LR, temperature=PULSE_TEMPERATURE)

    train_sync_spikes = simulate_batch(reservoir, train_sync)
    train_order_spikes = simulate_batch(reservoir, train_order)
    val_sync_spikes = simulate_batch(reservoir, val_sync)
    test_sync_spikes = simulate_batch(reservoir, test_sync)
    test_order_spikes = simulate_batch(reservoir, test_order)
    shifted_sync_spikes = simulate_batch(reservoir, shifted_sync)
    shifted_order_spikes = simulate_batch(reservoir, shifted_order)
    pulse_train_sync_x = features_from_spikes(pulse, train_sync_spikes)
    pulse_train_order_x = features_from_spikes(pulse, train_order_spikes)
    val_sync_x = features_from_spikes(pulse, val_sync_spikes)
    test_sync_x = features_from_spikes(pulse, test_sync_spikes)
    test_order_x = features_from_spikes(pulse, test_order_spikes)
    shifted_sync_x = features_from_spikes(pulse, shifted_sync_spikes)
    shifted_order_x = features_from_spikes(pulse, shifted_order_spikes)

    y_train_sync = np.asarray([t.target_class for t in train_sync], dtype=np.int64)
    y_train_order = np.asarray([t.target_class for t in train_order], dtype=np.int64)
    epoch_orders_sync = make_epoch_orders(seed + 40101, len(train_sync), TRAIN_PASSES)
    epoch_orders_order = make_epoch_orders(seed + 40201, len(train_order), TRAIN_PASSES)

    # Pulse model: same learner, train budget, and examples as v0.28.1.
    pulse_action_rng = np.random.default_rng(seed + 40002)
    train_pulse(pulse, train_sync_spikes, y_train_sync, epoch_orders_sync, pulse_action_rng)
    pulse_val_before = macro_metrics(np.asarray([pulse.predict(spikes)[0] for spikes in val_sync_spikes]), val_sync)
    pulse_test_before = macro_metrics(np.asarray([pulse.predict(spikes)[0] for spikes in test_sync_spikes]), test_sync)

    # Linear/logistic readout on the exact same reservoir temporal features.
    linear = LinearSoftmaxReadout(n_features=pulse.n_synapses)
    linear.fit_sequential(pulse_train_sync_x, y_train_sync, epoch_orders_sync)
    linear_val_before = macro_metrics(linear.predict(val_sync_x), val_sync)
    linear_test_before = macro_metrics(linear.predict(test_sync_x), test_sync)

    # Eligibility is decided from validation only, before task-2 training; test
    # performance is never used to select which seeds count for the conditional
    # forgetting estimate. All-seed results remain primary descriptive outputs.
    pulse_eligible = pulse_val_before["accuracy"] >= ELIGIBILITY_THRESHOLD
    linear_eligible = linear_val_before["accuracy"] >= ELIGIBILITY_THRESHOLD

    # Sequential phase: train on task 2 only. No task-1 replay.
    train_pulse(pulse, train_order_spikes, y_train_order, epoch_orders_order, pulse_action_rng)
    pulse_test_after = macro_metrics(np.asarray([pulse.predict(spikes)[0] for spikes in test_sync_spikes]), test_sync)
    pulse_shifted_sync_test = macro_metrics(np.asarray([pulse.predict(spikes)[0] for spikes in shifted_sync_spikes]), shifted_sync)
    pulse_task2_test = macro_metrics(np.asarray([pulse.predict(spikes)[0] for spikes in test_order_spikes]), test_order)
    pulse_shifted_test = macro_metrics(np.asarray([pulse.predict(spikes)[0] for spikes in shifted_order_spikes]), shifted_order)

    linear.fit_sequential(pulse_train_order_x, y_train_order, epoch_orders_order)
    linear_test_after = macro_metrics(linear.predict(test_sync_x), test_sync)
    linear_shifted_sync_test = macro_metrics(linear.predict(shifted_sync_x), shifted_sync)
    linear_task2_test = macro_metrics(linear.predict(test_order_x), test_order)
    linear_shifted_test = macro_metrics(linear.predict(shifted_order_x), shifted_order)

    pulse_forgetting = 100.0 * (pulse_test_before["accuracy"] - pulse_test_after["accuracy"])
    linear_forgetting = 100.0 * (linear_test_before["accuracy"] - linear_test_after["accuracy"])

    # Exact-sequence disjointness is asserted and recorded for each split.
    integrity = {
        "train_val_overlap_task1": len(train_sync_keys & val_sync_keys),
        "train_test_overlap_task1": len(train_sync_keys & test_sync_keys),
        "val_test_overlap_task1": len(val_sync_keys & test_sync_keys),
        "train_shifted_overlap_task1": len(train_sync_keys & dataset_key_set(shifted_sync)),
        "test_shifted_overlap_task1": len(test_sync_keys & dataset_key_set(shifted_sync)),
        "val_shifted_overlap_task1": len(val_sync_keys & dataset_key_set(shifted_sync)),
        "train_val_overlap_task2": len(train_order_keys & val_order_keys),
        "train_test_overlap_task2": len(train_order_keys & test_order_keys),
        "val_test_overlap_task2": len(val_order_keys & test_order_keys),
        "train_shifted_overlap_task2": len(train_order_keys & dataset_key_set(shifted_order)),
        "test_shifted_overlap_task2": len(test_order_keys & dataset_key_set(shifted_order)),
        "val_shifted_overlap_task2": len(val_order_keys & dataset_key_set(shifted_order)),
    }
    if any(integrity.values()):
        raise RuntimeError(f"Sequence leakage/overlap detected for seed {seed}: {integrity}")

    row = {
        "seed": seed,
        "eligibility": {"threshold": ELIGIBILITY_THRESHOLD,
                        "pulse": bool(pulse_eligible), "linear": bool(linear_eligible),
                        "matched": bool(pulse_eligible and linear_eligible),
                        "basis": "task-1 validation accuracy before task 2; never test accuracy"},
        "data_integrity": integrity,
        "pulse": {
            "task1_validation_before_task2": pulse_val_before,
            "task1_test_before_task2": pulse_test_before,
            "task1_test_after_task2": pulse_test_after,
            "task1_shifted_phase_test": pulse_shifted_sync_test,
            "task2_test_after_training": pulse_task2_test,
            "task2_shifted_phase_test": pulse_shifted_test,
            "forgetting_task1_pp": pulse_forgetting,
            "weight_min": float(pulse.w.min()), "weight_max": float(pulse.w.max()),
        },
        "linear": {
            "task1_validation_before_task2": linear_val_before,
            "task1_test_before_task2": linear_test_before,
            "task1_test_after_task2": linear_test_after,
            "task1_shifted_phase_test": linear_shifted_sync_test,
            "task2_test_after_training": linear_task2_test,
            "task2_shifted_phase_test": linear_shifted_test,
            "forgetting_task1_pp": linear_forgetting,
            "weight_min": float(linear.w.min()), "weight_max": float(linear.w.max()),
        },
        "elapsed_seconds": float(time.time() - start),
    }
    return row


def flatten_rows(rows: Sequence[dict]) -> list[dict]:
    flattened = []
    for row in rows:
        item = {"seed": row["seed"],
                "pulse_eligible_validation_ge_85": row["eligibility"]["pulse"],
                "linear_eligible_validation_ge_85": row["eligibility"]["linear"],
                "matched_eligible": row["eligibility"]["matched"],
                "pulse_task1_val_accuracy_pct": 100 * row["pulse"]["task1_validation_before_task2"]["accuracy"],
                "linear_task1_val_accuracy_pct": 100 * row["linear"]["task1_validation_before_task2"]["accuracy"],
                "pulse_task1_test_before_pct": 100 * row["pulse"]["task1_test_before_task2"]["accuracy"],
                "pulse_task1_test_before_recall0_pct": 100 * row["pulse"]["task1_test_before_task2"]["class_0_recall"],
                "pulse_task1_test_before_recall1_pct": 100 * row["pulse"]["task1_test_before_task2"]["class_1_recall"],
                "pulse_task1_test_before_predicted_positive_pct": 100 * row["pulse"]["task1_test_before_task2"]["predicted_positive_rate"],
                "pulse_task1_test_after_pct": 100 * row["pulse"]["task1_test_after_task2"]["accuracy"],
                "pulse_task1_test_after_recall0_pct": 100 * row["pulse"]["task1_test_after_task2"]["class_0_recall"],
                "pulse_task1_test_after_recall1_pct": 100 * row["pulse"]["task1_test_after_task2"]["class_1_recall"],
                "pulse_task1_shifted_pct": 100 * row["pulse"]["task1_shifted_phase_test"]["accuracy"],
                "pulse_task2_test_pct": 100 * row["pulse"]["task2_test_after_training"]["accuracy"],
                "pulse_task2_test_recall0_pct": 100 * row["pulse"]["task2_test_after_training"]["class_0_recall"],
                "pulse_task2_test_recall1_pct": 100 * row["pulse"]["task2_test_after_training"]["class_1_recall"],
                "pulse_task2_shifted_pct": 100 * row["pulse"]["task2_shifted_phase_test"]["accuracy"],
                "pulse_forgetting_pp": row["pulse"]["forgetting_task1_pp"],
                "linear_task1_test_before_pct": 100 * row["linear"]["task1_test_before_task2"]["accuracy"],
                "linear_task1_test_before_recall0_pct": 100 * row["linear"]["task1_test_before_task2"]["class_0_recall"],
                "linear_task1_test_before_recall1_pct": 100 * row["linear"]["task1_test_before_task2"]["class_1_recall"],
                "linear_task1_test_before_predicted_positive_pct": 100 * row["linear"]["task1_test_before_task2"]["predicted_positive_rate"],
                "linear_task1_test_after_pct": 100 * row["linear"]["task1_test_after_task2"]["accuracy"],
                "linear_task1_test_after_recall0_pct": 100 * row["linear"]["task1_test_after_task2"]["class_0_recall"],
                "linear_task2_test_pct": 100 * row["linear"]["task2_test_after_training"]["accuracy"],
                "linear_task2_test_recall0_pct": 100 * row["linear"]["task2_test_after_training"]["class_0_recall"],
                "linear_task2_test_recall1_pct": 100 * row["linear"]["task2_test_after_training"]["class_1_recall"],
                "linear_task2_shifted_pct": 100 * row["linear"]["task2_shifted_phase_test"]["accuracy"],
                "linear_forgetting_pp": row["linear"]["forgetting_task1_pp"],
                "elapsed_seconds": row["elapsed_seconds"]}
        flattened.append(item)
    return flattened


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    flat = flatten_rows(rows)
    if not flat:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)


def render_report(results: dict) -> str:
    summary = results["summary"]

    def fmt(name: str, metric: str) -> str:
        value = summary[name][metric]
        return fmt_value(value)

    def fmt_value(value: dict) -> str:
        if value["mean"] is None:
            return "غير متاح (لا توجد بذور مؤهلة)"
        if value["ci95_low"] is None:
            return f"{value['mean']:.2f} (بذرة واحدة فقط؛ لا يمكن حساب CI)"
        return f"{value['mean']:.2f}، CI95% [{value['ci95_low']:.2f}, {value['ci95_high']:.2f}]، n={value['n_seeds']}"

    lines = [
        "# تقرير تجربة التعلم المتتابع v0.29",
        "",
        "## النتيجة المنهجية الأساسية",
        "",
        f"عدد البذور: **{len(results['seeds'])}**. كل مهمة لها **{N_TEST_PER_TASK}** مثال اختبار IID مستقل، ومجموعة تحقق مستقلة لمهمة 1 من **{N_VALIDATION_TASK1}** مثال. حد أهلية تحليل النسيان هو **{ELIGIBILITY_THRESHOLD:.0%}** على مجموعة التحقق قبل تعلم المهمة 2. لا تُستبعد أي بذرة من جداول الأداء الكاملة.",
        "",
        "القارئ الخطي المدرّب بإشراف يستخدم **نفس ميزات الخزان النبضي الثابت** التي تدخل إلى قارئ النبضات؛ لذلك هو baseline أوضح من MLP على ترميز الإدخال الخام. المتعلم في نموذج النبضات لم يتغير عن v0.28.1. لا replay للمهمة 1 أثناء تعلم المهمة 2.",
        "",
        "## النتائج على كل البذور (متوسط دقة %، CI 95% عبر البذور)",
        "",
        "| المقياس | النبضات | القارئ الخطي |",
        "|---|---:|---:|",
        f"| المهمة 1 على التحقق قبل المهمة 2 | {fmt('pulse','task1_validation_before_task2_percent')} | {fmt('linear','task1_validation_before_task2_percent')} |",
        f"| المهمة 1 على الاختبار قبل المهمة 2 | {fmt('pulse','task1_test_before_task2_percent')} | {fmt('linear','task1_test_before_task2_percent')} |",
        f"| المهمة 1 على الاختبار بعد المهمة 2 | {fmt('pulse','task1_test_after_task2_percent')} | {fmt('linear','task1_test_after_task2_percent')} |",
        f"| المهمة 1 بعد إزاحة الطور | {fmt('pulse','task1_shifted_phase_test_percent')} | {fmt('linear','task1_shifted_phase_test_percent')} |",
        f"| المهمة 2 على الاختبار | {fmt('pulse','task2_test_after_training_percent')} | {fmt('linear','task2_test_after_training_percent')} |",
        f"| المهمة 2 بعد إزاحة الطور | {fmt('pulse','task2_shifted_phase_test_percent')} | {fmt('linear','task2_shifted_phase_test_percent')} |",
        f"| النسيان على كل البذور (نقطة مئوية) | {fmt('pulse','forgetting_task1_pp_all_seeds')} | {fmt('linear','forgetting_task1_pp_all_seeds')} |",
        "",
        "## فروق زوجية وصفية عبر كل البذور",
        "",
        "الفروق التالية تقارن النموذجين داخل البذرة نفسها، لكنّها تظل وصفية لأن شرط تعلم المهمة الأساسية لم يتحقق في أي بذرة.",
        "",
        "| الفرق (الخطي − النبضات، نقطة مئوية) | القيمة مع CI95% |",
        "|---|---:|",
        f"| المهمة 1 قبل المهمة 2 | {fmt_value(summary['paired_all_seed_differences']['task1_before_linear_minus_pulse_pp'])} |",
        f"| المهمة 1 بعد المهمة 2 | {fmt_value(summary['paired_all_seed_differences']['task1_after_linear_minus_pulse_pp'])} |",
        f"| المهمة 2 | {fmt_value(summary['paired_all_seed_differences']['task2_linear_minus_pulse_pp'])} |",
        f"| المهمة 2 بعد إزاحة الطور | {fmt_value(summary['paired_all_seed_differences']['task2_shifted_linear_minus_pulse_pp'])} |",
        f"| النسيان (الخطي − النبضات؛ الموجب يعني نسيانًا أعلى للخطي) | {fmt_value(summary['paired_all_seed_differences']['forgetting_linear_minus_pulse_pp'])} |",
        "",
        "## استدعاء الفئتين ومعدل التنبؤ الإيجابي",
        "",
        "| المؤشر | النبضات | القارئ الخطي |",
        "|---|---:|---:|",
        f"| استدعاء الفئة 0، المهمة 1 قبل | {fmt('pulse','task1_test_before_task2_class_0_recall_percent')} | {fmt('linear','task1_test_before_task2_class_0_recall_percent')} |",
        f"| استدعاء الفئة 1، المهمة 1 قبل | {fmt('pulse','task1_test_before_task2_class_1_recall_percent')} | {fmt('linear','task1_test_before_task2_class_1_recall_percent')} |",
        f"| معدل التنبؤ الإيجابي، المهمة 1 قبل | {fmt('pulse','task1_test_before_task2_predicted_positive_rate_percent')} | {fmt('linear','task1_test_before_task2_predicted_positive_rate_percent')} |",
        f"| استدعاء الفئة 0، المهمة 2 | {fmt('pulse','task2_test_after_training_class_0_recall_percent')} | {fmt('linear','task2_test_after_training_class_0_recall_percent')} |",
        f"| استدعاء الفئة 1، المهمة 2 | {fmt('pulse','task2_test_after_training_class_1_recall_percent')} | {fmt('linear','task2_test_after_training_class_1_recall_percent')} |",
        f"| معدل التنبؤ الإيجابي، المهمة 2 | {fmt('pulse','task2_test_after_training_predicted_positive_rate_percent')} | {fmt('linear','task2_test_after_training_predicted_positive_rate_percent')} |",
        "",
        "## تحليل النسيان بشرط التعلم المسبق",
        "",
        f"بذور تحقق فيها نموذج النبضات دقة تحقق المهمة 1 ≥85%: **{summary['pulse']['eligible_seed_count']} من {len(results['seeds'])}**. القارئ الخطي: **{summary['linear']['eligible_seed_count']} من {len(results['seeds'])}**. البذور المشتركة المؤهلة للمقارنة الزوجية: **{summary['matched_eligible_seed_count']}**.",
        f"متوسط نسيان النبضات ضمن بذورها المؤهلة: {fmt('pulse','forgetting_task1_pp_eligible_only')}.",
        f"متوسط نسيان القارئ الخطي ضمن بذوره المؤهلة: {fmt('linear','forgetting_task1_pp_eligible_only')}.",
        "فرق النسيان (الخطي − النبضات؛ الموجب يعني أن القارئ الخطي نسي أكثر) على البذور المؤهلة المشتركة: سيُدرج CI إذا كان عدد البذور كافيًا.",
        "",
        "**قاعدة القراءة:** لو لم توجد بذور مشتركة مؤهلة أو كان عددها قليلًا جدًا، لا يجوز ادعاء أن قاعدة تتفوق في النسيان؛ النتيجة الصحيحة أن المهمة الأساسية لم تجهز تجربة نسيان قابلة للتفسير بعد. شرط الأهلية يُحسب من التحقق قبل المهمة 2 وليس من الاختبار، لتجنب انتقاء البذور بناءً على نتيجة الاختبار.",
        "",
        "## تشخيص التحيز",
        "",
        "ملف CSV وملف JSON يحتويان على دقة كل بذرة، استدعاء الفئتين، ونسبة التنبؤات الإيجابية. افحص توازن الاستدعاء للفئتين؛ الدقة الإجمالية وحدها قد تخفي انحيازًا نحو الفئة الموجبة.",
        "",
        "## حدود الاستنتاج",
        "",
        "- هذه تجربة حسابية على خزّان عشوائي ثابت؛ ليست بعدُ اختبارًا لشبكة نبضية ذات لدونة داخلية متعلمة.",
        "- القارئ الخطي baseline مضبوط على تمثيل الخزان نفسه، لكنه يتعلم بإشراف مباشر بينما قارئ النبضات يتعلم بمكافأة؛ المقارنة تفصل قيمة قاعدة التعلم عن قيمة تمثيل الخزان جزئيًا، ولا تجعل ميزانية المعلومات متطابقة تمامًا.",
        "- فواصل الثقة تُحسب بإعادة أخذ عينات على مستوى البذور من متوسطات البذور؛ وهي لا تثبت السببية أو التفوق لوحدها.",
        "- إزاحة الطور اختبار توزيع مختلف، وتُعرض منفصلة عن أداء IID.",
        "",
        f"مدة التشغيل: {results['runtime_seconds']:.1f} ثانية.",
        "",
    ]
    paired = summary["paired_forgetting_delta_linear_minus_pulse_pp"]
    if paired["mean"] is None:
        placeholder = "فرق النسيان (الخطي − النبضات؛ الموجب يعني أن القارئ الخطي نسي أكثر) على البذور المؤهلة المشتركة: سيُدرج CI إذا كان عدد البذور كافيًا."
        lines[lines.index(placeholder)] = "فرق النسيان المشروط (الخطي − النبضات): غير متاح، لأن لا بذرة اجتازت شرط الأهلية في النموذجين معًا."
    else:
        lines[lines.index("فرق النسيان (الخطي − النبضات؛ الموجب يعني أن القارئ الخطي نسي أكثر) على البذور المؤهلة المشتركة: سيُدرج CI إذا كان عدد البذور كافيًا.")] = (
            f"فرق النسيان (الخطي − النبضات؛ الموجب يعني أن القارئ الخطي نسي أكثر) على البذور المؤهلة المشتركة: {paired['mean']:.2f}، CI95% [{paired['ci95_low']:.2f}, {paired['ci95_high']:.2f}]، n={paired['n_seeds']}."
        )
    return "\n".join(lines)


def main() -> None:
    protocol = verify_preregistration()
    outdir = ROOT / "results" / "sequential_learning_v0_29"
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows: list[dict] = []
    for index, seed in enumerate(SEEDS, start=1):
        row = run_seed(seed, outdir)
        rows.append(row)
        partial = {"version": "0.29", "seeds_completed": len(rows), "seeds": SEEDS[:len(rows)],
                   "per_seed": rows, "summary": None, "partial": True,
                   "note": "Final bootstrap CIs are calculated once, after all preregistered seeds finish."}
        (outdir / "results_partial.json").write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            f"[{index}/{len(SEEDS)}] seed={seed} "
            f"pulse val={row['pulse']['task1_validation_before_task2']['accuracy']:.3f}, "
            f"test1={row['pulse']['task1_test_before_task2']['accuracy']:.3f}->"
            f"{row['pulse']['task1_test_after_task2']['accuracy']:.3f}, "
            f"shift1={row['pulse']['task1_shifted_phase_test']['accuracy']:.3f}, "
            f"test2={row['pulse']['task2_test_after_training']['accuracy']:.3f}; "
            f"linear val={row['linear']['task1_validation_before_task2']['accuracy']:.3f}, "
            f"test2={row['linear']['task2_test_after_training']['accuracy']:.3f}; "
            f"eligible={row['eligibility']['matched']}; {row['elapsed_seconds']:.1f}s",
            flush=True,
        )

    summary = summarise(rows)
    results = {
        "version": "0.29",
        "seeds": SEEDS,
        "per_seed": rows,
        "summary": summary,
        "runtime_seconds": float(time.time() - started),
        "protocol": {
            "train_per_task": N_TRAIN_PER_TASK,
            "validation_task1": N_VALIDATION_TASK1,
            "test_per_task": N_TEST_PER_TASK,
            "shifted_per_task": N_SHIFTED_PER_TASK,
            "passes_per_task": TRAIN_PASSES,
            "hidden_neurons": N_HIDDEN,
            "pulse_lr": PULSE_LR,
            "pulse_temperature": PULSE_TEMPERATURE,
            "linear_lr": LINEAR_LR,
            "linear_weight_decay": LINEAR_WEIGHT_DECAY,
            "eligibility_threshold": ELIGIBILITY_THRESHOLD,
            "preregistration_sha256": sha256_file(outdir / "preregistration.json"),
            "source_hashes": protocol["sha256"],
            "test_based_seed_exclusions": 0,
        },
    }
    (outdir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "results_partial.json").unlink(missing_ok=True)
    write_csv(outdir / "per_seed_summary.csv", rows)
    (outdir / "report_ar.md").write_text(render_report(results), encoding="utf-8")
    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
