"""Nucleus Pulse v0.34: paired test of structured recurrent inhibition.

Compares an all-excitatory recurrent control to a Dale-style reservoir with
25% inhibitory source neurons. Connectivity, base connection magnitudes,
input weights, thresholds, leak, neuron count, tasks, features, reader and
training schedule are paired. Each condition is normalized to the same target
spectral radius (0.72) to control gross recurrent gain/stability.
"""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir  # noqa: E402
from experiments.run_learning_curve_v0_30 import (  # noqa: E402
    N_HIDDEN,
    make_epoch_labels,
    make_nested_training_pool,
    rasterize_batch,
    trial_key,
)
from experiments.run_incremental_windows_v0_31 import (  # noqa: E402
    classification_metrics,
    make_eval_dataset,
)

VERSION = "0.34"
SEEDS = list(range(7000, 7040))
CONDITIONS = ["all_excitatory", "dale_inhibition_25"]
TASKS = ["sync", "order"]
EPOCH_CHECKPOINTS = [1, 3, 5, 10]
PRIMARY_EPOCH = 10
N_TRAIN = 6000
N_VALIDATION = 1000
N_TEST = 1000
N_WINDOWS = 16
FEATURE_SET = "spike_bins16_rate_384"
INHIBITORY_FRACTION = 0.25
TARGET_SPECTRAL_RADIUS = 0.72
TARGET_VALIDATION_ACCURACY = 0.85
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20341024


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol_config() -> dict:
    return {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "tasks": TASKS,
        "unique_training_examples_per_task_seed": N_TRAIN,
        "validation_per_task_seed": N_VALIDATION,
        "final_test_per_task_seed": N_TEST,
        "training_pass_checkpoints": EPOCH_CHECKPOINTS,
        "primary_epoch": PRIMARY_EPOCH,
        "hidden_neurons": N_HIDDEN,
        "feature_set": FEATURE_SET,
        "feature_definition": "16 equal temporal bins, 2 ticks per bin, hidden spike counts divided by 2",
        "control": "all existing recurrent edges positive; shared connectivity mask and magnitudes before common spectral-radius normalization",
        "treatment": {
            "network_rule": "Dale-style fixed source polarity",
            "inhibitory_source_fraction": INHIBITORY_FRACTION,
            "inhibitory_source_count": int(round(N_HIDDEN * INHIBITORY_FRACTION)),
            "source_mask_seed": "reservoir_seed_plus_934001",
            "all_outgoing_edges_from_inhibitory_sources_negative": True,
            "all_outgoing_edges_from_excitatory_sources_positive": True,
        },
        "spectral_radius_target_both_conditions": TARGET_SPECTRAL_RADIUS,
        "normalization": "each condition independently scaled to identical target spectral radius; same topology and pre-normalization absolute edge magnitudes",
        "input_weights_thresholds_leaks_and_refractory_identical_between_conditions": True,
        "readout": {
            "class": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss", "penalty": "l2", "alpha": 1e-4,
            "learning_rate": "constant", "eta0": 0.01,
            "average": True, "shuffle": True, "fit_method": "partial_fit",
            "same_model_random_seed_paired_between_conditions": True,
        },
        "validation_used_for_threshold_analysis_only": True,
        "final_test_used_for_tuning_or_selection": False,
        "bootstrap_unit": "paired seed",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def verify_preregistration() -> dict:
    path = ROOT / "results" / "inhibition_structure_v0_34" / "preregistration.json"
    if not path.exists():
        raise RuntimeError("Missing frozen v0.34 preregistration; refusing full run.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Current run configuration differs from preregistration.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return doc


def atomic_json(path: Path, payload) -> None:
    def safe(v):
        if isinstance(v, dict):
            return {str(k): safe(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [safe(x) for x in v]
        if isinstance(v, np.generic):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            return None
        return v
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode("utf-8")).digest()[:4], "little")


def bootstrap_ci(values: Sequence[float], seed: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean() * 100.0)
    if arr.size == 1:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPS, 1000):
        n = min(1000, BOOTSTRAP_REPS - start)
        idx = rng.integers(0, arr.size, size=(n, arr.size))
        draws[start:start+n] = arr[idx].mean(axis=1) * 100.0
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return float(max(0, center - half) * 100), float(min(1, center + half) * 100)


def spectral_radius(weights: np.ndarray) -> float:
    values = np.linalg.eigvals(np.asarray(weights, dtype=np.float64))
    return float(np.max(np.abs(values))) if values.size else 0.0


def make_recurrent_weight_matrices(reservoir: DynamicPulseReservoir, seed: int) -> tuple[dict[str, np.ndarray], dict]:
    """Build paired all-excitatory and Dale-style E/I matrices.

    Existing topology and initial absolute edge values are shared. In this
    matrix convention, columns are presynaptic source neurons, so polarity is
    applied by source column. Both conditions are normalized to radius 0.72.
    """
    base = np.abs(np.asarray(reservoir.w_rec, dtype=np.float64))
    edge_mask = base != 0.0
    n = reservoir.n_hidden
    n_inhib = max(1, int(round(n * INHIBITORY_FRACTION)))
    rng = np.random.default_rng(int(seed) + 934001)
    eligible_sources = np.flatnonzero(np.any(edge_mask, axis=0))
    if eligible_sources.size < n_inhib:
        raise RuntimeError("Not enough recurrent source neurons with outgoing edges for inhibitory assignment")
    inhibitory_sources = np.sort(rng.choice(eligible_sources, size=n_inhib, replace=False))
    source_sign = np.ones(n, dtype=np.float64)
    source_sign[inhibitory_sources] = -1.0

    raw = {
        "all_excitatory": base.copy(),
        "dale_inhibition_25": base * source_sign[None, :],
    }
    matrices = {}
    diagnostics = {
        "inhibitory_source_indices": inhibitory_sources.astype(int).tolist(),
        "inhibitory_source_count": int(n_inhib),
        "hidden_neuron_count": int(n),
        "inhibitory_source_fraction": float(n_inhib / n),
        "existing_recurrent_edges": int(edge_mask.sum()),
        "topology_equal": True,
        "pre_normalization_absolute_magnitudes_equal": True,
        "target_spectral_radius": TARGET_SPECTRAL_RADIUS,
        "conditions": {},
    }
    for name, matrix in raw.items():
        if not np.array_equal(matrix != 0.0, edge_mask):
            raise RuntimeError("E/I structure changed recurrent connectivity mask")
        radius_before = spectral_radius(matrix)
        if radius_before <= 1e-12:
            raise RuntimeError(f"Cannot normalize degenerate recurrent matrix for {name}")
        scale = TARGET_SPECTRAL_RADIUS / radius_before
        normalized = (matrix * scale).astype(np.float32)
        radius_after = spectral_radius(normalized)
        if not np.isclose(radius_after, TARGET_SPECTRAL_RADIUS, atol=5e-4):
            raise RuntimeError(f"Spectral radius normalization failed for {name}: {radius_after}")
        if name == "all_excitatory":
            expected_negative = 0
        else:
            expected_negative = int(np.sum(edge_mask[:, inhibitory_sources]))
        observed_negative = int(np.sum(normalized < 0))
        if observed_negative != expected_negative:
            raise RuntimeError(f"Recurrent sign pattern invariant failed for {name}")
        matrices[name] = normalized
        diagnostics["conditions"][name] = {
            "spectral_radius_before_normalization": radius_before,
            "normalization_scale": float(scale),
            "spectral_radius_after_normalization": radius_after,
            "positive_edge_count": int(np.sum(normalized > 0)),
            "negative_edge_count": observed_negative,
            "negative_edge_fraction": float(observed_negative / edge_mask.sum()),
            "absolute_weight_sum": float(np.sum(np.abs(normalized))),
            "mean_absolute_edge_weight": float(np.mean(np.abs(normalized[edge_mask]))),
        }
    # Both conditions retain exactly the same graph support, including zero diagonal.
    if not np.array_equal(matrices[CONDITIONS[0]] != 0, matrices[CONDITIONS[1]] != 0):
        raise RuntimeError("Connectivity topology differs across conditions")
    return matrices, diagnostics


def simulate_batch_with_recurrent_weights(reservoir: DynamicPulseReservoir,
                                          trials: Sequence,
                                          recurrent_weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized LIF simulation with an explicitly supplied recurrent matrix."""
    if recurrent_weights.shape != reservoir.w_rec.shape:
        raise ValueError("recurrent_weights must match reservoir.w_rec shape")
    if not np.isfinite(recurrent_weights).all():
        raise ValueError("recurrent_weights contains non-finite values")
    if not trials:
        return (np.empty((0, reservoir.duration, 4), np.float32),
                np.empty((0, reservoir.duration, reservoir.n_hidden), np.float32))
    inputs = rasterize_batch(trials, reservoir.duration)
    count, duration, _ = inputs.shape
    hidden = reservoir.n_hidden
    voltage = np.zeros((count, hidden), dtype=np.float32)
    i_syn = np.zeros((count, hidden), dtype=np.float32)
    refractory = np.zeros((count, hidden), dtype=np.int8)
    previous_spikes = np.zeros((count, hidden), dtype=np.float32)
    spikes = np.zeros((count, duration, hidden), dtype=np.float32)
    w_in_t = reservoir.w_in.T
    w_rec_t = np.asarray(recurrent_weights, dtype=np.float32).T
    for t in range(duration):
        i_syn = (reservoir.syn_leak[None, :] * i_syn
                 + inputs[:, t, :] @ w_in_t
                 + previous_spikes @ w_rec_t)
        voltage = reservoir.reset_leak[None, :] * voltage + (1.0 - reservoir.reset_leak[None, :]) * i_syn
        active = (voltage >= reservoir.threshold[None, :]) & (refractory == 0)
        spikes[:, t, :] = active.astype(np.float32)
        voltage[active] = 0.0
        refractory = np.maximum(refractory - 1, 0)
        refractory[active] = 1
        previous_spikes = active.astype(np.float32)
    return inputs, spikes


def extract_features(spikes: np.ndarray) -> np.ndarray:
    if spikes.ndim != 3 or spikes.shape[1:] != (32, N_HIDDEN):
        raise ValueError(f"Expected spike tensor (N,32,{N_HIDDEN}), got {spikes.shape}")
    values = spikes.reshape(spikes.shape[0], N_WINDOWS, 2, N_HIDDEN).sum(axis=2) / 2.0
    values = values.reshape(spikes.shape[0], N_WINDOWS * N_HIDDEN).astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise RuntimeError("Temporal feature matrix contains non-finite values")
    return values


def run_reader(features: dict[str, np.ndarray], labels: dict[str, np.ndarray], seed: int, task_i: int) -> list[dict]:
    from sklearn.linear_model import SGDClassifier
    x_train, x_val, x_test = features["train"], features["validation"], features["test"]
    y_train, y_val, y_test = labels["train"], labels["validation"], labels["test"]
    model_seed = int(seed + 940001 + task_i * 1000)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          learning_rate="constant", eta0=0.01, average=True,
                          shuffle=True, random_state=model_seed)
    rows = []
    cumulative_seconds = 0.0
    for epoch in range(1, PRIMARY_EPOCH + 1):
        started = time.time()
        if epoch == 1:
            model.partial_fit(x_train, y_train, classes=np.asarray([0, 1], dtype=np.int64))
        else:
            model.partial_fit(x_train, y_train)
        pass_seconds = time.time() - started
        cumulative_seconds += pass_seconds
        if epoch not in EPOCH_CHECKPOINTS:
            continue
        val_metrics = classification_metrics(y_val, model.predict(x_val))
        test_metrics = classification_metrics(y_test, model.predict(x_test))
        rows.append({
            "seed": int(seed), "task": TASKS[task_i], "condition": "",
            "epochs": int(epoch), "n_train_unique": N_TRAIN,
            "cumulative_example_exposures": int(N_TRAIN * epoch),
            **{f"validation_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
            "epoch_fit_seconds": round(pass_seconds, 6),
            "cumulative_fit_seconds": round(cumulative_seconds, 6),
        })
    return rows


def run_seed(seed: int) -> tuple[list[dict], dict]:
    started = time.time()
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    recurrent_matrices, recurrent_diagnostics = make_recurrent_weight_matrices(reservoir, seed)
    rows: list[dict] = []
    integrity: dict = {
        "seed": int(seed), "reservoir_seed": int(seed),
        "same_topology_and_pre_normalization_magnitudes": True,
        "w_in_thresholds_leaks_and_neuron_count_shared": True,
        "recurrent_diagnostics": recurrent_diagnostics,
        "tasks": {},
    }
    for task_i, task in enumerate(TASKS):
        train_pool = make_nested_training_pool(seed + 11111 + task_i * 100, task, N_TRAIN)
        train_keys = {trial_key(tr) for tr in train_pool}
        val_trials = make_eval_dataset(seed + 52222 + task_i * 100, task, N_VALIDATION, set(train_keys))
        val_keys = {trial_key(tr) for tr in val_trials}
        test_trials = make_eval_dataset(seed + 63333 + task_i * 100, task, N_TEST, set(train_keys) | val_keys)
        test_keys = {trial_key(tr) for tr in test_trials}
        if len(train_keys) != N_TRAIN or len(val_keys) != N_VALIDATION or len(test_keys) != N_TEST:
            raise RuntimeError(f"Non-unique sequences at seed={seed}, task={task}")
        if (train_keys & val_keys) or (train_keys & test_keys) or (val_keys & test_keys):
            raise RuntimeError(f"Split overlap at seed={seed}, task={task}")
        trial_sets = {"train": train_pool, "validation": val_trials, "test": test_trials}
        labels = {split: make_epoch_labels(trials) for split, trials in trial_sets.items()}
        for split, ys in labels.items():
            if int(np.sum(ys == 0)) != len(ys) // 2 or int(np.sum(ys == 1)) != len(ys) // 2:
                raise RuntimeError(f"Class imbalance at seed={seed}, task={task}, split={split}")
        concatenated = train_pool + val_trials + test_trials
        bounds = {
            "train": (0, N_TRAIN),
            "validation": (N_TRAIN, N_TRAIN + N_VALIDATION),
            "test": (N_TRAIN + N_VALIDATION, N_TRAIN + N_VALIDATION + N_TEST),
        }
        features_by_condition: dict[str, dict[str, np.ndarray]] = {}
        activity = {}
        expected_inputs = None
        for condition in CONDITIONS:
            inputs, spikes = simulate_batch_with_recurrent_weights(reservoir, concatenated, recurrent_matrices[condition])
            if expected_inputs is None:
                expected_inputs = inputs
            elif not np.array_equal(inputs, expected_inputs):
                raise RuntimeError("Input raster changed across recurrent conditions")
            feat = {}
            for split, (left, right) in bounds.items():
                feat[split] = extract_features(spikes[left:right])
            features_by_condition[condition] = feat
            total_per_trial = np.sum(spikes, axis=(1, 2))
            per_cell = np.sum(spikes, axis=(0, 1))
            activity[condition] = {
                "mean_spikes_per_trial": float(np.mean(total_per_trial)),
                "fraction_hidden_cells_with_at_least_one_spike": float(np.mean(per_cell > 0)),
                "fraction_trials_with_zero_hidden_spikes": float(np.mean(total_per_trial == 0)),
                "mean_spike_rate_per_hidden_tick": float(np.mean(spikes)),
            }
            del inputs, spikes
        integrity["tasks"][task] = {
            "n_train_unique": len(train_keys), "n_validation_unique": len(val_keys), "n_test_unique": len(test_keys),
            "train_validation_overlap": len(train_keys & val_keys),
            "train_test_overlap": len(train_keys & test_keys),
            "validation_test_overlap": len(val_keys & test_keys),
            "class_balance": {split: {"0": int(np.sum(ys == 0)), "1": int(np.sum(ys == 1))} for split, ys in labels.items()},
            "activity_diagnostics": activity,
        }
        for condition in CONDITIONS:
            result_rows = run_reader(features_by_condition[condition], labels, seed, task_i)
            for row in result_rows:
                row["condition"] = condition
            rows.extend(result_rows)
        del features_by_condition
    integrity["elapsed_seconds"] = round(time.time() - started, 3)
    return rows, integrity


def aggregate_records(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["task"], row["condition"], row["epochs"])].append(row)
    metrics = ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate")
    result = []
    for task, condition, epoch in sorted(groups):
        group = groups[(task, condition, epoch)]
        summary = {"task": task, "condition": condition, "epochs": epoch,
                   "n_seeds": len({r["seed"] for r in group}),
                   "unique_training_examples_per_seed": N_TRAIN,
                   "example_exposures_per_seed": N_TRAIN * epoch}
        for split in ("validation", "test"):
            for metric in metrics:
                mean, low, high = bootstrap_ci(
                    [r[f"{split}_{metric}"] for r in group],
                    stable_seed(task, condition, str(epoch), split, metric),
                )
                summary[f"{split}_{metric}_mean_percent"] = mean
                summary[f"{split}_{metric}_ci95_low_percent"] = low
                summary[f"{split}_{metric}_ci95_high_percent"] = high
        result.append(summary)
    return result


def paired_delta_summary(records: Sequence[dict]) -> list[dict]:
    record_map = {(r["seed"], r["task"], r["condition"], r["epochs"]): r for r in records}
    seeds = sorted({r["seed"] for r in records})
    output = []
    for task in TASKS:
        for epoch in EPOCH_CHECKPOINTS:
            deltas = []
            for seed in seeds:
                control = record_map[(seed, task, "all_excitatory", epoch)]["test_accuracy"]
                treatment = record_map[(seed, task, "dale_inhibition_25", epoch)]["test_accuracy"]
                deltas.append(treatment - control)
            mean, low, high = bootstrap_ci(deltas, stable_seed(task, "paired_inhibition_delta", str(epoch)))
            output.append({
                "task": task, "epochs": epoch,
                "contrast": "dale_inhibition_25 - all_excitatory",
                "n_paired_seeds": len(deltas), "mean_delta_percentage_points": mean,
                "ci95_low_percentage_points": low, "ci95_high_percentage_points": high,
                "positive_seed_deltas": int(np.sum(np.asarray(deltas) > 0)),
                "negative_seed_deltas": int(np.sum(np.asarray(deltas) < 0)),
                "zero_seed_deltas": int(np.sum(np.asarray(deltas) == 0)),
            })
    return output


def threshold_analysis(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["seed"], row["task"], row["condition"])].append(row)
    per_seed = []
    summary = []
    for (seed, task, condition), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["epochs"])
        hit = next((r for r in rows if r["validation_accuracy"] >= TARGET_VALIDATION_ACCURACY), None)
        per_seed.append({
            "seed": seed, "task": task, "condition": condition,
            "reached_85_validation": hit is not None,
            "first_epoch_reaching_85": int(hit["epochs"]) if hit else None,
            "validation_accuracy_at_selection_percent": 100 * hit["validation_accuracy"] if hit else None,
            "independent_test_accuracy_at_selected_epoch_percent": 100 * hit["test_accuracy"] if hit else None,
        })
    for task in TASKS:
        for condition in CONDITIONS:
            subset = [r for r in per_seed if r["task"] == task and r["condition"] == condition]
            hits = [r for r in subset if r["reached_85_validation"]]
            low, high = wilson_ci(len(hits), len(subset))
            test_vals = [r["independent_test_accuracy_at_selected_epoch_percent"] / 100 for r in hits]
            test_mean, test_low, test_high = bootstrap_ci(test_vals, stable_seed(task, condition, "threshold_test"))
            ep = [r["first_epoch_reaching_85"] for r in hits]
            summary.append({
                "task": task, "condition": condition, "n_seeds": len(subset), "n_reached_85": len(hits),
                "reach_rate_percent": 100 * len(hits) / len(subset),
                "reach_rate_ci95_wilson_low_percent": low, "reach_rate_ci95_wilson_high_percent": high,
                "mean_first_epoch_among_reached": float(np.mean(ep)) if ep else None,
                "median_first_epoch_among_reached": float(np.median(ep)) if ep else None,
                "independent_test_accuracy_at_selected_epoch_mean_percent": test_mean if hits else None,
                "independent_test_accuracy_at_selected_epoch_ci95_low_percent": test_low if hits else None,
                "independent_test_accuracy_at_selected_epoch_ci95_high_percent": test_high if hits else None,
            })
    return per_seed, summary


def render_report(records: Sequence[dict], summaries: Sequence[dict], deltas: Sequence[dict],
                  threshold_rows: Sequence[dict], integrity: Sequence[dict], runtime: float, workers: int) -> str:
    lookup = {(r["task"], r["condition"], r["epochs"]): r for r in summaries}
    lines = [
        "# Nucleus Pulse v0.34 — اختبار تنظيم التثبيط داخل الخزان", "",
        "## السؤال المسجل مسبقًا", "",
        "هل تنظيم 25% من الخلايا كمصادر تثبيطية وفق قاعدة Dale يحسن تمثيل التوقيت مقارنة بخزان وصلاته الراجعة موجبة فقط، عندما تُثبت الشبكة والقارئ والبيانات قدر الإمكان؟", "",
        "## البروتوكول", "",
        f"- البذور: **{len(SEEDS)}** ({SEEDS[0]}–{SEEDS[-1]})، والمقارنتان زوجيتان داخل كل بذرة.",
        f"- لكل مهمة وبذرة: {N_TRAIN} مثال تدريب فريد، و{N_VALIDATION} تحقق، و{N_TEST} اختبار نهائي؛ كلها متوازنة ومنفصلة.",
        "- التحكم: كل الوصلات الراجعة الموجودة موجبة. التدخل: 25% من مصادر المشابك لها وصلات خارجة سالبة، و75% موجبة (قاعدة Dale).",
        "- يُحفظ قناع الاتصال وقيم الأوزان المطلقة قبل التطبيع بين الشرطين، ثم يُطبّع كل خزان إلى نصف قطر طيفي 0.72 للتحكم في الكسب الراجع الإجمالي.",
        "- أوزان المدخلات، العتبات، التسريبات، فترة الجموح وعدد الخلايا متطابقة داخل كل بذرة.",
        f"- قارئ SGD لوجستي بميزات 16 نافذة زمنية؛ نقاط القياس بعد {EPOCH_CHECKPOINTS} مرورّات على نفس مجموعة التدريب.",
        f"- المقارنة الأساسية محددة مسبقًا عند {PRIMARY_EPOCH} مرورّات؛ التحقق وحده يُستخدم لتحليل عتبة 85%.",
        f"- التنفيذ: {workers} عمليات متوازية؛ الزمن الإجمالي {runtime:.1f} ثانية.",
        "", "## دقة الاختبار المستقل عند 10 مرورّات", "",
        "| المهمة | الشرط | الدقة [فاصل ثقة 95% عبر البذور] |", "|---|---|---:|",
    ]
    names = {"all_excitatory": "خزان استثاري فقط", "dale_inhibition_25": "25% خلايا تثبيطية (Dale)"}
    for task in TASKS:
        for condition in CONDITIONS:
            row = lookup[(task, condition, PRIMARY_EPOCH)]
            cell = f"{row['test_accuracy_mean_percent']:.2f}% [{row['test_accuracy_ci95_low_percent']:.2f}, {row['test_accuracy_ci95_high_percent']:.2f}]"
            lines.append(f"| {task} | {names[condition]} | {cell} |")
    lines += ["", "## الفرق الزوجي الأساسي: تثبيط منظم ناقص خزان استثاري فقط", "",
              "الفرق محسوب داخل كل بذرة على نفس مجموعة الاختبار، ثم يُلخّص على 40 بذرة. الوحدات نقاط مئوية.",
              "", "| المهمة | المرورّات | الفرق [فاصل ثقة 95%] | بذور: موجب / سالب / صفر |", "|---|---:|---:|---:|"]
    for d in deltas:
        lines.append(f"| {d['task']} | {d['epochs']} | {d['mean_delta_percentage_points']:+.2f} [{d['ci95_low_percentage_points']:+.2f}, {d['ci95_high_percentage_points']:+.2f}] | {d['positive_seed_deltas']} / {d['negative_seed_deltas']} / {d['zero_seed_deltas']} |")
    lines += ["", "## الوصول إلى 85% على التحقق", "",
              "يُعرض أول وصول في التحقق فقط. لا تُخفى البذور التي لم تصل، ولا يُستخدم الاختبار لاختيار نقطة المرور.",
              "", "| المهمة | الشرط | وصلت | نسبة الوصول [Wilson 95%] | الاختبار عند أول وصول [CI 95%] |", "|---|---|---:|---:|---:|"]
    for row in threshold_rows:
        if row["n_reached_85"]:
            test_cell = f"{row['independent_test_accuracy_at_selected_epoch_mean_percent']:.2f}% [{row['independent_test_accuracy_at_selected_epoch_ci95_low_percent']:.2f}, {row['independent_test_accuracy_at_selected_epoch_ci95_high_percent']:.2f}]"
        else:
            test_cell = "لم تصل أي بذرة"
        lines.append(f"| {row['task']} | {names[row['condition']]} | {row['n_reached_85']}/{row['n_seeds']} | {row['reach_rate_percent']:.1f}% [{row['reach_rate_ci95_wilson_low_percent']:.1f}, {row['reach_rate_ci95_wilson_high_percent']:.1f}] | {test_cell} |")
    overlap = sum(t["train_validation_overlap"] + t["train_test_overlap"] + t["validation_test_overlap"]
                  for item in integrity for t in item["tasks"].values())
    expected = len(SEEDS) * len(TASKS) * len(CONDITIONS) * len(EPOCH_CHECKPOINTS)
    lines += ["", "## تشخيص نشاط الخزان", "",
              "متوسط النشاط لكل شرط موجود في `results.json` مع تفاصيل كل بذرة. عدد النبضات ليس مقياسًا لجودة التعلّم وحده، لكنه يساعد على معرفة هل أدى التثبيط إلى شبكة صامتة أو مفرطة النشاط.",
              "", "## سلامة التجربة", "",
              f"- السجلات: **{len(records)} / {expected}**؛ البذور: **{len(integrity)} / {len(SEEDS)}**.",
              f"- تقاطعات تسلسلات التدريب/التحقق/الاختبار: **{overlap}**.",
              "- مصفوفة الاتصال نفسها مستعملة في الشرطين؛ قيم الأوزان المطلقة قبل التطبيع مشتركة، ويُستخدم نصف قطر طيفي مستهدف واحد.",
              "- فواصل الثقة للمتوسطات والفروق مبنية على إعادة سحب البذور، ومعدل بلوغ العتبة يستخدم Wilson.",
              "", "## القيود والتفسير", "",
              "- المقارنة تختبر شبكة صغيرة ذات وصلات راجعة ثابتة، ولا تختبر لدونة الخزان أو STDP.",
              "- الخزان الاستثاري فقط مرجع مضبوط للمقارنة وليس خط أساس عامًا لكل شبكات النبضات.",
              "- تطبيع كل شرط إلى نصف قطر طيفي واحد يعني أن عامل التطبيع الكلي قد يختلف قليلًا بسبب البنية الإشارية؛ هذه المعلومة موثقة لكل بذرة.",
              "- حتى لو ظهر فرق، فهو خاص بهاتين المهمتين الاصطناعيتين وهذا التكوين، ولا يثبت قدرة على التفكير الهندسي العام.",
              ""]
    return "\n".join(lines)


def plot_curves(summaries: Sequence[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"all_excitatory": "All-excitatory recurrence", "dale_inhibition_25": "25% inhibitory sources (Dale)"}
    for task in TASKS:
        for split in ("validation", "test"):
            fig, ax = plt.subplots(figsize=(8.4, 5.4))
            for condition in CONDITIONS:
                rows = sorted((r for r in summaries if r["task"] == task and r["condition"] == condition), key=lambda r: r["epochs"])
                xs = [r["epochs"] for r in rows]
                ys = [r[f"{split}_accuracy_mean_percent"] for r in rows]
                lo = [r[f"{split}_accuracy_ci95_low_percent"] for r in rows]
                hi = [r[f"{split}_accuracy_ci95_high_percent"] for r in rows]
                ax.plot(xs, ys, marker="o", label=names[condition])
                ax.fill_between(xs, lo, hi, alpha=0.15)
            ax.axhline(TARGET_VALIDATION_ACCURACY * 100, linestyle="--", linewidth=1, label="85% validation target")
            ax.axhline(50, linestyle=":", linewidth=1, label="chance")
            ax.set_xticks(EPOCH_CHECKPOINTS)
            ax.set_xlabel("Passes over the same 6,000 unique examples")
            ax.set_ylabel(f"{split.capitalize()} accuracy (%)")
            ax.set_title(f"Nucleus Pulse v0.34 — {task} — {split}")
            ax.set_ylim(35, 101)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(outdir / f"learning_curve_{task}_{split}.png", dpi=160)
            plt.close(fig)


def run_full(outdir: Path, resume: bool, workers: int) -> dict:
    verify_preregistration()
    outdir.mkdir(parents=True, exist_ok=True)
    partial = outdir / "partial_seed_results.json"
    done: dict[int, dict] = {}
    if resume and partial.exists():
        loaded = json.loads(partial.read_text(encoding="utf-8"))
        for item in loaded:
            done[int(item["integrity"]["seed"])] = item
    missing = [s for s in SEEDS if s not in done]
    started = time.time()
    print(f"Nucleus Pulse v{VERSION}: {len(done)} seeds already saved; {len(missing)} remaining; workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in missing}
        for future in as_completed(futures):
            seed = futures[future]
            seed_rows, seed_integrity = future.result()
            done[seed] = {"records": seed_rows, "integrity": seed_integrity}
            atomic_json(partial, [done[k] for k in sorted(done)])
            print(f"Completed seed {seed}: {len(done)}/{len(SEEDS)}", flush=True)
    items = [done[k] for k in sorted(done)]
    records = [r for item in items for r in item["records"]]
    integrity = [item["integrity"] for item in items]
    expected = len(SEEDS) * len(TASKS) * len(CONDITIONS) * len(EPOCH_CHECKPOINTS)
    if len(records) != expected or len(integrity) != len(SEEDS):
        raise RuntimeError(f"Incomplete run: records={len(records)}/{expected}, seeds={len(integrity)}/{len(SEEDS)}")
    summaries = aggregate_records(records)
    deltas = paired_delta_summary(records)
    threshold_per_seed, threshold_rows = threshold_analysis(records)
    diagnostics = {
        condition: {
            "mean_spikes_per_trial": float(np.mean([
                item["integrity"]["tasks"][task]["activity_diagnostics"][condition]["mean_spikes_per_trial"]
                for item in items for task in TASKS]))
        } for condition in CONDITIONS
    }
    result = {
        "version": VERSION, "protocol": protocol_config(), "seeds": SEEDS,
        "conditions": CONDITIONS, "n_hidden": N_HIDDEN, "n_train": N_TRAIN,
        "n_validation": N_VALIDATION, "n_test": N_TEST, "feature_set": FEATURE_SET,
        "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
        "records": records, "summary": summaries, "paired_differences": deltas,
        "threshold_per_seed": threshold_per_seed, "threshold_summary": threshold_rows,
        "integrity": integrity, "activity_diagnostic_means": diagnostics,
        "runtime_seconds": time.time() - started, "workers": workers,
    }
    atomic_json(outdir / "results.json", result)
    write_csv(outdir / "per_seed_inhibition.csv", records)
    write_csv(outdir / "inhibition_summary.csv", summaries)
    write_csv(outdir / "paired_inhibition_differences.csv", deltas)
    write_csv(outdir / "per_seed_threshold.csv", threshold_per_seed)
    write_csv(outdir / "threshold_summary.csv", threshold_rows)
    (outdir / "report_ar.md").write_text(render_report(records, summaries, deltas, threshold_rows, integrity,
                                                       result["runtime_seconds"], workers), encoding="utf-8")
    plot_curves(summaries, outdir)
    partial.unlink(missing_ok=True)
    print("FINAL RESULTS SAVED", flush=True)
    for row in deltas:
        if row["epochs"] == PRIMARY_EPOCH:
            print(f"PRIMARY {row['task']}: delta={row['mean_delta_percentage_points']:+.2f} pp "
                  f"CI95=[{row['ci95_low_percentage_points']:+.2f}, {row['ci95_high_percentage_points']:+.2f}], "
                  f"positive/negative/zero={row['positive_seed_deltas']}/{row['negative_seed_deltas']}/{row['zero_seed_deltas']}", flush=True)
    return result


def run_smoke() -> None:
    seed = 6990  # outside preregistered range
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, diag = make_recurrent_weight_matrices(reservoir, seed)
    mask = reservoir.w_rec != 0
    assert np.array_equal(matrices["all_excitatory"] != 0, mask)
    assert np.array_equal(matrices["dale_inhibition_25"] != 0, mask)
    assert np.all(matrices["all_excitatory"][mask] > 0)
    assert np.allclose(spectral_radius(matrices["all_excitatory"]), TARGET_SPECTRAL_RADIUS, atol=5e-4)
    assert np.allclose(spectral_radius(matrices["dale_inhibition_25"]), TARGET_SPECTRAL_RADIUS, atol=5e-4)
    inhibited = diag["inhibitory_source_indices"]
    for source in range(N_HIDDEN):
        values = matrices["dale_inhibition_25"][:, source]
        if source in inhibited:
            assert np.all(values[values != 0] < 0)
        else:
            assert np.all(values[values != 0] > 0)
    trials = make_nested_training_pool(seed + 11, "sync", 200)
    inputs, spikes = simulate_batch_with_recurrent_weights(reservoir, trials, matrices["all_excitatory"])
    assert inputs.shape == (200, 32, 4) and spikes.shape == (200, 32, N_HIDDEN)
    assert np.isfinite(spikes).all()
    assert extract_features(spikes).shape == (200, 384)
    assert np.isfinite(extract_features(spikes)).all()
    assert len({trial_key(t) for t in trials}) == 200
    print("SMOKE PASSED: polarity, topology, spectral-radius control, simulation and features", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="engineering-only smoke test")
    parser.add_argument("--resume", action="store_true", help="resume preregistered run from saved seed results")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full(ROOT / "results" / "inhibition_structure_v0_34", resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
