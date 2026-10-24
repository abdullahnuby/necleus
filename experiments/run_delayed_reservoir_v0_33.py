"""Nucleus Pulse v0.33: paired test of heterogeneous recurrent synaptic delays.

Only recurrent edge delays change: the same per-seed LIF reservoir weights,
input weights, thresholds, leak, spike generation and 16-window SGD readout are
used in both conditions. Baseline recurrence is delay=1 on every existing
edge; treatment assigns each existing recurrent edge a fixed delay uniformly
from 1..4 ticks. Training, validation and final-test trials are paired across
conditions. Primary comparison is test performance at the preregistered 10th
pass; validation alone is used for the secondary 85% threshold analysis.
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
    simulate_batch_vectorized,
    trial_key,
)
from experiments.run_incremental_windows_v0_31 import (  # noqa: E402
    classification_metrics,
    make_eval_dataset,
    new_online_reader,
)

VERSION = "0.33"
SEEDS = list(range(6000, 6040))
CONDITIONS = ["baseline_delay1", "heterogeneous_delay1_4"]
EPOCH_CHECKPOINTS = [1, 3, 5, 10]
PRIMARY_EPOCH = 10
N_TRAIN = 6000
N_VALIDATION = 1000
N_TEST = 1000
FEATURE_SET = "spike_bins16_rate_384"
N_WINDOWS = 16
DELAY_MIN = 1
DELAY_MAX = 4
TARGET_VALIDATION_ACCURACY = 0.85
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20331024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_config() -> dict:
    return {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "unique_training_examples_per_task_seed": N_TRAIN,
        "validation_per_task_seed": N_VALIDATION,
        "final_test_per_task_seed": N_TEST,
        "training_pass_checkpoints": EPOCH_CHECKPOINTS,
        "primary_epoch": PRIMARY_EPOCH,
        "hidden_neurons": N_HIDDEN,
        "feature_set": FEATURE_SET,
        "feature_definition": "16 equal temporal bins, 2 ticks per bin, hidden spike counts divided by 2",
        "baseline_recurrent_delays": {"existing_edges": 1, "missing_edges": 0},
        "heterogeneous_recurrent_delays": {
            "distribution": "uniform discrete integer",
            "minimum_tick": DELAY_MIN,
            "maximum_tick": DELAY_MAX,
            "assigned_only_to_existing_edges": True,
            "delay_matrix_seed": "trial_seed_plus_930331",
        },
        "reservoir_weights_tau_threshold_and_input_weights_identical_between_conditions": True,
        "readout": {
            "class": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss", "penalty": "l2", "alpha": 1e-4,
            "learning_rate": "constant", "eta0": 0.01,
            "average": True, "shuffle": True, "fit_method": "partial_fit",
            "same_model_random_seed_paired_between_conditions": True,
        },
        "validation_used_for_threshold_analysis_only": True,
        "final_test_used_for_tuning_or_selection": False,
        "primary_comparison_is_fixed_epoch": PRIMARY_EPOCH,
        "no_inhibition_or_neuron_count_change": True,
        "bootstrap_unit": "paired seed",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def verify_preregistration() -> dict:
    path = ROOT / "results" / "delayed_reservoir_v0_33" / "preregistration.json"
    if not path.exists():
        raise RuntimeError("Missing frozen v0.33 preregistration; refusing full run.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Current run configuration differs from preregistration.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return doc


def atomic_json(path: Path, payload: dict) -> None:
    def safe(value):
        if isinstance(value, dict):
            return {k: safe(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [safe(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return None
        if isinstance(value, np.generic):
            return value.item()
        return value
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stable_seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode("utf-8")).digest()[:4], "little")


def bootstrap_ci(values: Sequence[float], seed: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(arr.mean() * 100.0)
    if arr.size == 1:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPS, 1000):
        n = min(1000, BOOTSTRAP_REPS - start)
        indices = rng.integers(0, arr.size, size=(n, arr.size))
        draws[start:start+n] = arr[indices].mean(axis=1) * 100.0
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n))) / denom
    return float(max(0.0, center - half) * 100), float(min(1.0, center + half) * 100)


def make_recurrent_delay_matrix(reservoir: DynamicPulseReservoir, condition: str, seed: int) -> np.ndarray:
    """Assign discrete delays to the already-frozen recurrent edges only.

    Zero is a sentinel for non-edges. This function never mutates reservoir
    weights or any other dynamic parameter.
    """
    edge_mask = np.asarray(reservoir.w_rec != 0.0)
    delays = np.zeros(reservoir.w_rec.shape, dtype=np.int8)
    if condition == "baseline_delay1":
        delays[edge_mask] = 1
    elif condition == "heterogeneous_delay1_4":
        rng = np.random.default_rng(int(seed) + 930331)
        delays[edge_mask] = rng.integers(DELAY_MIN, DELAY_MAX + 1, size=int(edge_mask.sum()), dtype=np.int8)
    else:
        raise ValueError(f"Unknown reservoir condition: {condition}")
    if np.any(delays[edge_mask] < 1) or np.any(delays[edge_mask] > DELAY_MAX):
        raise RuntimeError("Existing recurrent edges must receive valid positive delays")
    if np.any(delays[~edge_mask] != 0):
        raise RuntimeError("Non-edges must have zero delay sentinel")
    return delays


def delay_counts(delay_matrix: np.ndarray) -> dict[str, int]:
    return {str(delay): int(np.sum(delay_matrix == delay)) for delay in range(1, DELAY_MAX + 1)}


def simulate_batch_with_recurrent_delays(reservoir: DynamicPulseReservoir,
                                         trials: Sequence,
                                         rec_delays: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized LIF simulation with per-edge recurrent delay.

    Input paths and all cell parameters remain identical to the existing
    simulator. For delay d, only recurrent edges tagged d transmit the spike
    vector from t-d. The delay-1 configuration must reproduce the original
    simulator exactly up to floating point tolerance.
    """
    if rec_delays.shape != reservoir.w_rec.shape:
        raise ValueError("rec_delays must have the same shape as w_rec")
    if np.any((rec_delays != 0) & ((rec_delays < 1) | (rec_delays > DELAY_MAX))):
        raise ValueError("Recurrent delays must be zero or in the configured positive delay range")
    edge_mask = reservoir.w_rec != 0.0
    if np.any((rec_delays > 0) != edge_mask):
        raise ValueError("Every existing recurrent edge needs a delay; non-edges must not have one")
    if not trials:
        return (np.empty((0, reservoir.duration, 4), np.float32),
                np.empty((0, reservoir.duration, reservoir.n_hidden), np.float32))

    inputs = rasterize_batch(trials, reservoir.duration)
    count, duration, _ = inputs.shape
    hidden = reservoir.n_hidden
    voltage = np.zeros((count, hidden), dtype=np.float32)
    i_syn = np.zeros((count, hidden), dtype=np.float32)
    refractory = np.zeros((count, hidden), dtype=np.int8)
    spikes = np.zeros((count, duration, hidden), dtype=np.float32)
    w_in_t = reservoir.w_in.T
    recurrent_parts = {
        delay: (reservoir.w_rec * (rec_delays == delay)).T.astype(np.float32, copy=False)
        for delay in range(DELAY_MIN, DELAY_MAX + 1)
    }
    for t in range(duration):
        rec_current = np.zeros((count, hidden), dtype=np.float32)
        for delay, weight_t in recurrent_parts.items():
            source_t = t - delay
            if source_t >= 0:
                rec_current += spikes[:, source_t, :] @ weight_t
        i_syn = (reservoir.syn_leak[None, :] * i_syn
                 + inputs[:, t, :] @ w_in_t
                 + rec_current)
        voltage = reservoir.reset_leak[None, :] * voltage + (1.0 - reservoir.reset_leak[None, :]) * i_syn
        active = (voltage >= reservoir.threshold[None, :]) & (refractory == 0)
        spikes[:, t, :] = active.astype(np.float32)
        voltage[active] = 0.0
        refractory = np.maximum(refractory - 1, 0)
        refractory[active] = 1
    return inputs, spikes


def extract_16window_rate_features(spikes: np.ndarray) -> np.ndarray:
    """Same 16-bin rate view used in v0.31/v0.32, with no fitted transform."""
    if spikes.ndim != 3 or spikes.shape[1:] != (32, N_HIDDEN):
        raise ValueError(f"Expected spike tensor (N,32,{N_HIDDEN}), got {spikes.shape}")
    n = spikes.shape[0]
    features = spikes.reshape(n, N_WINDOWS, 2, N_HIDDEN).sum(axis=2) / 2.0
    features = features.reshape(n, N_WINDOWS * N_HIDDEN).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise RuntimeError("Temporal feature matrix contains non-finite values")
    return features


def run_condition(features: dict[str, np.ndarray], labels: dict[str, np.ndarray],
                  seed: int, task_i: int) -> list[dict]:
    from sklearn.linear_model import SGDClassifier
    X_train = features["train"]
    X_val = features["validation"]
    X_test = features["test"]
    y_train, y_val, y_test = labels["train"], labels["validation"], labels["test"]
    model_seed = int(seed + 840001 + task_i * 1000)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          learning_rate="constant", eta0=0.01, average=True,
                          shuffle=True, random_state=model_seed)
    rows = []
    cumulative_seconds = 0.0
    for epoch in range(1, PRIMARY_EPOCH + 1):
        started = time.time()
        if epoch == 1:
            model.partial_fit(X_train, y_train, classes=np.asarray([0, 1], dtype=np.int64))
        else:
            model.partial_fit(X_train, y_train)
        pass_seconds = time.time() - started
        cumulative_seconds += pass_seconds
        if epoch not in EPOCH_CHECKPOINTS:
            continue
        validation_metrics = classification_metrics(y_val, model.predict(X_val))
        test_metrics = classification_metrics(y_test, model.predict(X_test))
        rows.append({
            "seed": int(seed), "task": ("sync", "order")[task_i],
            "condition": "", "epochs": int(epoch),
            "n_train_unique": N_TRAIN, "cumulative_example_exposures": N_TRAIN * epoch,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
            "epoch_fit_seconds": round(pass_seconds, 6),
            "cumulative_fit_seconds": round(cumulative_seconds, 6),
        })
    return rows


def run_seed(seed: int) -> tuple[list[dict], dict]:
    started = time.time()
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    delay_matrices = {condition: make_recurrent_delay_matrix(reservoir, condition, seed)
                      for condition in CONDITIONS}
    rows: list[dict] = []
    integrity: dict = {
        "seed": int(seed), "reservoir_seed": int(seed),
        "same_base_reservoir_parameters_between_conditions": True,
        "recurrent_edge_count": int(np.sum(reservoir.w_rec != 0.0)),
        "delay_counts": {condition: delay_counts(mat) for condition, mat in delay_matrices.items()},
        "tasks": {},
    }

    for task_i, task in enumerate(("sync", "order")):
        train_pool = make_nested_training_pool(seed + 11111 + task_i * 100, task, N_TRAIN)
        train_keys = {trial_key(trial) for trial in train_pool}
        val_trials = make_eval_dataset(seed + 52222 + task_i * 100, task, N_VALIDATION, set(train_keys))
        val_keys = {trial_key(trial) for trial in val_trials}
        test_trials = make_eval_dataset(seed + 63333 + task_i * 100, task, N_TEST, set(train_keys) | val_keys)
        test_keys = {trial_key(trial) for trial in test_trials}
        if len(train_keys) != N_TRAIN or len(val_keys) != N_VALIDATION or len(test_keys) != N_TEST:
            raise RuntimeError(f"Non-unique sequences at seed={seed}, task={task}")
        if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
            raise RuntimeError(f"Train/validation/test overlap at seed={seed}, task={task}")

        trial_sets = {"train": train_pool, "validation": val_trials, "test": test_trials}
        labels = {split: make_epoch_labels(trials) for split, trials in trial_sets.items()}
        for split, ys in labels.items():
            if np.sum(ys == 0) != len(ys) // 2 or np.sum(ys == 1) != len(ys) // 2:
                raise RuntimeError(f"Class imbalance at seed={seed}, task={task}, split={split}")

        concatenated = train_pool + val_trials + test_trials
        split_indices = {"train": (0, N_TRAIN),
                         "validation": (N_TRAIN, N_TRAIN + N_VALIDATION),
                         "test": (N_TRAIN + N_VALIDATION, N_TRAIN + N_VALIDATION + N_TEST)}
        features_by_condition: dict[str, dict[str, np.ndarray]] = {}
        spike_diagnostics: dict[str, dict] = {}
        base_inputs, base_spikes = simulate_batch_vectorized(reservoir, concatenated)
        for condition in CONDITIONS:
            if condition == "baseline_delay1":
                xs, spikes = base_inputs, base_spikes
            else:
                xs, spikes = simulate_batch_with_recurrent_delays(reservoir, concatenated,
                                                                  delay_matrices[condition])
                if not np.array_equal(xs, base_inputs):
                    raise RuntimeError("Input raster changed between delay conditions")
            matrices = {}
            for split, (left, right) in split_indices.items():
                matrices[split] = extract_16window_rate_features(spikes[left:right])
            features_by_condition[condition] = matrices
            active = np.sum(spikes, axis=(0, 1))
            spike_diagnostics[condition] = {
                "mean_spikes_per_trial": float(np.mean(np.sum(spikes, axis=(1, 2)))),
                "fraction_hidden_cells_with_at_least_one_spike": float(np.mean(active > 0)),
                "fraction_trials_with_zero_hidden_spikes": float(np.mean(np.sum(spikes, axis=(1, 2)) == 0)),
                "mean_spike_rate_per_hidden_tick": float(np.mean(spikes)),
            }
            if condition == "heterogeneous_delay1_4":
                del xs, spikes
        # Both conditions have been made; don't let arrays leak across tasks.
        del base_inputs, base_spikes

        integrity["tasks"][task] = {
            "n_train_unique": len(train_keys), "n_validation_unique": len(val_keys), "n_test_unique": len(test_keys),
            "train_validation_overlap": len(train_keys & val_keys),
            "train_test_overlap": len(train_keys & test_keys),
            "validation_test_overlap": len(val_keys & test_keys),
            "class_balance": {split: {"0": int(np.sum(ys == 0)), "1": int(np.sum(ys == 1))}
                              for split, ys in labels.items()},
            "spike_diagnostics": spike_diagnostics,
        }

        for condition in CONDITIONS:
            result_rows = run_condition(features_by_condition[condition], labels, seed, task_i)
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
    result: list[dict] = []
    for task, condition, epoch in sorted(groups):
        group = groups[(task, condition, epoch)]
        summary = {"task": task, "condition": condition, "epochs": epoch,
                   "n_seeds": len({row["seed"] for row in group}),
                   "unique_training_examples_per_seed": N_TRAIN,
                   "example_exposures_per_seed": N_TRAIN * epoch}
        for split in ("validation", "test"):
            for metric in metrics:
                mean, low, high = bootstrap_ci(
                    [row[f"{split}_{metric}"] for row in group],
                    stable_seed(task, condition, str(epoch), split, metric),
                )
                summary[f"{split}_{metric}_mean_percent"] = mean
                summary[f"{split}_{metric}_ci95_low_percent"] = low
                summary[f"{split}_{metric}_ci95_high_percent"] = high
        result.append(summary)
    return result


def paired_delta_summary(records: Sequence[dict]) -> list[dict]:
    """Heterogeneous minus baseline paired by seed, primary endpoint fixed at epoch 10."""
    record_map = {(r["seed"], r["task"], r["condition"], r["epochs"]): r for r in records}
    output = []
    seed_ids = sorted({row["seed"] for row in records})
    tasks_present = sorted({row["task"] for row in records}, key=("sync", "order").index)
    epochs_present = sorted({row["epochs"] for row in records})
    for task in tasks_present:
        for epoch in epochs_present:
            deltas = []
            for seed in seed_ids:
                base = record_map[(seed, task, "baseline_delay1", epoch)]["test_accuracy"]
                hetero = record_map[(seed, task, "heterogeneous_delay1_4", epoch)]["test_accuracy"]
                deltas.append(hetero - base)
            mean, low, high = bootstrap_ci(deltas, stable_seed(task, "paired_delta", str(epoch)))
            output.append({"task": task, "epochs": epoch, "contrast": "heterogeneous_delay1_4 - baseline_delay1",
                           "n_paired_seeds": len(deltas), "mean_delta_percentage_points": mean,
                           "ci95_low_percentage_points": low, "ci95_high_percentage_points": high,
                           "positive_seed_deltas": int(np.sum(np.asarray(deltas) > 0)),
                           "negative_seed_deltas": int(np.sum(np.asarray(deltas) < 0)),
                           "zero_seed_deltas": int(np.sum(np.asarray(deltas) == 0))})
    return output


def threshold_analysis(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["seed"], row["task"], row["condition"])].append(row)
    per_seed = []
    summary = []
    for (seed, task, condition), rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda r: r["epochs"])
        hit = next((r for r in rows if r["validation_accuracy"] >= TARGET_VALIDATION_ACCURACY), None)
        per_seed.append({
            "seed": seed, "task": task, "condition": condition,
            "reached_85_validation": hit is not None,
            "first_epoch_reaching_85": int(hit["epochs"]) if hit else None,
            "validation_accuracy_at_selection_percent": 100 * hit["validation_accuracy"] if hit else None,
            "independent_test_accuracy_at_selected_epoch_percent": 100 * hit["test_accuracy"] if hit else None,
        })
    for task in ("sync", "order"):
        for condition in CONDITIONS:
            subset = [r for r in per_seed if r["task"] == task and r["condition"] == condition]
            hits = [r for r in subset if r["reached_85_validation"]]
            low, high = wilson_ci(len(hits), len(subset))
            test_vals = [r["independent_test_accuracy_at_selected_epoch_percent"] / 100.0 for r in hits]
            test_mean, test_low, test_high = bootstrap_ci(test_vals, stable_seed(task, condition, "test_at_threshold"))
            epochs = [r["first_epoch_reaching_85"] for r in hits]
            summary.append({
                "task": task, "condition": condition, "n_seeds": len(subset), "n_reached_85": len(hits),
                "reach_rate_percent": 100 * len(hits) / len(subset),
                "reach_rate_ci95_wilson_low_percent": low, "reach_rate_ci95_wilson_high_percent": high,
                "mean_first_epoch_among_reached": float(np.mean(epochs)) if epochs else None,
                "median_first_epoch_among_reached": float(np.median(epochs)) if epochs else None,
                "independent_test_accuracy_at_selected_epoch_mean_percent": test_mean if hits else None,
                "independent_test_accuracy_at_selected_epoch_ci95_low_percent": test_low if hits else None,
                "independent_test_accuracy_at_selected_epoch_ci95_high_percent": test_high if hits else None,
            })
    return per_seed, summary


def render_report(records: Sequence[dict], summaries: Sequence[dict], deltas: Sequence[dict],
                  threshold_rows: Sequence[dict], integrity: Sequence[dict], runtime: float, workers: int) -> str:
    lookup = {(r["task"], r["condition"], r["epochs"]): r for r in summaries}
    lines = [
        "# Nucleus Pulse v0.33 — هل التأخيرات المشبكية المتنوعة تساعد؟", "",
        "## السؤال المسجل مسبقًا", "",
        "هل توزيع تأخير المشابك الراجعة على 1–4 خطوات يحسن التعلّم مقارنة بتأخير خطوة واحدة، عندما تظل بقية ديناميكا الخزان والقارئ والبيانات ثابتة؟", "",
        "## البروتوكول", "",
        f"- بذور جديدة: **{len(SEEDS)}** ({SEEDS[0]}–{SEEDS[-1]}). كل بذرة تنتج المقارنتين على نفس الأمثلة.",
        f"- لكل مهمة وبذرة: {N_TRAIN} تدريب فريد، {N_VALIDATION} تحقق، و{N_TEST} اختبار نهائي؛ المجموعات منفصلة ومتوازنة.",
        "- المتغير الوحيد: تأخير كل مشبك راجع موجود. التحكم = تأخير 1؛ التجربة = تأخيرات صحيحة موزعة بانتظام من 1 إلى 4.",
        "- الأوزان الراجعة، أوزان المدخلات، العتبات، تسريبات الغشاء/المشبك وعدد الخلايا متطابقة بين الشرطين داخل كل بذرة.",
        f"- قارئ SGD لوجستي، ميزات 16 نافذة زمنية، ونقاط قياس بعد {EPOCH_CHECKPOINTS} مرورّات على نفس مجموعة التدريب.",
        f"- المقارنة الأساسية محددة مسبقًا عند {PRIMARY_EPOCH} مرورّات. الاختبار النهائي لا يختار إعدادات ولا نقطة مرور.",
        f"- التنفيذ: {workers} عمليات متوازية؛ الزمن الإجمالي {runtime:.1f} ثانية.",
        "", "## دقة الاختبار المستقل عند 10 مرورّات", "",
        "| المهمة | الخزان | الدقة [فاصل ثقة 95% عبر البذور] |", "|---|---|---:|",
    ]
    for task in ("sync", "order"):
        for condition in CONDITIONS:
            row = lookup[(task, condition, PRIMARY_EPOCH)]
            cell = f"{row['test_accuracy_mean_percent']:.2f}% [{row['test_accuracy_ci95_low_percent']:.2f}, {row['test_accuracy_ci95_high_percent']:.2f}]"
            lines.append(f"| {task} | {condition} | {cell} |")
    lines += ["", "## الفرق الزوجي الأساسي: تأخير متنوع ناقص تأخير خطوة واحدة", "",
              "الفرق محسوب داخل كل بذرة على اختبارها نفسه، ثم يُلخّص عبر 40 بذرة. الوحدات نقاط مئوية.",
              "", "| المهمة | المرورّات | الفرق المتوسط [فاصل ثقة 95%] | البذور: موجب / سالب / صفر |", "|---|---:|---:|---:|"]
    for d in deltas:
        lines.append(f"| {d['task']} | {d['epochs']} | {d['mean_delta_percentage_points']:+.2f} [{d['ci95_low_percentage_points']:+.2f}, {d['ci95_high_percentage_points']:+.2f}] | {d['positive_seed_deltas']} / {d['negative_seed_deltas']} / {d['zero_seed_deltas']} |")
    lines += ["", "## بلوغ 85% على التحقق", "", "يُبلّغ عن عدد البذور التي بلغت العتبة في أي نقطة تحقق. تُعرض دقة الاختبار المنفصل عند أول نقطة تحقق مؤهلة فقط، ولا تُخفى البذور التي لم تصل.",
              "", "| المهمة | الخزان | وصلت | نسبة الوصول [Wilson 95%] | الاختبار عند أول وصول [CI 95%] |", "|---|---|---:|---:|---:|"]
    for row in threshold_rows:
        if row["n_reached_85"]:
            test_cell = f"{row['independent_test_accuracy_at_selected_epoch_mean_percent']:.2f}% [{row['independent_test_accuracy_at_selected_epoch_ci95_low_percent']:.2f}, {row['independent_test_accuracy_at_selected_epoch_ci95_high_percent']:.2f}]"
        else:
            test_cell = "لا توجد بذور وصلت"
        lines.append(f"| {row['task']} | {row['condition']} | {row['n_reached_85']}/{row['n_seeds']} | {row['reach_rate_percent']:.1f}% [{row['reach_rate_ci95_wilson_low_percent']:.1f}, {row['reach_rate_ci95_wilson_high_percent']:.1f}] | {test_cell} |")
    overlap_total = sum(t["train_validation_overlap"] + t["train_test_overlap"] + t["validation_test_overlap"]
                        for item in integrity for t in item["tasks"].values())
    expected = len(SEEDS) * 2 * len(CONDITIONS) * len(EPOCH_CHECKPOINTS)
    lines += ["", "## سلامة التجربة", "",
              f"- السجلات: **{len(records)} / {expected}**؛ البذور: **{len(integrity)} / {len(SEEDS)}**.",
              f"- تداخل تسلسلات التدريب/التحقق/الاختبار: **{overlap_total}**.",
              "- كل شرط يستخدم نفس أمثلة التدريب والتحقق والاختبار لكل بذرة؛ المقارنة زوجية.",
              "- فواصل الثقة bootstrap مبنية على البذور كوحدة مستقلة، وفاصل نسبة بلوغ العتبة هو Wilson.",
              "- التأخير غير المتجانس لا يغيّر أوزان المشابك ولا يضيف تثبيطًا أو خلايا جديدة.",
              "", "## التفسير والقيود", "",
              "- هذا اختبار واحد لتوزيع تأخير ثابت عشوائي 1–4 داخل خزان صغير، وليس بحثًا عن التأخيرات المثلى.",
              "- إذا تحسن الأداء، فهذا يدعم فائدة التوقيت المتنوع في هذا السيناريو فقط؛ ولا يثبت تفوقًا عامًا للنموذج النبضي.",
              "- القارئ هو SGD لوجستي متعلم بالتدرج؛ أوزان الخزان تظل ثابتة، ولذلك لا تختبر هذه التجربة STDP أو لدونة الخزان.",
              "- مهمتا sync وorder اصطناعيتان وتستندان إلى تسلسلات مولدة؛ لا تُعمم النتائج مباشرة إلى تطبيق هندسي واقعي.",
              "", "## الملفات", "",
              "- `per_seed_delays.csv`: كل نتيجة لكل بذرة ومهمة وشرط ونقطة مرور.",
              "- `paired_delay_differences.csv`: فروق الاختبار الزوجية.",
              "- `delay_summary.csv`: المتوسطات وفواصل الثقة.",
              "- `threshold_summary.csv` و`per_seed_threshold.csv`: تحليل عتبة التحقق.",
              "- `results.json`: النتائج وفحوص السلامة وإحصاءات التأخيرات.", ""]
    return "\n".join(lines)


def plot_curves(summaries: Sequence[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"baseline_delay1": "all recurrent delays = 1", "heterogeneous_delay1_4": "heterogeneous delays 1–4"}
    for task in ("sync", "order"):
        for split in ("validation", "test"):
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            for condition in CONDITIONS:
                rows = sorted([r for r in summaries if r["task"] == task and r["condition"] == condition],
                              key=lambda r: r["epochs"])
                xs = [r["epochs"] for r in rows]
                ys = [r[f"{split}_accuracy_mean_percent"] for r in rows]
                lo = [r[f"{split}_accuracy_ci95_low_percent"] for r in rows]
                hi = [r[f"{split}_accuracy_ci95_high_percent"] for r in rows]
                ax.plot(xs, ys, marker="o", label=labels[condition])
                ax.fill_between(xs, lo, hi, alpha=0.15)
            ax.axhline(TARGET_VALIDATION_ACCURACY * 100, linestyle="--", linewidth=1, label="85% validation target")
            ax.axhline(50.0, linestyle=":", linewidth=1, label="chance")
            ax.set_xticks(EPOCH_CHECKPOINTS)
            ax.set_xlabel("Passes over the same 6,000 unique examples")
            ax.set_ylabel(f"{split.capitalize()} accuracy (%)")
            ax.set_title(f"Nucleus Pulse v0.33 — {task} — {split}")
            ax.set_ylim(35, 101)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(outdir / f"learning_curve_{task}_{split}.png", dpi=160)
            plt.close(fig)


def run_full(outdir: Path, resume: bool, workers: int) -> dict:
    protocol = verify_preregistration()
    workers = max(1, min(int(workers), 3, os.cpu_count() or 1))
    outdir.mkdir(parents=True, exist_ok=True)
    partial_path = outdir / "partial_results.json"
    records: list[dict] = []
    integrity: list[dict] = []
    completed: set[int] = set()
    if resume and partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        records = partial.get("records", [])
        integrity = partial.get("integrity", [])
        completed = set(partial.get("completed_seeds", []))
        if not completed.issubset(set(SEEDS)):
            raise RuntimeError("Partial result contains seed outside frozen protocol")
        print(f"Resuming {len(completed)}/{len(SEEDS)} seeds", flush=True)

    started = time.time()
    remaining = [seed for seed in SEEDS if seed not in completed]
    print(f"Running v0.33: {len(remaining)} seeds; paired baseline vs heterogeneous delays; workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in remaining}
        for future in as_completed(futures):
            seed = futures[future]
            seed_rows, seed_integrity = future.result()
            records.extend(seed_rows)
            integrity.append(seed_integrity)
            completed.add(seed)
            atomic_json(partial_path, {
                "version": VERSION,
                "completed_seeds": [s for s in SEEDS if s in completed],
                "records": records, "integrity": integrity, "partial": True, "workers": workers,
            })
            print(f"[{len(completed)}/{len(SEEDS)}] seed={seed}; records={len(records)}; elapsed={time.time()-started:.1f}s", flush=True)

    seed_order = {seed: i for i, seed in enumerate(SEEDS)}
    task_order = {"sync": 0, "order": 1}
    condition_order = {condition: i for i, condition in enumerate(CONDITIONS)}
    records.sort(key=lambda r: (seed_order[r["seed"]], task_order[r["task"]], condition_order[r["condition"]], r["epochs"]))
    integrity.sort(key=lambda r: seed_order[r["seed"]])
    summaries = aggregate_records(records)
    deltas = paired_delta_summary(records)
    threshold_per_seed, threshold_rows = threshold_analysis(records)
    overlap_total = sum(t["train_validation_overlap"] + t["train_test_overlap"] + t["validation_test_overlap"]
                        for item in integrity for t in item["tasks"].values())
    expected = len(SEEDS) * 2 * len(CONDITIONS) * len(EPOCH_CHECKPOINTS)
    if len(records) != expected or len(integrity) != len(SEEDS) or overlap_total != 0:
        raise RuntimeError(f"Final integrity failure: records={len(records)}/{expected}, seeds={len(integrity)}, overlap={overlap_total}")
    for seed_item in integrity:
        delays = seed_item["delay_counts"]["heterogeneous_delay1_4"]
        if sum(delays.values()) != seed_item["recurrent_edge_count"]:
            raise RuntimeError(f"Heterogeneous delay assignment misses/adds edges for seed {seed_item['seed']}")

    result = {
        "version": VERSION, "seeds": SEEDS, "conditions": CONDITIONS,
        "epochs": EPOCH_CHECKPOINTS, "primary_epoch": PRIMARY_EPOCH,
        "n_train_per_task_seed": N_TRAIN, "n_validation_per_task_seed": N_VALIDATION,
        "n_test_per_task_seed": N_TEST, "feature_set": FEATURE_SET,
        "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
        "records": records, "summary": summaries, "paired_differences": deltas,
        "threshold_per_seed": threshold_per_seed, "threshold_summary": threshold_rows,
        "integrity": integrity, "runtime_seconds": time.time() - started,
        "protocol": protocol, "workers": workers,
    }
    atomic_json(outdir / "results.json", result)
    write_csv(outdir / "per_seed_delays.csv", records)
    write_csv(outdir / "delay_summary.csv", summaries)
    write_csv(outdir / "paired_delay_differences.csv", deltas)
    write_csv(outdir / "per_seed_threshold.csv", threshold_per_seed)
    write_csv(outdir / "threshold_summary.csv", threshold_rows)
    (outdir / "report_ar.md").write_text(render_report(records, summaries, deltas, threshold_rows, integrity,
                                                       result["runtime_seconds"], workers), encoding="utf-8")
    plot_curves(summaries, outdir)
    partial_path.unlink(missing_ok=True)
    print("FINAL RESULTS SAVED", flush=True)
    for row in deltas:
        if row["epochs"] == PRIMARY_EPOCH:
            print(f"PRIMARY {row['task']}: delta={row['mean_delta_percentage_points']:+.2f} pp "
                  f"CI95=[{row['ci95_low_percentage_points']:+.2f}, {row['ci95_high_percentage_points']:+.2f}], "
                  f"positive/negative/zero={row['positive_seed_deltas']}/{row['negative_seed_deltas']}/{row['zero_seed_deltas']}", flush=True)
    return result


def run_smoke() -> None:
    seed = 5990  # engineering-only seed outside the preregistered range
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    base_delays = make_recurrent_delay_matrix(reservoir, "baseline_delay1", seed)
    hetero_delays = make_recurrent_delay_matrix(reservoir, "heterogeneous_delay1_4", seed)
    assert np.array_equal(make_recurrent_delay_matrix(reservoir, "baseline_delay1", seed), base_delays)
    assert int(np.sum(base_delays > 0)) == int(np.sum(reservoir.w_rec != 0.0))
    assert int(np.sum(hetero_delays > 0)) == int(np.sum(reservoir.w_rec != 0.0))
    assert set(np.unique(hetero_delays[hetero_delays > 0])).issubset({1, 2, 3, 4})
    trials = make_nested_training_pool(seed + 11, "sync", 200)
    _, original_spikes = simulate_batch_vectorized(reservoir, trials)
    _, new_base_spikes = simulate_batch_with_recurrent_delays(reservoir, trials, base_delays)
    _, hetero_spikes = simulate_batch_with_recurrent_delays(reservoir, trials, hetero_delays)
    assert np.allclose(original_spikes, new_base_spikes, rtol=0, atol=1e-7)
    assert hetero_spikes.shape == original_spikes.shape
    assert np.isfinite(hetero_spikes).all()
    feats = extract_16window_rate_features(hetero_spikes)
    assert feats.shape == (200, 384) and np.isfinite(feats).all()
    labels = make_epoch_labels(trials)
    assert int(np.sum(labels == 0)) == 100 and int(np.sum(labels == 1)) == 100
    print("SMOKE PASSED: delay assignment is edge-only, delay=1 reproduces legacy simulation, features are finite", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="engineering-only smoke test")
    parser.add_argument("--resume", action="store_true", help="resume preregistered run from saved seed results")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full(ROOT / "results" / "delayed_reservoir_v0_33", resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
