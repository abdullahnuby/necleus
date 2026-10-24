"""Nucleus Pulse v0.35: paired test of activity-dependent inhibitory gain.

Both conditions use the exact same Dale-style recurrent matrix from v0.34.
The baseline keeps inhibitory gain fixed at 1.0. The treatment dynamically
scales only the negative recurrent current using an exponential moving average
of recent population activity. No weights, connectivity, features, reader, or
training data differ between conditions.
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
from experiments.run_inhibition_structure_v0_34 import (  # noqa: E402
    EPOCH_CHECKPOINTS,
    N_TRAIN,
    N_VALIDATION,
    N_TEST,
    TASKS,
    N_WINDOWS,
    TARGET_VALIDATION_ACCURACY,
    BOOTSTRAP_REPS,
    stable_seed,
    bootstrap_ci,
    wilson_ci,
    atomic_json,
    write_csv,
    spectral_radius,
    make_recurrent_weight_matrices,
    simulate_batch_with_recurrent_weights,
    extract_features,
    run_reader as run_base_reader,
)

VERSION = "0.35"
SEEDS = list(range(8000, 8040))
CONDITIONS = ["dale_fixed_25", "dale_activity_adaptive_25"]
PRIMARY_EPOCH = 10
FEATURE_SET = "spike_bins16_rate_384"
TARGET_SPECTRAL_RADIUS = 0.72
# Frozen before the v0.35 final run. The target is rounded from v0.34's
# observed Dale-reservoir activity (about 0.129 spikes/cell/tick).
TARGET_ACTIVITY_RATE = 0.13
ACTIVITY_TRACE_TAU_TICKS = 4.0
INHIBITORY_GAIN_SLOPE = 4.0
INHIBITORY_GAIN_MIN = 0.5
INHIBITORY_GAIN_MAX = 2.0
BOOTSTRAP_SEED = 20351024


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
        "shared_base_recurrent_matrix": {
            "source": "v0.34 Dale-style 25% inhibitory matrix, generated once per reservoir seed",
            "same_matrix_for_both_conditions": True,
            "same_topology_and_signed_edge_weights": True,
            "base_spectral_radius_target": TARGET_SPECTRAL_RADIUS,
        },
        "baseline": "Dale-style 25% inhibitory sources with fixed inhibitory gain = 1.0",
        "treatment": {
            "name": "global activity-dependent inhibitory gain",
            "rule": "gain_t = clip(1 + slope * (ema_population_rate_t - target_rate), min_gain, max_gain)",
            "ema_tau_ticks": ACTIVITY_TRACE_TAU_TICKS,
            "target_spikes_per_neuron_per_tick": TARGET_ACTIVITY_RATE,
            "slope": INHIBITORY_GAIN_SLOPE,
            "gain_bounds": [INHIBITORY_GAIN_MIN, INHIBITORY_GAIN_MAX],
            "only_negative_recurrent_current_is_scaled": True,
            "activity_trace_initialization": "target_rate, producing initial gain 1.0 before observed activity",
            "parameter_source": "predefined from v0.34 activity diagnostic; no v0.35 validation/test tuning",
        },
        "same_input_weights_thresholds_leaks_refractory_mask_features_reader_and_training_order": True,
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
    path = ROOT / "results" / "activity_adaptive_inhibition_v0_35" / "preregistration.json"
    if not path.exists():
        raise RuntimeError("Missing frozen v0.35 preregistration; refusing official run.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Current run configuration differs from v0.35 preregistration.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return doc


def make_shared_dale_matrix(reservoir: DynamicPulseReservoir, seed: int):
    """Use v0.34's Dale matrix once; both v0.35 conditions receive exact copies."""
    matrices, diagnostics = make_recurrent_weight_matrices(reservoir, seed)
    dale = np.asarray(matrices["dale_inhibition_25"], dtype=np.float32).copy()
    radius = spectral_radius(dale)
    if not np.isclose(radius, TARGET_SPECTRAL_RADIUS, atol=5e-4):
        raise RuntimeError(f"Base Dale matrix has unexpected spectral radius: {radius}")
    paired = {
        "dale_fixed_25": dale.copy(),
        "dale_activity_adaptive_25": dale.copy(),
    }
    if not np.array_equal(paired[CONDITIONS[0]], paired[CONDITIONS[1]]):
        raise RuntimeError("The v0.35 conditions must start from exactly the same matrix")
    return paired, diagnostics


def simulate_batch_activity_adaptive_inhibition(
    reservoir: DynamicPulseReservoir,
    trials: Sequence,
    recurrent_weights: np.ndarray,
    *,
    target_rate: float = TARGET_ACTIVITY_RATE,
    trace_tau_ticks: float = ACTIVITY_TRACE_TAU_TICKS,
    slope: float = INHIBITORY_GAIN_SLOPE,
    gain_min: float = INHIBITORY_GAIN_MIN,
    gain_max: float = INHIBITORY_GAIN_MAX,
    return_diagnostics: bool = False,
):
    """Vectorized LIF simulation with a bounded, activity-dependent E/I gain.

    Recurrent columns are presynaptic sources. Positive recurrent current is
    unchanged; only the negative current gets a scalar gain per trial/tick.
    The EMA and gain are reset for each input sequence (no cross-trial state).
    """
    if recurrent_weights.shape != reservoir.w_rec.shape:
        raise ValueError("recurrent_weights must match reservoir.w_rec shape")
    if not np.isfinite(recurrent_weights).all():
        raise ValueError("recurrent_weights contains non-finite values")
    if not (0 < gain_min <= 1.0 <= gain_max):
        raise ValueError("gain bounds must straddle 1.0")
    if trace_tau_ticks <= 0 or slope < 0 or not (0 <= target_rate <= 1):
        raise ValueError("invalid adaptive inhibition parameters")
    if len(trials) == 0:
        empty_inputs = np.empty((0, reservoir.duration, 4), np.float32)
        empty_spikes = np.empty((0, reservoir.duration, reservoir.n_hidden), np.float32)
        empty_diag = {"mean_inhibitory_gain": None, "min_inhibitory_gain": None,
                      "max_inhibitory_gain": None, "fraction_gain_at_min": None,
                      "fraction_gain_at_max": None}
        return (empty_inputs, empty_spikes, empty_diag) if return_diagnostics else (empty_inputs, empty_spikes)

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
    rho = float(np.exp(-1.0 / trace_tau_ticks))
    activity_trace = np.full(count, target_rate, dtype=np.float32)
    gain_total = 0.0
    first_tick_gain_mean = None
    gain_min_seen = float("inf")
    gain_max_seen = float("-inf")
    at_min = 0
    at_max = 0
    n_gain_values = count * duration

    for t in range(duration):
        # Apply the gain from the EMA accumulated through the previous tick.
        # Therefore the initialized trace starts at target_rate and gain is
        # exactly 1.0 on tick 0, before any spikes from this trial are observed.
        gain = np.clip(1.0 + slope * (activity_trace - target_rate), gain_min, gain_max).astype(np.float32)
        gain_total += float(gain.sum())
        if t == 0:
            first_tick_gain_mean = float(gain.mean())
        gain_min_seen = min(gain_min_seen, float(gain.min()))
        gain_max_seen = max(gain_max_seen, float(gain.max()))
        at_min += int(np.sum(gain <= gain_min + 1e-7))
        at_max += int(np.sum(gain >= gain_max - 1e-7))
        excitatory_current = previous_spikes @ w_exc_t
        inhibitory_current = previous_spikes @ w_inh_t
        recurrent_current = excitatory_current + inhibitory_current * gain[:, None]
        i_syn = (reservoir.syn_leak[None, :] * i_syn
                 + inputs[:, t, :] @ w_in_t
                 + recurrent_current)
        voltage = reservoir.reset_leak[None, :] * voltage + (1.0 - reservoir.reset_leak[None, :]) * i_syn
        active = (voltage >= reservoir.threshold[None, :]) & (refractory == 0)
        spikes[:, t, :] = active.astype(np.float32)
        voltage[active] = 0.0
        refractory = np.maximum(refractory - 1, 0)
        refractory[active] = 1
        # Update the EMA after observing this tick's spikes. This state feeds
        # the gain at the following tick, preserving causal feedback.
        activity_trace = rho * activity_trace + (1.0 - rho) * active.mean(axis=1)
        previous_spikes = active.astype(np.float32)

    diag = {
        "target_activity_rate_per_neuron_per_tick": float(target_rate),
        "activity_trace_tau_ticks": float(trace_tau_ticks),
        "gain_slope": float(slope),
        "gain_bounds": [float(gain_min), float(gain_max)],
        "first_tick_inhibitory_gain_mean": float(first_tick_gain_mean),
        "mean_inhibitory_gain": float(gain_total / n_gain_values),
        "min_inhibitory_gain": float(gain_min_seen),
        "max_inhibitory_gain": float(gain_max_seen),
        "fraction_gain_at_min": float(at_min / n_gain_values),
        "fraction_gain_at_max": float(at_max / n_gain_values),
    }
    if not np.isfinite(spikes).all() or not np.isfinite(list(diag[k] for k in (
            "mean_inhibitory_gain", "min_inhibitory_gain", "max_inhibitory_gain"))).all():
        raise RuntimeError("Non-finite adaptive simulation state")
    return (inputs, spikes, diag) if return_diagnostics else (inputs, spikes)


def make_data(seed: int, task: str):
    task_i = TASKS.index(task)
    train_pool = make_nested_training_pool(seed + 11111 + task_i * 100, task, N_TRAIN)
    train_keys = {trial_key(tr) for tr in train_pool}
    val_trials = make_eval_dataset(seed + 52222 + task_i * 100, task, N_VALIDATION, set(train_keys))
    val_keys = {trial_key(tr) for tr in val_trials}
    test_trials = make_eval_dataset(seed + 63333 + task_i * 100, task, N_TEST, set(train_keys) | val_keys)
    test_keys = {trial_key(tr) for tr in test_trials}
    if (len(train_keys) != N_TRAIN or len(val_keys) != N_VALIDATION or len(test_keys) != N_TEST
            or train_keys & val_keys or train_keys & test_keys or val_keys & test_keys):
        raise RuntimeError(f"Invalid split sizes or overlap at seed={seed}, task={task}")
    trial_sets = {"train": train_pool, "validation": val_trials, "test": test_trials}
    labels = {split: make_epoch_labels(trials) for split, trials in trial_sets.items()}
    for split, ys in labels.items():
        if int(np.sum(ys == 0)) != len(ys) // 2 or int(np.sum(ys == 1)) != len(ys) // 2:
            raise RuntimeError(f"Class imbalance at seed={seed}, task={task}, split={split}")
    bounds = {
        "train": (0, N_TRAIN),
        "validation": (N_TRAIN, N_TRAIN + N_VALIDATION),
        "test": (N_TRAIN + N_VALIDATION, N_TRAIN + N_VALIDATION + N_TEST),
    }
    integrity = {
        "n_train_unique": len(train_keys), "n_validation_unique": len(val_keys), "n_test_unique": len(test_keys),
        "train_validation_overlap": len(train_keys & val_keys),
        "train_test_overlap": len(train_keys & test_keys),
        "validation_test_overlap": len(val_keys & test_keys),
        "class_balance": {split: {"0": int(np.sum(ys == 0)), "1": int(np.sum(ys == 1))} for split, ys in labels.items()},
    }
    return train_pool + val_trials + test_trials, bounds, labels, integrity


def run_seed(seed: int) -> tuple[list[dict], dict]:
    started = time.time()
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, matrix_diag = make_shared_dale_matrix(reservoir, seed)
    shared_weights = matrices[CONDITIONS[0]]
    rows = []
    integrity = {
        "seed": int(seed), "reservoir_seed": int(seed),
        "same_exact_recurrent_matrix_between_conditions": bool(np.array_equal(matrices[CONDITIONS[0]], matrices[CONDITIONS[1]])),
        "same_input_weights_thresholds_leaks_refractory_and_neuron_count": True,
        "base_spectral_radius": spectral_radius(shared_weights),
        "base_recurrent_matrix_diagnostics": matrix_diag["conditions"]["dale_inhibition_25"],
        "inhibitory_source_indices": matrix_diag["inhibitory_source_indices"],
        "tasks": {},
    }
    for task_i, task in enumerate(TASKS):
        concatenated, bounds, labels, split_integrity = make_data(seed, task)
        features_by_condition = {}
        activity = {}
        expected_inputs = None
        for condition in CONDITIONS:
            if condition == "dale_fixed_25":
                inputs, spikes = simulate_batch_with_recurrent_weights(reservoir, concatenated, shared_weights)
                gain_diag = {
                    "mean_inhibitory_gain": 1.0,
                    "min_inhibitory_gain": 1.0,
                    "max_inhibitory_gain": 1.0,
                    "fraction_gain_at_min": 0.0,
                    "fraction_gain_at_max": 0.0,
                    "gain_is_fixed": True,
                }
            else:
                inputs, spikes, gain_diag = simulate_batch_activity_adaptive_inhibition(
                    reservoir, concatenated, shared_weights, return_diagnostics=True
                )
                gain_diag["gain_is_fixed"] = False
            if expected_inputs is None:
                expected_inputs = inputs
            elif not np.array_equal(inputs, expected_inputs):
                raise RuntimeError("Input raster changed across v0.35 conditions")
            split_features = {}
            for split, (left, right) in bounds.items():
                split_features[split] = extract_features(spikes[left:right])
            features_by_condition[condition] = split_features
            total_per_trial = np.sum(spikes, axis=(1, 2))
            activity[condition] = {
                **gain_diag,
                "mean_spikes_per_trial": float(np.mean(total_per_trial)),
                "fraction_hidden_cells_with_at_least_one_spike": float(np.mean(np.sum(spikes, axis=(0, 1)) > 0)),
                "fraction_trials_with_zero_hidden_spikes": float(np.mean(total_per_trial == 0)),
                "mean_spike_rate_per_hidden_tick": float(np.mean(spikes)),
            }
            del inputs, spikes
        integrity["tasks"][task] = {**split_integrity, "activity_diagnostics": activity}
        for condition in CONDITIONS:
            result_rows = run_base_reader(features_by_condition[condition], labels, seed, task_i)
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
    out = []
    for task, condition, epoch in sorted(groups):
        group = groups[(task, condition, epoch)]
        item = {"task": task, "condition": condition, "epochs": int(epoch), "n_seeds": len({r["seed"] for r in group}),
                "unique_training_examples_per_seed": N_TRAIN, "example_exposures_per_seed": N_TRAIN * int(epoch)}
        for split in ("validation", "test"):
            for metric in metrics:
                mean, low, high = bootstrap_ci([r[f"{split}_{metric}"] for r in group],
                                               stable_seed("v035", task, condition, str(epoch), split, metric))
                item[f"{split}_{metric}_mean_percent"] = mean
                item[f"{split}_{metric}_ci95_low_percent"] = low
                item[f"{split}_{metric}_ci95_high_percent"] = high
        out.append(item)
    return out


def paired_delta_summary(records: Sequence[dict]) -> list[dict]:
    lookup = {(r["seed"], r["task"], r["condition"], r["epochs"]): r for r in records}
    seeds = sorted({int(r["seed"]) for r in records})
    out = []
    for task in TASKS:
        for epoch in EPOCH_CHECKPOINTS:
            deltas = np.asarray([
                lookup[(seed, task, "dale_activity_adaptive_25", epoch)]["test_accuracy"]
                - lookup[(seed, task, "dale_fixed_25", epoch)]["test_accuracy"]
                for seed in seeds
            ], dtype=float)
            mean, low, high = bootstrap_ci(deltas, stable_seed("v035-paired", task, str(epoch)))
            out.append({
                "task": task, "epochs": int(epoch),
                "contrast": "dale_activity_adaptive_25 - dale_fixed_25",
                "n_paired_seeds": len(deltas), "mean_delta_percentage_points": mean,
                "ci95_low_percentage_points": low, "ci95_high_percentage_points": high,
                "positive_seed_deltas": int(np.sum(deltas > 0)),
                "negative_seed_deltas": int(np.sum(deltas < 0)),
                "zero_seed_deltas": int(np.sum(deltas == 0)),
            })
    return out


def threshold_analysis(records: Sequence[dict]):
    groups = defaultdict(list)
    for row in records:
        groups[(row["seed"], row["task"], row["condition"])].append(row)
    per_seed, summaries = [], []
    for (seed, task, condition), group in sorted(groups.items()):
        group = sorted(group, key=lambda x: x["epochs"])
        hit = next((r for r in group if r["validation_accuracy"] >= TARGET_VALIDATION_ACCURACY), None)
        per_seed.append({
            "seed": int(seed), "task": task, "condition": condition,
            "reached_85_validation": hit is not None,
            "first_epoch_reaching_85": int(hit["epochs"]) if hit else None,
            "validation_accuracy_at_selection_percent": 100 * hit["validation_accuracy"] if hit else None,
            "independent_test_accuracy_at_selected_epoch_percent": 100 * hit["test_accuracy"] if hit else None,
        })
    for task in TASKS:
        for condition in CONDITIONS:
            subset = [x for x in per_seed if x["task"] == task and x["condition"] == condition]
            hits = [x for x in subset if x["reached_85_validation"]]
            low, high = wilson_ci(len(hits), len(subset))
            test_values = [x["independent_test_accuracy_at_selected_epoch_percent"] / 100 for x in hits]
            test_mean, test_low, test_high = bootstrap_ci(test_values, stable_seed("v035-threshold", task, condition))
            epochs = [x["first_epoch_reaching_85"] for x in hits]
            summaries.append({
                "task": task, "condition": condition, "n_seeds": len(subset), "n_reached_85": len(hits),
                "reach_rate_percent": 100 * len(hits) / len(subset),
                "reach_rate_ci95_wilson_low_percent": low, "reach_rate_ci95_wilson_high_percent": high,
                "mean_first_epoch_among_reached": float(np.mean(epochs)) if epochs else None,
                "median_first_epoch_among_reached": float(np.median(epochs)) if epochs else None,
                "independent_test_accuracy_at_selected_epoch_mean_percent": test_mean if hits else None,
                "independent_test_accuracy_at_selected_epoch_ci95_low_percent": test_low if hits else None,
                "independent_test_accuracy_at_selected_epoch_ci95_high_percent": test_high if hits else None,
            })
    return per_seed, summaries


def render_report(records, summaries, deltas, threshold_rows, integrity, runtime, workers):
    names = {"dale_fixed_25": "ثابت: تثبيط Dale بنسبة 25%",
             "dale_activity_adaptive_25": "تكيفي: كسب التثبيط تابع للنشاط"}
    lines = [
        "# تقرير Nucleus Pulse v0.35 — التثبيط المعتمد على النشاط", "",
        "## ملخص التجربة", "",
        f"- البذور: **{len({r['seed'] for r in records})}**؛ السجلات: **{len(records)}**.",
        f"- مجموعة التدريب: **{N_TRAIN}** مثالًا فريدًا لكل مهمة/بذرة؛ التحقق والاختبار: **{N_VALIDATION}** لكل منهما.",
        f"- المرورّات: **{EPOCH_CHECKPOINTS}**؛ الميزات: **{FEATURE_SET}**.",
        f"- زمن التنفيذ في التشغيل النهائي: **{runtime:.2f} ثانية**؛ العمليات المتوازية: **{workers}**.", "",
        "## التغيير الوحيد", "",
        "الشرطان يبدأان من **نفس مصفوفة Dale بالضبط**، بنفس طوبولوجيا الاتصال والأوزان وقطبية المصادر ونصف القطر الطيفي الأساسي 0.72. الشرط المرجعي يستخدم كسب تثبيط ثابتًا 1.0. الشرط التكيفي يضرب التيار الراجع السالب في كسب يعتمد على متوسط نشاط الخلايا خلال EMA بزمن ثابت 4 ticks، مستهدفه 0.13 نبضة/خلية/tick، ومحدود بين 0.5 و2.0. المعاملات مجمدة مسبقًا ولم تُضبط على تحقق أو اختبار v0.35.", "",
        "## دقة الاختبار المستقل", "",
        "| المهمة | الشرط | المرورّات | متوسط الدقة [95% CI] | متوسط الدقة المتوازنة | استدعاء الفئة 0 | استدعاء الفئة 1 |", "|---|---|---:|---:|---:|---:|---:|"
    ]
    for task in TASKS:
        for condition in CONDITIONS:
            r = next(x for x in summaries if x["task"] == task and x["condition"] == condition and x["epochs"] == PRIMARY_EPOCH)
            lines.append(f"| {task} | {names[condition]} | {PRIMARY_EPOCH} | {r['test_accuracy_mean_percent']:.2f}% [{r['test_accuracy_ci95_low_percent']:.2f}, {r['test_accuracy_ci95_high_percent']:.2f}] | {r['test_balanced_accuracy_mean_percent']:.2f}% | {r['test_class_0_recall_mean_percent']:.2f}% | {r['test_class_1_recall_mean_percent']:.2f}% |")
    lines += ["", "## فرق الدقة الزوجي (التكيفي − الثابت)", "", "| المهمة | المرورّات | فرق متوسط الدقة بالنقاط المئوية [95% CI] | بذور تحسنت/تراجعت/تعادلت |", "|---|---:|---:|---:|"]
    for r in deltas:
        lines.append(f"| {r['task']} | {r['epochs']} | {r['mean_delta_percentage_points']:+.2f} [{r['ci95_low_percentage_points']:+.2f}, {r['ci95_high_percentage_points']:+.2f}] | {r['positive_seed_deltas']}/{r['negative_seed_deltas']}/{r['zero_seed_deltas']} |")
    lines += ["", "## بلوغ 85% على التحقق", "", "| المهمة | الشرط | بذور وصلت/40 | نسبة الوصول [Wilson 95%] |", "|---|---|---:|---:|"]
    for r in threshold_rows:
        lines.append(f"| {r['task']} | {names[r['condition']]} | {r['n_reached_85']}/{r['n_seeds']} | {r['reach_rate_percent']:.1f}% [{r['reach_rate_ci95_wilson_low_percent']:.1f}, {r['reach_rate_ci95_wilson_high_percent']:.1f}] |")
    overlap = sum(t["train_validation_overlap"] + t["train_test_overlap"] + t["validation_test_overlap"] for item in integrity for t in item["tasks"].values())
    lines += ["", "## تشخيص نشاط الخزان", "", "متوسط نشاط الشرطين وكسب التثبيط وتكرار الوصول إلى حدود الكسب محفوظ في results.json وملف per-seed. هذه المقاييس توضح أثر آلية التحكم، ولا تُعد وحدها مقياسًا للطاقة أو الفاعلية.", "", "## سلامة البيانات والقيود", "",
              f"- السجلات: **{len(records)} / {len(SEEDS)*len(TASKS)*len(CONDITIONS)*len(EPOCH_CHECKPOINTS)}**؛ البذور: **{len(integrity)} / {len(SEEDS)}**.",
              f"- تقاطعات train/validation/test: **{overlap}**.",
              "- جميع أمثلة الشرطين متطابقة؛ الاختلاف الوحيد في الشرط التكيفي هو كسب التيار التثبيطي المتغير زمنيًا.",
              "- فواصل الثقة للمتوسطات والفروق تعيد سحب البذور، والفروق محسوبة زوجيًا على البذور نفسها.",
              "- الاختبار يقيّم قارئًا خطيًا فوق خزان صغير ومهمتين اصطناعيتين، ولا يثبت قدرة على التفكير الهندسي أو تعميمًا عامًا.",
              "- كسب التثبيط التكيفي heuristic محدود، وليس نموذجًا كاملًا للتثبيط البيولوجي أو تعلمًا مشبكيًا.", ""]
    return "\n".join(lines)


def plot_curves(summaries: Sequence[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"dale_fixed_25": "Fixed Dale inhibition", "dale_activity_adaptive_25": "Activity-adaptive inhibition"}
    for task in TASKS:
        for split in ("validation", "test"):
            fig, ax = plt.subplots(figsize=(8.4, 5.4))
            for condition in CONDITIONS:
                group = sorted((x for x in summaries if x["task"] == task and x["condition"] == condition), key=lambda x: x["epochs"])
                xs = [x["epochs"] for x in group]
                ys = [x[f"{split}_accuracy_mean_percent"] for x in group]
                low = [x[f"{split}_accuracy_ci95_low_percent"] for x in group]
                high = [x[f"{split}_accuracy_ci95_high_percent"] for x in group]
                ax.plot(xs, ys, marker="o", label=names[condition])
                ax.fill_between(xs, low, high, alpha=0.15)
            ax.axhline(TARGET_VALIDATION_ACCURACY * 100, linestyle="--", linewidth=1, label="85% target")
            ax.axhline(50, linestyle=":", linewidth=1, label="chance")
            ax.set_xticks(EPOCH_CHECKPOINTS)
            ax.set_xlabel("Passes over the same 6,000 unique examples")
            ax.set_ylabel(f"{split.capitalize()} accuracy (%)")
            ax.set_title(f"Nucleus Pulse v0.35 — {task} — {split}")
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
    done = {}
    if resume and partial.exists():
        for item in json.loads(partial.read_text(encoding="utf-8")):
            done[int(item["integrity"]["seed"])] = item
    missing = [s for s in SEEDS if s not in done]
    start = time.time()
    print(f"Nucleus Pulse v{VERSION}: {len(done)} saved seeds; {len(missing)} remaining; workers={workers}", flush=True)
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
        raise RuntimeError(f"Incomplete v0.35 run: records={len(records)}/{expected}; seeds={len(integrity)}/{len(SEEDS)}")
    summaries = aggregate_records(records)
    deltas = paired_delta_summary(records)
    threshold_per_seed, threshold_rows = threshold_analysis(records)
    activity_means = {}
    for condition in CONDITIONS:
        entries = [item["integrity"]["tasks"][task]["activity_diagnostics"][condition]
                   for item in items for task in TASKS]
        activity_means[condition] = {
            "mean_spikes_per_trial": float(np.mean([x["mean_spikes_per_trial"] for x in entries])),
            "mean_spike_rate_per_hidden_tick": float(np.mean([x["mean_spike_rate_per_hidden_tick"] for x in entries])),
            "mean_inhibitory_gain": float(np.mean([x["mean_inhibitory_gain"] for x in entries])),
            "mean_fraction_gain_at_min": float(np.mean([x["fraction_gain_at_min"] for x in entries])),
            "mean_fraction_gain_at_max": float(np.mean([x["fraction_gain_at_max"] for x in entries])),
        }
    runtime = time.time() - start
    result = {
        "version": VERSION, "protocol": protocol_config(), "seeds": SEEDS,
        "conditions": CONDITIONS, "n_hidden": N_HIDDEN, "n_train": N_TRAIN,
        "n_validation": N_VALIDATION, "n_test": N_TEST, "feature_set": FEATURE_SET,
        "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
        "records": records, "summary": summaries, "paired_differences": deltas,
        "threshold_per_seed": threshold_per_seed, "threshold_summary": threshold_rows,
        "integrity": integrity, "activity_diagnostic_means": activity_means,
        "runtime_seconds": runtime, "workers": workers,
    }
    atomic_json(outdir / "results.json", result)
    write_csv(outdir / "per_seed_activity_adaptive.csv", records)
    write_csv(outdir / "activity_adaptive_summary.csv", summaries)
    write_csv(outdir / "paired_activity_adaptive_differences.csv", deltas)
    write_csv(outdir / "per_seed_threshold.csv", threshold_per_seed)
    write_csv(outdir / "threshold_summary.csv", threshold_rows)
    (outdir / "report_ar.md").write_text(render_report(records, summaries, deltas, threshold_rows, integrity, runtime, workers), encoding="utf-8")
    plot_curves(summaries, outdir)
    partial.unlink(missing_ok=True)
    print("FINAL RESULTS SAVED", flush=True)
    for r in deltas:
        if r["epochs"] == PRIMARY_EPOCH:
            print(f"PRIMARY {r['task']}: delta={r['mean_delta_percentage_points']:+.2f} pp "
                  f"CI95=[{r['ci95_low_percentage_points']:+.2f}, {r['ci95_high_percentage_points']:+.2f}], "
                  f"positive/negative/zero={r['positive_seed_deltas']}/{r['negative_seed_deltas']}/{r['zero_seed_deltas']}", flush=True)
    print("ACTIVITY", json.dumps(activity_means, sort_keys=True), flush=True)
    return result


def run_smoke() -> None:
    seed = 7990  # outside the frozen official seed range
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, diagnostics = make_shared_dale_matrix(reservoir, seed)
    fixed = matrices["dale_fixed_25"]
    adaptive_base = matrices["dale_activity_adaptive_25"]
    assert np.array_equal(fixed, adaptive_base)
    assert np.isclose(spectral_radius(fixed), TARGET_SPECTRAL_RADIUS, atol=5e-4)
    assert diagnostics["topology_equal"]
    trials = make_nested_training_pool(seed + 11, "sync", 200)
    fixed_in, fixed_sp = simulate_batch_with_recurrent_weights(reservoir, trials, fixed)
    adaptive_in, adaptive_sp, diag = simulate_batch_activity_adaptive_inhibition(reservoir, trials, adaptive_base, return_diagnostics=True)
    assert np.array_equal(fixed_in, adaptive_in)
    assert fixed_sp.shape == adaptive_sp.shape == (200, 32, N_HIDDEN)
    assert np.isfinite(adaptive_sp).all()
    assert extract_features(adaptive_sp).shape == (200, 16 * N_HIDDEN)
    assert INHIBITORY_GAIN_MIN <= diag["min_inhibitory_gain"] <= diag["max_inhibitory_gain"] <= INHIBITORY_GAIN_MAX
    # Gain slope zero removes adaptive modulation and should reproduce the fixed simulation.
    _, no_adaptation = simulate_batch_activity_adaptive_inhibition(reservoir, trials, fixed, slope=0.0)
    assert np.allclose(fixed_sp, no_adaptation, rtol=0.0, atol=1e-7)
    print("SMOKE PASSED: same base matrix, bounded adaptive gain, simulation and feature checks", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full(ROOT / "results" / "activity_adaptive_inhibition_v0_35", args.resume, args.workers)


if __name__ == "__main__":
    main()
