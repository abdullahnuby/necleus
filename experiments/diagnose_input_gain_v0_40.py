"""Nucleus Pulse v0.40 Stage 1: component-only input-gain robustness diagnostic.

One diagnostic seed, 32 class-balanced examples per task, and input-weight scales
0.75/1.0/1.25. Compares fixed inhibitory gain with the local adaptive rule using
slope=2 and EMA tau=4 (the simpler setting retained after v0.39 found no clear
evidence that tau=8 is better). No readout is trained and no final-test metrics
or hyperparameter selection are performed.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir  # noqa: E402
from experiments import run_activity_adaptive_inhibition_v0_35 as v35  # noqa: E402
from experiments.tune_local_inhibition_v0_37 import simulate_local  # noqa: E402
from experiments.run_inhibition_structure_v0_34 import (  # noqa: E402
    extract_features,
    simulate_batch_with_recurrent_weights,
    spectral_radius,
)
from experiments.run_learning_curve_v0_30 import N_HIDDEN  # noqa: E402

VERSION = "0.40-stage1-input-gain-component"
SEED = 14040  # reserved diagnostic seed; not used for development selection or formal testing
TASKS = ("sync", "order")
INPUT_SCALES = (0.75, 1.0, 1.25)
CONDITIONS = ("fixed_inhibition", "local_adaptive_inhibition")
N_PER_CLASS = 16
LOCAL_SLOPE = 2.0
LOCAL_TAU = 4.0  # conservative/simple choice after v0.39 did not confirm tau=8 superiority
TARGET_RADIUS = v35.TARGET_SPECTRAL_RADIUS
TARGET_RATE = v35.TARGET_ACTIVITY_RATE
GAIN_MIN, GAIN_MAX = v35.INHIBITORY_GAIN_MIN, v35.INHIBITORY_GAIN_MAX
OUTDIR = ROOT / "results" / "input_gain_v0_40" / "component_smoke"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def scale_input_weights(reservoir: DynamicPulseReservoir, scale: float) -> DynamicPulseReservoir:
    """Return a shallow reservoir copy with input weights scaled; never mutate source."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("input scale must be finite and positive")
    clone = copy.copy(reservoir)
    clone.w_in = (np.asarray(reservoir.w_in, dtype=np.float32) * np.float32(scale)).astype(np.float32)
    return clone


def balanced_train_slice(all_trials, labels, n_per_class: int = N_PER_CLASS):
    y = np.asarray(labels["train"])
    if y.size > len(all_trials):
        raise RuntimeError("training labels exceed trial collection")
    i0 = np.flatnonzero(y == 0)[:n_per_class]
    i1 = np.flatnonzero(y == 1)[:n_per_class]
    if len(i0) != n_per_class or len(i1) != n_per_class:
        raise RuntimeError("not enough examples in each class for the component diagnostic")
    idx = np.sort(np.concatenate([i0, i1]))
    selected_y = y[idx]
    if int(np.sum(selected_y == 0)) != n_per_class or int(np.sum(selected_y == 1)) != n_per_class:
        raise RuntimeError("component diagnostic slice is not class-balanced")
    return [all_trials[int(i)] for i in idx], selected_y


def _mean_spikes(spikes: np.ndarray) -> float:
    return float(np.asarray(spikes, dtype=np.float64).sum(axis=(1, 2)).mean())


def run() -> dict[str, Any]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base_reservoir = DynamicPulseReservoir(seed=SEED, n_hidden=N_HIDDEN)
    original_w_in = np.asarray(base_reservoir.w_in).copy()
    matrices, matrix_diag = v35.make_shared_dale_matrix(base_reservoir, SEED)
    weights = np.asarray(matrices["dale_fixed_25"], dtype=np.float32).copy()
    radius = float(spectral_radius(weights))
    if not np.isclose(radius, TARGET_RADIUS, atol=5e-4):
        raise RuntimeError(f"unexpected recurrent spectral radius {radius}")

    rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {
        "diagnostic_seed": SEED,
        "tasks": {},
        "same_recurrent_matrix_for_all_scales_and_conditions": True,
        "same_event_sequences_for_all_scales": True,
        "input_weight_scaling_only": True,
        "original_reservoir_input_weights_unchanged": True,
        "recurrent_matrix_sha256": array_sha256(weights),
        "scaled_input_weight_sha256_by_scale": {
            str(scale): array_sha256(scale_input_weights(base_reservoir, scale).w_in)
            for scale in INPUT_SCALES
        },
        "readout_training_performed": False,
        "official_test_used": False,
        "selected_parameters": {"local_slope": LOCAL_SLOPE, "local_ema_tau_ticks": LOCAL_TAU},
    }

    for task in TASKS:
        all_trials, bounds, labels, split_diag = v35.make_data(SEED, task)
        trials, y = balanced_train_slice(all_trials, labels)
        task_outputs: dict[tuple[str, float], dict[str, np.ndarray]] = {}
        input_raster_reference = None
        for scale in INPUT_SCALES:
            scaled = scale_input_weights(base_reservoir, scale)
            if not np.array_equal(scaled.w_in, original_w_in * np.float32(scale)):
                raise RuntimeError("scaled reservoir weights do not match requested input scale")
            if not np.array_equal(base_reservoir.w_in, original_w_in):
                raise RuntimeError("scaling helper mutated the original reservoir")

            # Fixed inhibitory gain control.
            fixed_inputs, fixed_spikes = simulate_batch_with_recurrent_weights(scaled, trials, weights)
            if input_raster_reference is None:
                input_raster_reference = fixed_inputs.copy()
            elif not np.array_equal(input_raster_reference, fixed_inputs):
                raise RuntimeError("event raster changed across input-amplitude conditions")
            if not np.isfinite(fixed_spikes).all():
                raise RuntimeError("fixed condition produced non-finite spikes")
            task_outputs[("fixed_inhibition", scale)] = {
                "spikes": fixed_spikes.copy(),
                "features": extract_features(fixed_spikes),
            }
            rows.append({
                "seed": SEED, "task": task, "condition": "fixed_inhibition",
                "input_scale": scale, "n_trials": len(trials),
                "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
                "spectral_radius": radius, "mean_spikes_per_trial": _mean_spikes(fixed_spikes),
                "mean_spike_rate_per_hidden_tick": float(np.mean(fixed_spikes)),
                "first_tick_gain_mean": 1.0,
                "mean_gain": 1.0, "min_gain_seen": 1.0, "max_gain_seen": 1.0,
                "fraction_gain_at_min": 0.0, "fraction_gain_at_max": 0.0,
                "feature_mean_std": float(np.std(task_outputs[("fixed_inhibition", scale)]["features"], axis=0).mean()),
            })

            # Local activity-adaptive inhibition; only input magnitude differs from nominal.
            local_inputs, local_spikes, diag = simulate_local(
                scaled, trials, weights, slope=LOCAL_SLOPE, target_rate=TARGET_RATE,
                tau=LOCAL_TAU, gain_min=GAIN_MIN, gain_max=GAIN_MAX, return_diagnostics=True,
            )
            if not np.array_equal(local_inputs, fixed_inputs):
                raise RuntimeError("fixed and local conditions received different event rasters")
            if not np.isfinite(local_spikes).all():
                raise RuntimeError("local adaptive condition produced non-finite spikes")
            if not np.isclose(float(diag["first_tick_gain_mean"]), 1.0, atol=1e-7):
                raise RuntimeError("local condition initial inhibitory gain must equal 1.0")
            feats = extract_features(local_spikes)
            task_outputs[("local_adaptive_inhibition", scale)] = {"spikes": local_spikes.copy(), "features": feats}
            rows.append({
                "seed": SEED, "task": task, "condition": "local_adaptive_inhibition",
                "input_scale": scale, "n_trials": len(trials),
                "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
                "spectral_radius": radius, "mean_spikes_per_trial": _mean_spikes(local_spikes),
                "mean_spike_rate_per_hidden_tick": float(np.mean(local_spikes)),
                "first_tick_gain_mean": float(diag["first_tick_gain_mean"]),
                "mean_gain": float(diag["mean_gain"]),
                "min_gain_seen": float(diag["min_gain_seen"]),
                "max_gain_seen": float(diag["max_gain_seen"]),
                "fraction_gain_at_min": float(diag["fraction_at_min"]),
                "fraction_gain_at_max": float(diag["fraction_at_max"]),
                "feature_mean_std": float(np.std(feats, axis=0).mean()),
            })

        # Quantify change from nominal input scale (1.0), separately for the two conditions.
        for condition in CONDITIONS:
            nominal = task_outputs[(condition, 1.0)]
            for row in rows:
                if row["task"] != task or row["condition"] != condition:
                    continue
                scale = float(row["input_scale"])
                current = task_outputs[(condition, scale)]
                nominal_count = _mean_spikes(nominal["spikes"])
                current_count = _mean_spikes(current["spikes"])
                row["relative_spike_count_delta_vs_nominal"] = float(
                    (current_count - nominal_count) / max(nominal_count, 1e-12)
                )
                row["spike_pattern_disagreement_vs_nominal"] = float(
                    np.mean(current["spikes"] != nominal["spikes"])
                )
                row["feature_relative_l2_delta_vs_nominal"] = float(
                    np.linalg.norm(current["features"] - nominal["features"])
                    / max(float(np.linalg.norm(nominal["features"])), 1e-12)
                )

        integrity["tasks"][task] = {
            "n_trials": len(trials),
            "class_balance": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
            "input_scales": list(INPUT_SCALES),
            "same_event_raster_across_scales": True,
            "training_example_keys_sha256": hashlib.sha256(
                "\n".join(sorted(repr(v35.trial_key(t)) for t in all_trials[:v35.N_TRAIN])).encode("utf-8")
            ).hexdigest(),
            "selected_diagnostic_trial_keys_sha256": hashlib.sha256(
                "\n".join(sorted(repr(v35.trial_key(t)) for t in trials)).encode("utf-8")
            ).hexdigest(),
            "input_raster_sha256": array_sha256(input_raster_reference),
            "train_validation_overlap": split_diag["train_validation_overlap"],
            "train_test_overlap": split_diag["train_test_overlap"],
            "validation_test_overlap": split_diag["validation_test_overlap"],
            "nominal_spike_counts_by_condition": {
                condition: int(task_outputs[(condition, 1.0)]["spikes"].sum()) for condition in CONDITIONS
            },
        }

    if not np.array_equal(base_reservoir.w_in, original_w_in):
        raise RuntimeError("source reservoir was mutated during diagnostic")
    integrity["original_reservoir_input_weights_unchanged"] = True
    payload = {
        "title": "Nucleus Pulse v0.40 Stage 1: input-gain component diagnostic",
        "stage": "component diagnostic only; one reserved seed; no readout training; not a tuning result",
        "seed": SEED,
        "tasks": list(TASKS),
        "conditions": list(CONDITIONS),
        "input_scales": list(INPUT_SCALES),
        "n_per_class": N_PER_CLASS,
        "n_hidden": N_HIDDEN,
        "spectral_radius": radius,
        "local_slope": LOCAL_SLOPE,
        "local_ema_tau_ticks": LOCAL_TAU,
        "gain_bounds": [GAIN_MIN, GAIN_MAX],
        "rows": rows,
        "integrity": integrity,
        "interpretation_limit": "This diagnostic measures changes in firing activity/features under input amplitude scaling; it makes no learning-accuracy or energy-efficiency claim.",
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    fields = [key for key, value in rows[0].items() if not isinstance(value, (dict, list))]
    with (OUTDIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Summary rows describe how spike count responds across input scales; no accuracy metric is computed.
    summary_rows = []
    for task in TASKS:
        for condition in CONDITIONS:
            by_scale = {float(r["input_scale"]): r for r in rows if r["task"] == task and r["condition"] == condition}
            nominal = float(by_scale[1.0]["mean_spikes_per_trial"])
            scale_deviation = [abs(float(by_scale[s]["mean_spikes_per_trial"]) / max(nominal, 1e-12) - 1.0) for s in INPUT_SCALES]
            summary_rows.append({
                "task": task, "condition": condition,
                "nominal_mean_spikes_per_trial": nominal,
                "scale_075_mean_spikes_per_trial": float(by_scale[0.75]["mean_spikes_per_trial"]),
                "scale_125_mean_spikes_per_trial": float(by_scale[1.25]["mean_spikes_per_trial"]),
                "max_absolute_relative_spike_count_change_vs_nominal": float(max(scale_deviation)),
                "mean_gain_at_scale_075": float(by_scale[0.75]["mean_gain"]),
                "mean_gain_at_scale_100": float(by_scale[1.0]["mean_gain"]),
                "mean_gain_at_scale_125": float(by_scale[1.25]["mean_gain"]),
            })
    (OUTDIR / "summary.json").write_text(json.dumps(summary_rows, indent=2, allow_nan=False), encoding="utf-8")
    with (OUTDIR / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    # Separate small plots for each task; these are descriptive activity charts only.
    import matplotlib.pyplot as plt
    for task in TASKS:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for condition in CONDITIONS:
            subset = [r for r in rows if r["task"] == task and r["condition"] == condition]
            subset.sort(key=lambda r: float(r["input_scale"]))
            ax.plot(
                [float(r["input_scale"]) for r in subset],
                [float(r["mean_spikes_per_trial"]) for r in subset],
                marker="o", label=condition,
            )
        ax.set_xlabel("Input weight scale")
        ax.set_ylabel("Mean spikes per trial")
        ax.set_title(f"Nucleus Pulse v0.40 — {task} input-gain response")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTDIR / f"{task}_input_gain_response.png", dpi=160)
        plt.close(fig)

    report = [
        "# Nucleus Pulse v0.40 — المرحلة الأولى: فحص تأثير قوة المدخل", "",
        "## هدف الفحص", "",
        "نختبر مكوّنًا واحدًا فقط: هل يتغير نشاط الخزان مع تغيير مقدار قوة المدخل، وهل يختلف رد الفعل بين التثبيط الثابت والتثبيط المحلي التكيفي؟ لا يوجد تدريب لقارئ أو ادعاء عن دقة التعلّم.", "",
        "## الإعداد المثبّت", "",
        f"- بذرة تشخيص واحدة محجوزة: {SEED}.",
        f"- مهمتان: {', '.join(TASKS)}؛ {N_PER_CLASS} مثالًا لكل فئة في كل مهمة.",
        f"- مضاعفات أوزان المدخل: {list(INPUT_SCALES)}؛ جميع أحداث الإدخال وتوقيتها نفسها.",
        f"- التثبيط المحلي ثابت عند الميل {LOCAL_SLOPE} وtau={LOCAL_TAU}، لأن v0.39 لم يقدم دليلًا على تفوق tau=8.",
        "- مصفوفة الخزان والعتبات والتسريبات ثابتة؛ المتغير التشخيصي فقط هو مقياس أوزان الإدخال.",
        "- لا تدريب لقارئ ولا استخدام لمجموعة اختبار نهائي في هذا الفحص.", "",
        "## النتائج", "",
        "| المهمة | الشرط | نبضات/مثال عند 0.75× | عند 1× | عند 1.25× | أكبر تغير نسبي عن 1× |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summary_rows:
        report.append(
            f"| {r['task']} | {r['condition']} | {r['scale_075_mean_spikes_per_trial']:.2f} | "
            f"{r['nominal_mean_spikes_per_trial']:.2f} | {r['scale_125_mean_spikes_per_trial']:.2f} | "
            f"{100*r['max_absolute_relative_spike_count_change_vs_nominal']:.2f}% |"
        )
    report += [
        "", "## حدود الاستنتاج", "",
        "هذا فحص مكوّن صغير على بذرة واحدة و32 مثالًا متوازنًا لكل مهمة. يبين فقط كيف يتغير نشاط النبضات والميزات عند تغيير قوة المدخل. لا يثبت أن التثبيط التكيفي أكثر متانة، ولا يثبت تحسن التعلّم أو توفير الطاقة. لو اجتاز مراجعة المكوّن، فالخطوة التالية تكون تطويرًا متعدد البذور لاختيار مقياس اختبار مناسب، مع بقاء الاختبار النهائي مغلقًا.", "",
    ]
    (OUTDIR / "report_ar.md").write_text("\n".join(report), encoding="utf-8")

    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "passed", "version": VERSION, "seed": SEED,
                      "rows": len(result["rows"]), "tasks": list(result["tasks"]),
                      "conditions": list(result["conditions"])}, indent=2))
