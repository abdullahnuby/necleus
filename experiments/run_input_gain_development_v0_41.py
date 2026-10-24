"""Nucleus Pulse v0.41 Stage 2: multi-seed input-gain component diagnostic.

Development-only component diagnostic. It uses only samples taken from the
training pool, with no readout fitting, no validation/test set generation, and
no accuracy-based selection. It repeats the v0.40 input-amplitude sensitivity
check across eight independent development seeds.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir  # noqa: E402
from experiments import run_activity_adaptive_inhibition_v0_35 as v35  # noqa: E402
from experiments.diagnose_input_gain_v0_40 import scale_input_weights, balanced_train_slice  # noqa: E402
from experiments.tune_local_inhibition_v0_37 import simulate_local  # noqa: E402
from experiments.run_inhibition_structure_v0_34 import (  # noqa: E402
    extract_features,
    simulate_batch_with_recurrent_weights,
    spectral_radius,
)
from experiments.run_learning_curve_v0_30 import (  # noqa: E402
    N_HIDDEN,
    N_TRAIN_POOL,
    make_epoch_labels,
    make_nested_training_pool,
    trial_key,
)

VERSION = "0.41-stage2-input-gain-development"
DEV_SEEDS = [14111, 14129, 14147, 14165, 14183, 14201, 14219, 14237]
TASKS = ("sync", "order")
INPUT_SCALES = (0.75, 1.0, 1.25)
CONDITIONS = ("fixed_inhibition", "local_adaptive_inhibition")
N_PER_CLASS = 16
LOCAL_SLOPE = 2.0
LOCAL_TAU = 4.0
TARGET_RADIUS = v35.TARGET_SPECTRAL_RADIUS
TARGET_RATE = v35.TARGET_ACTIVITY_RATE
GAIN_MIN, GAIN_MAX = v35.INHIBITORY_GAIN_MIN, v35.INHIBITORY_GAIN_MAX
BOOTSTRAP_REPS = 20000
BOOTSTRAP_SEED = 20411041
OUTDIR = ROOT / "results" / "input_gain_v0_41" / "development_diagnostic"
PREREG = ROOT / "results" / "input_gain_v0_41" / "preregistration.json"
SOURCE_FILES = (
    "experiments/run_input_gain_development_v0_41.py",
    "tests/test_input_gain_development_v0_41.py",
    "docs/benchmark_protocol_v0_41_STAGE2_AR.md",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def verify_preregistration() -> dict[str, Any]:
    if not PREREG.exists():
        raise RuntimeError("Missing preregistration.json; the experiment cannot start")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    expected = {
        "development_seeds": DEV_SEEDS,
        "tasks": list(TASKS),
        "input_scales": list(INPUT_SCALES),
        "conditions": list(CONDITIONS),
        "n_per_class": N_PER_CLASS,
        "local_slope": LOCAL_SLOPE,
        "local_tau": LOCAL_TAU,
        "target_rate": TARGET_RATE,
        "gain_bounds": [GAIN_MIN, GAIN_MAX],
        "target_spectral_radius": TARGET_RADIUS,
        "n_hidden": N_HIDDEN,
        "reader_training": False,
        "validation_or_test_sets_used": False,
    }
    if prereg.get("version") != VERSION or prereg.get("config") != expected:
        raise RuntimeError("Experiment settings differ from preregistered configuration")
    expected_hashes = prereg.get("sha256", {})
    for relative in SOURCE_FILES:
        actual = sha256_file(ROOT / relative)
        if expected_hashes.get(relative) != actual:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return prereg


def mean_ci(values: list[float] | np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2 or not np.isfinite(arr).all():
        raise ValueError("mean_ci needs at least two finite scalar observations")
    mean = float(arr.mean())
    se = float(stats.sem(arr))
    half = float(stats.t.ppf((1 + confidence) / 2, df=arr.size - 1) * se)
    return mean, mean - half, mean + half


def relative_delta(value: float, nominal: float) -> float:
    return float((value - nominal) / max(abs(nominal), 1e-12))


def make_diagnostics(spikes: np.ndarray, features: np.ndarray) -> dict[str, float]:
    return {
        "mean_spikes_per_trial": float(np.asarray(spikes, dtype=np.float64).sum(axis=(1, 2)).mean()),
        "mean_spike_rate_per_hidden_tick": float(np.asarray(spikes, dtype=np.float64).mean()),
        "feature_std_mean": float(np.std(features, axis=0).mean()),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metric_names = (
        "mean_spikes_per_trial", "mean_spike_rate_per_hidden_tick", "feature_std_mean",
        "relative_spike_count_delta_vs_nominal", "spike_pattern_disagreement_vs_nominal",
        "feature_relative_l2_delta_vs_nominal", "fraction_gain_at_min", "fraction_gain_at_max",
    )
    for task in TASKS:
        for condition in CONDITIONS:
            for scale in INPUT_SCALES:
                subset = [r for r in rows if r["task"] == task and r["condition"] == condition and r["input_scale"] == scale]
                out: dict[str, Any] = {"task": task, "condition": condition, "input_scale": scale, "n_seeds": len(subset)}
                for metric in metric_names:
                    values = [float(r[metric]) for r in subset]
                    mean, low, high = mean_ci(values)
                    out[f"{metric}_mean"] = mean
                    out[f"{metric}_ci95_low"] = low
                    out[f"{metric}_ci95_high"] = high
                output.append(out)
    return output


def _sensitivity_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for task in TASKS:
        for condition in CONDITIONS:
            for seed in DEV_SEEDS:
                trio = {float(r["input_scale"]): r for r in rows if r["task"] == task and r["condition"] == condition and int(r["seed"]) == seed}
                low = float(trio[0.75]["relative_spike_count_delta_vs_nominal"])
                high = float(trio[1.25]["relative_spike_count_delta_vs_nominal"])
                abs_low, abs_high = abs(low), abs(high)
                per_seed.append({
                    "seed": seed, "task": task, "condition": condition,
                    "low_scale_relative_spike_delta": low,
                    "high_scale_relative_spike_delta": high,
                    "max_absolute_spike_sensitivity": max(abs_low, abs_high),
                    "high_minus_low_spike_delta": high - low,
                    "max_absolute_feature_l2_delta": max(
                        abs(float(trio[0.75]["feature_relative_l2_delta_vs_nominal"])),
                        abs(float(trio[1.25]["feature_relative_l2_delta_vs_nominal"])),
                    ),
                })
            subset = [r for r in per_seed if r["task"] == task and r["condition"] == condition]
            for metric in ("max_absolute_spike_sensitivity", "high_minus_low_spike_delta", "max_absolute_feature_l2_delta"):
                vals = [float(r[metric]) for r in subset]
                m, lo, hi = mean_ci(vals)
                summary.append({"task": task, "condition": condition, "metric": metric, "mean": m,
                                "ci95_low": lo, "ci95_high": hi, "n_seed_pairs": len(vals)})
    # Paired local-minus-fixed sensitivity comparisons on each seed-task pair.
    paired: list[dict[str, Any]] = []
    for task in TASKS:
        for seed in DEV_SEEDS:
            fixed = next(r for r in per_seed if r["seed"] == seed and r["task"] == task and r["condition"] == "fixed_inhibition")
            local = next(r for r in per_seed if r["seed"] == seed and r["task"] == task and r["condition"] == "local_adaptive_inhibition")
            paired.append({"seed": seed, "task": task,
                           "fixed_max_abs_spike_sensitivity": fixed["max_absolute_spike_sensitivity"],
                           "local_max_abs_spike_sensitivity": local["max_absolute_spike_sensitivity"],
                           "local_minus_fixed_max_abs_spike_sensitivity": local["max_absolute_spike_sensitivity"] - fixed["max_absolute_spike_sensitivity"],
                           "fixed_max_abs_feature_l2_delta": fixed["max_absolute_feature_l2_delta"],
                           "local_max_abs_feature_l2_delta": local["max_absolute_feature_l2_delta"],
                           "local_minus_fixed_max_abs_feature_l2_delta": local["max_absolute_feature_l2_delta"] - fixed["max_absolute_feature_l2_delta"]})
    return per_seed, summary, paired


def _plot(summary: list[dict[str, Any]], task: str, metric: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    for condition, label in (("fixed_inhibition", "Fixed inhibition"), ("local_adaptive_inhibition", "Local adaptive inhibition")):
        sub = sorted([r for r in summary if r["task"] == task and r["condition"] == condition], key=lambda r: r["input_scale"])
        xs = [float(r["input_scale"]) for r in sub]
        ys = [float(r[f"{metric}_mean"]) for r in sub]
        lows = [float(r[f"{metric}_ci95_low"]) for r in sub]
        highs = [float(r[f"{metric}_ci95_high"]) for r in sub]
        ax.plot(xs, ys, marker="o", label=label)
        ax.fill_between(xs, lows, highs, alpha=0.16)
    ax.set_xlabel("Input-weight scale")
    ax.set_ylabel(ylabel)
    ax.set_title(f"v0.41 development diagnostic: {task}")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run() -> dict[str, Any]:
    prereg = verify_preregistration()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    integrity: dict[str, Any] = {
        "version": VERSION,
        "development_seeds": DEV_SEEDS,
        "tasks": list(TASKS),
        "input_scales": list(INPUT_SCALES),
        "conditions": list(CONDITIONS),
        "n_per_class": N_PER_CLASS,
        "readout_training_performed": False,
        "validation_sets_generated_or_used": False,
        "test_sets_generated_or_used": False,
        "condition_recurrent_matrix_equal_within_seed": True,
        "input_raster_equal_across_scales_and_conditions": True,
        "source_weights_mutated": False,
        "seeds": {},
    }
    for seed_idx, seed in enumerate(DEV_SEEDS):
        reservoir = DynamicPulseReservoir(seed=int(seed), n_hidden=N_HIDDEN)
        original_w_in = np.asarray(reservoir.w_in).copy()
        matrices, matrix_diag = v35.make_shared_dale_matrix(reservoir, int(seed))
        weights = np.asarray(matrices["dale_fixed_25"], dtype=np.float32).copy()
        radius = float(spectral_radius(weights))
        if not np.isclose(radius, TARGET_RADIUS, atol=5e-4):
            raise RuntimeError(f"unexpected recurrent spectral radius for seed {seed}: {radius}")
        seed_integrity: dict[str, Any] = {
            "recurrent_matrix_sha256": array_sha256(weights),
            "recurrent_spectral_radius": radius,
            "tasks": {},
        }
        for task_idx, task in enumerate(TASKS):
            # Use only the generated training pool; do not create validation or test sets.
            pool_seed = int(seed + 11111 + task_idx * 100)
            train_pool = make_nested_training_pool(pool_seed, task, N_TRAIN_POOL)
            labels = make_epoch_labels(train_pool)
            trials, y = balanced_train_slice(train_pool, {"train": labels}, N_PER_CLASS)
            selected_keys = [repr(trial_key(t)) for t in trials]
            task_rows: dict[tuple[str, float], dict[str, Any]] = {}
            reference_inputs: np.ndarray | None = None
            input_hash_by_scale: dict[str, str] = {}
            for scale in INPUT_SCALES:
                scaled = scale_input_weights(reservoir, float(scale))
                if not np.array_equal(reservoir.w_in, original_w_in):
                    raise RuntimeError("input-scaling helper mutated the original reservoir")
                inputs_ref = None
                for condition in CONDITIONS:
                    if condition == "fixed_inhibition":
                        inputs, spikes = simulate_batch_with_recurrent_weights(scaled, trials, weights)
                        diag = {
                            "first_tick_gain_mean": 1.0, "mean_gain": 1.0,
                            "min_gain_seen": 1.0, "max_gain_seen": 1.0,
                            "fraction_at_min": 0.0, "fraction_at_max": 0.0,
                        }
                    else:
                        inputs, spikes, diag = simulate_local(
                            scaled, trials, weights, slope=LOCAL_SLOPE,
                            target_rate=TARGET_RATE, tau=LOCAL_TAU,
                            gain_min=GAIN_MIN, gain_max=GAIN_MAX,
                            return_diagnostics=True,
                        )
                    if not np.isfinite(spikes).all():
                        raise RuntimeError("non-finite spike raster")
                    if inputs_ref is None:
                        inputs_ref = inputs.copy()
                    elif not np.array_equal(inputs_ref, inputs):
                        raise RuntimeError("condition event raster differs for same scale")
                    if reference_inputs is None:
                        reference_inputs = inputs.copy()
                    elif not np.array_equal(reference_inputs, inputs):
                        raise RuntimeError("event input raster changed across input scales")
                    if condition == "local_adaptive_inhibition" and not np.isclose(diag["first_tick_gain_mean"], 1.0, atol=1e-7):
                        raise RuntimeError("local adaptive inhibition does not start at gain 1.0")
                    feats = extract_features(spikes)
                    metrics = make_diagnostics(spikes, feats)
                    task_rows[(condition, float(scale))] = {"spikes": spikes.copy(), "features": feats.copy(), **metrics, **diag}
                input_hash_by_scale[str(scale)] = array_sha256(reference_inputs)
            nominal = {condition: task_rows[(condition, 1.0)] for condition in CONDITIONS}
            for condition in CONDITIONS:
                nominal_spikes = nominal[condition]["spikes"]
                nominal_features = nominal[condition]["features"]
                nominal_count = float(nominal[condition]["mean_spikes_per_trial"])
                for scale in INPUT_SCALES:
                    cell = task_rows[(condition, float(scale))]
                    row = {
                        "seed": seed, "task": task, "condition": condition,
                        "input_scale": float(scale), "n_trials": len(trials),
                        "class_0": int(np.sum(y == 0)), "class_1": int(np.sum(y == 1)),
                        "recurrent_spectral_radius": radius,
                        "mean_spikes_per_trial": float(cell["mean_spikes_per_trial"]),
                        "mean_spike_rate_per_hidden_tick": float(cell["mean_spike_rate_per_hidden_tick"]),
                        "feature_std_mean": float(cell["feature_std_mean"]),
                        "first_tick_gain_mean": float(cell["first_tick_gain_mean"]),
                        "mean_gain": float(cell["mean_gain"]),
                        "min_gain_seen": float(cell["min_gain_seen"]),
                        "max_gain_seen": float(cell["max_gain_seen"]),
                        "fraction_gain_at_min": float(cell["fraction_at_min"]),
                        "fraction_gain_at_max": float(cell["fraction_at_max"]),
                        "relative_spike_count_delta_vs_nominal": relative_delta(float(cell["mean_spikes_per_trial"]), nominal_count),
                        "spike_pattern_disagreement_vs_nominal": float(np.mean(cell["spikes"] != nominal_spikes)),
                        "feature_relative_l2_delta_vs_nominal": float(np.linalg.norm(cell["features"] - nominal_features) / max(float(np.linalg.norm(nominal_features)), 1e-12)),
                    }
                    rows.append(row)
            seed_integrity["tasks"][task] = {
                "pool_seed": pool_seed,
                "n_training_pool_examples": len(train_pool),
                "n_selected_examples": len(trials),
                "class_balance": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))},
                "selected_trial_keys_sha256": hashlib.sha256("\n".join(sorted(selected_keys)).encode()).hexdigest(),
                "input_raster_sha256_by_scale": input_hash_by_scale,
                "no_validation_or_test_examples_generated": True,
            }
        if not np.array_equal(reservoir.w_in, original_w_in):
            raise RuntimeError("source reservoir input weights were mutated")
        integrity["seeds"][str(seed)] = seed_integrity
        # Save the raw partial CSV after each seed so a failed session cannot discard completed work.
        current = [r for r in rows if int(r["seed"]) in DEV_SEEDS[:seed_idx + 1]]
        write_csv(OUTDIR / "per_seed_input_gain.csv", current)
        print(f"development seed {seed} complete ({seed_idx + 1}/{len(DEV_SEEDS)})", flush=True)

    summary = _summary_rows(rows)
    sensitivity, sensitivity_summary, paired = _sensitivity_rows(rows)
    write_csv(OUTDIR / "input_gain_summary.csv", summary)
    write_csv(OUTDIR / "per_seed_sensitivity.csv", sensitivity)
    write_csv(OUTDIR / "sensitivity_summary.csv", sensitivity_summary)
    write_csv(OUTDIR / "paired_sensitivity_differences.csv", paired)
    for task in TASKS:
        _plot(summary, task, "mean_spikes_per_trial", "Mean spikes per trial", OUTDIR / f"{task}_input_gain_response.png")
        _plot(summary, task, "feature_relative_l2_delta_vs_nominal", "Relative feature L2 delta vs nominal", OUTDIR / f"{task}_feature_sensitivity.png")

    # Paired bootstrap intervals across development seeds, kept separate from t intervals.
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap: dict[str, Any] = {}
    for task in TASKS:
        subset = [r for r in paired if r["task"] == task]
        for metric in ("local_minus_fixed_max_abs_spike_sensitivity", "local_minus_fixed_max_abs_feature_l2_delta"):
            values = np.asarray([r[metric] for r in subset], dtype=np.float64)
            indices = rng.integers(0, len(values), size=(BOOTSTRAP_REPS, len(values)))
            draws = values[indices].mean(axis=1)
            bootstrap[f"{task}:{metric}"] = {
                "mean": float(values.mean()),
                "percentile_ci95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                "n_seed_pairs": len(values),
                "bootstrap_reps": BOOTSTRAP_REPS,
            }
    payload = {
        "version": VERSION,
        "stage": "multi-seed development component diagnostic only",
        "config": {
            "development_seeds": DEV_SEEDS, "tasks": list(TASKS), "input_scales": list(INPUT_SCALES),
            "conditions": list(CONDITIONS), "n_per_class": N_PER_CLASS,
            "local_slope": LOCAL_SLOPE, "local_tau": LOCAL_TAU,
            "target_rate": TARGET_RATE, "gain_bounds": [GAIN_MIN, GAIN_MAX],
            "target_spectral_radius": TARGET_RADIUS, "n_hidden": N_HIDDEN,
        },
        "n_rows": len(rows), "rows": rows, "summary": summary,
        "per_seed_sensitivity": sensitivity, "sensitivity_summary": sensitivity_summary,
        "paired_sensitivity_differences": paired, "paired_bootstrap": bootstrap,
        "integrity": integrity,
        "readout_training_performed": False,
        "validation_or_test_sets_generated_or_used": False,
        "preregistration_sha256": sha256_file(PREREG),
        "preregistered_source_sha256": prereg["sha256"],
        "interpretation_limit": "This is a development component diagnostic of activity/feature sensitivity only. No reader is trained, and it makes no accuracy, robustness-generalization, or energy-efficiency claim.",
    }
    (OUTDIR / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report = [
        "# Nucleus Pulse v0.41 — تقرير تشخيص قوة المدخل عبر بذور تطوير",
        "",
        "## التصميم",
        f"- بذور التطوير: {len(DEV_SEEDS)} ({DEV_SEEDS[0]}–{DEV_SEEDS[-1]}).",
        f"- سجلات القياس: {len(rows)} (8 بذور × مهمتين × 3 مستويات إدخال × شرطين).",
        "- لكل بذرة/مهمة: 32 مثالًا متوازنًا مأخوذًا من training pool فقط، 16 من كل فئة.",
        "- مستويات الإدخال: 0.75× و1.0× و1.25×. الشرطان: تثبيط ثابت وتثبيط محلي تكيفي.",
        f"- التثبيط المحلي: slope={LOCAL_SLOPE}، tau={LOCAL_TAU}، حدود الكسب [{GAIN_MIN},{GAIN_MAX}]، هدف النشاط {TARGET_RATE}.",
        "- لم يتم إنشاء أو استخدام مجموعات تحقق أو اختبار، ولم يتم تدريب قارئ.",
        "",
        "## النتائج الأساسية",
        "متوسط النشاط وفواصل الثقة عبر البذور موجود في `input_gain_summary.csv`. حساسية كل بذرة موجودة في `per_seed_sensitivity.csv`، والمقارنة الزوجية في `paired_sensitivity_differences.csv`.",
        "",
        "## حدود التفسير",
        "هذه التجربة لا تقيس دقة تعلم أو متانة تعميم. هي تشخيص مكوّن يختبر كيف يتغير النشاط والميزات عند تعديل مقياس قوة الإشارة الداخلة، على مجموعة من بذور التطوير. لا يجوز اعتبارها اختبارًا رسميًا أو دليلًا على تفوق أي شرط.",
        "",
        "## سلامة البيانات",
        "- جميع الشروط تستخدم مصفوفة الخزان نفسها داخل البذرة، وتسلسلات أحداث الإدخال نفسها بين مقاييس قوة الإدخال.",
        "- أوزان المدخل الأصلية لا تتغير؛ كل مقياس يعمل على نسخة مستقلة.",
        "- سجل النزاهة التفصيلي ومفاتيح الأمثلة وبصمات المصفوفات موجودة في `results.json`.",
    ]
    (OUTDIR / "report_ar.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTDIR / "run_summary.txt").write_text(
        f"version={VERSION}\nrows={len(rows)}\nseeds={len(DEV_SEEDS)}\ntasks={len(TASKS)}\n"
        f"input_scales={','.join(map(str, INPUT_SCALES))}\nconditions={','.join(CONDITIONS)}\n"
        "reader_training=False\nvalidation_or_test_sets_generated_or_used=False\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(json.dumps({"version": result["version"], "n_rows": result["n_rows"], "output_dir": str(OUTDIR)}, indent=2))
