"""Nucleus Pulse v0.36: local per-cell inhibitory gain vs global/fixed controls.

The local rule scales inhibitory current into each postsynaptic cell using that
cell's own recent firing-rate EMA. Its slope is selected on separate development
seeds using validation balanced accuracy only, then frozen before official tests.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments import run_activity_adaptive_inhibition_v0_35 as v35
from experiments.tune_local_inhibition_v0_36 import simulate_local
from experiments.run_inhibition_structure_v0_34 import (
    simulate_batch_with_recurrent_weights, extract_features, spectral_radius,
)
from experiments.run_learning_curve_v0_30 import N_HIDDEN

VERSION = "0.36"
SEEDS = list(range(10000, 10040))
TASKS = list(v35.TASKS)
CONDITIONS = ["dale_fixed_25", "dale_global_adaptive_25", "dale_local_adaptive_25"]
EPOCH_CHECKPOINTS = list(v35.EPOCH_CHECKPOINTS)
PRIMARY_EPOCH = 10
N_TRAIN, N_VALIDATION, N_TEST = v35.N_TRAIN, v35.N_VALIDATION, v35.N_TEST
N_WINDOWS = 16
FEATURE_SET = "spike_bins16_rate_384"
TARGET_SPECTRAL_RADIUS = v35.TARGET_SPECTRAL_RADIUS
TARGET_ACTIVITY_RATE = v35.TARGET_ACTIVITY_RATE
TRACE_TAU_TICKS = v35.ACTIVITY_TRACE_TAU_TICKS
GAIN_MIN, GAIN_MAX = v35.INHIBITORY_GAIN_MIN, v35.INHIBITORY_GAIN_MAX
GLOBAL_SLOPE = v35.INHIBITORY_GAIN_SLOPE
# Chosen using development seeds 9100-9105 and validation balanced accuracy only.
LOCAL_SLOPE = 6.0
DEV_SEEDS = list(range(9100, 9106))
DEV_CANDIDATES = [1.0, 2.0, 4.0, 6.0]
BOOTSTRAP_REPS = v35.BOOTSTRAP_REPS
BOOTSTRAP_SEED = 20361024
TARGET_VALIDATION_ACCURACY = v35.TARGET_VALIDATION_ACCURACY
OUTDIR = ROOT / "results" / "local_inhibition_v0_36" / "official_run"
PREREG = ROOT / "results" / "local_inhibition_v0_36" / "preregistration.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol_config() -> dict:
    selected_path = ROOT / "results" / "local_inhibition_v0_36" / "development_tuning" / "selected_slope.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8")) if selected_path.exists() else {}
    return {
        "version": VERSION,
        "official_seeds": SEEDS,
        "development_seeds": DEV_SEEDS,
        "development_candidate_slopes": DEV_CANDIDATES,
        "selected_local_slope": LOCAL_SLOPE,
        "slope_selection": "maximize mean validation balanced accuracy at epoch 10 across 6 development seeds × 2 tasks; ties choose smaller slope",
        "selection_result_slope": selected.get("selected_slope"),
        "conditions": CONDITIONS,
        "tasks": TASKS,
        "unique_training_examples_per_task_seed": N_TRAIN,
        "validation_examples_per_task_seed": N_VALIDATION,
        "final_test_examples_per_task_seed": N_TEST,
        "epochs": EPOCH_CHECKPOINTS,
        "primary_epoch": PRIMARY_EPOCH,
        "n_hidden": N_HIDDEN,
        "feature_set": FEATURE_SET,
        "same_matrix_and_topology_across_conditions": True,
        "base_spectral_radius_target": TARGET_SPECTRAL_RADIUS,
        "inhibitory_sources": "25% fixed Dale-style inhibitory presynaptic source neurons; same exact signed matrix across all conditions",
        "fixed_condition": "gain 1.0 for all postsynaptic cells and ticks",
        "global_condition": {
            "rule": "gain=clip(1+4*(EMA_population_rate-0.13), 0.5, 2.0)",
            "ema_tau_ticks": TRACE_TAU_TICKS,
        },
        "local_condition": {
            "rule": "gain_i=clip(1+selected_slope*(EMA_rate_i-0.13),0.5,2.0); only inhibitory current into postsynaptic cell i is scaled",
            "selected_slope": LOCAL_SLOPE,
            "ema_tau_ticks": TRACE_TAU_TICKS,
            "target_rate_per_cell_per_tick": TARGET_ACTIVITY_RATE,
            "initial_trace": "target rate, so initial gain is exactly 1.0",
            "bounds": [GAIN_MIN, GAIN_MAX],
        },
        "reader": {
            "class": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss", "penalty": "l2", "alpha": 1e-4,
            "learning_rate": "constant", "eta0": 0.01, "average": True,
            "shuffle": True, "fit_method": "partial_fit",
            "same_model_seed_paired_across_conditions": True,
        },
        "same_input_weights_thresholds_leaks_refractory_mask_training_order_and_features": True,
        "validation_for_threshold_analysis_only": True,
        "final_test_used_for_tuning_or_selection": False,
        "paired_bootstrap_unit": "seed",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def freeze_preregistration() -> dict:
    selected_path = ROOT / "results" / "local_inhibition_v0_36" / "development_tuning" / "selected_slope.json"
    if not selected_path.exists():
        raise RuntimeError("Run development tuning first; selected_slope.json is missing.")
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if float(selected.get("selected_slope", -1)) != LOCAL_SLOPE:
        raise RuntimeError(f"Code slope {LOCAL_SLOPE} does not match development-selected slope {selected.get('selected_slope')}")
    source_files = [
        "experiments/run_local_inhibition_v0_36.py",
        "experiments/tune_local_inhibition_v0_36.py",
        "experiments/run_activity_adaptive_inhibition_v0_35.py",
        "experiments/run_inhibition_structure_v0_34.py",
        "experiments/run_learning_curve_v0_30.py",
        "pulse_core/sequential_stdp.py",
        "results/local_inhibition_v0_36/development_tuning/selected_slope.json",
        "results/local_inhibition_v0_36/development_tuning/dev_tuning_results.json",
        "results/local_inhibition_v0_36/development_tuning/dev_tuning_results.csv",
        "docs/benchmark_protocol_v0_36_AR.md",
        "tests/test_local_inhibition_v0_36.py",
        "experiments/run_incremental_windows_v0_31.py",
        "pulse_core/tasks.py",
    ]
    doc = {
        "title": "Nucleus Pulse v0.36 local per-cell inhibitory gain — preregistration",
        "config": protocol_config(),
        "sha256": {rel: sha256_file(ROOT / rel) for rel in source_files},
    }
    PREREG.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    return doc


def verify_preregistration() -> dict:
    if not PREREG.exists():
        raise RuntimeError("Missing v0.36 preregistration. Freeze protocol before official run.")
    doc = json.loads(PREREG.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Current v0.36 protocol differs from frozen preregistration.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {relative}")
    return doc


def make_shared_dale_matrix(reservoir: DynamicPulseReservoir, seed: int):
    matrices, diagnostics = v35.make_shared_dale_matrix(reservoir, seed)
    base = matrices["dale_fixed_25"].copy()
    same = {name: base.copy() for name in CONDITIONS}
    if any(not np.array_equal(same[CONDITIONS[0]], same[c]) for c in CONDITIONS[1:]):
        raise RuntimeError("All v0.36 conditions must receive the exact same recurrent matrix")
    if not np.isclose(spectral_radius(base), TARGET_SPECTRAL_RADIUS, atol=5e-4):
        raise RuntimeError("Unexpected base recurrent spectral radius")
    return same, diagnostics


def run_seed(seed: int) -> tuple[list[dict], dict]:
    started = time.time()
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, matrix_diag = make_shared_dale_matrix(reservoir, seed)
    rows: list[dict] = []
    integrity = {
        "seed": int(seed), "reservoir_seed": int(seed),
        "same_exact_recurrent_matrix_across_conditions": all(
            np.array_equal(matrices[CONDITIONS[0]], matrices[c]) for c in CONDITIONS[1:]),
        "same_input_weights_thresholds_leaks_refractory_mask_and_neuron_count": True,
        "base_spectral_radius": spectral_radius(matrices[CONDITIONS[0]]),
        "matrix_diagnostics": matrix_diag,
        "inhibitory_source_indices": matrix_diag["inhibitory_source_indices"],
        "tasks": {},
    }
    for task_i, task in enumerate(TASKS):
        concatenated, bounds, labels, split_integrity = v35.make_data(seed, task)
        features_by_condition: dict[str, dict[str, np.ndarray]] = {}
        activity = {}
        expected_inputs = None
        for condition in CONDITIONS:
            weights = matrices[condition]
            if condition == "dale_fixed_25":
                inputs, spikes = simulate_batch_with_recurrent_weights(reservoir, concatenated, weights)
                diag = {"gain_type": "fixed", "mean_gain": 1.0, "min_gain_seen": 1.0,
                        "max_gain_seen": 1.0, "first_tick_gain_mean": 1.0,
                        "fraction_at_min": 0.0, "fraction_at_max": 0.0,
                        "mean_within_trial_gain_std": 0.0}
            elif condition == "dale_global_adaptive_25":
                inputs, spikes, raw = v35.simulate_batch_activity_adaptive_inhibition(
                    reservoir, concatenated, weights, return_diagnostics=True)
                diag = {"gain_type": "global", "mean_gain": raw["mean_inhibitory_gain"],
                        "min_gain_seen": raw["min_inhibitory_gain"], "max_gain_seen": raw["max_inhibitory_gain"],
                        "first_tick_gain_mean": raw["first_tick_inhibitory_gain_mean"],
                        "fraction_at_min": raw["fraction_gain_at_min"],
                        "fraction_at_max": raw["fraction_gain_at_max"],
                        "mean_within_trial_gain_std": 0.0}
            else:
                inputs, spikes, diag = simulate_local(reservoir, concatenated, weights, LOCAL_SLOPE,
                                                      return_diagnostics=True)
            if expected_inputs is None:
                expected_inputs = inputs.copy()
            elif not np.array_equal(inputs, expected_inputs):
                raise RuntimeError("Input raster changed across v0.36 conditions")
            split_features = {split: extract_features(spikes[left:right]) for split, (left, right) in bounds.items()}
            features_by_condition[condition] = split_features
            total_per_trial = np.sum(spikes, axis=(1, 2))
            per_cell = np.sum(spikes, axis=(0, 1))
            activity[condition] = {
                **diag,
                "mean_spikes_per_trial": float(np.mean(total_per_trial)),
                "fraction_hidden_cells_with_at_least_one_spike": float(np.mean(per_cell > 0)),
                "fraction_trials_with_zero_hidden_spikes": float(np.mean(total_per_trial == 0)),
                "mean_spike_rate_per_hidden_tick": float(np.mean(spikes)),
                "mean_spikes_per_cell": float(np.mean(per_cell)),
            }
            del inputs, spikes
        integrity["tasks"][task] = {**split_integrity, "activity_diagnostics": activity}
        for condition in CONDITIONS:
            result_rows = v35.run_base_reader(features_by_condition[condition], labels, seed, task_i)
            for row in result_rows:
                row["condition"] = condition
            rows.extend(result_rows)
        del features_by_condition
    integrity["elapsed_seconds"] = round(time.time() - started, 3)
    return rows, integrity


def aggregate_records(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    metrics = ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate")
    for r in records:
        groups[(r["task"], r["condition"], int(r["epochs"]))].append(r)
    out = []
    for task, condition, epoch in sorted(groups):
        group = groups[(task, condition, epoch)]
        item = {"task": task, "condition": condition, "epochs": int(epoch),
                "n_seeds": len({r["seed"] for r in group}), "unique_training_examples_per_seed": N_TRAIN,
                "example_exposures_per_seed": N_TRAIN * int(epoch)}
        for split in ("validation", "test"):
            for metric in metrics:
                mean, low, high = v35.bootstrap_ci(
                    [r[f"{split}_{metric}"] for r in group],
                    v35.stable_seed("v036", task, condition, str(epoch), split, metric))
                item[f"{split}_{metric}_mean_percent"] = mean
                item[f"{split}_{metric}_ci95_low_percent"] = low
                item[f"{split}_{metric}_ci95_high_percent"] = high
        out.append(item)
    return out


def paired_delta_summary(records: Sequence[dict]) -> list[dict]:
    lookup = {(r["seed"], r["task"], r["condition"], int(r["epochs"])): r for r in records}
    seeds = sorted({int(r["seed"]) for r in records})
    contrasts = [
        ("dale_global_adaptive_25", "dale_fixed_25"),
        ("dale_local_adaptive_25", "dale_fixed_25"),
        ("dale_local_adaptive_25", "dale_global_adaptive_25"),
    ]
    out = []
    for task in TASKS:
        for epoch in EPOCH_CHECKPOINTS:
            for treatment, control in contrasts:
                deltas = np.asarray([
                    lookup[(seed, task, treatment, epoch)]["test_accuracy"]
                    - lookup[(seed, task, control, epoch)]["test_accuracy"] for seed in seeds
                ], dtype=float)
                mean, low, high = v35.bootstrap_ci(deltas, v35.stable_seed("v036-paired", task, treatment, control, str(epoch)))
                out.append({"task": task, "epochs": int(epoch), "contrast": f"{treatment} - {control}",
                            "n_paired_seeds": len(deltas), "mean_delta_percentage_points": mean,
                            "ci95_low_percentage_points": low, "ci95_high_percentage_points": high,
                            "positive_seed_deltas": int(np.sum(deltas > 0)),
                            "negative_seed_deltas": int(np.sum(deltas < 0)),
                            "zero_seed_deltas": int(np.sum(deltas == 0))})
    return out


def threshold_analysis(records: Sequence[dict]):
    groups = defaultdict(list)
    for r in records:
        groups[(r["seed"], r["task"], r["condition"])].append(r)
    per_seed, summaries = [], []
    for (seed, task, condition), group in sorted(groups.items()):
        group = sorted(group, key=lambda x: x["epochs"])
        hit = next((r for r in group if r["validation_accuracy"] >= TARGET_VALIDATION_ACCURACY), None)
        per_seed.append({"seed": int(seed), "task": task, "condition": condition,
                         "reached_85_validation": bool(hit is not None),
                         "first_epoch_reaching_85": int(hit["epochs"]) if hit else None,
                         "validation_accuracy_at_selection_percent": 100 * hit["validation_accuracy"] if hit else None,
                         "independent_test_accuracy_at_selected_epoch_percent": 100 * hit["test_accuracy"] if hit else None})
    for task in TASKS:
        for condition in CONDITIONS:
            subset = [r for r in per_seed if r["task"] == task and r["condition"] == condition]
            hits = [r for r in subset if r["reached_85_validation"]]
            low, high = v35.wilson_ci(len(hits), len(subset))
            test_vals = [r["independent_test_accuracy_at_selected_epoch_percent"] / 100 for r in hits]
            test_mean, test_low, test_high = v35.bootstrap_ci(test_vals, v35.stable_seed("v036-threshold", task, condition)) if test_vals else (None, None, None)
            ep = [r["first_epoch_reaching_85"] for r in hits]
            summaries.append({"task": task, "condition": condition, "n_seeds": len(subset), "n_reached_85": len(hits),
                              "reach_rate_percent": 100 * len(hits) / len(subset),
                              "reach_rate_ci95_wilson_low_percent": low, "reach_rate_ci95_wilson_high_percent": high,
                              "mean_first_epoch_among_reached": float(np.mean(ep)) if ep else None,
                              "median_first_epoch_among_reached": float(np.median(ep)) if ep else None,
                              "independent_test_accuracy_at_selected_epoch_mean_percent": test_mean,
                              "independent_test_accuracy_at_selected_epoch_ci95_low_percent": test_low,
                              "independent_test_accuracy_at_selected_epoch_ci95_high_percent": test_high})
    return per_seed, summaries


def activity_summary(integrity: Sequence[dict]) -> dict:
    result = {}
    for condition in CONDITIONS:
        entries = [item["tasks"][task]["activity_diagnostics"][condition] for item in integrity for task in TASKS]
        result[condition] = {}
        for key in ("mean_spikes_per_trial", "mean_spike_rate_per_hidden_tick", "mean_gain", "fraction_at_min", "fraction_at_max", "mean_within_trial_gain_std", "fraction_hidden_cells_with_at_least_one_spike", "fraction_trials_with_zero_hidden_spikes"):
            vals = [float(e[key]) for e in entries if key in e]
            if vals:
                result[condition][key] = float(np.mean(vals))
    return result


def render_report(records, summaries, deltas, thresholds, integrity, activity, runtime, workers):
    names = {"dale_fixed_25": "تثبيط ثابت (كسب 1.0)",
             "dale_global_adaptive_25": "تثبيط تكيفي عالمي",
             "dale_local_adaptive_25": f"تثبيط تكيفي محلي (الميل {LOCAL_SLOPE:g})"}
    lines = ["# Nucleus Pulse v0.36 — تقرير التثبيط التكيفي المحلي", "",
             "## ملخص التجربة", "",
             f"- بذور التطوير المنفصلة لاختيار الميل: {DEV_SEEDS[0]}–{DEV_SEEDS[-1]}.",
             f"- البذور الرسمية: {SEEDS[0]}–{SEEDS[-1]} ({len(integrity)} بذرة).",
             f"- الأمثلة الفريدة لكل مهمة وبذرة: تدريب {N_TRAIN}، تحقق {N_VALIDATION}، اختبار {N_TEST}.",
             f"- الخلايا: {N_HIDDEN}؛ التمثيل: 16 نافذة زمنية؛ نقاط التقييم: {EPOCH_CHECKPOINTS}.",
             f"- المعامل المحلي المختار مسبقًا من تحقق مجموعة التطوير: الميل {LOCAL_SLOPE:g}، EMA tau={TRACE_TAU_TICKS:g} ticks، target={TARGET_ACTIVITY_RATE:g}، حدود الكسب [{GAIN_MIN:g}, {GAIN_MAX:g}].",
             f"- زمن المرحلة الرسمية: {runtime:.1f} ثانية، عدد العمال: {workers}.", "",
             "## نتيجة التطوير واختيار الميل", "",
             "اختير الميل 6.0 من بين [1, 2, 4, 6] باستخدام متوسط الدقة المتوازنة على التحقق بعد 10 مرورّات عبر 6 بذور تطوير ومهمتين. لم تستخدم مجموعة الاختبار الرسمية في اختيار المعامل.", "",
             "## دقة الاختبار المستقل", "",
             "| المهمة | الشرط | المرورّات | دقة الاختبار [فاصل bootstrap 95%] | استدعاء الفئة 0 | استدعاء الفئة 1 |", "|---|---|---:|---:|---:|---:|"]
    for r in summaries:
        lines.append(f"| {r['task']} | {names[r['condition']]} | {r['epochs']} | {r['test_accuracy_mean_percent']:.2f}% [{r['test_accuracy_ci95_low_percent']:.2f}, {r['test_accuracy_ci95_high_percent']:.2f}] | {r['test_class_0_recall_mean_percent']:.2f}% | {r['test_class_1_recall_mean_percent']:.2f}% |")
    lines += ["", "## الفروق الزوجية (نقاط مئوية) عند 10 مرورّات", "",
              "| المهمة | المقارنة | الفرق [95% CI] | بذور تحسنت/تراجعت/تعادلت |", "|---|---|---:|---:|"]
    for r in deltas:
        if r["epochs"] == PRIMARY_EPOCH:
            lines.append(f"| {r['task']} | {r['contrast']} | {r['mean_delta_percentage_points']:+.2f} [{r['ci95_low_percentage_points']:+.2f}, {r['ci95_high_percentage_points']:+.2f}] | {r['positive_seed_deltas']}/{r['negative_seed_deltas']}/{r['zero_seed_deltas']} |")
    lines += ["", "## الوصول إلى 85% على التحقق", "", "| المهمة | الشرط | بذور وصلت/40 | نسبة الوصول [Wilson 95%] |", "|---|---|---:|---:|"]
    for r in thresholds:
        lines.append(f"| {r['task']} | {names[r['condition']]} | {r['n_reached_85']}/{r['n_seeds']} | {r['reach_rate_percent']:.1f}% [{r['reach_rate_ci95_wilson_low_percent']:.1f}, {r['reach_rate_ci95_wilson_high_percent']:.1f}] |")
    overlaps = sum(t["train_validation_overlap"] + t["train_test_overlap"] + t["validation_test_overlap"] for i in integrity for t in i["tasks"].values())
    lines += ["", "## نشاط الخزان", "", "متوسطات تشخيص النشاط عبر البذور والمهمتين:", "", "| الشرط | نبضات/مثال | معدل النبضات/خلية/لحظة | متوسط الكسب | تشتت الكسب المحلي داخل العينة |", "|---|---:|---:|---:|---:|"]
    for cond in CONDITIONS:
        a = activity[cond]
        lines.append(f"| {names[cond]} | {a['mean_spikes_per_trial']:.2f} | {a['mean_spike_rate_per_hidden_tick']:.4f} | {a['mean_gain']:.4f} | {a.get('mean_within_trial_gain_std', 0):.4f} |")
    lines += ["", "## سلامة البيانات والقيود", "",
              f"- السجلات النهائية: **{len(records)} / {len(SEEDS)*len(TASKS)*len(CONDITIONS)*len(EPOCH_CHECKPOINTS)}**.",
              f"- تقاطعات train/validation/test عبر جميع البذور: **{overlaps}**.",
              "- شروط كل بذرة تستخدم نفس المصفوفة الراجعة الموقعة نفسها، ونفس raster المدخل والأوزان الداخلة والعتبات والتسريبات والحالة الحرارية والميزات والقارئ وترتيب التدريب.",
              "- الشرط المحلي يغير كسب التيار التثبيطي الوارد لكل خلية حسب EMA نشاط تلك الخلية؛ بقية المكونات مثبتة.",
              "- فواصل الثقة تسحب البذور كوحدة إعادة سحب، والفروق محسوبة زوجيًا على البذور نفسها.",
              "- هذا اختبار قارئ خطي فوق خزان اصطناعي ثابت ومهمتين اصطناعيتين، وليس برهانًا على التعلم الهندسي أو التفكير العام.",
              "- كسب التثبيط المحلي heuristic محوسب؛ لا يمثل نموذجًا بيولوجيًا كاملًا ولا يقيس الطاقة الفعلية.", ""]
    return "\n".join(lines)


def plot_curves(summaries, outdir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = {"dale_fixed_25": "Fixed inhibition", "dale_global_adaptive_25": "Global adaptive", "dale_local_adaptive_25": "Local adaptive"}
    for task in TASKS:
        for split in ("validation", "test"):
            fig, ax = plt.subplots(figsize=(8.6, 5.5))
            for condition in CONDITIONS:
                group = sorted([x for x in summaries if x["task"] == task and x["condition"] == condition], key=lambda r: r["epochs"])
                xs = [x["epochs"] for x in group]
                ys = [x[f"{split}_accuracy_mean_percent"] for x in group]
                low = [x[f"{split}_accuracy_ci95_low_percent"] for x in group]
                high = [x[f"{split}_accuracy_ci95_high_percent"] for x in group]
                ax.plot(xs, ys, marker="o", label=names[condition])
                ax.fill_between(xs, low, high, alpha=0.15)
            ax.axhline(85, linestyle="--", linewidth=1, label="85% target")
            ax.axhline(50, linestyle=":", linewidth=1, label="chance")
            ax.set_xticks(EPOCH_CHECKPOINTS)
            ax.set_xlabel("Passes over the same 6,000 unique training examples")
            ax.set_ylabel(f"{split.capitalize()} accuracy (%)")
            ax.set_title(f"Nucleus Pulse v0.36 — {task} — {split}")
            ax.set_ylim(35, 101); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(outdir / f"learning_curve_{task}_{split}.png", dpi=160); plt.close(fig)


def atomic_json(path: Path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def run_full(outdir: Path, resume: bool, workers: int):
    verify_preregistration()
    outdir.mkdir(parents=True, exist_ok=True)
    partial = outdir / "partial_seed_results.json"
    done = {}
    if resume and partial.exists():
        for item in json.loads(partial.read_text(encoding="utf-8")):
            done[int(item["integrity"]["seed"])] = item
    missing = [s for s in SEEDS if s not in done]
    start = time.time()
    print(f"Nucleus Pulse v{VERSION}: {len(done)} saved; {len(missing)} remaining; workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_seed, seed): seed for seed in missing}
        for future in as_completed(futures):
            seed = futures[future]
            rows, integ = future.result()
            done[seed] = {"records": rows, "integrity": integ}
            atomic_json(partial, [done[k] for k in sorted(done)])
            print(f"Completed official seed {seed}: {len(done)}/{len(SEEDS)}", flush=True)
    items = [done[k] for k in sorted(done)]
    records = [r for item in items for r in item["records"]]
    integrity = [item["integrity"] for item in items]
    expected = len(SEEDS) * len(TASKS) * len(CONDITIONS) * len(EPOCH_CHECKPOINTS)
    if len(records) != expected or len(integrity) != len(SEEDS):
        raise RuntimeError(f"Incomplete v0.36 run: records {len(records)}/{expected}; seeds {len(integrity)}/{len(SEEDS)}")
    summaries = aggregate_records(records)
    deltas = paired_delta_summary(records)
    threshold_per_seed, threshold_rows = threshold_analysis(records)
    activity = activity_summary(integrity)
    runtime = time.time() - start
    result = {"version": VERSION, "protocol": protocol_config(), "seeds": SEEDS, "conditions": CONDITIONS,
              "n_hidden": N_HIDDEN, "n_train": N_TRAIN, "n_validation": N_VALIDATION, "n_test": N_TEST,
              "feature_set": FEATURE_SET, "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
              "records": records, "summary": summaries, "paired_differences": deltas,
              "threshold_per_seed": threshold_per_seed, "threshold_summary": threshold_rows,
              "integrity": integrity, "activity_diagnostic_means": activity,
              "runtime_seconds": runtime, "workers": workers}
    atomic_json(outdir / "results.json", result)
    v35.write_csv(outdir / "per_seed_local_inhibition.csv", records)
    v35.write_csv(outdir / "local_inhibition_summary.csv", summaries)
    v35.write_csv(outdir / "paired_local_inhibition_differences.csv", deltas)
    v35.write_csv(outdir / "per_seed_threshold.csv", threshold_per_seed)
    v35.write_csv(outdir / "threshold_summary.csv", threshold_rows)
    (outdir / "report_ar.md").write_text(render_report(records, summaries, deltas, threshold_rows, integrity, activity, runtime, workers), encoding="utf-8")
    plot_curves(summaries, outdir)
    partial.unlink(missing_ok=True)
    print("OFFICIAL RESULTS SAVED", flush=True)
    for row in deltas:
        if row["epochs"] == PRIMARY_EPOCH:
            print(f"PRIMARY {row['task']} {row['contrast']}: delta={row['mean_delta_percentage_points']:+.2f}pp CI95=[{row['ci95_low_percentage_points']:+.2f},{row['ci95_high_percentage_points']:+.2f}]", flush=True)
    print("ACTIVITY", json.dumps(activity, sort_keys=True), flush=True)
    return result


def run_smoke():
    seed = 8990
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, diag = make_shared_dale_matrix(reservoir, seed)
    assert all(np.array_equal(matrices[CONDITIONS[0]], matrices[c]) for c in CONDITIONS[1:])
    assert np.isclose(spectral_radius(matrices[CONDITIONS[0]]), TARGET_SPECTRAL_RADIUS, atol=5e-4)
    trials = v35.make_data(seed, "sync")[0][:160]
    fixed_in, fixed_spikes = simulate_batch_with_recurrent_weights(reservoir, trials, matrices[CONDITIONS[0]])
    glob_in, glob_spikes, glob_diag = v35.simulate_batch_activity_adaptive_inhibition(
        reservoir, trials, matrices[CONDITIONS[1]], return_diagnostics=True)
    loc_in, loc_spikes, loc_diag = simulate_local(reservoir, trials, matrices[CONDITIONS[2]], LOCAL_SLOPE, return_diagnostics=True)
    assert np.array_equal(fixed_in, glob_in) and np.array_equal(fixed_in, loc_in)
    assert fixed_spikes.shape == glob_spikes.shape == loc_spikes.shape == (len(trials), 32, N_HIDDEN)
    assert np.isfinite(loc_spikes).all()
    assert loc_diag["first_tick_gain_mean"] == 1.0
    assert GAIN_MIN <= loc_diag["min_gain_seen"] <= loc_diag["max_gain_seen"] <= GAIN_MAX
    assert loc_diag["mean_within_trial_gain_std"] >= 0
    _, no_adapt = simulate_local(reservoir, trials, matrices[CONDITIONS[2]], slope=0.0)
    assert np.allclose(fixed_spikes, no_adapt, atol=1e-7, rtol=0.0)
    print("SMOKE PASSED: shared matrix/inputs, bounded per-cell gain, zero-slope identity", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    elif args.freeze:
        print(json.dumps(freeze_preregistration(), indent=2), flush=True)
    else:
        run_full(OUTDIR, args.resume, args.workers)

if __name__ == "__main__":
    main()
