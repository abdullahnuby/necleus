"""Development-only tuning of local inhibitory EMA tau for Nucleus Pulse v0.38.

The script deliberately constructs only train and validation splits. It does not
create or access any final-test examples. Primary selection: mean validation
balanced accuracy at epoch 10 across development seeds and both tasks; ties go
to the smaller tau. No official evaluation seeds are used.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir  # noqa: E402
from experiments import run_activity_adaptive_inhibition_v0_35 as v35  # noqa: E402
from experiments.tune_local_inhibition_v0_37 import simulate_local  # noqa: E402
from experiments.run_inhibition_structure_v0_34 import (  # noqa: E402
    extract_features, spectral_radius, BOOTSTRAP_REPS, bootstrap_ci, stable_seed,
)
from experiments.run_learning_curve_v0_30 import (  # noqa: E402
    N_HIDDEN, make_epoch_labels, make_nested_training_pool, trial_key,
)
from experiments.run_incremental_windows_v0_31 import make_eval_dataset, classification_metrics  # noqa: E402

VERSION = "0.38-stage2-development"
DEV_SEEDS = list(range(9300, 9308))
TAU_CANDIDATES = (2.0, 4.0, 8.0)
TASKS = ("sync", "order")
CHECKPOINTS = (1, 3, 5, 10)
LOCAL_SLOPE = 2.0
FEATURE_SET = "16 temporal bins; 2 ticks/bin; per-neuron spike rates"
OUTDIR = ROOT / "results" / "ema_tau_v0_38" / "development_tuning"
CHECKPOINT_PATH = OUTDIR / "checkpoint.json"
PREREG_PATH = OUTDIR / "preregistration.json"
BOOTSTRAP_SEED = 20381024
OFFICIAL_SEEDS = set(range(12000, 12040))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def protocol_config() -> dict:
    return {
        "version": VERSION,
        "stage": "development-only; no final test set generated or accessed",
        "development_seeds": list(DEV_SEEDS),
        "tau_candidates_ticks": list(TAU_CANDIDATES),
        "tasks": list(TASKS),
        "train_examples_per_task_seed": v35.N_TRAIN,
        "validation_examples_per_task_seed": v35.N_VALIDATION,
        "training_epochs": 10,
        "checkpoints": list(CHECKPOINTS),
        "primary_selection_metric": "mean validation balanced_accuracy at epoch 10 across all development seed-task pairs",
        "tie_break": "smaller tau",
        "fixed_local_slope": LOCAL_SLOPE,
        "target_activity_rate": v35.TARGET_ACTIVITY_RATE,
        "gain_bounds": [v35.INHIBITORY_GAIN_MIN, v35.INHIBITORY_GAIN_MAX],
        "recurrent_matrix": "same Dale 25% matrix within a seed across tau candidates; target spectral radius 0.72",
        "reader": {"class": "SGDClassifier", "loss": "log_loss", "penalty": "l2", "alpha": 1e-4,
                   "learning_rate": "constant", "eta0": 0.01, "average": True, "shuffle": True,
                   "same random seed across tau candidates": True},
        "validation_only": True,
        "final_test_created_or_accessed": False,
        "official_seed_range_reserved": [12000, 12039],
        "paired_bootstrap_unit": "development seed, grouped across both tasks where applicable",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def prereg_source_files() -> list[str]:
    return [
        "experiments/tune_ema_tau_v0_38.py",
        "experiments/diagnose_ema_tau_v0_38.py",
        "experiments/tune_local_inhibition_v0_37.py",
        "experiments/run_activity_adaptive_inhibition_v0_35.py",
        "experiments/run_inhibition_structure_v0_34.py",
        "experiments/run_learning_curve_v0_30.py",
        "experiments/run_incremental_windows_v0_31.py",
        "pulse_core/sequential_stdp.py",
        "docs/benchmark_protocol_v0_38_STAGE2_AR.md",
        "tests/test_ema_tau_v0_38_stage2.py",
    ]


def freeze_preregistration() -> dict:
    doc = {"config": protocol_config(),
           "sha256": {rel: sha256_file(ROOT / rel) for rel in prereg_source_files()}}
    PREREG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREREG_PATH.write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    return doc


def verify_preregistration() -> dict:
    if not PREREG_PATH.exists():
        raise RuntimeError("Missing stage-2 development preregistration; freeze before tuning")
    doc = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Current development protocol differs from frozen configuration")
    for rel, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {rel}")
    return doc


def atomic_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = []
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (list, dict)) and key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in fields})


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def make_development_splits(seed: int, task: str):
    """Build train and validation only; do not generate an evaluation-test set."""
    task_i = TASKS.index(task)
    train = make_nested_training_pool(seed + 11111 + task_i * 100, task, v35.N_TRAIN)
    train_keys = {trial_key(tr) for tr in train}
    if len(train_keys) != v35.N_TRAIN:
        raise RuntimeError("Duplicate training sequences in development split")
    validation = make_eval_dataset(seed + 52222 + task_i * 100, task, v35.N_VALIDATION, set(train_keys))
    val_keys = {trial_key(tr) for tr in validation}
    if len(val_keys) != v35.N_VALIDATION or train_keys & val_keys:
        raise RuntimeError("Development train/validation split overlap or wrong size")
    y_train = make_epoch_labels(train)
    y_val = make_epoch_labels(validation)
    for name, ys in (("train", y_train), ("validation", y_val)):
        counts = [int(np.sum(ys == c)) for c in (0, 1)]
        if counts != [len(ys) // 2, len(ys) // 2]:
            raise RuntimeError(f"Unbalanced {name} set: {counts}")
    return train, validation, y_train, y_val, {
        "train_count": len(train), "validation_count": len(validation),
        "train_unique": len(train_keys), "validation_unique": len(val_keys),
        "train_validation_overlap": len(train_keys & val_keys),
        "train_keys_sha256": hashlib.sha256("\n".join(sorted(map(repr, train_keys))).encode("utf-8")).hexdigest(),
        "validation_keys_sha256": hashlib.sha256("\n".join(sorted(map(repr, val_keys))).encode("utf-8")).hexdigest(),
        "train_class_counts": {"0": int(np.sum(y_train == 0)), "1": int(np.sum(y_train == 1))},
        "validation_class_counts": {"0": int(np.sum(y_val == 0)), "1": int(np.sum(y_val == 1))},
    }


def run_candidate(seed: int, task_i: int, task: str, tau: float):
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, matrix_diag = v35.make_shared_dale_matrix(reservoir, seed)
    weights = matrices["dale_fixed_25"]
    radius = float(spectral_radius(weights))
    if not np.isclose(radius, v35.TARGET_SPECTRAL_RADIUS, atol=5e-4):
        raise RuntimeError(f"unexpected recurrent radius: {radius}")
    train, validation, y_train, y_val, split_diag = make_development_splits(seed, task)
    all_trials = train + validation
    inputs, spikes, sim_diag = simulate_local(
        reservoir, all_trials, weights, slope=LOCAL_SLOPE, tau=tau,
        target_rate=v35.TARGET_ACTIVITY_RATE,
        gain_min=v35.INHIBITORY_GAIN_MIN, gain_max=v35.INHIBITORY_GAIN_MAX,
        return_diagnostics=True,
    )
    expected_shape = (len(all_trials), reservoir.duration, N_HIDDEN)
    if spikes.shape != expected_shape or not np.isfinite(spikes).all():
        raise RuntimeError(f"bad spike tensor for seed={seed}, task={task}, tau={tau}")
    if not np.isclose(sim_diag["first_tick_gain_mean"], 1.0, atol=1e-7):
        raise RuntimeError("first tick gain must equal 1.0")
    if not np.isclose(matrix_diag["conditions"]["dale_inhibition_25"]["spectral_radius_after_normalization"], radius, atol=5e-4):
        raise RuntimeError("spectral radius diagnostics mismatch")
    n_train = len(train)
    f_train = extract_features(spikes[:n_train])
    f_val = extract_features(spikes[n_train:])
    if f_train.shape[0] != len(y_train) or f_val.shape[0] != len(y_val):
        raise RuntimeError("feature and label counts mismatch")
    model_seed = int(seed + 940001 + task_i * 1000)
    model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                          learning_rate="constant", eta0=0.01, average=True,
                          shuffle=True, random_state=model_seed)
    metric_rows = []
    for epoch in range(1, 11):
        if epoch == 1:
            model.partial_fit(f_train, y_train, classes=np.asarray([0, 1], dtype=np.int64))
        else:
            model.partial_fit(f_train, y_train)
        if epoch in CHECKPOINTS:
            metrics = classification_metrics(y_val, model.predict(f_val))
            metric_rows.append({
                "seed": int(seed), "task": task, "tau_ticks": float(tau), "epochs": int(epoch),
                **{f"validation_{key}": float(value) for key, value in metrics.items()},
                "train_examples": len(train), "validation_examples": len(validation),
            })
    activity = {
        "seed": int(seed), "task": task, "tau_ticks": float(tau),
        "mean_spikes_per_trial": float(np.sum(spikes, axis=(1, 2)).mean()),
        "mean_spike_rate_per_hidden_tick": float(spikes.mean()),
        "mean_gain": float(sim_diag["mean_gain"]),
        "gain_min_seen": float(sim_diag["min_gain_seen"]),
        "gain_max_seen": float(sim_diag["max_gain_seen"]),
        "fraction_gain_at_min": float(sim_diag["fraction_at_min"]),
        "fraction_gain_at_max": float(sim_diag["fraction_at_max"]),
        "first_tick_gain_mean": float(sim_diag["first_tick_gain_mean"]),
        "tau_ticks": float(tau), "recurrent_spectral_radius": radius,
    }
    integrity = {"seed": int(seed), "task": task, "tau_ticks": float(tau), **split_diag,
                 "same_matrix_across_tau_by_seed": True,
                 "matrix_sha256": array_sha256(weights),
                 "input_raster_sha256": array_sha256(inputs),
                 "test_set_generated_or_accessed": False,
                 "feature_shape_train": list(f_train.shape), "feature_shape_validation": list(f_val.shape)}
    del inputs, spikes, f_train, f_val
    return metric_rows, activity, integrity


def summarize(rows: list[dict], activity_rows: list[dict]):
    summary = []
    for task in TASKS:
        for tau in TAU_CANDIDATES:
            for epoch in CHECKPOINTS:
                subset = [r for r in rows if r["task"] == task and r["tau_ticks"] == tau and r["epochs"] == epoch]
                item = {"task": task, "tau_ticks": tau, "epochs": epoch, "n_dev_seeds": len(subset)}
                for metric in ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate"):
                    mean, low, high = v35.bootstrap_ci(
                        [r[f"validation_{metric}"] for r in subset],
                        stable_seed("v038-stage2", task, str(tau), str(epoch), metric))
                    item[f"validation_{metric}_mean_percent"] = mean
                    item[f"validation_{metric}_ci95_low_percent"] = low
                    item[f"validation_{metric}_ci95_high_percent"] = high
                acts = [a for a in activity_rows if a["task"] == task and a["tau_ticks"] == tau]
                if acts:
                    for metric in ("mean_spikes_per_trial", "mean_spike_rate_per_hidden_tick", "mean_gain", "fraction_gain_at_min", "fraction_gain_at_max"):
                        vals = [a[metric] for a in acts]
                        mean, low, high = v35.bootstrap_ci(vals, stable_seed("v038-activity", task, str(tau), metric))
                        item[f"{metric}_mean"] = mean / 100.0
                        item[f"{metric}_ci95_low"] = low / 100.0
                        item[f"{metric}_ci95_high"] = high / 100.0
                summary.append(item)
    paired = []
    index = {(r["seed"], r["task"], r["tau_ticks"], r["epochs"]): r for r in rows}
    for task in TASKS:
        for epoch in CHECKPOINTS:
            for tau in TAU_CANDIDATES:
                if tau == 4.0:
                    continue
                diffs = np.asarray([index[(seed, task, tau, epoch)]["validation_balanced_accuracy"] -
                                    index[(seed, task, 4.0, epoch)]["validation_balanced_accuracy"]
                                    for seed in DEV_SEEDS], dtype=float)
                mean, low, high = v35.bootstrap_ci(diffs, stable_seed("v038-paired-delta", task, str(tau), str(epoch)))
                paired.append({"task": task, "tau_ticks_minus_tau4": tau,
                               "comparison": f"tau_{tau:g} - tau_4", "epochs": epoch,
                               "n_paired_dev_seeds": len(diffs), "mean_delta_balanced_accuracy_points": mean,
                               "ci95_low_points": low, "ci95_high_points": high,
                               "positive_seed_deltas": int(np.sum(diffs > 0)),
                               "negative_seed_deltas": int(np.sum(diffs < 0)),
                               "zero_seed_deltas": int(np.sum(diffs == 0))})
    # Selection is fixed in the preregistration: maximum mean validation balanced accuracy at epoch 10.
    scores = []
    for tau in TAU_CANDIDATES:
        subset = [r for r in rows if r["tau_ticks"] == tau and r["epochs"] == 10]
        scores.append({"tau_ticks": tau,
                       "mean_validation_balanced_accuracy_percent": float(np.mean([r["validation_balanced_accuracy"] for r in subset]) * 100),
                       "mean_validation_accuracy_percent": float(np.mean([r["validation_accuracy"] for r in subset]) * 100),
                       "n_seed_task_pairs": len(subset)})
    selected = sorted(scores, key=lambda x: (-x["mean_validation_balanced_accuracy_percent"], x["tau_ticks"]))[0]
    return summary, paired, scores, selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="freeze this development protocol and source hashes")
    parser.add_argument("--force-rerun", action="store_true", help="discard local development checkpoint and rerun all development candidates")
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        doc = freeze_preregistration()
        print(json.dumps({"frozen": True, "sources": len(doc["sha256"]), "path": str(PREREG_PATH)}, indent=2))
        return
    verify_preregistration()
    if args.force_rerun and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if checkpoint.get("protocol_sha256") != sha256_file(PREREG_PATH):
            raise RuntimeError("Development checkpoint belongs to another protocol hash")
    else:
        checkpoint = {"protocol_sha256": sha256_file(PREREG_PATH), "rows": [], "activity_rows": [], "integrity": []}
        atomic_json(CHECKPOINT_PATH, checkpoint)
    rows = checkpoint["rows"]
    activity_rows = checkpoint["activity_rows"]
    integrity_rows = checkpoint["integrity"]
    completed = {(int(x["seed"]), str(x["task"]), float(x["tau_ticks"])) for x in integrity_rows}
    started = time.time()
    for seed in DEV_SEEDS:
        for task_i, task in enumerate(TASKS):
            for tau in TAU_CANDIDATES:
                key = (seed, task, float(tau))
                if key in completed:
                    continue
                metric_rows, activity, integrity = run_candidate(seed, task_i, task, float(tau))
                rows.extend(metric_rows)
                activity_rows.append(activity)
                integrity_rows.append(integrity)
                completed.add(key)
                checkpoint = {"protocol_sha256": sha256_file(PREREG_PATH), "rows": rows,
                              "activity_rows": activity_rows, "integrity": integrity_rows}
                atomic_json(CHECKPOINT_PATH, checkpoint)
        print(f"development seed {seed} complete ({len({x['seed'] for x in integrity_rows})}/{len(DEV_SEEDS)})", flush=True)
    expected_combos = len(DEV_SEEDS) * len(TASKS) * len(TAU_CANDIDATES)
    if len(integrity_rows) != expected_combos or len(rows) != expected_combos * len(CHECKPOINTS):
        raise RuntimeError(f"Incomplete development results: combos={len(integrity_rows)}/{expected_combos}, rows={len(rows)}/{expected_combos*len(CHECKPOINTS)}")
    if any(x["train_validation_overlap"] != 0 or x["test_set_generated_or_accessed"] is not False for x in integrity_rows):
        raise RuntimeError("Development integrity check failed")
    for seed in DEV_SEEDS:
        for task in TASKS:
            group = [x for x in integrity_rows if x["seed"] == seed and x["task"] == task]
            if len(group) != len(TAU_CANDIDATES):
                raise RuntimeError(f"Expected all tau candidates for seed={seed}, task={task}")
            for key in ("matrix_sha256", "input_raster_sha256", "train_keys_sha256", "validation_keys_sha256"):
                if len({x[key] for x in group}) != 1:
                    raise RuntimeError(f"Mismatch in {key} across tau candidates for seed={seed}, task={task}")
    summary, paired, scores, selected = summarize(rows, activity_rows)
    results = {
        "title": "Nucleus Pulse v0.38 stage-2 EMA tau development tuning",
        "stage": "development only; no final test set generated or accessed",
        "protocol_sha256": sha256_file(PREREG_PATH),
        "development_seeds": list(DEV_SEEDS), "tau_candidates": list(TAU_CANDIDATES),
        "fixed_local_slope": LOCAL_SLOPE, "tasks": list(TASKS),
        "rows": rows, "activity_rows": activity_rows, "integrity": integrity_rows,
        "summary": summary, "paired_validation_differences": paired,
        "selection_scores": scores, "selected_tau_ticks": selected["tau_ticks"],
        "selection_rule": protocol_config()["primary_selection_metric"] + "; ties go to smaller tau",
        "elapsed_seconds_in_final_invocation": round(time.time() - started, 3),
    }
    atomic_json(OUTDIR / "dev_tuning_results.json", results)
    write_csv(OUTDIR / "per_seed_validation.csv", rows)
    write_csv(OUTDIR / "activity_diagnostics.csv", activity_rows)
    write_csv(OUTDIR / "summary_with_ci.csv", summary)
    write_csv(OUTDIR / "paired_validation_differences.csv", paired)
    atomic_json(OUTDIR / "selected_tau.json", {"stage": "development_only", "selected_tau_ticks": selected["tau_ticks"],
                  "selection_scores": scores, "selection_rule": results["selection_rule"],
                  "official_test_used": False, "official_tau_still_not_frozen": True})
    report = [
        "# Nucleus Pulse v0.38 — المرحلة الثانية: ضبط تطوير EMA tau", "",
        "## نطاق التجربة", "",
        f"- بذور التطوير: `{DEV_SEEDS}` ({len(DEV_SEEDS)} بذور).",
        f"- قيم tau: `{list(TAU_CANDIDATES)}` ticks؛ الميل المحلي ثابت عند {LOCAL_SLOPE}.",
        f"- لكل بذرة/مهمة: {v35.N_TRAIN} مثال تدريب و{v35.N_VALIDATION} مثال تحقق، من دون إنشاء مجموعة اختبار نهائي.",
        f"- القارئ: SGDClassifier بـ10 مرورّات، ومقاييس التحقق عند {list(CHECKPOINTS)} مرورّات.",
        "- الاختيار الأساسي حسب متوسط الدقة المتوازنة على التحقق عند المرور العاشر فقط؛ التعادل للميل الأقل.",
        "- المصفوفة الراجعة واحدة لجميع قيم tau داخل البذرة؛ المدخلات والميزات وإعدادات القارئ ثابتة.", "",
        "## ملخص التحقق عند المرور العاشر", "",
        "| المهمة | tau | الدقة المتوازنة % (95% CI) | الدقة % (95% CI) | نبضات/مثال | كسب متوسط | تشبّع أدنى | تشبّع أعلى |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        if r["epochs"] != 10:
            continue
        report.append(f"| {r['task']} | {r['tau_ticks']:g} | {r['validation_balanced_accuracy_mean_percent']:.2f} ({r['validation_balanced_accuracy_ci95_low_percent']:.2f}–{r['validation_balanced_accuracy_ci95_high_percent']:.2f}) | {r['validation_accuracy_mean_percent']:.2f} ({r['validation_accuracy_ci95_low_percent']:.2f}–{r['validation_accuracy_ci95_high_percent']:.2f}) | {r['mean_spikes_per_trial_mean']:.2f} | {r['mean_gain_mean']:.3f} | {100*r['fraction_gain_at_min_mean']:.2f}% | {100*r['fraction_gain_at_max_mean']:.2f}% |")
    report += ["", "## اختيار التطوير", "", "| tau | متوسط الدقة المتوازنة على التحقق عند 10 مرورّات | متوسط الدقة | أزواج بذرة/مهمة |", "|---:|---:|---:|---:|"]
    for s in scores:
        report.append(f"| {s['tau_ticks']:g} | {s['mean_validation_balanced_accuracy_percent']:.3f}% | {s['mean_validation_accuracy_percent']:.3f}% | {s['n_seed_task_pairs']} |")
    report += ["", f"**قيمة tau المختارة على التطوير: {selected['tau_ticks']:g}.** هذه عملية اختيار تطوير فقط وليست نتيجة اختبار نهائي. لا يوجد ادعاء بأن هذه القيمة أفضل على بيانات مستقلة؛ يجب تجميدها قبل تجربة رسمية جديدة.", "",
               "## حدود الاستنتاج", "", "لا تُنشأ ولا تُقرأ أي أمثلة اختبار نهائي في هذا السكريبت. فواصل الثقة تخص بذور التطوير الثماني، وهي تقديرات أولية محدودة العدد. ينتهي هذا الإصدار عند اختيار قيمة مرشحة؛ لا يُعلن نجاحًا رسميًا ولا يغيّر أي نتيجة من الإصدارات السابقة.", ""]
    (OUTDIR / "report_ar.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"status": "complete", "development_combinations": len(integrity_rows),
                      "metric_rows": len(rows), "selected_tau_ticks": selected["tau_ticks"],
                      "selection_scores": scores}, indent=2))


if __name__ == "__main__":
    main()
