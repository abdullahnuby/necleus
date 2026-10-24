"""Official confirmation of EMA tau=8 versus tau=4 selected in v0.38 development.

The official seeds were reserved in the v0.38 stage-2 development preregistration
and were not generated or accessed during tuning. Primary endpoint: paired
independent-test balanced-accuracy difference (tau 8 minus tau 4) at epoch 10,
averaged within seed across sync/order. All other metrics are secondary/descriptive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir  # noqa: E402
from experiments import run_activity_adaptive_inhibition_v0_35 as v35  # noqa: E402
from experiments.tune_local_inhibition_v0_37 import simulate_local  # noqa: E402
from experiments.run_inhibition_structure_v0_34 import (  # noqa: E402
    N_TRAIN, N_VALIDATION, N_TEST, TASKS, EPOCH_CHECKPOINTS, BOOTSTRAP_REPS,
    extract_features, spectral_radius, stable_seed, bootstrap_ci,
)
from experiments.run_learning_curve_v0_30 import N_HIDDEN, trial_key  # noqa: E402

VERSION = "0.39-official-ema-tau-confirmation"
OFFICIAL_SEEDS = list(range(12000, 12040))
TAU_CANDIDATES = (4.0, 8.0)
LOCAL_SLOPE = 2.0
PRIMARY_EPOCH = 10
TARGET_SPECTRAL_RADIUS = 0.72
TARGET_ACTIVITY_RATE = 0.13
GAIN_MIN = 0.5
GAIN_MAX = 2.0
BOOTSTRAP_SEED = 20391024
OUTDIR = ROOT / "results" / "ema_tau_v0_39_official"
CHECKPOINT_PATH = OUTDIR / "checkpoint.json"
PREREG_PATH = OUTDIR / "preregistration.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def trial_keys_sha256(trials: Sequence) -> str:
    keys = [repr(trial_key(tr)) for tr in trials]
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def protocol_config() -> dict:
    return {
        "version": VERSION,
        "stage": "official confirmation on held-out official seeds; parameters frozen",
        "official_seed_range": [OFFICIAL_SEEDS[0], OFFICIAL_SEEDS[-1]],
        "n_official_seeds": len(OFFICIAL_SEEDS),
        "seeds_reserved_in_v0_38_preregistration": True,
        "tasks": list(TASKS),
        "conditions": {"tau4": 4.0, "tau8": 8.0},
        "fixed_local_inhibition_slope": LOCAL_SLOPE,
        "gain_bounds": [GAIN_MIN, GAIN_MAX],
        "target_activity_rate": TARGET_ACTIVITY_RATE,
        "recurrent_spectral_radius_target": TARGET_SPECTRAL_RADIUS,
        "hidden_neurons": N_HIDDEN,
        "unique_training_examples_per_task_seed": N_TRAIN,
        "validation_examples_per_task_seed": N_VALIDATION,
        "independent_test_examples_per_task_seed": N_TEST,
        "checkpoints": list(EPOCH_CHECKPOINTS),
        "primary_endpoint": "paired independent-test balanced_accuracy difference tau8 minus tau4 at epoch 10; average sync/order differences within each seed, then bootstrap over seeds",
        "secondary_endpoints": [
            "task-specific independent-test accuracy and balanced accuracy at epoch 10",
            "paired test metric differences by task at epochs 1,3,5,10 (descriptive only)",
            "validation performance and threshold rate (descriptive only)",
            "activity and gain diagnostics (descriptive only)",
        ],
        "reader": {"class": "SGDClassifier", "loss": "log_loss", "penalty": "l2", "alpha": 1e-4,
                   "learning_rate": "constant", "eta0": 0.01, "average": True, "shuffle": True,
                   "same model random seed for both tau values": True},
        "same_dataset_reservoir_input_weights_thresholds_refractory_and_recurrent_matrix_within_seed": True,
        "only_changed_parameter": "local inhibitory EMA tau: 4 versus 8 ticks",
        "tuning_after_official_test": False,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "official seed; primary metric averages task deltas within seed",
    }


def source_files_for_preregistration() -> list[str]:
    return [
        "experiments/run_ema_tau_v0_39_confirmation.py",
        "experiments/tune_ema_tau_v0_38.py",
        "experiments/tune_local_inhibition_v0_37.py",
        "experiments/run_local_inhibition_v0_37.py",
        "experiments/run_activity_adaptive_inhibition_v0_35.py",
        "experiments/run_inhibition_structure_v0_34.py",
        "experiments/run_learning_curve_v0_30.py",
        "experiments/run_incremental_windows_v0_31.py",
        "pulse_core/sequential_stdp.py",
        "docs/benchmark_protocol_v0_39_AR.md",
        "tests/test_ema_tau_v0_39_confirmation.py",
        "results/ema_tau_v0_38/development_tuning/selected_tau.json",
        "results/ema_tau_v0_38/development_tuning/dev_tuning_results.json",
        "results/ema_tau_v0_38/development_tuning/preregistration.json",
    ]


def freeze_preregistration() -> dict:
    prereg = {
        "title": "Nucleus Pulse v0.39 official confirmation: EMA tau 8 versus 4",
        "config": protocol_config(),
        "source_sha256": {rel: sha256_file(ROOT / rel) for rel in source_files_for_preregistration()},
        "development_selection_evidence": json.loads((ROOT / "results/ema_tau_v0_38/development_tuning/selected_tau.json").read_text(encoding="utf-8")),
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    PREREG_PATH.write_text(json.dumps(prereg, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return prereg


def verify_preregistration() -> dict:
    if not PREREG_PATH.exists():
        raise RuntimeError("Missing frozen v0.39 official preregistration; freeze protocol before running")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    if prereg.get("config") != protocol_config():
        raise RuntimeError("Current official protocol differs from frozen v0.39 config")
    for rel, expected in prereg["source_sha256"].items():
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {rel}")
    expected_tau = prereg["development_selection_evidence"].get("selected_tau_ticks")
    if float(expected_tau) != 8.0:
        raise RuntimeError("Official v0.39 confirmation is only valid for development-selected tau=8")
    return prereg


def atomic_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key, value in row.items():
            if not isinstance(value, (list, dict)) and key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: v for k, v in row.items() if k in fields})


def run_candidate(seed: int, task_i: int, task: str, tau: float) -> tuple[list[dict], dict]:
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    matrices, matrix_diag = v35.make_shared_dale_matrix(reservoir, seed)
    weights = matrices["dale_fixed_25"]
    radius = float(spectral_radius(weights))
    if not np.isclose(radius, TARGET_SPECTRAL_RADIUS, atol=5e-4):
        raise RuntimeError(f"Unexpected spectral radius {radius} for seed={seed}")

    trials, bounds, labels, split_integrity = v35.make_data(seed, task)
    inputs, spikes, sim_diag = simulate_local(
        reservoir, trials, weights, slope=LOCAL_SLOPE, tau=tau,
        target_rate=TARGET_ACTIVITY_RATE, gain_min=GAIN_MIN, gain_max=GAIN_MAX,
        return_diagnostics=True,
    )
    expected_shape = (N_TRAIN + N_VALIDATION + N_TEST, reservoir.duration, N_HIDDEN)
    if spikes.shape != expected_shape or not np.isfinite(spikes).all():
        raise RuntimeError(f"Invalid spike tensor for seed={seed}, task={task}, tau={tau}")
    if not np.isclose(sim_diag["first_tick_gain_mean"], 1.0, atol=1e-7):
        raise RuntimeError("Initial gain must equal 1.0")
    if any(split_integrity[k] != 0 for k in ("train_validation_overlap", "train_test_overlap", "validation_test_overlap")):
        raise RuntimeError(f"Data split overlap at seed={seed}, task={task}")
    if not np.isclose(matrix_diag["conditions"]["dale_inhibition_25"]["spectral_radius_after_normalization"], radius, atol=5e-4):
        raise RuntimeError("Spectral-radius diagnostic mismatch")

    feature_splits = {
        split: extract_features(spikes[left:right])
        for split, (left, right) in bounds.items()
    }
    model_rows = v35.run_base_reader(feature_splits, labels, seed, task_i)
    condition = f"local_inhibition_tau_{tau:g}"
    for row in model_rows:
        row["condition"] = condition
        row["tau_ticks"] = float(tau)
        row["input_train_sha256"] = array_sha256(inputs[bounds["train"][0]:bounds["train"][1]])

    integrity = {
        "seed": int(seed), "task": task, "tau_ticks": float(tau),
        "same_recurrent_matrix_across_tau_by_seed": True,
        "matrix_sha256": array_sha256(weights),
        "input_weights_sha256": array_sha256(reservoir.w_in),
        "input_raster_sha256": array_sha256(inputs),
        "train_keys_sha256": trial_keys_sha256(trials[bounds["train"][0]:bounds["train"][1]]),
        "validation_keys_sha256": trial_keys_sha256(trials[bounds["validation"][0]:bounds["validation"][1]]),
        "test_keys_sha256": trial_keys_sha256(trials[bounds["test"][0]:bounds["test"][1]]),
        **split_integrity,
        "label_counts": {split: {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))} for split, y in labels.items()},
        "spike_shape": list(spikes.shape),
        "mean_spikes_per_trial": float(np.sum(spikes, axis=(1, 2)).mean()),
        "mean_spike_rate_per_hidden_tick": float(spikes.mean()),
        "mean_gain": float(sim_diag["mean_gain"]),
        "min_gain_seen": float(sim_diag["min_gain_seen"]),
        "max_gain_seen": float(sim_diag["max_gain_seen"]),
        "fraction_gain_at_min": float(sim_diag["fraction_at_min"]),
        "fraction_gain_at_max": float(sim_diag["fraction_at_max"]),
        "first_tick_gain_mean": float(sim_diag["first_tick_gain_mean"]),
        "spectral_radius": radius,
        "unique_training_examples": N_TRAIN,
        "validation_examples": N_VALIDATION,
        "independent_test_examples": N_TEST,
    }
    del feature_splits, inputs, spikes
    return model_rows, integrity


def paired_ci(deltas: Sequence[float], seed: int) -> tuple[float, float, float]:
    # v35.bootstrap_ci reports fractions as percentages, so input deltas are fractions.
    return bootstrap_ci(deltas, seed)


def summarize(records: list[dict]) -> tuple[list[dict], list[dict], dict]:
    rows_summary = []
    for task in TASKS:
        for tau in TAU_CANDIDATES:
            condition = f"local_inhibition_tau_{tau:g}"
            for epoch in EPOCH_CHECKPOINTS:
                group = [r for r in records if r["task"] == task and r["condition"] == condition and int(r["epochs"]) == epoch]
                if len(group) != len(OFFICIAL_SEEDS):
                    raise RuntimeError(f"Incomplete summary group task={task}, tau={tau}, epoch={epoch}")
                item = {"task": task, "tau_ticks": tau, "epochs": epoch, "n_seeds": len(group)}
                for split in ("validation", "test"):
                    for metric in ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate"):
                        mean, low, high = bootstrap_ci(
                            [r[f"{split}_{metric}"] for r in group],
                            stable_seed("v039", task, str(tau), str(epoch), split, metric))
                        item[f"{split}_{metric}_mean_percent"] = mean
                        item[f"{split}_{metric}_ci95_low_percent"] = low
                        item[f"{split}_{metric}_ci95_high_percent"] = high
                rows_summary.append(item)

    lookup = {(int(r["seed"]), r["task"], float(r["tau_ticks"]), int(r["epochs"])): r for r in records}
    paired = []
    for task in TASKS:
        for epoch in EPOCH_CHECKPOINTS:
            for metric in ("test_accuracy", "test_balanced_accuracy", "validation_accuracy", "validation_balanced_accuracy"):
                deltas = np.asarray([
                    lookup[(seed, task, 8.0, epoch)][metric] - lookup[(seed, task, 4.0, epoch)][metric]
                    for seed in OFFICIAL_SEEDS
                ], dtype=float)
                mean, low, high = paired_ci(deltas, stable_seed("v039-paired", task, str(epoch), metric))
                paired.append({"task": task, "epochs": int(epoch), "metric": metric,
                               "comparison": "tau8 - tau4", "n_paired_seeds": len(deltas),
                               "mean_delta_percentage_points": mean, "ci95_low_percentage_points": low,
                               "ci95_high_percentage_points": high,
                               "positive_seed_deltas": int(np.sum(deltas > 0)),
                               "negative_seed_deltas": int(np.sum(deltas < 0)),
                               "zero_seed_deltas": int(np.sum(deltas == 0))})

    # Primary inference: average the tau8-tau4 test balanced-accuracy delta
    # across both tasks inside each seed, then bootstrap the 40 paired seeds.
    seed_deltas = []
    for seed in OFFICIAL_SEEDS:
        task_deltas = []
        for task in TASKS:
            task_deltas.append(lookup[(seed, task, 8.0, PRIMARY_EPOCH)]["test_balanced_accuracy"] -
                               lookup[(seed, task, 4.0, PRIMARY_EPOCH)]["test_balanced_accuracy"])
        seed_deltas.append(float(np.mean(task_deltas)))
    mean, low, high = paired_ci(seed_deltas, BOOTSTRAP_SEED)
    primary = {
        "endpoint": protocol_config()["primary_endpoint"],
        "tau8_minus_tau4_mean_delta_percentage_points": mean,
        "ci95_low_percentage_points": low,
        "ci95_high_percentage_points": high,
        "n_paired_seeds": len(seed_deltas),
        "positive_seed_deltas": int(np.sum(np.asarray(seed_deltas) > 0)),
        "negative_seed_deltas": int(np.sum(np.asarray(seed_deltas) < 0)),
        "zero_seed_deltas": int(np.sum(np.asarray(seed_deltas) == 0)),
        "ci_excludes_zero": bool(low > 0 or high < 0),
    }
    return rows_summary, paired, primary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="freeze official confirmation preregistration")
    parser.add_argument("--force-rerun", action="store_true", help="discard checkpoint and rerun all official seeds")
    args = parser.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if args.freeze:
        doc = freeze_preregistration()
        print(json.dumps({"frozen": True, "source_files": len(doc["source_sha256"]), "path": str(PREREG_PATH)}, indent=2))
        return
    prereg = verify_preregistration()
    prereg_hash = sha256_file(PREREG_PATH)
    if args.force_rerun and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
    if CHECKPOINT_PATH.exists():
        ckpt = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if ckpt.get("preregistration_sha256") != prereg_hash:
            raise RuntimeError("Checkpoint belongs to a different frozen protocol")
    else:
        ckpt = {"preregistration_sha256": prereg_hash, "records": [], "integrity": []}
        atomic_json(CHECKPOINT_PATH, ckpt)
    records, integrity = ckpt["records"], ckpt["integrity"]
    completed = {(int(r["seed"]), r["task"], float(r["tau_ticks"])) for r in integrity}
    started = time.time()
    for seed in OFFICIAL_SEEDS:
        for task_i, task in enumerate(TASKS):
            for tau in TAU_CANDIDATES:
                key = (seed, task, float(tau))
                if key in completed:
                    continue
                new_records, new_integrity = run_candidate(seed, task_i, task, float(tau))
                records.extend(new_records)
                integrity.append(new_integrity)
                completed.add(key)
                atomic_json(CHECKPOINT_PATH, {"preregistration_sha256": prereg_hash,
                                              "records": records, "integrity": integrity})
        print(f"official seed {seed} complete ({len({r['seed'] for r in integrity})}/{len(OFFICIAL_SEEDS)})", flush=True)

    expected_integrity = len(OFFICIAL_SEEDS) * len(TASKS) * len(TAU_CANDIDATES)
    expected_records = expected_integrity * len(EPOCH_CHECKPOINTS)
    if len(integrity) != expected_integrity or len(records) != expected_records:
        raise RuntimeError(f"Incomplete official run: integrity={len(integrity)}/{expected_integrity}; records={len(records)}/{expected_records}")
    for x in integrity:
        if any(x[k] != 0 for k in ("train_validation_overlap", "train_test_overlap", "validation_test_overlap")):
            raise RuntimeError(f"Data overlap detected: seed={x['seed']} task={x['task']} tau={x['tau_ticks']}")
        if x["first_tick_gain_mean"] != 1.0:
            raise RuntimeError("Initial inhibitory gain differs from 1.0")
    for seed in OFFICIAL_SEEDS:
        for task in TASKS:
            pair = [x for x in integrity if int(x["seed"]) == seed and x["task"] == task]
            if len(pair) != len(TAU_CANDIDATES):
                raise RuntimeError(f"Missing tau condition: seed={seed}, task={task}")
            for key in ("matrix_sha256", "input_weights_sha256", "input_raster_sha256",
                        "train_keys_sha256", "validation_keys_sha256", "test_keys_sha256"):
                if len({x[key] for x in pair}) != 1:
                    raise RuntimeError(f"Mismatch in {key} across tau candidates at seed={seed}, task={task}")
    for seed in OFFICIAL_SEEDS:
        for task in TASKS:
            for tau in TAU_CANDIDATES:
                if len([r for r in records if int(r["seed"]) == seed and r["task"] == task and float(r["tau_ticks"]) == tau]) != len(EPOCH_CHECKPOINTS):
                    raise RuntimeError("Missing checkpoint rows")

    summary, paired, primary = summarize(records)
    results = {
        "title": "Nucleus Pulse v0.39 official EMA tau confirmation",
        "preregistration_sha256": prereg_hash,
        "official_seeds": OFFICIAL_SEEDS, "tasks": list(TASKS), "tau_candidates": list(TAU_CANDIDATES),
        "primary_endpoint": primary, "records": records, "integrity": integrity,
        "summary": summary, "paired_differences": paired,
        "n_official_seed_task_tau_combinations": len(integrity),
        "n_metric_rows": len(records),
        "all_splits_balanced_and_disjoint": True,
        "official_test_used_for_tuning": False,
        "elapsed_seconds_final_invocation_only": round(time.time() - started, 3),
    }
    atomic_json(OUTDIR / "results.json", results)
    write_csv(OUTDIR / "per_seed_confirmation.csv", records)
    write_csv(OUTDIR / "confirmation_summary.csv", summary)
    write_csv(OUTDIR / "paired_differences.csv", paired)
    write_csv(OUTDIR / "integrity_audit.csv", integrity)

    report = [
        "# Nucleus Pulse v0.39 — التأكيد الرسمي لاختيار EMA tau", "",
        "## التصميم", "",
        f"- البذور الرسمية: `{OFFICIAL_SEEDS[0]}–{OFFICIAL_SEEDS[-1]}` ({len(OFFICIAL_SEEDS)} بذرة).",
        f"- المقارنة الوحيدة: tau 8 مقابل tau 4، والميل المحلي ثابت عند {LOCAL_SLOPE}.",
        f"- لكل بذرة/مهمة: {N_TRAIN} تدريب + {N_VALIDATION} تحقق + {N_TEST} اختبار مستقل متوازن.",
        "- نقاط التدريب: 1 و3 و5 و10 مرورّات؛ نقطة النهاية الأساسية محددة مسبقًا عند المرور العاشر.",
        "- لم يستخدم الاختبار النهائي لاختيار tau أو تعديل أي معامل.", "",
        "## النتيجة الأساسية", "",
        f"- فرق الدقة المتوازنة على الاختبار (tau8 − tau4)، متوسطًا عبر المهمتين لكل بذرة: **{primary['tau8_minus_tau4_mean_delta_percentage_points']:+.3f} نقطة مئوية**.",
        f"- فاصل الثقة 95%: **[{primary['ci95_low_percentage_points']:+.3f}, {primary['ci95_high_percentage_points']:+.3f}]**.",
        f"- البذور التي كان فرقها موجبًا/سالبًا/صفرًا: {primary['positive_seed_deltas']} / {primary['negative_seed_deltas']} / {primary['zero_seed_deltas']}.",
        f"- هل استبعد فاصل الثقة الصفر؟ **{'نعم' if primary['ci_excludes_zero'] else 'لا'}**.", "",
        "## تفسير النتيجة", "",
        "تُقرأ المقارنة الأساسية فقط بوصفها اختبارًا لتأكيد اختيار التطوير. إذا شمل فاصل الثقة الصفر، فلا يوجد دليل واضح على أن tau=8 أفضل من tau=4؛ لا نختار بديلًا بعد الاطلاع على النتائج ولا نعيد ضبط المعاملات على مجموعة الاختبار.",
        "", "## سلامة البيانات", "",
        f"- تركيبات البذرة/المهمة/tau المكتملة: {len(integrity)} / {expected_integrity}.",
        f"- سجلات المقاييس: {len(records)} / {expected_records}.",
        f"- الاختبار: {N_TEST} مثال مستقل لكل مهمة وبذرة، متوازن الفئات.",
        "- تحقق آليًا: مصفوفة الخزان والمدخلات والتسلسلات والتقسيمات مطابقة بين tau داخل البذرة؛ لا تداخل بين train/validation/test.",
        "- الوحدة الإحصائية الأساسية: البذرة، مع تجميع فروق المهمتين داخل كل بذرة قبل bootstrap.",
    ]
    (OUTDIR / "report_ar.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUTDIR / "run.log").write_text(
        f"version={VERSION}\npreregistration_sha256={prereg_hash}\nseeds={len(OFFICIAL_SEEDS)}\nrecords={len(records)}\nintegrity={len(integrity)}\nprimary_delta={primary['tau8_minus_tau4_mean_delta_percentage_points']}\nprimary_ci95=[{primary['ci95_low_percentage_points']},{primary['ci95_high_percentage_points']}]\n",
        encoding="utf-8")
    print(json.dumps({"completed": True, "seeds": len(OFFICIAL_SEEDS), "records": len(records),
                      "primary_endpoint": primary, "preregistration_sha256": prereg_hash}, indent=2))


if __name__ == "__main__":
    main()
