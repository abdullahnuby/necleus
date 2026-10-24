"""Nucleus Pulse v0.32: repeated-pass learning curves for an online logistic readout.

Question: does revisiting the same 6,000 unique training examples for 1/3/5/10
passes improve a fixed SGD logistic readout over the v0.31 four-, eight-, and
sixteen-window features? The LIF reservoir and feature representations are fixed.
Validation selects the first checkpoint reaching 85%; final-test metrics never
select epochs or hyperparameters. Repeated passes are deliberate replay exposure.
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
    make_features,
    make_nested_training_pool,
    simulate_batch_vectorized,
    trial_key,
)
from experiments.run_incremental_windows_v0_31 import (  # noqa: E402
    classification_metrics,
    make_eval_dataset,
    new_online_reader,
    transform_features,
)

VERSION = "0.32"
SEEDS = list(range(5000, 5040))
EPOCH_CHECKPOINTS = [1, 3, 5, 10]
MAX_EPOCHS = max(EPOCH_CHECKPOINTS)
N_TRAIN = 6000
N_VALIDATION = 1000
N_TEST = 1000
FEATURE_SETS = ["phase4_96", "spike_bins8_rate_192", "spike_bins16_rate_384"]
TARGET_VALIDATION_ACCURACY = 0.85
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20321024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_config() -> dict:
    return {
        "seeds": SEEDS,
        "unique_training_examples_per_task_seed": N_TRAIN,
        "training_pass_checkpoints": EPOCH_CHECKPOINTS,
        "validation_per_task_per_seed": N_VALIDATION,
        "final_test_per_task_per_seed": N_TEST,
        "hidden_neurons": N_HIDDEN,
        "feature_sets": FEATURE_SETS,
        "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
        "readout": {
            "class": "sklearn.linear_model.SGDClassifier",
            "loss": "log_loss",
            "penalty": "l2",
            "alpha": 1e-4,
            "learning_rate": "constant",
            "eta0": 0.01,
            "average": True,
            "shuffle": True,
            "random_state_seeded_per_task_feature": True,
            "fit_method": "partial_fit",
        },
        "fixed_training_examples_revisited_each_pass": True,
        "reservoir_and_features_unchanged_from_v0_31": True,
        "validation_used_for_epoch_checkpoint_selection_only": True,
        "final_test_used_for_tuning_or_selection": False,
        "no_hyperparameter_tuning": True,
        "bootstrapping_unit": "seed",
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }


def verify_preregistration() -> dict:
    path = ROOT / "results" / "multi_epoch_v0_32" / "preregistration.json"
    if not path.exists():
        raise RuntimeError("Missing frozen v0.32 preregistration; refusing full run.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Current run configuration differs from the preregistration.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return doc


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _stable_seed(*parts: str) -> int:
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
        count = min(1000, BOOTSTRAP_REPS - start)
        indices = rng.integers(0, arr.size, size=(count, arr.size))
        draws[start:start + count] = arr[indices].mean(axis=1) * 100.0
    low, high = np.quantile(draws, [0.025, 0.975])
    return mean, float(low), float(high)


def run_seed(seed: int) -> tuple[list[dict], dict]:
    started = time.time()
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    records: list[dict] = []
    integrity: dict = {"seed": int(seed), "tasks": {}}

    for task_i, task in enumerate(("sync", "order")):
        train_pool = make_nested_training_pool(seed + 11111 + task_i * 100, task, N_TRAIN)
        train_keys = {trial_key(t) for t in train_pool}
        if len(train_keys) != N_TRAIN:
            raise RuntimeError(f"Duplicate training sequence: seed={seed} task={task}")
        val_trials = make_eval_dataset(seed + 52222 + task_i * 100, task, N_VALIDATION, set(train_keys))
        val_keys = {trial_key(t) for t in val_trials}
        test_trials = make_eval_dataset(seed + 63333 + task_i * 100, task, N_TEST, set(train_keys) | val_keys)
        test_keys = {trial_key(t) for t in test_trials}
        if (train_keys & val_keys) or (train_keys & test_keys) or (val_keys & test_keys):
            raise RuntimeError(f"Train/validation/test overlap: seed={seed} task={task}")

        y_train = make_epoch_labels(train_pool)
        y_val = make_epoch_labels(val_trials)
        y_test = make_epoch_labels(test_trials)
        for split_name, labels in (("train", y_train), ("validation", y_val), ("test", y_test)):
            if np.sum(labels == 0) != len(labels) // 2 or np.sum(labels == 1) != len(labels) // 2:
                raise RuntimeError(f"Class imbalance in {split_name}: seed={seed} task={task}")

        x_train, s_train = simulate_batch_vectorized(reservoir, train_pool)
        x_val, s_val = simulate_batch_vectorized(reservoir, val_trials)
        x_test, s_test = simulate_batch_vectorized(reservoir, test_trials)
        f_train = transform_features(make_features(x_train, s_train))
        f_val = transform_features(make_features(x_val, s_val))
        f_test = transform_features(make_features(x_test, s_test))

        integrity["tasks"][task] = {
            "n_train_unique": len(train_keys), "n_validation_unique": len(val_keys), "n_test_unique": len(test_keys),
            "train_validation_overlap": len(train_keys & val_keys),
            "train_test_overlap": len(train_keys & test_keys),
            "validation_test_overlap": len(val_keys & test_keys),
            "class_balance_train": {"0": int(np.sum(y_train == 0)), "1": int(np.sum(y_train == 1))},
            "class_balance_validation": {"0": int(np.sum(y_val == 0)), "1": int(np.sum(y_val == 1))},
            "class_balance_test": {"0": int(np.sum(y_test == 0)), "1": int(np.sum(y_test == 1))},
            "same_train_examples_reused_each_epoch": True,
        }

        for feature_i, feature_name in enumerate(FEATURE_SETS):
            model_seed = seed + 800001 + task_i * 1000 + feature_i * 100
            from sklearn.linear_model import SGDClassifier
            model = SGDClassifier(
                loss="log_loss", penalty="l2", alpha=1e-4,
                learning_rate="constant", eta0=0.01, average=True, shuffle=True,
                random_state=int(model_seed),
            )
            X_train = f_train[feature_name]
            X_val = f_val[feature_name]
            X_test = f_test[feature_name]
            cumulative_fit_seconds = 0.0
            for epoch in range(1, MAX_EPOCHS + 1):
                fit_start = time.time()
                if epoch == 1:
                    model.partial_fit(X_train, y_train, classes=np.asarray([0, 1], dtype=np.int64))
                else:
                    model.partial_fit(X_train, y_train)
                update_seconds = time.time() - fit_start
                cumulative_fit_seconds += update_seconds
                if epoch not in EPOCH_CHECKPOINTS:
                    continue
                val_metrics = classification_metrics(y_val, model.predict(X_val))
                test_metrics = classification_metrics(y_test, model.predict(X_test))
                records.append({
                    "seed": int(seed), "task": task, "feature_set": feature_name,
                    "epochs": int(epoch), "unique_training_examples": N_TRAIN,
                    "cumulative_example_exposures": int(N_TRAIN * epoch),
                    "mean_passes_per_example": float(epoch),
                    **{f"validation_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "last_epoch_fit_seconds": round(update_seconds, 5),
                    "cumulative_fit_seconds": round(cumulative_fit_seconds, 5),
                    "validation_reached_85": bool(val_metrics["accuracy"] >= TARGET_VALIDATION_ACCURACY),
                })
        del x_train, s_train, x_val, s_val, x_test, s_test, f_train, f_val, f_test

    integrity["elapsed_seconds"] = round(time.time() - started, 3)
    return records, integrity


def aggregate_records(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["task"], row["feature_set"], row["epochs"])].append(row)
    output: list[dict] = []
    metrics = ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate")
    for task, feature, epoch in sorted(groups):
        rows = groups[(task, feature, epoch)]
        summary = {"task": task, "feature_set": feature, "epochs": epoch,
                   "unique_training_examples_per_seed": N_TRAIN,
                   "cumulative_exposures_per_seed": N_TRAIN * epoch,
                   "n_seeds": len({row["seed"] for row in rows})}
        for split in ("validation", "test"):
            for metric in metrics:
                mean, low, high = bootstrap_ci(
                    [row[f"{split}_{metric}"] for row in rows],
                    _stable_seed(task, feature, str(epoch), split, metric),
                )
                summary[f"{split}_{metric}_mean_percent"] = mean
                summary[f"{split}_{metric}_ci95_low_percent"] = low
                summary[f"{split}_{metric}_ci95_high_percent"] = high
        output.append(summary)
    return output


def threshold_analysis(records: Sequence[dict]) -> tuple[list[dict], list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["task"], row["feature_set"], row["seed"])].append(row)
    per_seed = []
    summary_rows = []
    for (task, feature, seed), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda row: row["epochs"])
        reached = next((row for row in ordered if row["validation_accuracy"] >= TARGET_VALIDATION_ACCURACY), None)
        per_seed.append({
            "task": task, "feature_set": feature, "seed": seed,
            "reached_85_on_validation": bool(reached),
            "first_epoch_reaching_85": int(reached["epochs"]) if reached else None,
            "validation_accuracy_at_selection_percent": 100 * reached["validation_accuracy"] if reached else None,
            "independent_test_accuracy_at_selected_epoch_percent": 100 * reached["test_accuracy"] if reached else None,
        })
    for task in ("sync", "order"):
        for feature in FEATURE_SETS:
            subset = [row for row in per_seed if row["task"] == task and row["feature_set"] == feature]
            hits = [row for row in subset if row["reached_85_on_validation"]]
            hit_flags = [1.0 if row["reached_85_on_validation"] else 0.0 for row in subset]
            rate, low, high = bootstrap_ci(hit_flags, _stable_seed(task, feature, "threshold_hit_rate"))
            test_at_selection = [float(row["independent_test_accuracy_at_selected_epoch_percent"]) / 100.0 for row in hits]
            if hits:
                test_mean, test_low, test_high = bootstrap_ci(test_at_selection,
                    _stable_seed(task, feature, "test_at_validation_selected_epoch"))
                mean_epoch = float(np.mean([row["first_epoch_reaching_85"] for row in hits]))
                median_epoch = float(np.median([row["first_epoch_reaching_85"] for row in hits]))
            else:
                test_mean = test_low = test_high = float("nan")
                mean_epoch = median_epoch = float("nan")
            summary_rows.append({
                "task": task, "feature_set": feature, "n_seeds": len(subset), "n_reached_85": len(hits),
                "reach_rate_percent": rate, "reach_rate_ci95_low_percent": low, "reach_rate_ci95_high_percent": high,
                "mean_first_epoch_among_reached": mean_epoch, "median_first_epoch_among_reached": median_epoch,
                "independent_test_accuracy_at_selected_epoch_mean_percent": test_mean,
                "independent_test_accuracy_at_selected_epoch_ci95_low_percent": test_low,
                "independent_test_accuracy_at_selected_epoch_ci95_high_percent": test_high,
            })
    return per_seed, summary_rows


def render_report(records: Sequence[dict], summaries: Sequence[dict], threshold_rows: Sequence[dict],
                  integrity: Sequence[dict], runtime: float, workers: int) -> str:
    lookup = {(row["task"], row["feature_set"], row["epochs"]): row for row in summaries}
    lines = [
        "# تقرير Nucleus Pulse v0.32 — أثر تكرار المرور على البيانات", "",
        "## السؤال", "",
        "هل تكرار عرض نفس 6000 مثال (1/3/5/10 مرورّات) يساعد قارئ SGD اللوجستي على تعلم مهام التوقيت من ميزات الخزان؟", "",
        "## البروتوكول المسجل مسبقًا", "",
        f"- بذور جديدة: **{len(SEEDS)}** ({SEEDS[0]}–{SEEDS[-1]}).",
        f"- تدريب: **{N_TRAIN} مثال فريد لكل مهمة وبذرة**، مع نقاط فحص بعد {EPOCH_CHECKPOINTS} مرورّات.",
        f"- تحقق مستقل واختبار نهائي مستقل: **{N_VALIDATION} لكل منهما** لكل مهمة وبذرة، متوازنان ولا يتداخلان مع التدريب أو بعضهما.",
        "- لم يتغير الخزان أو تمثيل الميزات عن v0.31؛ المتغير الوحيد هو عدد مرات المرور على مجموعة التدريب نفسها.",
        "- القارئ: SGDClassifier بخسارة لوجستية وL2 وalpha=1e-4 وeta0=0.01 ومتوسط أوزان، وإعدادات ثابتة مجمدة.",
        f"- عتبة الاختيار: أول مرور يصل فيه **التحقق فقط** إلى {TARGET_VALIDATION_ACCURACY:.0%}; لا تُستخدم دقة الاختبار في اختيار المرور أو ضبط المعاملات.",
        "- لأن المرور يتكرر، عدد مرات التعرض الكلي للأمثلة هو 6,000 × عدد المرورّات. هذه التجربة تقيس أثر إعادة العرض، وليست تعلمًا من أمثلة جديدة.",
        f"- التنفيذ: {workers} عمليات متوازية؛ زمن التشغيل {runtime:.1f} ثانية.",
        "", "## منحنى دقة الاختبار المستقل (المتوسط [فاصل ثقة 95% عبر البذور])", "",
        "| المهمة | التمثيل | 1 مرور | 3 مرورّات | 5 مرورّات | 10 مرورّات |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in ("sync", "order"):
        for feature in FEATURE_SETS:
            cells = []
            for epoch in EPOCH_CHECKPOINTS:
                row = lookup[(task, feature, epoch)]
                cells.append(f"{row['test_accuracy_mean_percent']:.1f}% [{row['test_accuracy_ci95_low_percent']:.1f}, {row['test_accuracy_ci95_high_percent']:.1f}]")
            lines.append(f"| {task} | {feature} | " + " | ".join(cells) + " |")

    lines += ["", "## هل وصلت أي بذرة إلى 85% على التحقق؟", "",
              "الاختبار المنفصل أدناه يُقاس عند أول مرور اجتاز العتبة في التحقق. إذا لم تصل بذرة إلى 85% في أي نقطة، فهي تظل فشلًا معلنًا ولا يُختار لها مرور بديل بناءً على الاختبار.",
              "", "| المهمة | التمثيل | بذور وصلت | نسبة الوصول [CI 95%] | متوسط أول مرور بين الواصلين | اختبار عند المرور المختار [CI 95%] |",
              "|---|---|---:|---:|---:|---:|"]
    for row in threshold_rows:
        test_mean = row["independent_test_accuracy_at_selected_epoch_mean_percent"]
        test_low = row["independent_test_accuracy_at_selected_epoch_ci95_low_percent"]
        test_high = row["independent_test_accuracy_at_selected_epoch_ci95_high_percent"]
        if test_mean is None or not np.isfinite(test_mean):
            test_cell = "لا توجد بذور مؤهلة"
            epoch_cell = "—"
        else:
            test_cell = f"{test_mean:.1f}% [{test_low:.1f}, {test_high:.1f}]"
            epoch_cell = f"{row['mean_first_epoch_among_reached']:.1f}"
        lines.append(f"| {row['task']} | {row['feature_set']} | {row['n_reached_85']}/{row['n_seeds']} | "
                     f"{row['reach_rate_percent']:.1f}% [{row['reach_rate_ci95_low_percent']:.1f}, {row['reach_rate_ci95_high_percent']:.1f}] | {epoch_cell} | {test_cell} |")

    overlap_total = sum(task["train_validation_overlap"] + task["train_test_overlap"] + task["validation_test_overlap"]
                        for item in integrity for task in item["tasks"].values())
    expected = len(SEEDS) * 2 * len(FEATURE_SETS) * len(EPOCH_CHECKPOINTS)
    lines += ["", "## سلامة التجربة", "",
              f"- سجلات القياس: **{len(records)}** (المتوقع {expected}).",
              f"- البذور المكتملة: **{len(integrity)}/{len(SEEDS)}**.",
              f"- مجموع تداخل تسلسلات الأحداث بين التدريب والتحقق والاختبار: **{overlap_total}**.",
              "- كل مجموعة تدريب فريدة وموزونة فئويًا؛ إعادة المرور تستخدم الأمثلة نفسها عمدًا، وليست تكرارات متسللة بين مجموعات البيانات.",
              "- فواصل الثقة bootstrap 95% تعيد سحب البذور (40 وحدة مستقلة)، لا آلاف الأمثلة داخل البذرة.",
              "", "## التفسير والقيود", "",
              "- تحسن الأداء مع المرورّات يدعم أن القارئ لم يكن قد تقارب بعد بعد مرور واحد؛ لا يثبت أن 10 مرورّات هي الأمثل عمومًا.",
              "- هذا اختبار لقارئ لوجستي على خزان ثابت. لا يختبر تعديل أوزان الخزان أو STDP أو النسيان.",
              "- الاختبار النهائي يظهر عند كل مرور لوصف المنحنى، لكنه لا يشارك في اختيار المرور. لا تُعامل أفضل نقطة اختبار بعد الاطلاع عليها كأنها تقدير غير متحيز.",
              "- الوصول إلى 85% محدد مسبقًا على التحقق؛ عدم الوصول يُسجل صراحة ولا نغيّر الحد بعد ظهور النتائج.",
              "", "## الملفات", "",
              "- `per_seed_multi_epoch.csv`: القياسات لكل بذرة ومهمة وتمثيل ومرور.",
              "- `multi_epoch_summary.csv`: المتوسطات وفواصل الثقة عبر البذور.",
              "- `per_seed_threshold_crossings.csv` و`threshold_summary.csv`: تحليل العتبة المعتمد على التحقق فقط.",
              "- `results.json`: النتائج الكاملة وفحوص سلامة البيانات وبصمات البروتوكول.", ""]
    return "\n".join(lines)


def plot_curves(summaries: Sequence[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in ("sync", "order"):
        for split in ("validation", "test"):
            fig, ax = plt.subplots(figsize=(9, 6))
            for feature in FEATURE_SETS:
                subset = sorted([row for row in summaries if row["task"] == task and row["feature_set"] == feature],
                                key=lambda row: row["epochs"])
                xs = [row["epochs"] for row in subset]
                ys = [row[f"{split}_accuracy_mean_percent"] for row in subset]
                lo = [row[f"{split}_accuracy_ci95_low_percent"] for row in subset]
                hi = [row[f"{split}_accuracy_ci95_high_percent"] for row in subset]
                ax.plot(xs, ys, marker="o", label=feature)
                ax.fill_between(xs, lo, hi, alpha=0.15)
            ax.axhline(TARGET_VALIDATION_ACCURACY * 100.0, linestyle="--", linewidth=1,
                       label="85% validation selection threshold")
            ax.axhline(50.0, linestyle=":", linewidth=1, label="chance")
            ax.set_xticks(EPOCH_CHECKPOINTS)
            ax.set_xlabel("Passes over the same 6,000 unique training examples")
            ax.set_ylabel(f"{split.capitalize()} accuracy (%)")
            ax.set_title(f"Nucleus Pulse v0.32 — {task} — {split}")
            ax.set_ylim(35, 101)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
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
            raise RuntimeError("Partial output includes seeds outside frozen protocol")
        print(f"Resuming {len(completed)}/{len(SEEDS)} seeds", flush=True)

    started = time.time()
    remaining = [seed for seed in SEEDS if seed not in completed]
    print(f"Running v0.32: {len(remaining)} seeds, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in remaining}
        for future in as_completed(futures):
            seed = futures[future]
            seed_records, seed_integrity = future.result()
            records.extend(seed_records)
            integrity.append(seed_integrity)
            completed.add(seed)
            atomic_json(partial_path, {
                "version": VERSION,
                "completed_seeds": [seed_id for seed_id in SEEDS if seed_id in completed],
                "records": records, "integrity": integrity, "partial": True, "execution_workers": workers,
            })
            print(f"[{len(completed)}/{len(SEEDS)}] seed={seed} records={len(records)} elapsed={time.time()-started:.1f}s", flush=True)

    seed_order = {seed: idx for idx, seed in enumerate(SEEDS)}
    task_order = {"sync": 0, "order": 1}
    feature_order = {feature: idx for idx, feature in enumerate(FEATURE_SETS)}
    records.sort(key=lambda row: (seed_order[row["seed"]], task_order[row["task"]], feature_order[row["feature_set"]], row["epochs"]))
    integrity.sort(key=lambda row: seed_order[row["seed"]])
    summaries = aggregate_records(records)
    threshold_per_seed, threshold_rows = threshold_analysis(records)
    overlap_total = sum(task["train_validation_overlap"] + task["train_test_overlap"] + task["validation_test_overlap"]
                        for item in integrity for task in item["tasks"].values())
    expected = len(SEEDS) * 2 * len(FEATURE_SETS) * len(EPOCH_CHECKPOINTS)
    if len(records) != expected or len(integrity) != len(SEEDS) or overlap_total != 0:
        raise RuntimeError(f"Final integrity failure: records={len(records)}/{expected}, seeds={len(integrity)}, overlap={overlap_total}")

    result = {
        "version": VERSION, "seeds": SEEDS, "training_epochs": EPOCH_CHECKPOINTS,
        "unique_training_examples_per_task_seed": N_TRAIN, "feature_sets": FEATURE_SETS,
        "n_validation_per_task_per_seed": N_VALIDATION, "n_test_per_task_per_seed": N_TEST,
        "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
        "readout_config": {"type": "SGDClassifier", "loss": "log_loss", "penalty": "l2", "alpha": 1e-4,
                            "learning_rate": "constant", "eta0": 0.01, "average": True, "shuffle": True,
                            "revisit_same_training_set_each_epoch": True},
        "records": records, "summary": summaries, "threshold_crossings_per_seed": threshold_per_seed,
        "threshold_summary": threshold_rows, "integrity": integrity, "runtime_seconds": float(time.time() - started),
        "protocol": protocol, "execution_workers": workers,
    }
    atomic_json(outdir / "results.json", result)
    write_csv(outdir / "per_seed_multi_epoch.csv", records)
    write_csv(outdir / "multi_epoch_summary.csv", summaries)
    write_csv(outdir / "per_seed_threshold_crossings.csv", threshold_per_seed)
    write_csv(outdir / "threshold_summary.csv", threshold_rows)
    (outdir / "report_ar.md").write_text(render_report(records, summaries, threshold_rows, integrity,
                                                       result["runtime_seconds"], workers), encoding="utf-8")
    plot_curves(summaries, outdir)
    partial_path.unlink(missing_ok=True)
    print("FINAL RESULTS SAVED", flush=True)
    for row in threshold_rows:
        print(f"{row['task']:5s} {row['feature_set']:25s} reached={row['n_reached_85']}/{row['n_seeds']} "
              f"mean_epoch={row['mean_first_epoch_among_reached']} test@selected={row['independent_test_accuracy_at_selected_epoch_mean_percent']}", flush=True)
    return result


def run_smoke() -> None:
    seed = 4990  # engineering-only seed outside the official range
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    train = make_nested_training_pool(seed + 11, "sync", 200)
    train_keys = {trial_key(row) for row in train}
    validation = make_eval_dataset(seed + 21, "sync", 60, set(train_keys))
    val_keys = {trial_key(row) for row in validation}
    test = make_eval_dataset(seed + 31, "sync", 60, set(train_keys) | val_keys)
    x_train, spikes_train = simulate_batch_vectorized(reservoir, train)
    x_val, spikes_val = simulate_batch_vectorized(reservoir, validation)
    x_test, spikes_test = simulate_batch_vectorized(reservoir, test)
    f_train = transform_features(make_features(x_train, spikes_train))
    f_val = transform_features(make_features(x_val, spikes_val))
    f_test = transform_features(make_features(x_test, spikes_test))
    y_train, y_val, y_test = make_epoch_labels(train), make_epoch_labels(validation), make_epoch_labels(test)
    for i, feature in enumerate(FEATURE_SETS):
        model_seed = seed + i
        from sklearn.linear_model import SGDClassifier
        model = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, learning_rate="constant",
                              eta0=0.01, average=True, shuffle=True, random_state=model_seed)
        for epoch in range(3):
            if epoch == 0:
                model.partial_fit(f_train[feature], y_train, classes=np.asarray([0, 1]))
            else:
                model.partial_fit(f_train[feature], y_train)
        assert model.predict(f_val[feature]).shape == (60,)
        assert model.predict(f_test[feature]).shape == (60,)
    assert not (train_keys & val_keys or train_keys & {trial_key(row) for row in test} or val_keys & {trial_key(row) for row in test})
    print("SMOKE PASSED: repeated partial_fit passes, all features, balanced labels, and disjoint evaluation sets", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="engineering-only smoke test")
    parser.add_argument("--resume", action="store_true", help="resume the preregistered full run from partial results")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full(ROOT / "results" / "multi_epoch_v0_32", resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
