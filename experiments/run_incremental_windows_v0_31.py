"""Nucleus Pulse v0.31: online/streaming readout with 4/8/16 temporal windows.

Each unique training example is presented once, in nested balanced chunks. A single
SGD logistic readout per feature representation persists between checkpoints.
Validation selects the first sample budget reaching a preregistered 85% accuracy
threshold; a disjoint final test set estimates performance at that checkpoint.
The fixed LIF reservoir itself remains untrained.
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
from pulse_core.sequential_stdp import (  # noqa: E402
    DynamicPulseReservoir,
    TemporalTrial,
    make_disjoint_dataset,
)
from experiments.run_learning_curve_v0_30 import (  # noqa: E402
    N_HIDDEN,
    make_epoch_labels,
    make_features,
    make_nested_training_pool,
    simulate_batch_vectorized,
    trial_key,
)

SEEDS = list(range(4000, 4040))
TRAIN_SIZES = [200, 500, 1000, 2000, 6000]
N_TRAIN_POOL = max(TRAIN_SIZES)
N_VALIDATION = 1000
N_TEST = 1000
FEATURE_SETS = ["phase4_96", "spike_bins8_rate_192", "spike_bins16_rate_384"]
TARGET_VALIDATION_ACCURACY = 0.85
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20311024

# Frozen online logistic SGD settings. One partial_fit pass over each new chunk;
# no past sample is replayed and no hyperparameter is selected from final-test data.
SGD_ALPHA = 1e-4
SGD_ETA0 = 0.01
SGD_AVERAGE = True
SGD_PENALTY = "l2"
SGD_LOSS = "log_loss"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_config() -> dict:
    return {
        "seeds": SEEDS,
        "train_sizes": TRAIN_SIZES,
        "train_pool_per_task": N_TRAIN_POOL,
        "validation_per_task_per_seed": N_VALIDATION,
        "test_per_task_per_seed": N_TEST,
        "hidden_neurons": N_HIDDEN,
        "feature_sets": FEATURE_SETS,
        "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
        "sgd_loss": SGD_LOSS,
        "sgd_penalty": SGD_PENALTY,
        "sgd_alpha": SGD_ALPHA,
        "sgd_learning_rate": "constant",
        "sgd_eta0": SGD_ETA0,
        "sgd_average": SGD_AVERAGE,
        "one_pass_per_unique_example": True,
        "train_chunks_are_nested_and_class_balanced": True,
        "test_used_for_tuning_or_checkpoint_selection": False,
        "validation_used_for_hyperparameter_tuning": False,
        "threshold_selection_uses_validation_only": True,
    }


def verify_preregistration() -> dict:
    path = ROOT / "results" / "incremental_windows_v0_31" / "preregistration.json"
    if not path.exists():
        raise RuntimeError("Missing frozen v0.31 preregistration; refusing full run.")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Current run configuration differs from preregistration.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return doc


def make_eval_dataset(seed: int, task: str, n: int, forbidden: set[tuple]) -> list[TemporalTrial]:
    examples = make_disjoint_dataset(seed, task, n, "test", forbidden=forbidden)
    keys = [trial_key(t) for t in examples]
    if len(set(keys)) != n or set(keys) & forbidden:
        raise RuntimeError(f"Duplicate/overlap found while making {task} evaluation data")
    return examples


def transform_features(features: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Use fixed scale factors, not data-fitted preprocessing, for online stability."""
    out = {
        "phase4_96": features["phase4_96"].astype(np.float32, copy=False),
        # Each bin is respectively 4 or 2 ticks long; divide counts by bin width.
        "spike_bins8_rate_192": (features["spike_bins8_192"] / 4.0).astype(np.float32),
        "spike_bins16_rate_384": (features["spike_bins16_384"] / 2.0).astype(np.float32),
    }
    for name, values in out.items():
        if values.ndim != 2 or not np.isfinite(values).all():
            raise RuntimeError(f"Non-finite or non-matrix features: {name}")
    return out


def new_online_reader(seed: int):
    from sklearn.linear_model import SGDClassifier
    return SGDClassifier(
        loss=SGD_LOSS,
        penalty=SGD_PENALTY,
        alpha=SGD_ALPHA,
        learning_rate="constant",
        eta0=SGD_ETA0,
        average=SGD_AVERAGE,
        shuffle=True,
        random_state=int(seed),
    )


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recall0 = float(np.mean(y_pred[y_true == 0] == 0))
    recall1 = float(np.mean(y_pred[y_true == 1] == 1))
    return {
        "n_eval": int(y_true.size),
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": 0.5 * (recall0 + recall1),
        "class_0_recall": recall0,
        "class_1_recall": recall1,
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
    }


def run_seed(seed: int) -> tuple[list[dict], dict]:
    started = time.time()
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    rows: list[dict] = []
    integrity: dict = {"seed": seed, "tasks": {}}

    for task_i, task in enumerate(("sync", "order")):
        train_pool = make_nested_training_pool(seed + 11011 + task_i * 100, task, N_TRAIN_POOL)
        train_keys = {trial_key(t) for t in train_pool}
        if len(train_keys) != N_TRAIN_POOL:
            raise RuntimeError(f"Duplicate training sequence in seed={seed}, task={task}")

        val_trials = make_eval_dataset(seed + 52011 + task_i * 100, task, N_VALIDATION, set(train_keys))
        val_keys = {trial_key(t) for t in val_trials}
        test_trials = make_eval_dataset(seed + 63011 + task_i * 100, task, N_TEST, set(train_keys) | val_keys)
        test_keys = {trial_key(t) for t in test_trials}
        if train_keys & val_keys or train_keys & test_keys or val_keys & test_keys:
            raise RuntimeError("Exact sequence overlap between train/validation/test")

        y_train = make_epoch_labels(train_pool)
        y_val = make_epoch_labels(val_trials)
        y_test = make_epoch_labels(test_trials)
        for name, labels in (("train", y_train), ("validation", y_val), ("test", y_test)):
            if int(np.sum(labels == 0)) != len(labels) // 2 or int(np.sum(labels == 1)) != len(labels) // 2:
                raise RuntimeError(f"Class imbalance in {name} split, seed={seed}, task={task}")

        x_train, s_train = simulate_batch_vectorized(reservoir, train_pool)
        x_val, s_val = simulate_batch_vectorized(reservoir, val_trials)
        x_test, s_test = simulate_batch_vectorized(reservoir, test_trials)
        f_train = transform_features(make_features(x_train, s_train))
        f_val = transform_features(make_features(x_val, s_val))
        f_test = transform_features(make_features(x_test, s_test))

        integrity["tasks"][task] = {
            "n_train_pool": len(train_pool), "n_validation": len(val_trials), "n_test": len(test_trials),
            "train_unique_sequences": len(train_keys), "validation_unique_sequences": len(val_keys),
            "test_unique_sequences": len(test_keys),
            "train_validation_overlap": len(train_keys & val_keys),
            "train_test_overlap": len(train_keys & test_keys),
            "validation_test_overlap": len(val_keys & test_keys),
            "class_balance_train": {"0": int(np.sum(y_train == 0)), "1": int(np.sum(y_train == 1))},
            "class_balance_validation": {"0": int(np.sum(y_val == 0)), "1": int(np.sum(y_val == 1))},
            "class_balance_test": {"0": int(np.sum(y_test == 0)), "1": int(np.sum(y_test == 1))},
            "sample_order_nested_and_balanced": True,
        }

        for feature_i, feature_set in enumerate(FEATURE_SETS):
            model = new_online_reader(seed + 720001 + task_i * 1000 + feature_i * 100)
            X = f_train[feature_set]
            X_val = f_val[feature_set]
            X_test = f_test[feature_set]
            last_size = 0
            cumulative_fit_seconds = 0.0
            for train_size in TRAIN_SIZES:
                chunk_x = X[last_size:train_size]
                chunk_y = y_train[last_size:train_size]
                update_start = time.time()
                if last_size == 0:
                    model.partial_fit(chunk_x, chunk_y, classes=np.asarray([0, 1], dtype=np.int64))
                else:
                    model.partial_fit(chunk_x, chunk_y)
                update_seconds = time.time() - update_start
                cumulative_fit_seconds += update_seconds

                val_prediction = model.predict(X_val)
                test_prediction = model.predict(X_test)
                val_metrics = classification_metrics(y_val, val_prediction)
                test_metrics = classification_metrics(y_test, test_prediction)
                if train_size != last_size + len(chunk_x):
                    raise RuntimeError("Training exposure counter mismatch")

                rows.append({
                    "seed": int(seed), "task": task, "feature_set": feature_set,
                    "train_size": int(train_size), "incremental_chunk_size": int(len(chunk_x)),
                    "unique_examples_seen_once": int(train_size),
                    **{f"validation_{k}": v for k, v in val_metrics.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "update_seconds": round(update_seconds, 5),
                    "cumulative_fit_seconds": round(cumulative_fit_seconds, 5),
                    "validation_reached_85": bool(val_metrics["accuracy"] >= TARGET_VALIDATION_ACCURACY),
                })
                last_size = train_size

        del x_train, s_train, x_val, s_val, x_test, s_test, f_train, f_val, f_test

    integrity["elapsed_seconds"] = round(time.time() - started, 3)
    return rows, integrity


def _json_safe(value):
    """Convert missing/non-finite summary values to standards-compliant JSON null."""
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
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def bootstrap_ci(values: Sequence[float], seed: int) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return (float("nan"), float("nan"), float("nan"))
    mean = 100.0 * float(arr.mean())
    if len(arr) == 1:
        return (mean, float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    for start in range(0, BOOTSTRAP_REPS, 1000):
        count = min(1000, BOOTSTRAP_REPS - start)
        idx = rng.integers(0, len(arr), size=(count, len(arr)))
        draws[start:start + count] = arr[idx].mean(axis=1) * 100.0
    low, high = np.quantile(draws, [0.025, 0.975])
    return (mean, float(low), float(high))


def aggregate_records(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["task"], row["feature_set"], row["train_size"])].append(row)
    output = []
    metric_suffixes = ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall")
    for task, feature_set, train_size in sorted(groups):
        rows = groups[(task, feature_set, train_size)]
        summary = {"task": task, "feature_set": feature_set, "train_size": train_size,
                   "n_seeds": len({r["seed"] for r in rows})}
        for split in ("validation", "test"):
            for metric in metric_suffixes:
                mean, low, high = bootstrap_ci(
                    [r[f"{split}_{metric}"] for r in rows],
                    _stable_seed(task, feature_set, str(train_size), split, metric),
                )
                summary[f"{split}_{metric}_mean_percent"] = mean
                summary[f"{split}_{metric}_ci95_low_percent"] = low
                summary[f"{split}_{metric}_ci95_high_percent"] = high
        output.append(summary)
    return output


def crossing_summary(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["task"], row["feature_set"], row["seed"])].append(row)
    seed_rows = []
    for (task, feature_set, seed), rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda r: r["train_size"])
        reached = next((r for r in ordered if r["validation_accuracy"] >= TARGET_VALIDATION_ACCURACY), None)
        seed_rows.append({
            "task": task, "feature_set": feature_set, "seed": seed,
            "reached_85_on_validation": bool(reached),
            "first_train_size_reaching_85": reached["train_size"] if reached else None,
            "validation_accuracy_at_crossing_percent": 100.0 * reached["validation_accuracy"] if reached else None,
            "independent_test_accuracy_at_crossing_percent": 100.0 * reached["test_accuracy"] if reached else None,
        })

    result = []
    for task in ("sync", "order"):
        for feature_set in FEATURE_SETS:
            sub = [r for r in seed_rows if r["task"] == task and r["feature_set"] == feature_set]
            hits = [r for r in sub if r["reached_85_on_validation"]]
            hit_values = np.asarray([1.0 if r["reached_85_on_validation"] else 0.0 for r in sub])
            hit_mean, hit_low, hit_high = bootstrap_ci(hit_values, _stable_seed(task, feature_set, "hit_rate"))
            sizes = [float(r["first_train_size_reaching_85"]) for r in hits]
            test_at_cross = [float(r["independent_test_accuracy_at_crossing_percent"]) for r in hits]
            if test_at_cross:
                test_mean, test_low, test_high = bootstrap_ci(np.asarray(test_at_cross) / 100.0,
                    _stable_seed(task, feature_set, "test_at_validation_crossing"))
                mean_size = float(np.mean(sizes))
                median_size = float(np.median(sizes))
            else:
                test_mean = test_low = test_high = float("nan")
                mean_size = median_size = float("nan")
            result.append({
                "task": task, "feature_set": feature_set, "n_seeds": len(sub),
                "n_reached_85": len(hits), "reach_rate_percent": hit_mean,
                "reach_rate_ci95_low_percent": hit_low, "reach_rate_ci95_high_percent": hit_high,
                "mean_train_size_at_first_crossing_among_reached": mean_size,
                "median_train_size_at_first_crossing_among_reached": median_size,
                "independent_test_accuracy_at_validation_crossing_mean_percent": test_mean,
                "independent_test_accuracy_at_validation_crossing_ci95_low_percent": test_low,
                "independent_test_accuracy_at_validation_crossing_ci95_high_percent": test_high,
            })
    return seed_rows, result


def render_report(records: Sequence[dict], summaries: Sequence[dict], seed_crossings: Sequence[dict],
                  crossing: Sequence[dict], integrity: Sequence[dict], runtime: float,
                  workers: int) -> str:
    lookup = {(r["task"], r["feature_set"], r["train_size"]): r for r in summaries}
    lines = [
        "# تقرير v0.31 — التعلم التدريجي مع 4/8/16 نافذة", "",
        "## السؤال", "",
        "هل يستطيع قارئ لوجستي تدريجي أن يتعلم من ميزات زمنية أغنى، وكم مثالًا يحتاج للوصول إلى دقة تحقق 85%؟", "",
        "## البروتوكول المسجل مسبقًا", "",
        f"- البذور الجديدة: **{len(SEEDS)}** ({SEEDS[0]}–{SEEDS[-1]}).",
        f"- أحجام التدريب المتداخلة: **{TRAIN_SIZES}** لكل مهمة؛ كل مثال فريد يمر على القارئ مرة واحدة فقط.",
        f"- التحقق والاختبار: **{N_VALIDATION} مثال تحقق + {N_TEST} اختبار مستقل** لكل مهمة وبذرة، وكلاهما متوازن ومستقل عن كامل مجموعة التدريب.",
        "- التمثيلات: بنك الأثر بأربع نوافذ، 8 نوافذ، و16 نافذة؛ القارئ نفسه لكل تمثيل.",
        f"- القارئ: SGDClassifier logistic-loss، alpha={SGD_ALPHA:g}، eta0={SGD_ETA0:g}، average={SGD_AVERAGE}، دون إعادة تشغيل أمثلة سابقة.",
        f"- العتبة المحددة مسبقًا: **{TARGET_VALIDATION_ACCURACY:.0%} على التحقق فقط**. أول نقطة تحقق العتبة تحدد حجم العينة؛ الاختبار النهائي لا يختارها.",
        "- لا ضبط للمعاملات باستخدام التحقق أو الاختبار. تطبيع الميزات ثابت بحجم النافذة ولا يتعلم من مجموعات التقييم.",
        f"- التنفيذ: {workers} عمليات بالتوازي، بذرة مستقلة لكل مهمة؛ الزمن الكلي {runtime:.1f} ثانية.",
        "", "## متوسط منحنى التعلّم مع فاصل ثقة 95% عبر البذور", "",
        "كل خلية = المتوسط [الحد الأدنى، الحد الأعلى]؛ وحدة إعادة السحب هي البذرة لا المثال.", "",
    ]
    for task in ("sync", "order"):
        lines += [f"### {task}", "",
                  "| التمثيل | " + " | ".join(map(str, TRAIN_SIZES)) + " |",
                  "|---|" + "---:|" * len(TRAIN_SIZES)]
        for feature in FEATURE_SETS:
            cells = []
            for n in TRAIN_SIZES:
                r = lookup[(task, feature, n)]
                mean = r["test_accuracy_mean_percent"]
                lo = r["test_accuracy_ci95_low_percent"]
                hi = r["test_accuracy_ci95_high_percent"]
                val = r["validation_accuracy_mean_percent"]
                cells.append(f"{mean:.1f}% [{lo:.1f}, {hi:.1f}] (val {val:.1f}%)")
            lines.append(f"| {feature} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## هل وصل القارئ إلى 85% على التحقق؟", "",
              "الاختبار المستقل في عمود الاختبار التالي يُحسب عند أول حجم حقق العتبة في مجموعة التحقق لكل بذرة؛ لا يُستخدم لاختيار الحجم.",
              "", "| المهمة | التمثيل | بذور وصلت | نسبة الوصول [CI 95%] | متوسط أول حجم بين الواصلين | وسيط الحجم | دقة الاختبار عند نقطة الاختيار [CI 95%] |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in crossing:
        mean_test = row["independent_test_accuracy_at_validation_crossing_mean_percent"]
        low_test = row["independent_test_accuracy_at_validation_crossing_ci95_low_percent"]
        high_test = row["independent_test_accuracy_at_validation_crossing_ci95_high_percent"]
        size_mean = row["mean_train_size_at_first_crossing_among_reached"]
        size_median = row["median_train_size_at_first_crossing_among_reached"]
        test_cell = "لا توجد بذور مؤهلة" if np.isnan(mean_test) else f"{mean_test:.1f}% [{low_test:.1f}, {high_test:.1f}]"
        sizes = "—" if np.isnan(size_mean) else f"{size_mean:.0f} / {size_median:.0f}"
        lines.append(f"| {row['task']} | {row['feature_set']} | {row['n_reached_85']}/{row['n_seeds']} | "
                     f"{row['reach_rate_percent']:.1f}% [{row['reach_rate_ci95_low_percent']:.1f}, {row['reach_rate_ci95_high_percent']:.1f}] | "
                     f"{sizes} | {test_cell} |")

    overlap_total = sum(t["train_validation_overlap"] + t["train_test_overlap"] + t["validation_test_overlap"]
                        for per_seed in integrity for t in per_seed["tasks"].values())
    lines += ["", "## سلامة البيانات", "",
              f"- سجلات القياس: **{len(records)}** (المتوقع {len(SEEDS) * 2 * len(TRAIN_SIZES) * len(FEATURE_SETS)}).",
              f"- اكتمال البذور: **{len(integrity)}/{len(SEEDS)}**.",
              f"- مجموع تداخلات التسلسلات بين train/validation/test: **{overlap_total}**.",
              "- كل مجموعة اختبار متوازنة ومختلفة تمامًا عن التدريب والتحقق، ومشتركة عبر التمثيلات ونقاط التدريب داخل البذرة.",
              "- القارئ تدريجي فعليًا: يستقبل فقط الجزء الجديد عند كل نقطة، ولا يعاد تدريبه من البداية ولا تُعاد أمثلة قديمة.",
              "", "## التفسير وحدود الاستنتاج", "",
              "- إذا تفوقت 8/16 نافذة على الأربع عند نفس عدد الأمثلة، فهذا يدعم أن ميزات الأربع نوافذ تضغط معلومات زمنية نافعة.",
              "- نسبة الوصول إلى 85% تعني الوصول على مجموعة التحقق؛ تقدير الاختبار المنفصل بجانبها يوضح هل انتقل الأداء إلى بيانات مستقلة.",
              "- عدم الوصول إلى 85% لا يعني أن الخزان لا يحتوي المعلومة؛ يعني فقط أن قارئ SGD بهذه الإعدادات وعدد مرور واحد لم يصل إلى العتبة في هذه الميزانية.",
              "- هذه التجربة لا تختبر STDP أو النسيان أو الاستدلال العام؛ الخزان ثابت والمُتعلم هو قارئ لوجستي فقط.",
              "", "## مخرجات البذور", "",
              "`per_seed_incremental.csv` يحوي دقة التحقق والاختبار واستدعاء كل فئة وتوقيت التحديث لكل بذرة ونقطة تدريب. `per_seed_threshold_crossings.csv` يسجل أول نقطة اجتازت العتبة وفق التحقق فقط.", ""]
    return "\n".join(lines)


def plot_curves(summaries: Sequence[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for task in ("sync", "order"):
        for split in ("validation", "test"):
            fig, ax = plt.subplots(figsize=(9, 6))
            for feature in FEATURE_SETS:
                sub = sorted([r for r in summaries if r["task"] == task and r["feature_set"] == feature],
                             key=lambda r: r["train_size"])
                xs = [r["train_size"] for r in sub]
                ys = [r[f"{split}_accuracy_mean_percent"] for r in sub]
                lo = [r[f"{split}_accuracy_ci95_low_percent"] for r in sub]
                hi = [r[f"{split}_accuracy_ci95_high_percent"] for r in sub]
                ax.plot(xs, ys, marker="o", label=feature)
                ax.fill_between(xs, lo, hi, alpha=0.15)
            ax.axhline(TARGET_VALIDATION_ACCURACY * 100.0, linestyle="--", linewidth=1,
                       label="85% target (selection threshold on validation only)")
            ax.axhline(50.0, linestyle=":", linewidth=1, label="chance")
            ax.set_xscale("log")
            ax.set_xticks(TRAIN_SIZES)
            ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
            ax.set_xlabel("Unique training examples seen once")
            ax.set_ylabel(f"{split.capitalize()} accuracy (%)")
            ax.set_title(f"Nucleus Pulse v0.31 incremental learning — {task} — {split}")
            ax.set_ylim(35, 101)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
            fig.tight_layout()
            fig.savefig(outdir / f"learning_curve_{task}_{split}.png", dpi=160)
            plt.close(fig)


def run_full(outdir: Path, resume: bool = False, workers: int | None = None) -> dict:
    protocol = verify_preregistration()
    workers = max(1, min(int(workers or min(3, os.cpu_count() or 1)), 3, os.cpu_count() or 1))
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
            raise RuntimeError("Partial output includes seeds outside the frozen protocol")
        print(f"Resuming {len(completed)}/{len(SEEDS)} seeds", flush=True)

    start = time.time()
    remaining = [s for s in SEEDS if s not in completed]
    print(f"Running v0.31: {len(remaining)} seeds, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_seed, seed): seed for seed in remaining}
        for future in as_completed(futures):
            seed = futures[future]
            seed_records, seed_integrity = future.result()
            records.extend(seed_records)
            integrity.append(seed_integrity)
            completed.add(seed)
            atomic_json(partial_path, {"version": "0.31", "completed_seeds": [s for s in SEEDS if s in completed],
                                       "records": records, "integrity": integrity, "partial": True,
                                       "execution_workers": workers})
            print(f"[{len(completed)}/{len(SEEDS)}] seed={seed} records={len(records)} elapsed={time.time()-start:.1f}s", flush=True)

    seed_order = {seed: i for i, seed in enumerate(SEEDS)}
    task_order = {"sync": 0, "order": 1}
    feature_order = {name: i for i, name in enumerate(FEATURE_SETS)}
    records.sort(key=lambda r: (seed_order[r["seed"]], task_order[r["task"]],
                                feature_order[r["feature_set"]], r["train_size"]))
    integrity.sort(key=lambda r: seed_order[r["seed"]])
    summaries = aggregate_records(records)
    seed_crossings, crossing = crossing_summary(records)
    overlap_total = sum(t["train_validation_overlap"] + t["train_test_overlap"] + t["validation_test_overlap"]
                        for item in integrity for t in item["tasks"].values())
    if len(records) != len(SEEDS) * 2 * len(TRAIN_SIZES) * len(FEATURE_SETS):
        raise RuntimeError("Unexpected number of records")
    if len(integrity) != len(SEEDS) or overlap_total != 0:
        raise RuntimeError("Final integrity check failed")

    result = {
        "version": "0.31", "seeds": SEEDS, "train_sizes": TRAIN_SIZES,
        "feature_sets": FEATURE_SETS, "n_validation_per_task_per_seed": N_VALIDATION,
        "n_test_per_task_per_seed": N_TEST, "target_validation_accuracy": TARGET_VALIDATION_ACCURACY,
        "online_sgd_config": {"loss": SGD_LOSS, "penalty": SGD_PENALTY, "alpha": SGD_ALPHA,
                               "learning_rate": "constant", "eta0": SGD_ETA0, "average": SGD_AVERAGE,
                               "one_pass_per_unique_example": True},
        "records": records, "summary": summaries, "threshold_crossings_per_seed": seed_crossings,
        "threshold_summary": crossing, "integrity": integrity, "runtime_seconds": float(time.time() - start),
        "protocol": protocol, "execution_workers": workers,
    }
    atomic_json(outdir / "results.json", result)
    write_csv(outdir / "per_seed_incremental.csv", records)
    write_csv(outdir / "incremental_curve_summary.csv", summaries)
    write_csv(outdir / "per_seed_threshold_crossings.csv", seed_crossings)
    write_csv(outdir / "threshold_crossing_summary.csv", crossing)
    (outdir / "report_ar.md").write_text(render_report(records, summaries, seed_crossings, crossing,
                                                        integrity, result["runtime_seconds"], workers),
                                           encoding="utf-8")
    plot_curves(summaries, outdir)
    partial_path.unlink(missing_ok=True)
    print("FINAL RESULTS SAVED", flush=True)
    for row in crossing:
        test = row["independent_test_accuracy_at_validation_crossing_mean_percent"]
        print(f"{row['task']:5s} {row['feature_set']:25s} reached={row['n_reached_85']}/{row['n_seeds']} "
              f"mean_size={row['mean_train_size_at_first_crossing_among_reached']:.1f} "
              f"test@cross={test:.2f}", flush=True)
    return result


def run_smoke() -> None:
    # These engineering-only seeds are outside the frozen evaluation seeds.
    seed = 3990
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    for task_i, task in enumerate(("sync", "order")):
        train = make_nested_training_pool(seed + 11 + task_i, task, 200)
        keys_train = {trial_key(t) for t in train}
        val = make_eval_dataset(seed + 21 + task_i, task, 40, set(keys_train))
        keys_val = {trial_key(t) for t in val}
        test = make_eval_dataset(seed + 31 + task_i, task, 40, set(keys_train) | keys_val)
        inp, spk = simulate_batch_vectorized(reservoir, train)
        vi, vs = simulate_batch_vectorized(reservoir, val)
        ti, ts = simulate_batch_vectorized(reservoir, test)
        ftr = transform_features(make_features(inp, spk))
        fva = transform_features(make_features(vi, vs))
        fte = transform_features(make_features(ti, ts))
        ytr, yva, yte = make_epoch_labels(train), make_epoch_labels(val), make_epoch_labels(test)
        for feature_i, feature in enumerate(FEATURE_SETS):
            model = new_online_reader(seed + feature_i)
            model.partial_fit(ftr[feature][:100], ytr[:100], classes=np.asarray([0, 1]))
            model.partial_fit(ftr[feature][100:200], ytr[100:200])
            assert model.predict(fva[feature]).shape == (40,)
            assert model.predict(fte[feature]).shape == (40,)
            assert 0.0 <= classification_metrics(yva, model.predict(fva[feature]))["accuracy"] <= 1.0
        assert not (keys_train & keys_val or keys_train & {trial_key(t) for t in test}
                    or keys_val & {trial_key(t) for t in test})
    print("SMOKE PASSED: online updates, three feature widths, class labels, and disjoint evaluation sets", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="engineering smoke test; not benchmark evidence")
    parser.add_argument("--resume", action="store_true", help="resume frozen run from partial results")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
    else:
        run_full(ROOT / "results" / "incremental_windows_v0_31", resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
