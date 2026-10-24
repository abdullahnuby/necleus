"""Nucleus Pulse v0.30: learning curves for reservoir and raw-input readouts.

This is a diagnostic, not a continual-learning/STDP comparison. A fixed LIF
reservoir is simulated once per independent train/test set. Several readouts
are trained on nested, class-balanced sample budgets and evaluated on the same
held-out set. Test data never select hyperparameters or feature representations.
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
    DURATION,
    DynamicPulseReservoir,
    RewardSTDPReadout,
    TemporalTrial,
    make_disjoint_dataset,
    rasterize,
)

SEEDS = [104, 887, 2027, 90210, 731] + list(range(3000, 3035))
TRAIN_SIZES = [200, 500, 1000, 2000, 6000]
N_TRAIN_POOL = max(TRAIN_SIZES)
N_TEST_PER_TASK = 1000
N_HIDDEN = 24
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20301024

# These are frozen before the full run. Logistic regression is standardized;
# tree readouts use raw feature magnitudes. No parameter tuning uses test labels.
LOGISTIC_C = 1.0
LOGISTIC_MAX_ITER = 250
GBM_MAX_ITER = 80
GBM_MAX_LEAF_NODES = 15
GBM_MIN_SAMPLES_LEAF = 20
GBM_L2 = 1.0
PIPELINES = [
    {"reader": "linear", "feature_set": "phase4_96"},
    {"reader": "linear", "feature_set": "spike_counts_24"},
    {"reader": "linear", "feature_set": "spike_bins8_192"},
    {"reader": "linear", "feature_set": "spike_bins16_384"},
    {"reader": "linear", "feature_set": "spike_raster_768"},
    {"reader": "gbm", "feature_set": "spike_raster_768"},
    {"reader": "linear", "feature_set": "raw_input_128"},
    {"reader": "gbm", "feature_set": "raw_input_128"},
]


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
        "test_per_task_per_seed": N_TEST_PER_TASK,
        "hidden_neurons": N_HIDDEN,
        "pipelines": PIPELINES,
        "logistic_C": LOGISTIC_C,
        "logistic_max_iter": LOGISTIC_MAX_ITER,
        "gbm_max_iter": GBM_MAX_ITER,
        "gbm_max_leaf_nodes": GBM_MAX_LEAF_NODES,
        "gbm_min_samples_leaf": GBM_MIN_SAMPLES_LEAF,
        "gbm_l2": GBM_L2,
        "test_used_for_tuning": False,
        "nested_class_balanced_training_subsets": True,
    }


def verify_preregistration() -> dict:
    p = ROOT / "results" / "learning_curve_v0_30" / "preregistration.json"
    if not p.exists():
        raise RuntimeError("Missing frozen v0.30 preregistration; refusing full run.")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("The current run configuration differs from preregistration.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return doc


def trial_key(trial: TemporalTrial) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (trial.a, trial.b)


def make_nested_training_pool(seed: int, task: str, n: int = N_TRAIN_POOL) -> list[TemporalTrial]:
    """Return nested prefixes that are class balanced at every even train size."""
    pool = make_disjoint_dataset(seed, task, n, "train")
    by_class = {0: [t for t in pool if t.target_class == 0],
                1: [t for t in pool if t.target_class == 1]}
    if len(by_class[0]) != n // 2 or len(by_class[1]) != n // 2 or n % 2:
        raise RuntimeError("The training pool must have exactly balanced classes and even size.")
    nested: list[TemporalTrial] = []
    for i in range(n // 2):
        nested.append(by_class[0][i])
        nested.append(by_class[1][i])
    return nested


def rasterize_batch(trials: Sequence[TemporalTrial], duration: int = DURATION) -> np.ndarray:
    x = np.zeros((len(trials), duration, 4), dtype=np.float32)
    for i, trial in enumerate(trials):
        if trial.a:
            x[i, np.asarray(trial.a, dtype=int), 0] = 1.0
        if trial.b:
            x[i, np.asarray(trial.b, dtype=int), 1] = 1.0
        x[i, 0, 2 if trial.task == "sync" else 3] = 1.0
    return x


def simulate_batch_vectorized(reservoir: DynamicPulseReservoir,
                              trials: Sequence[TemporalTrial]) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized, state-isolated simulation equivalent to reservoir.simulate.

    The sample dimension is batched for runtime; each trial maintains its own
    membrane, synaptic current and refractory state. No state crosses examples.
    Returns (input raster, hidden spike raster).
    """
    if not trials:
        return (np.empty((0, reservoir.duration, 4), np.float32),
                np.empty((0, reservoir.duration, reservoir.n_hidden), np.float32))
    inputs = rasterize_batch(trials, reservoir.duration)
    count = len(trials)
    h = reservoir.n_hidden
    v = np.zeros((count, h), dtype=np.float32)
    i_syn = np.zeros((count, h), dtype=np.float32)
    refractory = np.zeros((count, h), dtype=np.int8)
    previous_spikes = np.zeros((count, h), dtype=np.float32)
    spikes = np.zeros((count, reservoir.duration, h), dtype=np.float32)
    w_in_t = reservoir.w_in.T
    w_rec_t = reservoir.w_rec.T
    for t in range(reservoir.duration):
        i_syn = (reservoir.syn_leak[None, :] * i_syn
                 + inputs[:, t, :] @ w_in_t
                 + previous_spikes @ w_rec_t)
        v = reservoir.reset_leak[None, :] * v + (1.0 - reservoir.reset_leak[None, :]) * i_syn
        active = (v >= reservoir.threshold[None, :]) & (refractory == 0)
        spikes[:, t, :] = active.astype(np.float32)
        v[active] = 0.0
        refractory = np.maximum(refractory - 1, 0)
        refractory[active] = 1
        previous_spikes = active.astype(np.float32)
    return inputs, spikes


def make_features(inputs: np.ndarray, spikes: np.ndarray) -> dict[str, np.ndarray]:
    """Build frozen feature views from the same raw input and reservoir run."""
    n, duration, hidden = spikes.shape
    if duration != 32 or hidden != N_HIDDEN:
        raise ValueError(f"Expected (N,32,{N_HIDDEN}) spikes; received {spikes.shape}")

    phase_parts = []
    for start, stop in RewardSTDPReadout.PHASE_BINS:
        marker = stop - 1
        times = np.arange(start, stop, dtype=np.float32)
        kernel = np.exp(-(marker - times) / RewardSTDPReadout.TRACE_TAU).astype(np.float32)
        phase_parts.append(np.sum(spikes[:, start:stop, :] * kernel[None, :, None], axis=1))
    phase4 = np.concatenate(phase_parts, axis=1)
    phase4 /= np.maximum(np.linalg.norm(phase4, axis=1, keepdims=True), 1e-8)

    bins8 = spikes.reshape(n, 8, 4, hidden).sum(axis=2)
    bins16 = spikes.reshape(n, 16, 2, hidden).sum(axis=2)
    features = {
        "phase4_96": phase4.astype(np.float32),
        "spike_counts_24": spikes.sum(axis=1).astype(np.float32),
        "spike_bins8_192": bins8.reshape(n, -1).astype(np.float32),
        "spike_bins16_384": bins16.reshape(n, -1).astype(np.float32),
        "spike_raster_768": spikes.reshape(n, -1).astype(np.float32),
        "raw_input_128": inputs.reshape(n, -1).astype(np.float32),
    }
    return features


def make_epoch_labels(trials: Sequence[TemporalTrial]) -> np.ndarray:
    return np.asarray([t.target_class for t in trials], dtype=np.int64)


def new_reader(reader: str, seed: int):
    if reader == "linear":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=LOGISTIC_C, max_iter=LOGISTIC_MAX_ITER,
                               solver="lbfgs", random_state=int(seed)),
        )
    if reader == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=GBM_MAX_ITER,
            learning_rate=0.1,
            max_leaf_nodes=GBM_MAX_LEAF_NODES,
            min_samples_leaf=GBM_MIN_SAMPLES_LEAF,
            l2_regularization=GBM_L2,
            early_stopping=False,
            random_state=int(seed),
        )
    raise ValueError(f"Unknown reader: {reader}")


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    r0 = float(np.mean(y_pred[y_true == 0] == 0))
    r1 = float(np.mean(y_pred[y_true == 1] == 1))
    return {
        "n_test": int(y_true.size),
        "accuracy": float(np.mean(y_pred == y_true)),
        "balanced_accuracy": 0.5 * (r0 + r1),
        "class_0_recall": r0,
        "class_1_recall": r1,
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
    }


def make_test_dataset(seed: int, task: str, train_trials: Sequence[TemporalTrial], n: int) -> list[TemporalTrial]:
    forbidden = {trial_key(t) for t in train_trials}
    test = make_disjoint_dataset(seed, task, n, "test", forbidden=forbidden)
    test_keys = {trial_key(t) for t in test}
    if len(test_keys) != n or test_keys & forbidden:
        raise RuntimeError(f"Train/test overlap or duplicate test sequences in {task}.")
    return test


def summarize_group(values: Sequence[float], seed: int) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    if not len(arr):
        return {"n_seeds": 0, "mean_percent": None, "ci95_low_percent": None, "ci95_high_percent": None}
    if len(arr) == 1:
        return {"n_seeds": 1, "mean_percent": 100.0 * float(arr.mean()),
                "ci95_low_percent": None, "ci95_high_percent": None}
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPS, dtype=np.float64)
    chunk = 1000
    for start in range(0, BOOTSTRAP_REPS, chunk):
        count = min(chunk, BOOTSTRAP_REPS - start)
        indices = rng.integers(0, len(arr), size=(count, len(arr)))
        draws[start:start + count] = arr[indices].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return {"n_seeds": len(arr), "mean_percent": 100.0 * float(arr.mean()),
            "ci95_low_percent": 100.0 * float(lo), "ci95_high_percent": 100.0 * float(hi)}


def _stable_seed(*parts: str) -> int:
    raw = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def run_seed(seed: int, train_sizes: Sequence[int], n_test: int,
             progress=None) -> tuple[list[dict], dict]:
    started = time.time()
    reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
    seed_records: list[dict] = []
    integrity: dict = {"seed": int(seed), "tasks": {}}
    for task_i, task in enumerate(("sync", "order")):
        # Training sample pool is full-size and nested; test is generated against
        # every training pattern, including training points beyond smaller budgets.
        train_pool = make_nested_training_pool(seed + 11011 + task_i * 100, task, N_TRAIN_POOL)
        test_trials = make_test_dataset(seed + 51011 + task_i * 100, task, train_pool, n_test)
        train_inputs, train_spikes = simulate_batch_vectorized(reservoir, train_pool)
        test_inputs, test_spikes = simulate_batch_vectorized(reservoir, test_trials)
        train_features = make_features(train_inputs, train_spikes)
        test_features = make_features(test_inputs, test_spikes)
        y_pool = make_epoch_labels(train_pool)
        y_test = make_epoch_labels(test_trials)

        # Structural integrity: no exact sequence overlap and sample sets are nested.
        pool_keys = {trial_key(t) for t in train_pool}
        test_keys = {trial_key(t) for t in test_trials}
        assert len(pool_keys) == N_TRAIN_POOL
        assert len(test_keys) == n_test
        assert not (pool_keys & test_keys)
        for n in train_sizes:
            assert n % 2 == 0 and 0 < n <= N_TRAIN_POOL
            assert int(np.sum(y_pool[:n] == 0)) == n // 2
            assert int(np.sum(y_pool[:n] == 1)) == n // 2

        integrity["tasks"][task] = {
            "n_train_pool": len(train_pool), "n_test": len(test_trials),
            "train_pool_unique_sequences": len(pool_keys), "test_unique_sequences": len(test_keys),
            "train_test_overlap": len(pool_keys & test_keys),
            "class_balance_train_pool": {"0": int(np.sum(y_pool == 0)), "1": int(np.sum(y_pool == 1))},
            "class_balance_test": {"0": int(np.sum(y_test == 0)), "1": int(np.sum(y_test == 1))},
            "nested_train_sizes_class_balanced": True,
        }

        for train_size in train_sizes:
            y_train = y_pool[:train_size]
            for pipe in PIPELINES:
                reader = pipe["reader"]
                feature_set = pipe["feature_set"]
                x_train = train_features[feature_set][:train_size]
                x_test = test_features[feature_set]
                model = new_reader(reader, seed=seed + 700001 + task_i * 1000 + train_size)
                fit_started = time.time()
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                metrics = classification_metrics(y_test, y_pred)
                seed_records.append({
                    "seed": int(seed), "task": task, "train_size": int(train_size),
                    "reader": reader, "feature_set": feature_set,
                    **metrics,
                    "fit_seconds": round(time.time() - fit_started, 4),
                })
            if progress:
                progress({"seed": seed, "task": task, "train_size": train_size,
                          "completed_models": len(seed_records)})
    integrity["elapsed_seconds"] = round(time.time() - started, 3)
    return seed_records, integrity


def aggregate_results(records: Sequence[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[(row["task"], row["reader"], row["feature_set"], row["train_size"])].append(row)
    summaries = []
    for (task, reader, feature, n), rows in sorted(groups.items()):
        summary = {
            "task": task, "reader": reader, "feature_set": feature,
            "train_size": n, "n_seeds": len({r["seed"] for r in rows}),
        }
        key_parts = (task, reader, feature, str(n))
        for metric in ("accuracy", "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate"):
            ci = summarize_group([r[metric] for r in rows], _stable_seed(*key_parts, metric))
            summary[f"{metric}_mean_percent"] = ci["mean_percent"]
            summary[f"{metric}_ci95_low_percent"] = ci["ci95_low_percent"]
            summary[f"{metric}_ci95_high_percent"] = ci["ci95_high_percent"]
        summaries.append(summary)
    return summaries


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_records_csv(path: Path, records: Sequence[dict]) -> None:
    fields = ["seed", "task", "train_size", "reader", "feature_set", "n_test", "accuracy",
              "balanced_accuracy", "class_0_recall", "class_1_recall", "predicted_positive_rate", "fit_seconds"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def write_summary_csv(path: Path, summaries: Sequence[dict]) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def render_report(seeds: Sequence[int], records: Sequence[dict], summaries: Sequence[dict],
                  integrity: Sequence[dict], runtime: float, train_sizes: Sequence[int], n_test: int) -> str:
    lookup = {(r["task"], r["reader"], r["feature_set"], r["train_size"]): r for r in summaries}
    lines = [
        "# تقرير منحنى التعلّم — Nucleus Pulse v0.30", "",
        "## سؤال التجربة", "",
        "هل ضعف الأداء في v0.29 ناتج عن ميزانية أمثلة صغيرة، أو ضغط التوقيت في ميزات المرحلة الأربع، أم محدودية الخزان نفسه؟",
        "هذه التجربة تشخّص قابلية فكّ الترميز (decodability) وكفاءة العينات. **لا تختبر النسيان ولا تغيّر قاعدة STDP.**",
        "", "## البروتوكول المجمد", "",
        f"- البذور: **{len(seeds)}**؛ نفس قائمة البذور المسجلة في البروتوكول.",
        f"- أحجام التدريب لكل مهمة: **{', '.join(map(str, train_sizes))}**، ومجموعات متداخلة ومتوازنة الفئتين.",
        f"- الاختبار: **{n_test} مثال لكل مهمة ولكل بذرة**، منفصل عن كل أنماط التدريب، ويظل نفسه عبر جميع القارئات وأحجام التدريب في البذرة.",
        f"- مجموعات المهام: sync (التزامن) وorder (A قبل B بفارق 2–4 نبضات).", 
        "- 40 خزّانًا ثابتًا مختلفًا حسب البذرة؛ محاكاة الخزان لا تتدرّب على الملصقات.",
        "- القارئ الخطي: StandardScaler + LogisticRegression، C=1.0، max_iter=250.",
        f"- القارئ غير الخطي: HistGradientBoosting، {GBM_MAX_ITER} تكرارًا، و{GBM_MAX_LEAF_NODES} ورقة كحد أقصى، دون early stopping.",
        "- أحجام التدريب متداخلة: عينة بحجم أصغر جزء من العينة الأكبر، وبكل حجم عدد متساوٍ من الفئتين.",
        "- لا تستخدم نتيجة الاختبار لضبط الميزات أو المعاملات؛ كل CI يُحسب بإعادة أخذ عينات من متوسط كل بذرة.",
        "- تعديل جدولة التنفيذ: 13 بذرة اكتملت بالمشغل التسلسلي الأول؛ بقية البذور عولجت بالتوازي على عمليات مستقلة. لم تتغير بيانات البذور أو نماذجها أو إعداداتها أو الاختبارات.",
        "", "## النتائج: دقة الاختبار ومتوسط CI 95% على مستوى البذور", "",
        "كل خلية تعرض المتوسط والنطاق المئوي 95%. النتائج الوصفية الكاملة لكل بذرة موجودة في CSV.", ""
    ]
    for task in ("sync", "order"):
        lines.extend([f"### المهمة: {task}", "",
                      "| القارئ / الميزات | " + " | ".join(map(str, train_sizes)) + " |",
                      "|---|" + "---:|" * len(train_sizes)])
        for pipe in PIPELINES:
            cells = []
            for n in train_sizes:
                r = lookup[(task, pipe["reader"], pipe["feature_set"], n)]
                lo, mean, hi = (r["accuracy_ci95_low_percent"], r["accuracy_mean_percent"], r["accuracy_ci95_high_percent"])
                cells.append(f"{mean:.1f}% [{lo:.1f}, {hi:.1f}]")
            lines.append(f"| {pipe['reader']} / {pipe['feature_set']} | " + " | ".join(cells) + " |")
        lines.append("")

    # Endpoint interpretation table to make the data-size trend easy to inspect.
    lines.extend(["## مقارنة قصيرة: 200 مقابل 6000 مثال", "",
                  "| المهمة | القارئ / الميزات | 200 مثال | 6000 مثال | الفرق (نقطة) |", "|---|---|---:|---:|---:|"])
    for task in ("sync", "order"):
        for pipe in PIPELINES:
            low = lookup[(task, pipe["reader"], pipe["feature_set"], 200)]["accuracy_mean_percent"]
            high = lookup[(task, pipe["reader"], pipe["feature_set"], 6000)]["accuracy_mean_percent"]
            lines.append(f"| {task} | {pipe['reader']} / {pipe['feature_set']} | {low:.2f}% | {high:.2f}% | {high-low:+.2f} |")
    lines.extend([
        "", "## سلامة البيانات", "",
        f"عدد سجلات القياس: **{len(records)}**؛ عدد البذور: **{len(seeds)}**. لكل بذرة ومهمة، التقرير يثبت تفرد أمثلة التدريب والاختبار وعدم تداخل التسلسلات؛ ومجموعة التدريب متوازنة في كل حجم.",
        f"تداخلات أنماط train/test المسجلة في جميع المهام: **{sum(t['train_test_overlap'] for row in integrity for t in row['tasks'].values())}**.",
        "", "## كيف نقرأ منحنى التعلّم؟", "",
        "- إذا تحسنت ميزات الأربع نوافذ بشدة عند تكبير العينة، فميزانية الأمثلة جزء من المشكلة.",
        "- إذا تفوقت ميزات raster أو bins الأغنى على phase4 عند الحجم نفسه، فالضغط في بنك الأثر الأربع قد يهدر توقيتًا مفيدًا.",
        "- إذا تفوق القارئ غير الخطي على الخطي على التمثيل نفسه، فهذا يشير إلى علاقة قرار غير خطية أو تفاعلات لم يستغلها القارئ الخطي.",
        "- إذا اقترب القارئ الخطي من غير الخطي عند حجم كبير، فهذا يشير إلى أن تمثيل الخزان قابل للفك خطيًا على هذه المهمة، لا أن قاعدة تعلم نبضية قد تعلمت ذلك بكفاءة.",
        "- نتيجة المدخل الخام هي baseline منفصل: لا تثبت أن الخزان أفضل أو أسوأ دون قراءة النتائج وفواصل الثقة.",
        "", "## الحدود", "",
        "هذه المقارنة تقيس تمثيلات وقراء supervised، لا قاعدة التعلم بالمكافأة نفسها. لا نستنتج منها تفوق STDP أو كفاءة التعلّم من أمثلة قليلة. لا تُستخدم نتائج منحنى التطوير كاختبار نهائي مستقل بعد ضبط ميزات أو معاملات بناءً عليها.",
        f"", f"زمن التشغيل: **{runtime:.1f} ثانية**.", ""
    ])
    return "\n".join(lines)


def plot_learning_curves(summaries: Sequence[dict], outdir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for task in ("sync", "order"):
        fig, ax = plt.subplots(figsize=(11, 7))
        rows = [r for r in summaries if r["task"] == task]
        for pipe in PIPELINES:
            sub = sorted([r for r in rows if r["reader"] == pipe["reader"] and r["feature_set"] == pipe["feature_set"]], key=lambda r: r["train_size"])
            xs = [r["train_size"] for r in sub]
            ys = [r["accuracy_mean_percent"] for r in sub]
            los = [r["accuracy_ci95_low_percent"] for r in sub]
            his = [r["accuracy_ci95_high_percent"] for r in sub]
            ax.plot(xs, ys, marker="o", label=f"{pipe['reader']} / {pipe['feature_set']}")
            ax.fill_between(xs, los, his, alpha=0.10)
        ax.axhline(50.0, linestyle="--", linewidth=1, label="chance (balanced classes)")
        ax.set_xscale("log")
        ax.set_xticks(TRAIN_SIZES if train_sizes_in_summaries(summaries) == TRAIN_SIZES else sorted({r["train_size"] for r in rows}))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.set_xlabel("Training examples per task (log scale)")
        ax.set_ylabel("Test accuracy (%)")
        ax.set_title(f"Nucleus Pulse v0.30 learning curve — {task}")
        ax.set_ylim(35, 101)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        fig.savefig(outdir / f"learning_curve_{task}.png", dpi=160)
        plt.close(fig)


def train_sizes_in_summaries(summaries: Sequence[dict]) -> list[int]:
    return sorted({int(r["train_size"]) for r in summaries})


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
            raise RuntimeError("Partial output contains seed(s) not in the frozen protocol.")
        print(f"Resuming with {len(completed)}/{len(SEEDS)} completed seeds.", flush=True)

    start = time.time()
    remaining = [s for s in SEEDS if s not in completed]
    print(f"Executing {len(remaining)} remaining seeds with {workers} worker(s).", flush=True)
    # Each seed is independent. Parallelizing seed jobs changes only scheduling;
    # model hyperparameters, per-seed random streams and all train/test data stay fixed.
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(run_seed, seed, TRAIN_SIZES, N_TEST_PER_TASK): seed for seed in remaining}
        for future in as_completed(futures):
            seed = futures[future]
            seed_rows, seed_integrity = future.result()
            records.extend(seed_rows)
            integrity.append(seed_integrity)
            completed.add(seed)
            atomic_json(partial_path, {"version": "0.30", "completed_seeds": [s for s in SEEDS if s in completed],
                                       "records": records, "integrity": integrity, "partial": True,
                                       "execution_workers": workers})
            print(f"[{len(completed)}/{len(SEEDS)}] seed={seed} records={len(records)} "
                  f"total={time.time()-start:.1f}s", flush=True)

    seed_order = {seed: i for i, seed in enumerate(SEEDS)}
    task_order = {"sync": 0, "order": 1}
    pipe_order = {(p["reader"], p["feature_set"]): i for i, p in enumerate(PIPELINES)}
    records.sort(key=lambda r: (seed_order[r["seed"]], task_order[r["task"]], r["train_size"],
                                pipe_order[(r["reader"], r["feature_set"])]))
    integrity.sort(key=lambda r: seed_order[r["seed"]])
    summaries = aggregate_results(records)
    result = {
        "version": "0.30",
        "seeds": SEEDS,
        "train_sizes": TRAIN_SIZES,
        "pipelines": PIPELINES,
        "n_test_per_task_per_seed": N_TEST_PER_TASK,
        "records": records,
        "summary": summaries,
        "integrity": integrity,
        "runtime_seconds": float(time.time() - start),
        "protocol": protocol,
        "execution_metadata": {
            "worker_processes_for_remaining_seeds": workers,
            "parallel_execution": True,
            "note": "13 preregistered seeds completed under the original sequential runner; the remaining seeds used process-level parallel scheduling. Per-seed model, feature extraction, data generation, test set, and metrics are unchanged.",
        },
    }
    atomic_json(outdir / "results.json", result)
    write_records_csv(outdir / "per_seed_learning_curve.csv", records)
    write_summary_csv(outdir / "learning_curve_summary.csv", summaries)
    (outdir / "report_ar.md").write_text(render_report(SEEDS, records, summaries, integrity,
                                                        result["runtime_seconds"], TRAIN_SIZES,
                                                        N_TEST_PER_TASK), encoding="utf-8")
    plot_learning_curves(summaries, outdir)
    partial_path.unlink(missing_ok=True)
    print("FINAL RESULTS SAVED", flush=True)
    for row in summaries:
        print(f"{row['task']:5s} {row['reader']:6s} {row['feature_set']:20s} n={row['train_size']:4d} "
              f"acc={row['accuracy_mean_percent']:.2f}% CI=[{row['accuracy_ci95_low_percent']:.2f},"
              f"{row['accuracy_ci95_high_percent']:.2f}] nseed={row['n_seeds']}", flush=True)
    return result


def run_smoke() -> None:
    """Small test run; never presented as benchmark evidence."""
    seeds = [104]
    sizes = [40, 80]
    records: list[dict] = []
    integrity = []
    start = time.time()
    for seed in seeds:
        # Smoke uses 1000 pool examples, then smaller nested prefixes.
        reservoir = DynamicPulseReservoir(seed=seed, n_hidden=N_HIDDEN)
        for task_i, task in enumerate(("sync", "order")):
            train_pool = make_nested_training_pool(seed + 900 + task_i, task, 1000)
            test = make_test_dataset(seed + 1900 + task_i, task, train_pool, 100)
            xtr, strn = simulate_batch_vectorized(reservoir, train_pool)
            xte, ste = simulate_batch_vectorized(reservoir, test)
            ftr, fte = make_features(xtr, strn), make_features(xte, ste)
            ytr, yte = make_epoch_labels(train_pool), make_epoch_labels(test)
            assert not ({trial_key(t) for t in train_pool} & {trial_key(t) for t in test})
            for size in sizes:
                for pipe in PIPELINES:
                    model = new_reader(pipe["reader"], seed + task_i + size)
                    model.fit(ftr[pipe["feature_set"]][:size], ytr[:size])
                    metrics = classification_metrics(yte, model.predict(fte[pipe["feature_set"]]))
                    assert 0 <= metrics["accuracy"] <= 1
        print(f"SMOKE PASSED seed={seed} tasks=2 sizes={sizes} elapsed={time.time()-start:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run a small engineering smoke check; not benchmark evidence")
    parser.add_argument("--resume", action="store_true", help="resume a full frozen run from partial_results.json")
    parser.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1),
                        help="parallel seed workers (capped at available CPUs and 3)")
    args = parser.parse_args()
    if args.smoke:
        run_smoke()
        return
    run_full(ROOT / "results" / "learning_curve_v0_30", resume=args.resume, workers=args.workers)


if __name__ == "__main__":
    main()
