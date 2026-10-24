"""Nucleus Pulse v0.42: preregistered low-shot STDP-vs-linear readout comparison.

The reservoir is frozen. Both learners receive the same 16-window hidden-spike
rate features and the same class-balanced training examples. The STDP-style
reader updates online once per example using reward-modulated pre/post timing
eligibility; the conventional linear baseline is a fixed, regularized logistic
regression fitted on each nested sample budget. No test-based tuning is allowed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pulse_core.sequential_stdp import (  # noqa: E402
    DURATION,
    DynamicPulseReservoir,
    TemporalTrial,
    make_disjoint_dataset,
)
from experiments.run_learning_curve_v0_30 import (  # noqa: E402
    N_HIDDEN,
    make_nested_training_pool,
    simulate_batch_vectorized,
    trial_key,
)

# Confirmatory protocol; do not adjust after inspecting results.
SEEDS = list(range(5200, 5240))
SMOKE_SEEDS = [5000, 5001, 5100, 5101]
TRAIN_SIZES = [50, 100, 200]
N_TRAIN_POOL = 200
N_TEST_PER_TASK = 1000
TASKS = ("sync", "order")
N_WINDOWS = 16
WINDOW_WIDTH = 2
STDP_LR = 0.08
STDP_TEMPERATURE = 0.70
STDP_TRACE_TAU_WINDOWS = 4.0
STDP_INITIAL_WEIGHT_STD = 0.05
STDP_REWARD_BASELINE_MOMENTUM = 0.97
STDP_WEIGHT_CLIP = 1.5
LINEAR_C = 1.0
LINEAR_MAX_ITER = 250
BOOTSTRAP_REPS = 10000
BOOTSTRAP_SEED = 20421024
MIN_PRACTICAL_GAIN_PP = 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_config() -> dict:
    return {
        "seeds": SEEDS,
        "tasks": list(TASKS),
        "train_sizes": TRAIN_SIZES,
        "train_pool_per_task_per_seed": N_TRAIN_POOL,
        "test_examples_per_task_per_seed": N_TEST_PER_TASK,
        "test_is_iid_same_onset_range": True,
        "hidden_neurons": N_HIDDEN,
        "duration_ticks": DURATION,
        "feature_representation": "16-window hidden spike rates; each window is 2 ticks; 16*24=384 dimensions",
        "class_balance": "exactly balanced at every nested training checkpoint and in each test set",
        "train_subsets": "nested prefixes; each unique example exposed once to STDP",
        "stdp": {
            "type": "reward-modulated pair-timing eligibility / three-factor readout abstraction",
            "lr": STDP_LR,
            "softmax_temperature": STDP_TEMPERATURE,
            "trace_tau_windows": STDP_TRACE_TAU_WINDOWS,
            "initial_weight_std": STDP_INITIAL_WEIGHT_STD,
            "reward_baseline_momentum": STDP_REWARD_BASELINE_MOMENTUM,
            "weight_clip": STDP_WEIGHT_CLIP,
            "epochs_per_example": 1,
            "test_prediction": "deterministic argmax",
        },
        "linear": {
            "model": "StandardScaler + LogisticRegression(lbfgs, L2)",
            "C": LINEAR_C,
            "max_iter": LINEAR_MAX_ITER,
            "fit_each_budget_from_scratch": True,
        },
        "primary_endpoint": "seed-cluster paired difference (STDP - linear) in mean test accuracy across sync/order at n=200",
        "pass_gate": "mean paired gain >= 2.0 percentage points AND seed-cluster bootstrap 95% CI lower bound > 0",
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "test_used_for_tuning": False,
        "test_used_for_checkpoint_selection": False,
        "validation_split": "none; fixed parameters carried over / specified before this run",
    }


def verify_preregistration() -> dict:
    p = ROOT / "results" / "stdp_vs_linear_v0_42" / "preregistration.json"
    if not p.exists():
        raise RuntimeError("Missing frozen v0.42 preregistration; refusing official run.")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("config") != protocol_config():
        raise RuntimeError("Run configuration differs from preregistration; refusing to run.")
    for relative, expected in doc.get("sha256", {}).items():
        actual = sha256_file(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"Frozen-source hash mismatch: {relative}")
    return doc


def extract_16window_rate_features(spikes: np.ndarray) -> np.ndarray:
    """Return unfit 16-window rate features, shape (N, 384), values in [0, 1]."""
    if spikes.ndim != 3 or spikes.shape[1:] != (DURATION, N_HIDDEN):
        raise ValueError(f"Expected (N,{DURATION},{N_HIDDEN}) spikes, got {spikes.shape}")
    n = spikes.shape[0]
    x = spikes.reshape(n, N_WINDOWS, WINDOW_WIDTH, N_HIDDEN).sum(axis=2) / float(WINDOW_WIDTH)
    x = x.reshape(n, N_WINDOWS * N_HIDDEN).astype(np.float32, copy=False)
    if not np.isfinite(x).all():
        raise RuntimeError("Non-finite 16-window feature values")
    if x.size and (float(x.min()) < 0.0 or float(x.max()) > 1.0):
        raise RuntimeError("Window rate feature outside [0, 1]")
    return x


def encode_trials(reservoir: DynamicPulseReservoir, trials: Sequence[TemporalTrial], chunk_size: int = 256) -> np.ndarray:
    """Simulate independent trials in memory-bounded batches using one frozen reservoir."""
    parts: list[np.ndarray] = []
    for left in range(0, len(trials), chunk_size):
        batch = trials[left:left + chunk_size]
        _inputs, spikes = simulate_batch_vectorized(reservoir, batch)
        parts.append(extract_16window_rate_features(spikes))
    if not parts:
        return np.empty((0, N_WINDOWS * N_HIDDEN), dtype=np.float32)
    return np.concatenate(parts, axis=0)


def stdp_eligibility(features: np.ndarray) -> np.ndarray:
    """Apply fixed pre-to-terminal-post eligibility decay to the 16-window rates.

    This is an explicit computational abstraction: each window's rate is a local
    presynaptic event magnitude; the output action is at the terminal window.
    It is not claimed to be a biologically complete STDP implementation.
    """
    x = np.asarray(features, dtype=np.float32)
    one = x.ndim == 1
    if one:
        x = x[None, :]
    expected = N_WINDOWS * N_HIDDEN
    if x.ndim != 2 or x.shape[1] != expected:
        raise ValueError(f"Expected feature shape (N,{expected}), got {x.shape}")
    windows = x.reshape(-1, N_WINDOWS, N_HIDDEN)
    window_idx = np.arange(N_WINDOWS, dtype=np.float32)
    kernel = np.exp(-(N_WINDOWS - 1 - window_idx) / STDP_TRACE_TAU_WINDOWS).astype(np.float32)
    eligible = (windows * kernel[None, :, None]).reshape(-1, expected)
    norm = np.maximum(np.linalg.norm(eligible, axis=1, keepdims=True), 1e-8)
    eligible = eligible / norm
    return eligible[0] if one else eligible


class RewardSTDP16WindowReadout:
    """Online reward-modulated STDP-style linear synaptic readout over 16 windows."""
    def __init__(self, seed: int):
        self.rng = np.random.default_rng(int(seed))
        n_features = N_WINDOWS * N_HIDDEN
        self.w = self.rng.normal(0.0, STDP_INITIAL_WEIGHT_STD, (2, n_features)).astype(np.float32)
        self.reward_baseline = 0.0
        self.correct_actions: list[int] = []

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        centered = scores - float(np.mean(scores))
        scale = max(float(np.std(scores)), 0.05)
        z = centered / (scale * max(STDP_TEMPERATURE, 1e-4))
        e = np.exp(np.clip(z, -20.0, 20.0))
        return e / np.sum(e)

    def score(self, features: np.ndarray) -> np.ndarray:
        e = stdp_eligibility(features)
        return (self.w @ e).astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:
        e = stdp_eligibility(features)
        scores = e @ self.w.T
        return np.argmax(scores, axis=1).astype(np.int64)

    def train_one(self, features: np.ndarray, target: int) -> int:
        eligibility = stdp_eligibility(features)
        scores = self.w @ eligibility
        p = self._softmax(scores)
        action = int(self.rng.choice(2, p=p))
        reward = 1.0 if action == int(target) else -1.0
        reward_error = reward - self.reward_baseline
        policy_factor = -p.copy()
        policy_factor[action] += 1.0
        self.w += STDP_LR * reward_error * policy_factor[:, None] * eligibility[None, :]
        np.clip(self.w, -STDP_WEIGHT_CLIP, STDP_WEIGHT_CLIP, out=self.w)
        self.reward_baseline = (
            STDP_REWARD_BASELINE_MOMENTUM * self.reward_baseline
            + (1.0 - STDP_REWARD_BASELINE_MOMENTUM) * reward
        )
        self.correct_actions.append(int(action == int(target)))
        return action


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    recall0 = float(np.mean(y_pred[y_true == 0] == 0)) if np.any(y_true == 0) else float("nan")
    recall1 = float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else float("nan")
    return {
        "n_test": int(y_true.size),
        "accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": 0.5 * (recall0 + recall1),
        "class_0_recall": recall0,
        "class_1_recall": recall1,
        "predicted_positive_rate": float(np.mean(y_pred == 1)),
    }


def evaluate_rows(seed: int, task: str, train_size: int, method: str,
                  y_test: np.ndarray, prediction: np.ndarray, training_action_accuracy: float | None) -> dict:
    metrics = classification_metrics(y_test, prediction)
    return {
        "seed": int(seed), "task": task, "train_size": int(train_size), "method": method,
        **metrics,
        "training_action_accuracy": training_action_accuracy,
    }


def run_seed(seed: int, outdir: Path, test_n: int = N_TEST_PER_TASK) -> tuple[list[dict], dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    started = time.time()
    reservoir = DynamicPulseReservoir(seed=int(seed), n_hidden=N_HIDDEN)
    rows: list[dict] = []
    integrity: dict = {"seed": int(seed), "reservoir_seed": int(seed), "tasks": {}}

    for task_i, task in enumerate(TASKS):
        train_seed = int(seed + 100_001 + 1000 * task_i)
        test_seed = int(seed + 200_001 + 1000 * task_i)
        train_trials = make_nested_training_pool(train_seed, task, N_TRAIN_POOL)
        train_keys = {trial_key(t) for t in train_trials}
        if len(train_keys) != N_TRAIN_POOL:
            raise RuntimeError(f"Duplicate training sequence: seed={seed}, task={task}")
        test_trials = make_disjoint_dataset(test_seed, task, test_n, "test", forbidden=train_keys)
        test_keys = {trial_key(t) for t in test_trials}
        if len(test_keys) != test_n or (test_keys & train_keys):
            raise RuntimeError(f"Train/test overlap or duplicates: seed={seed}, task={task}")

        y_train = np.asarray([t.target_class for t in train_trials], dtype=np.int64)
        y_test = np.asarray([t.target_class for t in test_trials], dtype=np.int64)
        for size in TRAIN_SIZES:
            prefix = y_train[:size]
            if int(np.sum(prefix == 0)) != size // 2 or int(np.sum(prefix == 1)) != size // 2:
                raise RuntimeError(f"Training prefix not balanced: seed={seed}, task={task}, n={size}")
        if int(np.sum(y_test == 0)) != test_n // 2 or int(np.sum(y_test == 1)) != test_n // 2:
            raise RuntimeError(f"Test set not balanced: seed={seed}, task={task}")

        x_train = encode_trials(reservoir, train_trials)
        x_test = encode_trials(reservoir, test_trials)
        if x_train.shape != (N_TRAIN_POOL, N_WINDOWS * N_HIDDEN):
            raise RuntimeError(f"Unexpected train feature shape: {x_train.shape}")
        if x_test.shape != (test_n, N_WINDOWS * N_HIDDEN):
            raise RuntimeError(f"Unexpected test feature shape: {x_test.shape}")

        # STDP sees each unique example exactly once; evaluate at nested checkpoints.
        learner = RewardSTDP16WindowReadout(seed + 300_001 + 1000 * task_i)
        train_action_correct = 0
        checkpoint_training_accuracy: dict[int, float] = {}
        checkpoint_predictions: dict[int, np.ndarray] = {}
        for i in range(N_TRAIN_POOL):
            action = learner.train_one(x_train[i], int(y_train[i]))
            train_action_correct += int(action == int(y_train[i]))
            seen = i + 1
            if seen in TRAIN_SIZES:
                checkpoint_training_accuracy[seen] = train_action_correct / seen
                checkpoint_predictions[seen] = learner.predict(x_test)

        for size in TRAIN_SIZES:
            rows.append(evaluate_rows(
                seed, task, size, "stdp16", y_test, checkpoint_predictions[size],
                checkpoint_training_accuracy[size],
            ))

            # Strong, conventional linear baseline; fixed settings are carried over
            # from the earlier learning-curve protocol, not selected on these tests.
            linear = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=LINEAR_C, max_iter=LINEAR_MAX_ITER,
                                   solver="lbfgs", random_state=int(seed + task_i * 1000 + size)),
            )
            linear.fit(x_train[:size], y_train[:size])
            pred = linear.predict(x_test).astype(np.int64)
            rows.append(evaluate_rows(seed, task, size, "linear_logistic", y_test, pred, None))

        integrity["tasks"][task] = {
            "n_train": len(train_trials), "n_test": len(test_trials),
            "unique_train_sequences": len(train_keys), "unique_test_sequences": len(test_keys),
            "train_test_exact_overlap": len(train_keys & test_keys),
            "train_class_counts": {"0": int(np.sum(y_train == 0)), "1": int(np.sum(y_train == 1))},
            "test_class_counts": {"0": int(np.sum(y_test == 0)), "1": int(np.sum(y_test == 1))},
            "train_feature_shape": list(x_train.shape), "test_feature_shape": list(x_test.shape),
            "feature_min": float(min(float(x_train.min()), float(x_test.min()))),
            "feature_max": float(max(float(x_train.max()), float(x_test.max()))),
        }

    integrity["elapsed_seconds"] = round(time.time() - started, 4)
    return rows, integrity


def bootstrap_mean_ci(values: Sequence[float], seed: int, reps: int = BOOTSTRAP_REPS) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(arr))
    if arr.size == 1:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    # Chunk resamples to keep memory stable.
    draws = np.empty(reps, dtype=np.float64)
    chunk = 500
    for start in range(0, reps, chunk):
        count = min(chunk, reps - start)
        idx = rng.integers(0, arr.size, size=(count, arr.size))
        draws[start:start + count] = arr[idx].mean(axis=1)
    lo, hi = np.quantile(draws, [0.025, 0.975])
    return mean, float(lo), float(hi)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], outdir: Path) -> dict:
    lookup = {(r["seed"], r["task"], r["train_size"], r["method"]): r for r in rows}
    summary_rows: list[dict] = []
    paired_rows: list[dict] = []
    pooled_by_size: dict[int, dict] = {}

    # Descriptive accuracy summaries; bootstrap unit is seed, not task-row.
    for size in TRAIN_SIZES:
        for task in TASKS:
            for method in ("stdp16", "linear_logistic"):
                vals = [lookup[(s, task, size, method)]["accuracy"] for s in SEEDS]
                mean, lo, hi = bootstrap_mean_ci(vals, BOOTSTRAP_SEED + size + TASKS.index(task) * 100 + (0 if method == "stdp16" else 1))
                summary_rows.append({
                    "scope": task, "train_size": size, "method": method,
                    "n_seeds": len(vals), "mean_accuracy_pct": 100 * mean,
                    "sd_accuracy_pp": 100 * float(np.std(vals, ddof=1)),
                    "ci95_low_pct": 100 * lo, "ci95_high_pct": 100 * hi,
                    "mean_balanced_accuracy_pct": 100 * float(np.mean([
                        lookup[(s, task, size, method)]["balanced_accuracy"] for s in SEEDS
                    ])),
                })
        for method in ("stdp16", "linear_logistic"):
            # First average the two task results within each seed, then summarize seeds.
            seed_values = [float(np.mean([
                lookup[(s, task, size, method)]["accuracy"] for task in TASKS
            ])) for s in SEEDS]
            mean, lo, hi = bootstrap_mean_ci(seed_values, BOOTSTRAP_SEED + size + (0 if method == "stdp16" else 1) + 1000)
            summary_rows.append({
                "scope": "task_macro_mean", "train_size": size, "method": method,
                "n_seeds": len(seed_values), "mean_accuracy_pct": 100 * mean,
                "sd_accuracy_pp": 100 * float(np.std(seed_values, ddof=1)),
                "ci95_low_pct": 100 * lo, "ci95_high_pct": 100 * hi,
                "mean_balanced_accuracy_pct": 100 * float(np.mean([
                    np.mean([lookup[(s, task, size, method)]["balanced_accuracy"] for task in TASKS]) for s in SEEDS
                ])),
            })

        # Paired seed-cluster bootstrap: within each seed average sync/order deltas.
        seed_diffs = []
        for s in SEEDS:
            task_diffs = {}
            for task in TASKS:
                a = lookup[(s, task, size, "stdp16")]["accuracy"]
                b = lookup[(s, task, size, "linear_logistic")]["accuracy"]
                delta = a - b
                task_diffs[f"{task}_difference_pp"] = 100 * delta
                task_diffs[f"{task}_stdp_accuracy_pct"] = 100 * a
                task_diffs[f"{task}_linear_accuracy_pct"] = 100 * b
            pooled_delta = float(np.mean([
                lookup[(s, task, size, "stdp16")]["accuracy"] - lookup[(s, task, size, "linear_logistic")]["accuracy"]
                for task in TASKS
            ]))
            seed_diffs.append(pooled_delta)
            paired_rows.append({"seed": s, "train_size": size, **task_diffs,
                                "task_macro_difference_pp": 100 * pooled_delta})
        mean, lo, hi = bootstrap_mean_ci(seed_diffs, BOOTSTRAP_SEED + 5000 + size)
        pooled_by_size[size] = {
            "n_independent_seed_clusters": len(seed_diffs),
            "mean_paired_gain_pp": 100 * mean,
            "ci95_low_pp": 100 * lo,
            "ci95_high_pp": 100 * hi,
            "stdp_beats_linear_by_preregistered_gate": bool(
                size == 200 and 100 * mean >= MIN_PRACTICAL_GAIN_PP and 100 * lo > 0.0
            ),
        }

    write_csv(outdir / "per_seed_results.csv", rows)
    write_csv(outdir / "summary.csv", summary_rows)
    write_csv(outdir / "paired_differences.csv", paired_rows)

    primary = pooled_by_size[200]
    if primary["stdp_beats_linear_by_preregistered_gate"]:
        decision = "PASS_STDP_HAS_PRACTICALLY_MEANINGFUL_ADVANTAGE"
        decision_ar = "نجاح بوابة الاختبار: تفوق STDP عند 200 مثال بفارق عملي لا يقل عن نقطتين مئويتين، والحد الأدنى لفاصل الثقة أكبر من الصفر."
    else:
        decision = "FAIL_TO_JUSTIFY_STDP_OVER_LINEAR"
        decision_ar = "لم يحقق STDP معيار التفوق المسجل مسبقًا عند 200 مثال؛ هذا لا يثبت استحالة كل قواعد STDP، لكنه لا يبرر مواصلة المسار الحالي بهذه النسخة."

    result = {
        "protocol_version": "v0.42",
        "decision": decision,
        "decision_ar": decision_ar,
        "primary_endpoint_at_200_examples": primary,
        "paired_gains_by_train_size": pooled_by_size,
        "notes": [
            "Only the fixed STDP readout learns; reservoir weights remain frozen.",
            "Both methods use the same 16-window base features and the same training/test examples per seed/task.",
            "STDP is a simplified reward-modulated timing-eligibility rule, not a full biological simulation.",
            "Linear logistic regression is refit on each nested prefix using fixed C=1 and max_iter=250; no test result tunes either method.",
            "If the gate fails, interpret this as no evidence that this fixed STDP implementation justifies the pulse path, not proof that all STDP designs fail.",
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# Nucleus Pulse v0.42 — مقارنة قارئ STDP بالقارئ الخطي",
        "",
        "## السؤال",
        "هل يتفوق قارئ reward-modulated STDP على قارئ خطي عند 50 و100 و200 مثال، باستخدام ميزات خزان ثابت من 16 نافذة؟",
        "",
        "## البروتوكول المجمد",
        f"- {len(SEEDS)} بذرة مستقلة؛ مهمتا `sync` و`order`.",
        f"- {N_TRAIN_POOL} مثال تدريب متوازن لكل بذرة/مهمة، مع بادئات متداخلة عند {TRAIN_SIZES}.",
        f"- {N_TEST_PER_TASK} مثال اختبار متوازن مستقل لكل بذرة/مهمة في التشغيل الكامل.",
        "- نفس الخزان ونفس الميزات الأساسية (16 نافذة × 24 خلية = 384 ميزة) ونفس الأمثلة للطريقتين.",
        "- STDP يمر على كل مثال مرة واحدة؛ القارئ الخطي LogisticRegression مضبوط بإعدادات ثابتة سابقة ولا يُضبط من الاختبار.",
        "- الاختبار النهائي لا يختار معاملات ولا نقاط توقف. التباين وفاصل الثقة يعتمدان على 40 بذرة، مع تجميع المهمتين داخل كل بذرة للمقياس الأساسي.",
        "",
        "## بوابة القرار المسبقة",
        f"نجاح STDP عند 200 يتطلب متوسط تفوق لا يقل عن {MIN_PRACTICAL_GAIN_PP:.1f} نقطة مئوية وأن يكون الحد الأدنى لفاصل الثقة 95% أكبر من الصفر.",
        "",
        "## النتائج الأساسية",
        "| عدد الأمثلة | مكسب STDP على الخطي (نقطة مئوية) | فاصل الثقة 95% |",
        "|---:|---:|---:|",
    ]
    for size in TRAIN_SIZES:
        item = pooled_by_size[size]
        report.append(f"| {size} | {item['mean_paired_gain_pp']:.2f} | [{item['ci95_low_pp']:.2f}, {item['ci95_high_pp']:.2f}] |")
    report += ["", "## القرار", decision_ar, "", "## التحفظات", "- هذه المقارنة تقيس مهمة زمنية اصطناعية بتمثيل وخزان محددين، ولا تثبت ذكاءً عامًا أو كفاءة عتادية.", "- الفشل يعني أن هذه النسخة المجمدة لا تبرر الطريق النبضي فوق القارئ الخطي؛ لا يعني أن كل قاعدة STDP ممكنة قد فشلت.", "- لا تُعدّل المعاملات ثم تعِد استخدام مجموعة الاختبار نفسها لإثبات التفوق.", ""]
    (outdir / "report_ar.md").write_text("\n".join(report), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a small component smoke test; not official evidence")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not args.smoke:
        verify_preregistration()
        seeds = SEEDS
        test_n = N_TEST_PER_TASK
        outdir = args.output or (ROOT / "results" / "stdp_vs_linear_v0_42")
    else:
        seeds = SMOKE_SEEDS[:2]
        test_n = 80
        outdir = args.output or (ROOT / "results" / "stdp_vs_linear_v0_42" / "component_smoke")
    outdir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    all_rows: list[dict] = []
    integrity_rows: list[dict] = []
    for idx, seed in enumerate(seeds, start=1):
        rows, integrity = run_seed(seed, outdir, test_n=test_n)
        all_rows.extend(rows)
        integrity_rows.append(integrity)
        # Partial save after every complete seed so an interrupted run is recoverable.
        write_csv(outdir / "per_seed_results_partial.csv", all_rows)
        (outdir / "integrity_partial.json").write_text(
            json.dumps(integrity_rows, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{idx}/{len(seeds)}] seed={seed} done in {integrity['elapsed_seconds']:.1f}s", flush=True)

    (outdir / "integrity.json").write_text(json.dumps(integrity_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if not args.smoke:
        result = summarize(all_rows, outdir)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        write_csv(outdir / "per_seed_results.csv", all_rows)
        print("SMOKE ONLY: component checks completed; these results are not benchmark evidence.")
    run_meta = {
        "smoke_only": bool(args.smoke), "seeds_run": list(seeds), "test_per_task": test_n,
        "elapsed_seconds": round(time.time() - started, 4),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_verified": (not args.smoke),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
