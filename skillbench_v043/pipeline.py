"""Reproducible development pipeline for v0.43 Phase 3."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any, Callable, Sequence

import numpy as np

from .baselines import (
    Predictions,
    deepsets_predictions,
    predict_constant,
    predict_linear,
    predict_program_induction,
    predict_program_induction_task_level,
    predict_retrieval,
    predict_tree,
    select_threshold,
    summarize,
)
from .corpus import (
    DEFAULT_SEED,
    build_development_manifest,
    build_development_tasks,
    corpus_manifest_from_episodes,
    materialize_training_corpus,
    write_manifest,
)
from .deepsets import DeepSetsConfig, DeepSetsNumpy
from .generator import TaskRecord

BASELINES: dict[str, Callable[[TaskRecord], Predictions]] = {
    "constant": predict_constant,
    "retrieval": predict_retrieval,
    "linear": predict_linear,
    "decision_tree": predict_tree,
    "program_induction": predict_program_induction,
    "program_induction_task_level": predict_program_induction_task_level,
}


def split_development(records: Sequence[TaskRecord]) -> tuple[list[TaskRecord], list[TaskRecord]]:
    """Deterministically split each class into calibration and evaluation halves."""
    by_class: dict[str, list[TaskRecord]] = defaultdict(list)
    for record in records:
        by_class[record.classification_private].append(record)
    calibration: list[TaskRecord] = []
    evaluation: list[TaskRecord] = []
    for name in sorted(by_class):
        rows = by_class[name]
        cut = len(rows) // 2
        calibration.extend(rows[:cut])
        evaluation.extend(rows[cut:])
    return calibration, evaluation


def evaluate_baselines(records: Sequence[TaskRecord]) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for name, predictor in BASELINES.items():
        predictions = [predictor(task) for task in records]
        output[name] = summarize(records, predictions)
    return output


def deepsets_probabilities(model: DeepSetsNumpy, records: Sequence[TaskRecord]) -> list[np.ndarray]:
    probabilities: list[np.ndarray] = []
    for task in records:
        support_x = np.asarray([[[ex.x[0], ex.x[1]] for ex in task.support]], dtype=np.float64)
        support_y = np.asarray([[ex.y for ex in task.support]], dtype=np.float64)
        query_x = np.asarray([[[x[0], x[1]] for x in task.query_x]], dtype=np.float64)
        probabilities.append(model.predict_proba(support_x, support_y, query_x)[0])
    return probabilities


def _aggregate_seed_metrics(seed_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not seed_rows:
        return {}
    methods = list(seed_rows[0]["metrics"])
    output: dict[str, Any] = {}
    for method in methods:
        output[method] = {}
        classes = ["overall"] + sorted(seed_rows[0]["metrics"][method]["by_class"])
        for class_name in classes:
            if class_name == "overall":
                values = [row["metrics"][method]["overall"] for row in seed_rows]
            else:
                values = [row["metrics"][method]["by_class"].get(class_name, {}) for row in seed_rows]
            metrics: dict[str, Any] = {}
            for key in ("accuracy", "coverage", "selective_risk", "abstention"):
                vals = [v.get(key) for v in values if v.get(key) is not None]
                if vals:
                    metrics[key] = {
                        "mean": float(np.mean(vals)),
                        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "n_seeds": len(vals),
                    }
            output[method][class_name] = metrics
    return output


def run_development(*, seeds: Sequence[int] = (430043,), epochs: int = 3,
                    n_episodes: int = 10_000, hidden: int = 128,
                    output_dir: str | Path = "results/v043_development",
                    checkpoint_dir: str | Path = ".local_private/v043_development/checkpoints",
                    corpus_seed: int = DEFAULT_SEED,
                    dev_seed: int = DEFAULT_SEED,
                    identifiable_count: int = 8,
                    ambiguous_count: int = 8,
                    contradictory_count: int = 8,
                    ood_count: int = 4) -> dict[str, Any]:
    start = time.perf_counter()
    tracemalloc.start()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)

    corpus = materialize_training_corpus(n_episodes=n_episodes, seed=corpus_seed)
    corpus_manifest = corpus_manifest_from_episodes(
        corpus, requested_episodes=n_episodes, seed=corpus_seed
    )
    write_manifest(out / "corpus_manifest.json", corpus_manifest)

    dev_records, dev_audit = build_development_tasks(
        seed=dev_seed,
        identifiable_count=identifiable_count,
        ambiguous_count=ambiguous_count,
        contradictory_count=contradictory_count,
        ood_count=ood_count,
    )
    dev_manifest = build_development_manifest(
        seed=dev_seed,
        identifiable_count=identifiable_count,
        ambiguous_count=ambiguous_count,
        contradictory_count=contradictory_count,
        ood_count=ood_count,
        records=dev_records,
        audit=dev_audit,
    )
    write_manifest(out / "dev_manifest.json", dev_manifest)
    calibration, evaluation = split_development(dev_records)

    baseline_results = evaluate_baselines(evaluation)
    seed_rows: list[dict[str, Any]] = []
    for seed in seeds:
        config = DeepSetsConfig(
            hidden=hidden, seed=int(seed), learning_rate=1e-3,
            batch_size=32, epochs=epochs, l2=1e-4, gradient_clip=5.0,
        )
        model = DeepSetsNumpy(config)
        train_split = corpus[:max(1, int(len(corpus) * 0.95))]
        validation_split = corpus[max(1, int(len(corpus) * 0.95)):]
        history = model.fit(train_split, validation=validation_split)
        checkpoint = checkpoints / f"deepsets_seed_{seed}.npz"
        checkpoint_hash = model.save(checkpoint, metadata={
            "seed": int(seed), "corpus_sha256": corpus_manifest["episode_sha256"],
            "development_manifest": dev_manifest["tasks"],
        })
        calibration_probabilities = deepsets_probabilities(model, calibration)
        selection = select_threshold(calibration, calibration_probabilities)
        threshold = float(selection["chosen"]["threshold"])
        evaluation_probabilities = deepsets_probabilities(model, evaluation)
        deepsets_rows = [deepsets_predictions(p, threshold) for p in evaluation_probabilities]
        metrics = dict(baseline_results)
        metrics["deepsets"] = summarize(evaluation, deepsets_rows)
        seed_rows.append({
            "seed": int(seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "config": config.__dict__,
            "training_history": history,
            "calibration_task_ids": [task.task_id for task in calibration],
            "evaluation_task_ids": [task.task_id for task in evaluation],
            "threshold_selection": selection,
            "threshold": threshold,
            "metrics": metrics,
        })

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    results = {
        "schema": "nucleus-pulse-v0.43-development-results-v1",
        "phase": "Phase 3",
        "official_test_opened": False,
        "corpus_manifest": corpus_manifest,
        "development_manifest": dev_manifest,
        "development_split_audit": dev_audit,
        "calibration_task_count": len(calibration),
        "evaluation_task_count": len(evaluation),
        "baseline_results_evaluation": baseline_results,
        "seed_results": seed_rows,
        "aggregate": _aggregate_seed_metrics(seed_rows),
        "compute": {
            "runtime_seconds": round(time.perf_counter() - start, 3),
            "peak_tracemalloc_mb": round(peak / (1024 * 1024), 3),
            "corpus_episodes": len(corpus),
            "training_episodes": max(1, int(len(corpus) * 0.95)),
            "validation_episodes": len(corpus) - max(1, int(len(corpus) * 0.95)),
            "seeds": list(seeds),
            "epochs": epochs,
            "hidden": hidden,
            "device": "local CPU, NumPy float64",
        },
        "notes": [
            "Development results only; the official v0.43 test was not opened.",
            "DeepSets thresholds are selected on the calibration half of development tasks.",
            "Baselines receive the same support inputs and labels as DeepSets.",
            "Program induction is a closed-world symbolic method; it is reported separately.",
        ],
    }
    (out / "development_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
