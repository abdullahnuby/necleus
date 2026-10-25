"""Reproducible development pipeline for v0.43 Phase 3."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any, Callable, Sequence

import numpy as np

from .baselines import (
    Predictions,
    deepsets_predictions,
    gate_status,
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
    DEVELOPMENT_SUBSETS,
    FAMILY_DEV_PROTOCOL,
    FAMILY_DEV_SCHEMA,
    build_development_manifest,
    build_development_tasks,
    build_family_aware_development_manifest,
    build_family_aware_development_tasks,
    corpus_manifest_from_episodes,
    family_aware_split_audit,
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


def evaluate_baselines(records: Sequence[TaskRecord], *, cluster_bootstrap: int = 0,
                       bootstrap_seed: int = 430043) -> dict[str, dict]:
    output: dict[str, dict] = {}
    for name, predictor in BASELINES.items():
        predictions = [predictor(task) for task in records]
        output[name] = summarize(
            records, predictions, cluster_bootstrap=cluster_bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
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


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def load_family_aware_subsets(
    manifest_path: str | Path, records: Sequence[TaskRecord]
) -> tuple[dict[str, list[TaskRecord]], dict[str, Any], dict[str, Any]]:
    """Load a frozen family-aware manifest and bind it to regenerated tasks."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != FAMILY_DEV_SCHEMA:
        raise ValueError(f"unexpected development manifest schema: {manifest.get('schema')}")
    manifest_rows = manifest.get("tasks", [])
    subset_of = {row["task_id"]: row["subset"] for row in manifest_rows}
    family_of = {row["task_id"]: row["family_id"] for row in manifest_rows}
    record_by_id = {record.task_id: record for record in records}
    if set(record_by_id) != set(subset_of):
        missing = sorted(set(record_by_id) ^ set(subset_of))[:3]
        raise ValueError(f"regenerated development tasks do not match frozen manifest: {missing}")
    subsets: dict[str, list[TaskRecord]] = {name: [] for name in DEVELOPMENT_SUBSETS}
    for task_id, subset in subset_of.items():
        record = record_by_id[task_id]
        if record.family_id != family_of[task_id]:
            raise ValueError(f"family mismatch for frozen task {task_id}")
        if subset not in subsets:
            raise ValueError(f"unknown frozen development subset: {subset}")
        subsets[subset].append(record)
    for name in DEVELOPMENT_SUBSETS:
        subsets[name].sort(key=lambda record: record.task_id)
    audit = family_aware_split_audit(records, subsets)
    return subsets, manifest, audit


def run_family_aware_development(
    *, seeds: Sequence[int] = (430043, 430044, 430045), epochs: int = 10,
    n_episodes: int = 10_000, hidden: int = 128,
    output_dir: str | Path = "results/v043_task010",
    checkpoint_dir: str | Path = ".local_private/v043_task010/checkpoints",
    corpus_seed: int = DEFAULT_SEED, dev_seed: int = DEFAULT_SEED,
    manifest_path: str | Path | None = None,
    cluster_bootstrap: int = 500, bootstrap_seed: int = 430043,
    force_manifest: bool = False,
) -> dict[str, Any]:
    """Run the v0.43-family-aware-dev-v1 development protocol.

    The manifest is frozen before evaluation. Thresholds are selected only on
    calibration families. If no feasible threshold exists, primary evaluation
    abstains on every query and the diagnostic fallback is reported separately.
    """
    start = time.perf_counter()
    tracemalloc.start()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    if manifest_path is None:
        manifest_path = out / "dev_manifest.json"
    manifest_file = Path(manifest_path)

    corpus = materialize_training_corpus(n_episodes=n_episodes, seed=corpus_seed)
    corpus_manifest = corpus_manifest_from_episodes(
        corpus, requested_episodes=n_episodes, seed=corpus_seed
    )
    write_manifest(out / "corpus_manifest.json", corpus_manifest)

    dev_records, generated_subsets, generated_audit = build_family_aware_development_tasks(
        seed=dev_seed
    )
    if manifest_file.exists() and not force_manifest:
        subsets, manifest, split_audit = load_family_aware_subsets(manifest_file, dev_records)
    else:
        subsets = generated_subsets
        split_audit = generated_audit
        manifest = build_family_aware_development_manifest(
            seed=dev_seed, records=dev_records, subsets=subsets, audit=split_audit
        )
        write_manifest(manifest_file, manifest)
    manifest_sha256 = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    if manifest["task_count"] != len(dev_records):
        raise ValueError("frozen development manifest task count does not match generated tasks")
    if manifest["family_count"] != len({record.family_id for record in dev_records}):
        raise ValueError("frozen development manifest family count does not match generated tasks")
    calibration = subsets["calibration"]
    evaluation = subsets["evaluation"]
    fit = subsets["fit"]

    baseline_results = evaluate_baselines(
        evaluation, cluster_bootstrap=cluster_bootstrap, bootstrap_seed=bootstrap_seed
    )
    seed_rows: list[dict[str, Any]] = []
    train_split = corpus[:max(1, int(len(corpus) * 0.95))]
    validation_split = corpus[max(1, int(len(corpus) * 0.95)):]
    for seed in seeds:
        config = DeepSetsConfig(
            hidden=hidden, seed=int(seed), learning_rate=1e-3,
            batch_size=32, epochs=epochs, l2=1e-4, gradient_clip=5.0,
        )
        model = DeepSetsNumpy(config)
        history = model.fit(train_split, validation=validation_split)
        checkpoint = checkpoints / f"family_v2_deepsets_seed_{seed}.npz"
        checkpoint_hash = model.save(checkpoint, metadata={
            "seed": int(seed),
            "protocol_version": FAMILY_DEV_PROTOCOL,
            "manifest_sha256": manifest_sha256,
            "corpus_manifest_sha256": corpus_manifest["episode_sha256"],
        })
        calibration_probabilities = deepsets_probabilities(model, calibration)
        selection = select_threshold(calibration, calibration_probabilities)
        evaluation_probabilities = deepsets_probabilities(model, evaluation)
        if selection["feasible"]:
            threshold = float(selection["chosen_threshold"])
            primary_predictions = [
                deepsets_predictions(prob, threshold) for prob in evaluation_probabilities
            ]
            fallback_threshold = None
            fallback_metrics = None
        else:
            threshold = None
            primary_predictions = [
                deepsets_predictions(prob, 1.01) for prob in evaluation_probabilities
            ]
            fallback = selection["diagnostic_fallback"]
            fallback_threshold = None if fallback is None else float(fallback["threshold"])
            fallback_metrics = None
            if fallback_threshold is not None:
                fallback_predictions = [
                    deepsets_predictions(prob, fallback_threshold)
                    for prob in evaluation_probabilities
                ]
                fallback_metrics = summarize(
                    evaluation, fallback_predictions,
                    cluster_bootstrap=cluster_bootstrap, bootstrap_seed=bootstrap_seed,
                )
        primary_metrics = summarize(
            evaluation, primary_predictions,
            cluster_bootstrap=cluster_bootstrap, bootstrap_seed=bootstrap_seed,
        )
        gate = gate_status(primary_metrics["by_class"])
        seed_rows.append({
            "seed": int(seed),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "config": dict(config.__dict__),
            "training_history": history,
            "calibration_task_ids": [task.task_id for task in calibration],
            "evaluation_task_ids": [task.task_id for task in evaluation],
            "threshold_selection": selection,
            "threshold": threshold,
            "threshold_feasible": bool(selection["feasible"]),
            "diagnostic_fallback_threshold": fallback_threshold,
            "diagnostic_fallback_metrics": fallback_metrics,
            "metrics": dict(baseline_results, deepsets=primary_metrics),
            "gate": gate,
        })

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    config_payload = {
        "protocol_version": FAMILY_DEV_PROTOCOL,
        "manifest_sha256": manifest_sha256,
        "corpus_seed": corpus_seed,
        "n_episodes": n_episodes,
        "seeds": list(seeds),
        "epochs": epochs,
        "hidden": hidden,
        "cluster_bootstrap": cluster_bootstrap,
        "bootstrap_seed": bootstrap_seed,
        "train_episodes": len(train_split),
        "validation_episodes": len(validation_split),
    }
    results = {
        "schema": "nucleus-pulse-v0.43-family-aware-development-results-v1",
        "protocol_version": FAMILY_DEV_PROTOCOL,
        "official_test_opened": False,
        "official_test_ready": False,
        "run_metadata": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "config_sha256": _json_sha256(config_payload),
            "config": config_payload,
            "manifest_path": str(manifest_file),
            "manifest_sha256": manifest_sha256,
        },
        "development_manifest_sha256": manifest_sha256,
        "development_manifest": manifest,
        "development_split_audit": split_audit,
        "corpus_manifest": corpus_manifest,
        "subset_task_counts": {name: len(subset) for name, subset in subsets.items()},
        "subset_family_counts": {
            name: len({record.family_id for record in subset})
            for name, subset in subsets.items()
        },
        "fit_task_count": len(fit),
        "calibration_task_count": len(calibration),
        "evaluation_task_count": len(evaluation),
        "baseline_results_evaluation": baseline_results,
        "seed_results": seed_rows,
        "aggregate": _aggregate_seed_metrics(seed_rows),
        "gate_summary": {
            str(row["seed"]): row["gate"] for row in seed_rows
        },
        "compute": {
            "runtime_seconds": round(time.perf_counter() - start, 3),
            "peak_tracemalloc_mb": round(peak / (1024 * 1024), 3),
            "corpus_episodes": len(corpus),
            "training_episodes": len(train_split),
            "validation_episodes": len(validation_split),
            "seeds": list(seeds),
            "epochs": epochs,
            "hidden": hidden,
            "device": "local CPU, NumPy float64",
        },
        "notes": [
            "Development results only; the official v0.43 test was not opened.",
            "Families are disjoint across fit, calibration, and evaluation.",
            "Thresholds are selected on calibration families only.",
            "If no threshold is feasible, primary evaluation abstains on every query.",
            "Diagnostic fallback metrics are reported separately and are not a pass.",
            "Query-level Wilson intervals remain descriptive; family-clustered intervals are the uncertainty estimate.",
        ],
    }
    (out / "development_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results


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
        feasible = bool(selection["feasible"])
        threshold = (
            float(selection["chosen_threshold"]) if feasible else None
        )
        evaluation_probabilities = deepsets_probabilities(model, evaluation)
        if threshold is None:
            deepsets_rows = [deepsets_predictions(p, 1.01) for p in evaluation_probabilities]
        else:
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
            "threshold_feasible": feasible,
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
