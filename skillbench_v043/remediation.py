"""Bounded DeepSets remediation experiments for TASK-010.

The plan is written and hashed before execution. The current 10-epoch
checkpoints are the unchanged control. Each candidate is evaluated on the same
frozen family-aware calibration/evaluation split and the frozen gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
import tracemalloc
from typing import Any, Sequence

import numpy as np

from .baselines import (
    deepsets_predictions,
    gate_status,
    select_threshold,
    summarize,
)
from .corpus import (
    FAMILY_DEV_PROTOCOL,
    build_family_aware_development_tasks,
    corpus_manifest_from_episodes,
    materialize_training_corpus,
)
from .deepsets import DeepSetsConfig, DeepSetsNumpy
from .generator import TaskRecord
from .pipeline import deepsets_probabilities, load_family_aware_subsets

REMEDIATION_EXPERIMENTS: tuple[dict[str, Any], ...] = (
    {
        "experiment_id": "E0_control",
        "primary_factor": "none (unchanged 10-epoch control checkpoint)",
        "corpus_episodes": 10_000,
        "epochs": 10,
        "checkpoint_selection": "existing control checkpoint",
        "secondary_differences": "none",
    },
    {
        "experiment_id": "E1_longer_schedule",
        "primary_factor": "training epochs: 20 instead of 10",
        "corpus_episodes": 10_000,
        "epochs": 20,
        "checkpoint_selection": "lower validation loss among epoch 10 and epoch 20",
        "secondary_differences": "none; same corpus, architecture, optimizer, and seed",
    },
    {
        "experiment_id": "E2_more_episodes",
        "primary_factor": "training episodes: 20,000 instead of 10,000",
        "corpus_episodes": 20_000,
        "epochs": 10,
        "checkpoint_selection": "final epoch 10",
        "secondary_differences": "none; same architecture, optimizer, epochs, and seed",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_abstain(probabilities: Sequence[np.ndarray]):
    return [deepsets_predictions(prob, 1.01) for prob in probabilities]


def _evaluate_checkpoint(model: DeepSetsNumpy, calibration: Sequence[TaskRecord],
                         evaluation: Sequence[TaskRecord], *, cluster_bootstrap: int,
                         bootstrap_seed: int) -> dict[str, Any]:
    calibration_probabilities = deepsets_probabilities(model, calibration)
    selection = select_threshold(calibration, calibration_probabilities)
    evaluation_probabilities = deepsets_probabilities(model, evaluation)
    if selection["feasible"]:
        threshold = float(selection["chosen_threshold"])
        predictions = [deepsets_predictions(prob, threshold) for prob in evaluation_probabilities]
        fallback_threshold = None
        fallback_metrics = None
    else:
        threshold = None
        predictions = _all_abstain(evaluation_probabilities)
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
    metrics = summarize(
        evaluation, predictions,
        cluster_bootstrap=cluster_bootstrap, bootstrap_seed=bootstrap_seed,
    )
    return {
        "threshold_selection": selection,
        "threshold": threshold,
        "threshold_feasible": bool(selection["feasible"]),
        "diagnostic_fallback_threshold": fallback_threshold,
        "diagnostic_fallback_metrics": fallback_metrics,
        "metrics": metrics,
        "gate": gate_status(metrics["by_class"]),
    }


def _aggregate_experiment(seed_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not seed_rows:
        return {}
    classes = ["overall"] + sorted(seed_rows[0]["metrics"]["by_class"])
    output: dict[str, Any] = {"seed_count": len(seed_rows), "classes": {}}
    for class_name in classes:
        rows = [
            row["metrics"]["overall"] if class_name == "overall"
            else row["metrics"]["by_class"].get(class_name, {})
            for row in seed_rows
        ]
        metrics: dict[str, Any] = {}
        for key in ("accuracy", "coverage", "selective_risk", "abstention", "accepted", "n_families"):
            values = [row.get(key) for row in rows if row.get(key) is not None]
            if values:
                metrics[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "n_seeds": len(values),
                }
        output["classes"][class_name] = metrics
    output["gates_passed"] = sum(1 for row in seed_rows if row["gate"]["passed"])
    output["all_seeds_passed"] = output["gates_passed"] == len(seed_rows)
    output["thresholds"] = [row["threshold"] for row in seed_rows]
    output["feasible_thresholds"] = sum(1 for row in seed_rows if row["threshold_feasible"])
    return output


def run_remediation(
    *, plan_path: str | Path = "docs/V0_43_DEEPSETS_REMEDIATION_PLAN.md",
    preregistered_plan_sha256: str | None = None,
    seeds: Sequence[int] = (430043, 430044, 430045),
    output_dir: str | Path = "results/v043_task010",
    manifest_path: str | Path = "results/v043_task010/dev_manifest.json",
    control_checkpoint_dir: str | Path = ".local_private/v043_task010/checkpoints",
    remediation_checkpoint_dir: str | Path = ".local_private/v043_task010/remediation/checkpoints",
    cluster_bootstrap: int = 500, bootstrap_seed: int = 430043,
) -> dict[str, Any]:
    start_all = time.perf_counter()
    plan_file = Path(plan_path)
    if not plan_file.exists():
        raise FileNotFoundError(f"preregistered plan is missing: {plan_file}")
    plan_sha256 = _sha256(plan_file)
    if preregistered_plan_sha256 is not None and plan_sha256 != preregistered_plan_sha256:
        raise ValueError(
            f"remediation plan hash mismatch: expected {preregistered_plan_sha256}, got {plan_sha256}"
        )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = Path(remediation_checkpoint_dir)
    checkpoints.mkdir(parents=True, exist_ok=True)
    records, generated_subsets, generated_audit = build_family_aware_development_tasks(seed=430043)
    subsets, manifest, audit = load_family_aware_subsets(manifest_path, records)
    calibration = subsets["calibration"]
    evaluation = subsets["evaluation"]
    manifest_sha256 = _sha256(Path(manifest_path))

    experiment_rows: list[dict[str, Any]] = []
    corpus_manifests: dict[str, Any] = {}
    for experiment in REMEDIATION_EXPERIMENTS:
        experiment_id = experiment["experiment_id"]
        experiment_start = time.perf_counter()
        seed_rows: list[dict[str, Any]] = []
        corpus = None
        corpus_manifest = None
        if experiment_id != "E0_control":
            corpus = materialize_training_corpus(
                n_episodes=experiment["corpus_episodes"], seed=430043
            )
            corpus_manifest = corpus_manifest_from_episodes(
                corpus, requested_episodes=experiment["corpus_episodes"], seed=430043
            )
            corpus_manifests[experiment_id] = corpus_manifest
        for seed in seeds:
            seed_start = time.perf_counter()
            tracemalloc.start()
            config = DeepSetsConfig(
                hidden=128, seed=int(seed), learning_rate=1e-3,
                batch_size=32, epochs=int(experiment["epochs"]), l2=1e-4,
                gradient_clip=5.0,
            )
            selected_epoch: int | None = None
            validation_losses: dict[str, float] = {}
            history: list[dict[str, float]] = []
            if experiment_id == "E0_control":
                checkpoint = Path(control_checkpoint_dir) / f"family_v2_deepsets_seed_{seed}.npz"
                if not checkpoint.exists():
                    raise FileNotFoundError(f"control checkpoint is missing: {checkpoint}")
                model = DeepSetsNumpy.load(checkpoint, config)
                history = model.history
                selected_epoch = int(history[-1]["epoch"]) if history else None
            else:
                if corpus is None:
                    raise RuntimeError("training corpus is missing")
                train_split = corpus[:max(1, int(len(corpus) * 0.95))]
                validation_split = corpus[max(1, int(len(corpus) * 0.95)):]
                model = DeepSetsNumpy(config)
                if experiment_id == "E1_longer_schedule":
                    prefix = checkpoints / f"E1_longer_schedule_seed_{seed}.npz"
                    history = model.fit(
                        train_split, validation=validation_split,
                        checkpoint_prefix=prefix, checkpoint_epochs=(10, 20),
                    )
                    candidates = []
                    for epoch in (10, 20):
                        candidate_path = prefix.with_name(
                            f"{prefix.stem}_epoch{epoch}{prefix.suffix or '.npz'}"
                        )
                        candidate = DeepSetsNumpy.load(candidate_path, config)
                        loss = candidate.evaluate_loss(validation_split)
                        validation_losses[str(epoch)] = loss
                        candidates.append((loss, epoch, candidate_path, candidate))
                    _, selected_epoch, checkpoint, model = min(
                        candidates, key=lambda item: (item[0], item[1])
                    )
                else:
                    checkpoint = checkpoints / f"E2_more_episodes_seed_{seed}.npz"
                    history = model.fit(train_split, validation=validation_split)
                    selected_epoch = int(history[-1]["epoch"]) if history else None
                    model.save(checkpoint, metadata={
                        "seed": int(seed),
                        "experiment_id": experiment_id,
                        "corpus_manifest_sha256": corpus_manifest["episode_sha256"],
                        "manifest_sha256": manifest_sha256,
                    })
            evaluation_result = _evaluate_checkpoint(
                model, calibration, evaluation,
                cluster_bootstrap=cluster_bootstrap, bootstrap_seed=bootstrap_seed,
            )
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            seed_rows.append({
                "experiment_id": experiment_id,
                "seed": int(seed),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "config": dict(config.__dict__),
                "selected_epoch": selected_epoch,
                "validation_losses": validation_losses,
                "training_history": history,
                "runtime_seconds": round(time.perf_counter() - seed_start, 3),
                "peak_tracemalloc_mb": round(peak / (1024 * 1024), 3),
                **evaluation_result,
            })
        experiment_rows.append({
            **experiment,
            "seeds": list(seeds),
            "seed_results": seed_rows,
            "aggregate": _aggregate_experiment(seed_rows),
            "runtime_seconds": round(time.perf_counter() - experiment_start, 3),
        })

    results = {
        "schema": "nucleus-pulse-v0.43-deepsets-remediation-v1",
        "protocol_version": FAMILY_DEV_PROTOCOL,
        "official_test_opened": False,
        "official_test_ready": False,
        "preregistered_plan_path": str(plan_file),
        "preregistered_plan_sha256": plan_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "development_split_audit": audit,
        "experiments": experiment_rows,
        "corpus_manifests": corpus_manifests,
        "decision": {
            "status": "PASS" if any(
                experiment["aggregate"].get("all_seeds_passed")
                for experiment in experiment_rows
            ) else "FAIL",
            "reason": (
                "At least one bounded candidate passed the frozen gates for every seed."
                if any(experiment["aggregate"].get("all_seeds_passed")
                       for experiment in experiment_rows)
                else "No bounded candidate passed the frozen gates for every seed; stop optimization."
            ),
        },
        "compute": {
            "runtime_seconds": round(time.perf_counter() - start_all, 3),
            "seeds": list(seeds),
            "cluster_bootstrap": cluster_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "device": "local CPU, NumPy float64",
        },
        "notes": [
            "Development only; the official v0.43 test was not opened.",
            "The control checkpoint is unchanged.",
            "Checkpoint selection for E1 uses validation loss only.",
            "No evaluation-fold query label is used for threshold fitting or checkpoint selection.",
            "If no feasible threshold exists, primary evaluation abstains on every query.",
        ],
    }
    (out / "remediation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
