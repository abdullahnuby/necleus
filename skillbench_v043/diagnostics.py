"""DeepSets diagnostics for the v0.43 family-aware development protocol."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .baselines import (
    deepsets_predictions,
    gate_status,
    select_threshold,
    summarize,
)
from .corpus import (
    DEVELOPMENT_SUBSETS,
    FAMILY_DEV_PROTOCOL,
    build_family_aware_development_tasks,
)
from .deepsets import DeepSetsConfig, DeepSetsNumpy
from .generator import TaskRecord
from .pipeline import deepsets_probabilities, load_family_aware_subsets

ABSTAIN_CLASSES = ("IN_SCOPE_AMBIGUOUS", "OUT_OF_SCOPE", "CONTRADICTORY_SUPPORT")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(checkpoint: str | Path) -> tuple[DeepSetsNumpy, dict[str, Any]]:
    path = Path(checkpoint)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    config = DeepSetsConfig(**meta["config"])
    model = DeepSetsNumpy.load(path, config)
    return model, meta


def _all_abstain(probabilities: Sequence[np.ndarray]):
    return [deepsets_predictions(prob, 1.01) for prob in probabilities]


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float | None:
    if len(probabilities) == 0:
        return None
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    predictions = (probabilities >= 0.5).astype(np.int64)
    correct = (predictions == labels).astype(np.float64)
    edges = np.linspace(0.5, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        total += (count / len(confidence)) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(total)


def _class_probability_summary(records: Sequence[TaskRecord],
                               probabilities: Sequence[np.ndarray]) -> dict[str, Any]:
    by_class: dict[str, dict[str, list[float]]] = {}
    for task, prob in zip(records, probabilities):
        bucket = by_class.setdefault(task.classification_private, {"p": [], "y": []})
        bucket["p"].extend(float(value) for value in prob)
        bucket["y"].extend(float(value) for value in task.query_y_private)
    output: dict[str, Any] = {}
    for name, bucket in sorted(by_class.items()):
        p = np.asarray(bucket["p"], dtype=np.float64)
        y = np.asarray(bucket["y"], dtype=np.float64)
        confidence = np.maximum(p, 1.0 - p)
        output[name] = {
            "queries": len(p),
            "mean_probability": float(p.mean()),
            "probability_std": float(p.std()),
            "mean_confidence": float(confidence.mean()),
            "confidence_quantiles": {
                "p00": float(np.percentile(confidence, 0)),
                "p05": float(np.percentile(confidence, 5)),
                "p25": float(np.percentile(confidence, 25)),
                "p50": float(np.percentile(confidence, 50)),
                "p75": float(np.percentile(confidence, 75)),
                "p95": float(np.percentile(confidence, 95)),
                "p100": float(np.percentile(confidence, 100)),
            },
            "brier": float(np.mean((p - y) ** 2)),
            "ece_10_bin": _ece(p, y, bins=10),
            "positive_rate": float(y.mean()),
        }
    return output


def _curve_row(records: Sequence[TaskRecord], probabilities: Sequence[np.ndarray],
               threshold: float) -> dict[str, Any]:
    predictions = [deepsets_predictions(prob, threshold) for prob in probabilities]
    summary = summarize(records, predictions)
    by_class = summary["by_class"]
    ident = by_class.get("IN_SCOPE_IDENTIFIABLE", {})
    row: dict[str, Any] = {
        "threshold": float(threshold),
        "identifiable_coverage": ident.get("coverage"),
        "identifiable_selective_risk": ident.get("selective_risk"),
        "identifiable_accuracy": ident.get("accuracy"),
        "identifiable_accepted": ident.get("accepted"),
        "identifiable_queries": ident.get("queries"),
    }
    for name in ABSTAIN_CLASSES:
        item = by_class.get(name, {})
        row[f"{name}_abstention"] = item.get("abstention")
        row[f"{name}_false_acceptance"] = item.get("coverage")
        row[f"{name}_accepted"] = item.get("accepted")
    violations = []
    coverage = row["identifiable_coverage"]
    risk = row["identifiable_selective_risk"]
    if coverage is None or coverage < 0.90:
        violations.append("identifiable_coverage<0.90")
    if risk is None:
        violations.append("identifiable_selective_risk_undefined")
    elif risk > 0.05:
        violations.append("identifiable_selective_risk>0.05")
    for name in ABSTAIN_CLASSES:
        abstention = row[f"{name}_abstention"]
        false_acceptance = row[f"{name}_false_acceptance"]
        if abstention is None or abstention < 0.90:
            violations.append(f"{name}_abstention<0.90")
        if false_acceptance is None or false_acceptance > 0.10:
            violations.append(f"{name}_false_acceptance>0.10")
    row["violations"] = violations
    row["feasible"] = not violations
    return row


def risk_coverage_curve(records: Sequence[TaskRecord], probabilities: Sequence[np.ndarray],
                        thresholds: Iterable[float] | None = None) -> list[dict[str, Any]]:
    if thresholds is None:
        thresholds = np.round(np.linspace(0.50, 0.999, 50), 3)
    return [_curve_row(records, probabilities, float(threshold)) for threshold in thresholds]


def _task_scores(records: Sequence[TaskRecord], probabilities: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    rows = []
    for task, prob in zip(records, probabilities):
        y = np.asarray(task.query_y_private, dtype=np.int64)
        confidence = np.maximum(prob, 1.0 - prob)
        clipped = np.clip(prob, 1e-12, 1.0 - 1e-12)
        entropy = -(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped))
        predicted = (prob >= 0.5).astype(np.int64)
        rows.append({
            "task_id": task.task_id,
            "family_id": task.family_id,
            "class": task.classification_private,
            "queries": len(prob),
            "mean_confidence": float(confidence.mean()),
            "mean_entropy": float(entropy.mean()),
            "accuracy": float((predicted == y).mean()),
            "positive_rate": float(y.mean()),
        })
    return rows


def _auc(rows: Sequence[dict[str, Any]], positive: str, negative: str,
         field: str) -> dict[str, Any]:
    selected = [row for row in rows if row["class"] in (positive, negative)]
    labels = [1 if row["class"] == positive else 0 for row in selected]
    scores = [row[field] for row in selected]
    if len(set(labels)) < 2:
        return {"positive": positive, "negative": negative, "field": field,
                "auc": None, "n_positive": int(sum(labels)), "n_negative": int(len(labels) - sum(labels))}
    return {
        "positive": positive,
        "negative": negative,
        "field": field,
        "auc": float(roc_auc_score(labels, scores)),
        "n_positive": int(sum(labels)),
        "n_negative": int(len(labels) - sum(labels)),
    }


def _broken_fallback(sweep: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    """Reproduce the historical broken score for diagnostic comparison only."""
    if not sweep:
        return None
    best = None
    best_score = None
    for row in sweep:
        score = (
            (row.get("identifiable_coverage") or 0.0)
            + (row.get("ambiguous_abstention") or 0.0)
            - (row.get("identifiable_selective_risk") or 0.0)
        )
        if best is None or score > best_score:
            best = row
            best_score = score
    return best


def _evaluate_threshold(records: Sequence[TaskRecord], probabilities: Sequence[np.ndarray],
                        threshold: float | None) -> dict[str, Any]:
    if threshold is None:
        predictions = _all_abstain(probabilities)
    else:
        predictions = [deepsets_predictions(prob, threshold) for prob in probabilities]
    metrics = summarize(records, predictions)
    return {"threshold": threshold, "metrics": metrics, "gate": gate_status(metrics["by_class"])}


def diagnose_checkpoint(checkpoint: str | Path, calibration: Sequence[TaskRecord],
                        evaluation: Sequence[TaskRecord]) -> dict[str, Any]:
    path = Path(checkpoint)
    model, meta = load_checkpoint(path)
    calibration_probabilities = deepsets_probabilities(model, calibration)
    evaluation_probabilities = deepsets_probabilities(model, evaluation)
    selection = select_threshold(calibration, calibration_probabilities)
    broken = _broken_fallback(selection["sweep"])
    fixed_eval = _evaluate_threshold(evaluation, evaluation_probabilities,
                                     selection["chosen_threshold"])
    broken_eval = _evaluate_threshold(evaluation, evaluation_probabilities,
                                      None if broken is None else broken["threshold"])
    calibration_curve = risk_coverage_curve(calibration, calibration_probabilities)
    evaluation_curve = risk_coverage_curve(evaluation, evaluation_probabilities)
    task_scores = _task_scores(evaluation, evaluation_probabilities)
    aucs = [
        _auc(task_scores, "IN_SCOPE_IDENTIFIABLE", "IN_SCOPE_AMBIGUOUS", "mean_confidence"),
        _auc(task_scores, "IN_SCOPE_IDENTIFIABLE", "IN_SCOPE_AMBIGUOUS", "mean_entropy"),
        _auc(task_scores, "IN_SCOPE_IDENTIFIABLE", "OUT_OF_SCOPE", "mean_confidence"),
        _auc(task_scores, "IN_SCOPE_IDENTIFIABLE", "CONTRADICTORY_SUPPORT", "mean_confidence"),
    ]
    risk_at_90 = [
        row for row in evaluation_curve
        if row["identifiable_coverage"] is not None and row["identifiable_coverage"] >= 0.90
    ]
    best_risk_at_90 = min(
        (row["identifiable_selective_risk"] for row in risk_at_90
         if row["identifiable_selective_risk"] is not None),
        default=None,
    )
    max_coverage_at_risk_5 = max(
        (row["identifiable_coverage"] for row in evaluation_curve
         if row["identifiable_selective_risk"] is not None
         and row["identifiable_selective_risk"] <= 0.05),
        default=None,
    )
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256(path),
        "checkpoint_metadata_sha256": _sha256(path.with_suffix(".json")),
        "config": meta.get("config"),
        "training_history": meta.get("history", []),
        "calibration_probability_summary": _class_probability_summary(
            calibration, calibration_probabilities
        ),
        "evaluation_probability_summary": _class_probability_summary(
            evaluation, evaluation_probabilities
        ),
        "calibration_risk_coverage": calibration_curve,
        "evaluation_risk_coverage": evaluation_curve,
        "task_scores_evaluation": task_scores,
        "discrimination_auc": aucs,
        "fixed_threshold_selection": selection,
        "broken_fallback_row": broken,
        "fixed_evaluation": fixed_eval,
        "broken_fallback_evaluation": broken_eval,
        "diagnostic_indicators": {
            "fixed_calibration_feasible": bool(selection["feasible"]),
            "broken_fallback_threshold": None if broken is None else broken["threshold"],
            "broken_fallback_feasible": bool(broken["feasible"]) if broken else None,
            "evaluation_curve_feasible_thresholds": sum(1 for row in evaluation_curve if row["feasible"]),
            "calibration_curve_feasible_thresholds": sum(1 for row in calibration_curve if row["feasible"]),
            "best_evaluation_risk_at_coverage_90": best_risk_at_90,
            "max_evaluation_coverage_at_risk_5": max_coverage_at_risk_5,
            "identifiable_vs_ambiguous_confidence_auc": aucs[0]["auc"],
            "identifiable_vs_ambiguous_entropy_auc": aucs[1]["auc"],
            "identifiable_vs_ood_confidence_auc": aucs[2]["auc"],
            "identifiable_vs_contradictory_confidence_auc": aucs[3]["auc"],
        },
    }


def run_deepsets_diagnostics(
    *, checkpoint_dir: str | Path = ".local_private/v043_development/checkpoints",
    seeds: Sequence[int] = (430043, 430044, 430045),
    manifest_path: str | Path = "results/v043_task010/dev_manifest.json",
    output_dir: str | Path = "results/v043_task010",
    checkpoint_pattern: str = "deepsets_seed_{seed}.npz",
    output_name: str = "deepsets_diagnostics.json",
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records, generated_subsets, generated_audit = build_family_aware_development_tasks(seed=430043)
    subsets, manifest, audit = load_family_aware_subsets(manifest_path, records)
    calibration = subsets["calibration"]
    evaluation = subsets["evaluation"]
    checkpoint_root = Path(checkpoint_dir)
    seed_rows: list[dict[str, Any]] = []
    not_run: list[dict[str, Any]] = []
    for seed in seeds:
        checkpoint = checkpoint_root / checkpoint_pattern.format(seed=seed)
        if not checkpoint.exists():
            not_run.append({"seed": int(seed), "checkpoint": str(checkpoint), "reason": "missing checkpoint"})
            continue
        seed_rows.append(diagnose_checkpoint(checkpoint, calibration, evaluation))
    manifest_sha256 = _sha256(Path(manifest_path))
    results = {
        "schema": "nucleus-pulse-v0.43-deepsets-diagnostics-v1",
        "protocol_version": FAMILY_DEV_PROTOCOL,
        "official_test_opened": False,
        "official_test_ready": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "development_split_audit": audit,
        "calibration_task_count": len(calibration),
        "evaluation_task_count": len(evaluation),
        "seeds_requested": [int(seed) for seed in seeds],
        "seed_results": seed_rows,
        "not_run": not_run,
        "notes": [
            "Diagnostics only; no threshold is selected on evaluation families.",
            "Evaluation-curve feasibility is reported to assess attainability, not to select a threshold.",
            "Brier and ECE are query-level descriptive calibration metrics.",
        ],
    }
    (out / output_name).write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
