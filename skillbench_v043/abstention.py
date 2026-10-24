"""Phase 4 abstention evaluation on development data only."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .baselines import (
    deepsets_predictions,
    predict_program_induction,
    predict_program_induction_task_level,
    select_threshold,
    summarize,
)
from .corpus import build_development_tasks
from .deepsets import DeepSetsNumpy
from .pipeline import deepsets_probabilities, split_development


def _gate(metrics: dict[str, Any]) -> dict[str, Any]:
    identifiable = metrics.get("IN_SCOPE_IDENTIFIABLE", {})
    abstain_classes = ("IN_SCOPE_AMBIGUOUS", "OUT_OF_SCOPE", "CONTRADICTORY_SUPPORT")
    coverage = identifiable.get("coverage")
    risk = identifiable.get("selective_risk")
    ident_pass = coverage is not None and coverage >= 0.90 and risk is not None and risk <= 0.05
    abstain_rows = {}
    for name in abstain_classes:
        row = metrics.get(name, {})
        abstention = row.get("abstention")
        coverage_class = row.get("coverage")
        passed = (abstention is not None and abstention >= 0.90
                  and coverage_class is not None and coverage_class <= 0.10)
        abstain_rows[name] = {
            "abstention": abstention,
            "false_acceptance": coverage_class,
            "pass": bool(passed),
        }
    return {
        "identifiable_pass": bool(ident_pass),
        "identifiable_coverage": coverage,
        "identifiable_selective_risk": risk,
        "abstention_classes": abstain_rows,
        "all_pass": bool(ident_pass and all(row["pass"] for row in abstain_rows.values())),
    }


def run_abstention_evaluation(*, seed: int = 430043,
                              seeds: Sequence[int] = (430043, 430044, 430045),
                              checkpoint_dir: str | Path = ".local_private/v043_development/checkpoints",
                              output_dir: str | Path = "results/v043_development") -> dict[str, Any]:
    records, audit = build_development_tasks(seed=seed)
    calibration, evaluation = split_development(records)
    symbolic_task = [predict_program_induction_task_level(task) for task in evaluation]
    symbolic_query = [predict_program_induction(task) for task in evaluation]
    symbolic_task_metrics = summarize(evaluation, symbolic_task)
    symbolic_query_metrics = summarize(evaluation, symbolic_query)
    deepsets_rows: list[dict[str, Any]] = []
    for model_seed in seeds:
        checkpoint = Path(checkpoint_dir) / f"deepsets_seed_{model_seed}.npz"
        if not checkpoint.exists():
            deepsets_rows.append({"seed": int(model_seed), "status": "BLOCKED",
                                  "reason": f"checkpoint not found: {checkpoint}"})
            continue
        model = DeepSetsNumpy.load(checkpoint)
        calibration_probabilities = deepsets_probabilities(model, calibration)
        selection = select_threshold(calibration, calibration_probabilities)
        threshold = float(selection["chosen"]["threshold"])
        evaluation_probabilities = deepsets_probabilities(model, evaluation)
        predictions = [deepsets_predictions(p, threshold) for p in evaluation_probabilities]
        metrics = summarize(evaluation, predictions)
        deepsets_rows.append({
            "seed": int(model_seed),
            "status": "PASS",
            "checkpoint": str(checkpoint),
            "threshold": threshold,
            "threshold_selection": selection,
            "metrics": metrics,
            "gate": _gate(metrics["by_class"]),
        })
    result = {
        "schema": "nucleus-pulse-v0.43-abstention-v1",
        "official_test_opened": False,
        "development_split_audit": audit,
        "calibration_task_count": len(calibration),
        "evaluation_task_count": len(evaluation),
        "symbolic_task_level": {
            "metrics": symbolic_task_metrics,
            "gate": _gate(symbolic_task_metrics["by_class"]),
        },
        "symbolic_query_level": {
            "metrics": symbolic_query_metrics,
            "gate": _gate(symbolic_query_metrics["by_class"]),
        },
        "deepsets": deepsets_rows,
        "notes": [
            "All thresholds are selected on development calibration tasks only.",
            "Task-level symbolic abstention uses only support-consistency information available at runtime.",
            "The official v0.43 test was not opened.",
        ],
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "abstention_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result
