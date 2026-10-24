"""Equal-information baselines and development metrics for v0.43."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from .dsl import enumerate_programs, mask_predict, point_index, semantic_mask
from .generator import TaskRecord


@dataclass(frozen=True)
class Predictions:
    predicted: np.ndarray
    accepted: np.ndarray
    confidence: np.ndarray


def _support_arrays(task: TaskRecord) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[ex.x[0], ex.x[1]] for ex in task.support], dtype=np.float64)
    y = np.asarray([ex.y for ex in task.support], dtype=np.int64)
    return x, y


def _query_array(task: TaskRecord) -> np.ndarray:
    return np.asarray([[x[0], x[1]] for x in task.query_x], dtype=np.float64)


def predict_constant(task: TaskRecord) -> Predictions:
    x, y = _support_arrays(task)
    value = int(np.argmax(np.bincount(y, minlength=2)))
    n = len(task.query_x)
    return Predictions(
        predicted=np.full(n, value, dtype=np.int64),
        accepted=np.ones(n, dtype=bool),
        confidence=np.ones(n, dtype=np.float64),
    )


def predict_retrieval(task: TaskRecord) -> Predictions:
    support_x, support_y = _support_arrays(task)
    query_x = _query_array(task)
    distances = ((query_x[:, None, :] - support_x[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)
    predicted = support_y[nearest]
    return Predictions(
        predicted=predicted,
        accepted=np.ones(len(query_x), dtype=bool),
        confidence=np.ones(len(query_x), dtype=np.float64),
    )


def predict_linear(task: TaskRecord, *, seed: int = 430043) -> Predictions:
    support_x, support_y = _support_arrays(task)
    query_x = _query_array(task)
    if len(np.unique(support_y)) == 1:
        return Predictions(
            predicted=np.full(len(query_x), int(support_y[0]), dtype=np.int64),
            accepted=np.ones(len(query_x), dtype=bool),
            confidence=np.ones(len(query_x), dtype=np.float64),
        )
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    model.fit(support_x, support_y)
    proba = model.predict_proba(query_x)[:, 1]
    return Predictions(
        predicted=(proba >= 0.5).astype(np.int64),
        accepted=np.ones(len(query_x), dtype=bool),
        confidence=np.maximum(proba, 1.0 - proba),
    )


def predict_tree(task: TaskRecord, *, seed: int = 430043) -> Predictions:
    support_x, support_y = _support_arrays(task)
    query_x = _query_array(task)
    if len(np.unique(support_y)) == 1:
        return Predictions(
            predicted=np.full(len(query_x), int(support_y[0]), dtype=np.int64),
            accepted=np.ones(len(query_x), dtype=bool),
            confidence=np.ones(len(query_x), dtype=np.float64),
        )
    model = DecisionTreeClassifier(max_depth=3, random_state=seed)
    model.fit(support_x, support_y)
    proba = model.predict_proba(query_x)[:, 1]
    return Predictions(
        predicted=(proba >= 0.5).astype(np.int64),
        accepted=np.ones(len(query_x), dtype=bool),
        confidence=np.maximum(proba, 1.0 - proba),
    )


def _consistent_masks(task: TaskRecord) -> list[int]:
    support = [(ex.x, ex.y) for ex in task.support]
    consistent_masks: list[int] = []
    for expr in enumerate_programs():
        mask = semantic_mask(expr)
        if all(mask_predict(mask, x) == y for x, y in support):
            consistent_masks.append(mask)
    return consistent_masks


def predict_program_induction(task: TaskRecord) -> Predictions:
    """Exact closed-world induction; abstain when consistent programs disagree."""
    consistent_masks = _consistent_masks(task)
    if not consistent_masks:
        n = len(task.query_x)
        return Predictions(
            predicted=np.zeros(n, dtype=np.int64),
            accepted=np.zeros(n, dtype=bool),
            confidence=np.zeros(n, dtype=np.float64),
        )
    predicted: list[int] = []
    accepted: list[bool] = []
    for x in task.query_x:
        votes = [int((mask >> point_index(x)) & 1) for mask in consistent_masks]
        counts = np.bincount(votes, minlength=2)
        predicted.append(int(np.argmax(counts)))
        accepted.append(len(set(votes)) == 1)
    return Predictions(
        predicted=np.asarray(predicted, dtype=np.int64),
        accepted=np.asarray(accepted, dtype=bool),
        confidence=np.asarray(accepted, dtype=np.float64),
    )


def predict_program_induction_task_level(task: TaskRecord) -> Predictions:
    """Conservative closed-world policy: abstain on the whole task unless one
    semantic function is consistent with support."""
    consistent_masks = _consistent_masks(task)
    semantic_classes = set(consistent_masks)
    n = len(task.query_x)
    if len(semantic_classes) != 1:
        return Predictions(
            predicted=np.zeros(n, dtype=np.int64),
            accepted=np.zeros(n, dtype=bool),
            confidence=np.zeros(n, dtype=np.float64),
        )
    mask = next(iter(semantic_classes))
    predicted = np.asarray([int((mask >> point_index(x)) & 1) for x in task.query_x],
                           dtype=np.int64)
    return Predictions(
        predicted=predicted,
        accepted=np.ones(n, dtype=bool),
        confidence=np.ones(n, dtype=np.float64),
    )


def deepsets_predictions(proba: np.ndarray, threshold: float) -> Predictions:
    confidence = np.maximum(proba, 1.0 - proba)
    return Predictions(
        predicted=(proba >= 0.5).astype(np.int64),
        accepted=confidence >= threshold,
        confidence=confidence,
    )


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def summarize(records: Sequence[TaskRecord], predictions: Sequence[Predictions]) -> dict:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    per_class: dict[str, list[dict]] = {}
    all_rows: list[dict] = []
    for task, pred in zip(records, predictions):
        rows = []
        for y, p, a in zip(task.query_y_private, pred.predicted, pred.accepted):
            rows.append({"y": int(y), "pred": int(p), "accepted": bool(a)})
        all_rows.extend(rows)
        per_class.setdefault(task.classification_private, []).extend(rows)
    def _metrics(rows: list[dict]) -> dict:
        if not rows:
            return {"queries": 0, "accuracy": None, "coverage": None,
                    "selective_risk": None, "abstention": None, "accepted": 0,
                    "accuracy_ci95": None, "coverage_ci95": None,
                    "selective_risk_ci95": None, "confusion_matrix": None}
        y = np.asarray([r["y"] for r in rows])
        p = np.asarray([r["pred"] for r in rows])
        a = np.asarray([r["accepted"] for r in rows], dtype=bool)
        accepted_count = int(a.sum())
        correct = p == y
        selective_risk = (1.0 - float(correct[a].mean())) if accepted_count else None
        accepted_errors = int((~correct[a]).sum()) if accepted_count else 0
        cm = [[int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())],
              [int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())]]
        return {
            "queries": len(rows),
            "accuracy": float(correct.mean()),
            "accuracy_ci95": _wilson(int(correct.sum()), len(rows)),
            "coverage": float(a.mean()),
            "coverage_ci95": _wilson(accepted_count, len(rows)),
            "selective_risk": selective_risk,
            "selective_risk_ci95": _wilson(accepted_errors, accepted_count),
            "abstention": float(1.0 - a.mean()),
            "accepted": accepted_count,
            "confusion_matrix": cm,
        }
    return {
        "overall": _metrics(all_rows),
        "by_class": {name: _metrics(rows) for name, rows in sorted(per_class.items())},
        "tasks": len(records),
    }


def select_threshold(records: Sequence[TaskRecord], probabilities: Sequence[np.ndarray],
                     thresholds: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70,
                                                     0.75, 0.80, 0.85, 0.90, 0.95)) -> dict:
    """Select a development threshold using identifiable coverage/risk constraints.

    The target is identifiable coverage >= 0.90 with selective risk <= 0.05. Among
    feasible thresholds, choose the one with the highest identifiable coverage.
    If none is feasible, return the threshold with the best combined score and
    mark ``feasible=False``.
    """
    rows: list[dict] = []
    for threshold in thresholds:
        preds = [deepsets_predictions(p, threshold) for p in probabilities]
        summary = summarize(records, preds)
        ident = summary["by_class"].get("IN_SCOPE_IDENTIFIABLE", {})
        amb = summary["by_class"].get("IN_SCOPE_AMBIGUOUS", {})
        coverage = ident.get("coverage")
        risk = ident.get("selective_risk")
        abstention = amb.get("abstention")
        feasible = (
            coverage is not None and coverage >= 0.90
            and risk is not None and risk <= 0.05
            and abstention is not None and abstention >= 0.90
        )
        score = (coverage or 0.0) + (abstention or 0.0) - (risk or 0.0)
        rows.append({
            "threshold": float(threshold),
            "feasible": bool(feasible),
            "identifiable_coverage": coverage,
            "identifiable_selective_risk": risk,
            "ambiguous_abstention": abstention,
            "score": float(score),
        })
    feasible_rows = [r for r in rows if r["feasible"]]
    chosen = max(feasible_rows, key=lambda r: r["identifiable_coverage"]) if feasible_rows else max(rows, key=lambda r: r["score"])
    return {"chosen": chosen, "feasible": bool(feasible_rows), "sweep": rows}
