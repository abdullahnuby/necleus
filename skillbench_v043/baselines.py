"""Equal-information baselines and development metrics for v0.43."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

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


def _family_metric(rows: list[dict], name: str) -> float | None:
    if not rows:
        return None
    y = np.asarray([r["y"] for r in rows])
    p = np.asarray([r["pred"] for r in rows])
    a = np.asarray([r["accepted"] for r in rows], dtype=bool)
    if name == "accuracy":
        return float((p == y).mean())
    if name == "coverage":
        return float(a.mean())
    if name == "abstention":
        return float(1.0 - a.mean())
    if name == "selective_risk":
        if not a.any():
            return None
        return float(1.0 - (p[a] == y[a]).mean())
    raise ValueError(f"unknown family metric: {name}")


def _cluster_bootstrap(rows_by_family: dict[str, list[dict]], *, n_boot: int,
                       seed: int) -> dict:
    """Family-level cluster bootstrap for development metrics.

    Queries within a task/family are not independent. This resamples whole
    families, preserving all rows of each sampled family.
    """
    if n_boot <= 0 or not rows_by_family:
        return {}
    families = sorted(rows_by_family)
    rng = np.random.default_rng(seed)
    metrics = ("accuracy", "coverage", "abstention", "selective_risk")
    values: dict[str, list[float]] = {name: [] for name in metrics}
    for _ in range(n_boot):
        sampled = rng.choice(len(families), size=len(families), replace=True)
        rows: list[dict] = []
        for index in sampled:
            rows.extend(rows_by_family[families[int(index)]])
        for name in metrics:
            value = _family_metric(rows, name)
            if value is not None:
                values[name].append(value)
    out: dict[str, Any] = {
        "n_families": len(families),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
    }
    for name in metrics:
        if values[name]:
            out[f"{name}_ci95"] = [
                float(np.percentile(values[name], 2.5)),
                float(np.percentile(values[name], 97.5)),
            ]
        else:
            out[f"{name}_ci95"] = None
    out["valid_risk_samples"] = len(values["selective_risk"])
    return out


def summarize(records: Sequence[TaskRecord], predictions: Sequence[Predictions], *,
              cluster_bootstrap: int = 0, bootstrap_seed: int = 430043) -> dict:
    if len(records) != len(predictions):
        raise ValueError("records and predictions must have equal length")
    per_class: dict[str, list[dict]] = {}
    per_class_families: dict[str, dict[str, list[dict]]] = {}
    all_rows: list[dict] = []
    all_families: dict[str, list[dict]] = {}
    family_rows: dict[str, dict[str, Any]] = {}
    for task, pred in zip(records, predictions):
        rows = []
        for y, p, a in zip(task.query_y_private, pred.predicted, pred.accepted):
            rows.append({
                "task_id": task.task_id, "y": int(y), "pred": int(p),
                "accepted": bool(a),
            })
        all_rows.extend(rows)
        per_class.setdefault(task.classification_private, []).extend(rows)
        all_families.setdefault(task.family_id, []).extend(rows)
        per_class_families.setdefault(task.classification_private, {}).setdefault(
            task.family_id, []
        ).extend(rows)
        if task.family_id not in family_rows:
            family_rows[task.family_id] = {
                "family_id": task.family_id,
                "class": task.classification_private,
                "task_ids": [],
            }
        family_rows[task.family_id]["task_ids"].append(task.task_id)

    def _metrics(rows: list[dict], families: dict[str, list[dict]] | None = None) -> dict:
        if not rows:
            return {"queries": 0, "accuracy": None, "coverage": None,
                    "selective_risk": None, "abstention": None, "accepted": 0,
                    "accuracy_ci95": None, "coverage_ci95": None,
                    "selective_risk_ci95": None, "confusion_matrix": None,
                    "n_families": 0, "n_tasks": 0, "family_clustered": None}
        y = np.asarray([r["y"] for r in rows])
        p = np.asarray([r["pred"] for r in rows])
        a = np.asarray([r["accepted"] for r in rows], dtype=bool)
        accepted_count = int(a.sum())
        correct = p == y
        selective_risk = (1.0 - float(correct[a].mean())) if accepted_count else None
        accepted_errors = int((~correct[a]).sum()) if accepted_count else 0
        cm = [[int(((y == 0) & (p == 0)).sum()), int(((y == 0) & (p == 1)).sum())],
              [int(((y == 1) & (p == 0)).sum()), int(((y == 1) & (p == 1)).sum())]]
        n_families = len(families) if families is not None else 0
        n_tasks = len({r.get("task_id") for r in rows if r.get("task_id")})
        clustered = None
        if families is not None and cluster_bootstrap > 0:
            clustered = _cluster_bootstrap(
                families, n_boot=cluster_bootstrap, seed=bootstrap_seed
            )
            clustered["accepted"] = accepted_count
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
            "n_families": n_families,
            "n_tasks": n_tasks,
            "family_clustered": clustered,
        }

    per_family: dict[str, dict[str, Any]] = {}
    for fid, rows in sorted(all_families.items()):
        meta = family_rows[fid]
        per_family[fid] = {
            "family_id": fid,
            "class": meta["class"],
            "task_ids": meta["task_ids"],
            "n_tasks": len(meta["task_ids"]),
            "queries": len(rows),
            "accepted": int(sum(1 for row in rows if row["accepted"])),
            "accuracy": _family_metric(rows, "accuracy"),
            "coverage": _family_metric(rows, "coverage"),
            "selective_risk": _family_metric(rows, "selective_risk"),
            "abstention": _family_metric(rows, "abstention"),
        }

    return {
        "overall": _metrics(all_rows, all_families),
        "by_class": {
            name: _metrics(rows, per_class_families.get(name))
            for name, rows in sorted(per_class.items())
        },
        "per_family": per_family,
        "tasks": len(records),
    }


IDENTIFIABLE_COVERAGE_MIN = 0.90
IDENTIFIABLE_RISK_MAX = 0.05
ABSTENTION_MIN = 0.90
FALSE_ACCEPTANCE_MAX = 0.10


def gate_status(metrics: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen development gates without reinterpreting failures."""
    identifiable = metrics.get("IN_SCOPE_IDENTIFIABLE", {})
    coverage = identifiable.get("coverage")
    risk = identifiable.get("selective_risk")
    identifiable_pass = (
        coverage is not None and coverage >= IDENTIFIABLE_COVERAGE_MIN
        and risk is not None and risk <= IDENTIFIABLE_RISK_MAX
    )
    abstention_rows: dict[str, Any] = {}
    abstention_pass = True
    for name in ("IN_SCOPE_AMBIGUOUS", "OUT_OF_SCOPE", "CONTRADICTORY_SUPPORT"):
        row = metrics.get(name, {})
        abstention = row.get("abstention")
        false_acceptance = row.get("coverage")
        passed = (
            abstention is not None and abstention >= ABSTENTION_MIN
            and false_acceptance is not None and false_acceptance <= FALSE_ACCEPTANCE_MAX
        )
        abstention_rows[name] = {
            "abstention": abstention,
            "false_acceptance": false_acceptance,
            "passed": passed,
        }
        abstention_pass = abstention_pass and passed
    return {
        "passed": bool(identifiable_pass and abstention_pass),
        "identifiable": {
            "coverage": coverage,
            "selective_risk": risk,
            "passed": identifiable_pass,
        },
        "abstention": abstention_rows,
        "criteria": {
            "identifiable_coverage_min": IDENTIFIABLE_COVERAGE_MIN,
            "identifiable_selective_risk_max": IDENTIFIABLE_RISK_MAX,
            "abstention_min": ABSTENTION_MIN,
            "false_acceptance_max": FALSE_ACCEPTANCE_MAX,
        },
    }


def _threshold_violations(row: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    coverage = row.get("identifiable_coverage")
    risk = row.get("identifiable_selective_risk")
    abstention = row.get("ambiguous_abstention")
    if coverage is None or coverage < IDENTIFIABLE_COVERAGE_MIN:
        violations.append("identifiable_coverage<0.90")
    if risk is None:
        violations.append("identifiable_selective_risk_undefined")
    elif risk > IDENTIFIABLE_RISK_MAX:
        violations.append("identifiable_selective_risk>0.05")
    if abstention is None or abstention < ABSTENTION_MIN:
        violations.append("ambiguous_abstention<0.90")
    return violations


def _threshold_violation_magnitude(row: dict[str, Any]) -> float:
    coverage = row.get("identifiable_coverage")
    risk = row.get("identifiable_selective_risk")
    abstention = row.get("ambiguous_abstention")
    return (
        max(0.0, IDENTIFIABLE_COVERAGE_MIN - (coverage if coverage is not None else 0.0))
        + max(0.0, (risk if risk is not None else 1.0) - IDENTIFIABLE_RISK_MAX)
        + max(0.0, ABSTENTION_MIN - (abstention if abstention is not None else 0.0))
    )


def select_threshold(records: Sequence[TaskRecord], probabilities: Sequence[np.ndarray],
                     thresholds: Iterable[float] = (0.50, 0.55, 0.60, 0.65, 0.70,
                                                     0.75, 0.80, 0.85, 0.90, 0.95)) -> dict:
    """Select a development threshold using the frozen identifiable/abstention gates.

    A threshold is feasible only when every required criterion is satisfied.
    Undefined selective risk remains undefined and can never be treated as zero.
    If no threshold is feasible, ``chosen`` is ``None`` and the diagnostic
    fallback is reported separately; it is not a successful calibration.
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
        row = {
            "threshold": float(threshold),
            "identifiable_coverage": coverage,
            "identifiable_selective_risk": risk,
            "identifiable_accepted": ident.get("accepted"),
            "ambiguous_abstention": abstention,
            "overall_accepted": summary["overall"].get("accepted"),
            "overall_queries": summary["overall"].get("queries"),
        }
        violations = _threshold_violations(row)
        row["violations"] = violations
        row["feasible"] = not violations
        row["violation_count"] = len(violations)
        row["violation_magnitude"] = _threshold_violation_magnitude(row)
        rows.append(row)

    feasible_rows = [row for row in rows if row["feasible"]]
    chosen = None
    if feasible_rows:
        chosen = max(
            feasible_rows,
            key=lambda row: (
                row["identifiable_coverage"],
                -row["identifiable_selective_risk"],
                -row["threshold"],
            ),
        )
    diagnostic_fallback = None
    if rows:
        diagnostic_fallback = min(
            rows,
            key=lambda row: (
                row["violation_count"],
                row["violation_magnitude"],
                -(row["identifiable_coverage"] if row["identifiable_coverage"] is not None else 0.0),
                row["identifiable_selective_risk"] if row["identifiable_selective_risk"] is not None else 1.0,
                row["threshold"],
            ),
        )
    return {
        "chosen": chosen,
        "chosen_threshold": chosen["threshold"] if chosen is not None else None,
        "feasible": bool(feasible_rows),
        "criteria": {
            "identifiable_coverage_min": IDENTIFIABLE_COVERAGE_MIN,
            "identifiable_selective_risk_max": IDENTIFIABLE_RISK_MAX,
            "ambiguous_abstention_min": ABSTENTION_MIN,
        },
        "diagnostic_fallback": diagnostic_fallback,
        "sweep": rows,
    }
