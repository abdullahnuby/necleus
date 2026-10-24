"""Corrective feedback and retention loop for v0.43 Phase 5."""
from __future__ import annotations

from dataclasses import dataclass
import json
import random
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier

from .corpus import _targets_by_family
from .dsl import DOMAIN_SIZE, enumerate_programs, mask_predict, point_index, semantic_mask
from .generator import Example, build_family_catalog, generate_task


@dataclass(frozen=True)
class RetentionSkill:
    skill_index: int
    task_id: str
    family_id: str
    target_ast_hash: str
    support: tuple[Example, ...]
    interaction_rounds: tuple[tuple[tuple[int, int], ...], ...]
    interaction_labels: tuple[tuple[int, ...], ...]
    eval_x: tuple[tuple[int, int], ...]
    eval_y: tuple[int, ...]


def _features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x0 = x[:, 0] / 99.0
    x1 = x[:, 1] / 99.0
    return np.column_stack([
        x0, x1,
        (x[:, 0] % 2 == 0).astype(float), (x[:, 0] % 3 == 0).astype(float),
        (x[:, 0] % 5 == 0).astype(float), (x[:, 1] % 2 == 0).astype(float),
        (x[:, 1] % 3 == 0).astype(float), (x[:, 1] % 5 == 0).astype(float),
        (x[:, 0] > 25).astype(float), (x[:, 0] > 50).astype(float),
        (x[:, 0] > 75).astype(float), (x[:, 1] > 25).astype(float),
        (x[:, 1] > 50).astype(float), (x[:, 1] > 75).astype(float),
        x0 * x1,
    ])


class FeatureLinearLearner:
    def __init__(self, seed: int = 430043):
        self.model = SGDClassifier(
            loss="log_loss", learning_rate="constant", eta0=0.05,
            random_state=seed, max_iter=1, tol=None, shuffle=False,
        )
        self.fitted = False

    def update(self, examples: Sequence[Example]) -> None:
        if not examples:
            return
        x = _features(np.asarray([ex.x for ex in examples], dtype=np.float64))
        y = np.asarray([ex.y for ex in examples], dtype=np.int64)
        if not self.fitted:
            self.model.partial_fit(x, y, classes=np.asarray([0, 1]))
            self.fitted = True
        else:
            self.model.partial_fit(x, y)

    def predict(self, query_x: Sequence[tuple[int, int]]) -> np.ndarray:
        if not self.fitted:
            return np.zeros(len(query_x), dtype=np.int64)
        x = _features(np.asarray(query_x, dtype=np.float64))
        return self.model.predict(x).astype(np.int64)


def _knn_predict(memory: Sequence[Example], query_x: Sequence[tuple[int, int]], k: int = 1) -> np.ndarray:
    if not memory:
        return np.zeros(len(query_x), dtype=np.int64)
    mx = np.asarray([ex.x for ex in memory], dtype=np.float64)
    my = np.asarray([ex.y for ex in memory], dtype=np.int64)
    qx = np.asarray(query_x, dtype=np.float64)
    distances = ((qx[:, None, :] - mx[None, :, :]) ** 2).sum(axis=2)
    nearest = np.argsort(distances, axis=1)[:, :k]
    labels = my[nearest]
    return (labels.mean(axis=1) >= 0.5).astype(np.int64)


def _program_predict(memory: Sequence[Example], query_x: Sequence[tuple[int, int]]) -> np.ndarray:
    labels = [(ex.x, ex.y) for ex in memory]
    masks: list[int] = []
    for expr in enumerate_programs():
        mask = semantic_mask(expr)
        if all(mask_predict(mask, x) == y for x, y in labels):
            masks.append(mask)
    if not masks:
        return np.zeros(len(query_x), dtype=np.int64)
    predicted: list[int] = []
    for x in query_x:
        votes = [mask_predict(mask, x) for mask in masks]
        predicted.append(int(np.mean(votes) >= 0.5))
    return np.asarray(predicted, dtype=np.int64)


def _sample_queries(mask: int, rng: random.Random, excluded: set[tuple[int, int]],
                    count: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    positives = [(x0, x1) for x0 in range(DOMAIN_SIZE) for x1 in range(DOMAIN_SIZE)
                 if ((mask >> (x0 * DOMAIN_SIZE + x1)) & 1) and (x0, x1) not in excluded]
    negatives = [(x0, x1) for x0 in range(DOMAIN_SIZE) for x1 in range(DOMAIN_SIZE)
                 if not ((mask >> (x0 * DOMAIN_SIZE + x1)) & 1) and (x0, x1) not in excluded]
    half = count // 2
    points = rng.sample(positives, half) + rng.sample(negatives, half)
    rng.shuffle(points)
    return tuple(points), tuple((mask >> (x0 * DOMAIN_SIZE + x1)) & 1 for x0, x1 in points)


def build_retention_skills(*, n_skills: int = 20, seed: int = 430043,
                           task_class: str = "IN_SCOPE_AMBIGUOUS",
                           support_mode: str = "random",
                           min_disagreement: float = 0.40) -> list[RetentionSkill]:
    catalog = build_family_catalog()
    targets = _targets_by_family(catalog["dev"], "dev")
    if len(targets) < n_skills:
        raise RuntimeError(f"need {n_skills} development families, found {len(targets)}")
    skills: list[RetentionSkill] = []
    for index, target in enumerate(targets):
        if len(skills) >= n_skills:
            break
        try:
            task = generate_task(
                target, split="dev", seed=seed + index * 101,
                desired_class=task_class, support_mode=support_mode,
                max_attempts=100, task_index=index,
            )
        except (ValueError, RuntimeError):
            continue
        if (task_class == "IN_SCOPE_AMBIGUOUS"
                and float(task.audit_private.get("candidate_disagreement", 0.0)) < min_disagreement):
            continue
        mask = semantic_mask(target)
        rng = random.Random(seed + 500_000 + index)
        excluded = {ex.x for ex in task.support} | set(task.query_x)
        interaction_rounds: list[tuple[tuple[int, int], ...]] = []
        interaction_labels: list[tuple[int, ...]] = []
        for _ in range(4):
            points, labels = _sample_queries(mask, rng, excluded, 50)
            excluded.update(points)
            interaction_rounds.append(points)
            interaction_labels.append(labels)
        eval_x, eval_y = _sample_queries(mask, rng, excluded, 100)
        skills.append(RetentionSkill(
            skill_index=len(skills), task_id=task.task_id, family_id=task.family_id,
            target_ast_hash=task.ast_sha256, support=task.support,
            interaction_rounds=tuple(interaction_rounds),
            interaction_labels=tuple(interaction_labels),
            eval_x=eval_x, eval_y=eval_y,
        ))
        if len(skills) == n_skills:
            break
    if len(skills) < n_skills:
        raise RuntimeError(f"generated only {len(skills)} of {n_skills} retention skills")
    return skills


def _accuracy(predicted: np.ndarray, labels: Sequence[int]) -> float:
    return float(np.mean(np.asarray(predicted) == np.asarray(labels)))


def run_correction_retention(*, n_skills: int = 20, seed: int = 430043,
                             knn_k: int = 3, memory_mode: str = "program") -> dict[str, Any]:
    skills = build_retention_skills(n_skills=n_skills, seed=seed)
    memories: list[list[Example]] = [[] for _ in skills]
    parameter_model = FeatureLinearLearner(seed=seed)
    replay_buffer: list[Example] = []
    replay_rng = random.Random(seed + 900_000)
    round_accuracy = {"memory": [[0.0] * 4 for _ in skills],
                      "parameter": [[0.0] * 4 for _ in skills]}
    correction_log: list[dict[str, Any]] = []
    retention_matrix = {"memory": [], "parameter": []}
    memory_eval_predictions: list[np.ndarray] = []

    for i, skill in enumerate(skills):
        memory = list(skill.support)
        memories[i] = memory
        parameter_model.update(skill.support)
        for round_index in range(4):
            interaction_x = skill.interaction_rounds[round_index]
            interaction_y = skill.interaction_labels[round_index]
            round_corrections: list[Example] = []
            parameter_examples: list[Example] = []
            parameter_corrections = 0
            round_memory_pred = (_program_predict(memory, interaction_x)
                                 if memory_mode == "program" else None)
            for query_index, (x, y) in enumerate(zip(interaction_x, interaction_y)):
                if memory_mode == "program":
                    memory_pred = int(round_memory_pred[query_index])
                else:
                    memory_pred = int(_knn_predict(memory, [x], k=knn_k)[0])
                parameter_pred = int(parameter_model.predict([x])[0])
                # A correction is disclosed only for a wrong answer. Correct
                # answers reveal no extra label and do not update either learner.
                if memory_pred != y:
                    round_corrections.append(Example(x, y))
                    if memory_mode != "program":
                        memory.append(Example(x, y))
                if parameter_pred != y:
                    parameter_corrections += 1
                    correction = Example(x, y)
                    parameter_examples.append(correction)
                    parameter_model.update([correction])
            if memory_mode == "program":
                memory.extend(round_corrections)
            if replay_buffer:
                replay_count = min(8, len(replay_buffer))
                parameter_model.update(replay_rng.sample(replay_buffer, replay_count))
            replay_buffer.extend(parameter_examples)
            if len(replay_buffer) > 200:
                replay_buffer = replay_buffer[-200:]
            if memory_mode == "program":
                memory_eval = _program_predict(memory, skill.eval_x)
            else:
                memory_eval = _knn_predict(memory, skill.eval_x, k=knn_k)
            parameter_eval = parameter_model.predict(skill.eval_x)
            round_accuracy["memory"][i][round_index] = _accuracy(memory_eval, skill.eval_y)
            round_accuracy["parameter"][i][round_index] = _accuracy(parameter_eval, skill.eval_y)
            correction_log.append({
                "skill_index": i, "round": round_index,
                "interaction_queries": len(interaction_x),
                "memory_corrections": len(round_corrections),
                "parameter_corrections": parameter_corrections,
                "memory_size_after": len(memory),
            })
        memory_eval_predictions.append(memory_eval)
        # Retention: evaluate every skill seen so far after learning this skill.
        for model_name in ("memory", "parameter"):
            row = []
            for j in range(i + 1):
                if model_name == "memory":
                    pred = memory_eval_predictions[j]
                else:
                    pred = parameter_model.predict(skills[j].eval_x)
                row.append(_accuracy(pred, skills[j].eval_y))
            retention_matrix[model_name].append(row)

    def _forgetting(matrix: list[list[float]]) -> dict[str, float]:
        per_skill = []
        for j in range(len(matrix)):
            best = max(matrix[i][j] for i in range(j, len(matrix)))
            final = matrix[-1][j]
            per_skill.append(max(0.0, best - final))
        return {
            "average_forgetting": float(np.mean(per_skill)),
            "worst_task_forgetting": float(np.max(per_skill)),
            "per_skill": per_skill,
        }

    memory_improvement = [round_accuracy["memory"][i][3] - round_accuracy["memory"][i][0]
                          for i in range(len(skills))]
    parameter_improvement = [round_accuracy["parameter"][i][3] - round_accuracy["parameter"][i][0]
                             for i in range(len(skills))]
    return {
        "schema": "nucleus-pulse-v0.43-correction-retention-v1",
        "official_test_opened": False,
        "skill_count": len(skills),
        "rounds_per_skill": 4,
        "interaction_queries_per_round": 50,
        "independent_eval_queries_per_skill": 100,
        "skills": [
            {"skill_index": s.skill_index, "task_id": s.task_id,
             "family_id": s.family_id, "support_size": len(s.support)}
            for s in skills
        ],
        "round_accuracy": round_accuracy,
        "correction_log": correction_log,
        "retention_matrix": retention_matrix,
        "forgetting": {
            "memory": _forgetting(retention_matrix["memory"]),
            "parameter": _forgetting(retention_matrix["parameter"]),
        },
        "improvement_round0_to_round3": {
            "memory_mean": float(np.mean(memory_improvement)),
            "memory_worst": float(np.min(memory_improvement)),
            "parameter_mean": float(np.mean(parameter_improvement)),
            "parameter_worst": float(np.min(parameter_improvement)),
        },
        "notes": [
            "Memory adaptation stores corrected examples per skill; it is not parameter learning.",
            "Parameter learning updates one shared SGD classifier and can forget earlier skills.",
            "Evaluation queries are disjoint from support, interaction, and correction rows.",
            "The official v0.43 test was not opened.",
        ],
    }
