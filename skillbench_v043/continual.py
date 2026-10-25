"""Corrective feedback and retention loop for v0.43 Phase 5."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import SGDClassifier

from .corpus import _targets_by_family
from .dsl import (
    DOMAIN_SIZE, Expr, enumerate_programs, family_id, mask_predict, point_index,
    semantic_mask,
)
from .generator import Example, build_family_catalog, generate_task
from .integrity import EXPOSED_FAMILY_IDS


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


class BalancedReplayBuffer:
    """Per-skill correction buckets with equalized replay sampling."""

    def __init__(self, capacity_per_skill: int = 20):
        self.capacity_per_skill = capacity_per_skill
        self.buckets: dict[int, list[Example]] = defaultdict(list)

    def add(self, skill_index: int, example: Example) -> None:
        bucket = self.buckets[skill_index]
        bucket.append(example)
        if len(bucket) > self.capacity_per_skill:
            del bucket[: len(bucket) - self.capacity_per_skill]

    def sample(self, count: int, rng: random.Random,
               seen_skills: Sequence[int]) -> list[Example]:
        available = [skill for skill in seen_skills if self.buckets[skill]]
        if not available or count <= 0:
            return []
        order = list(available)
        rng.shuffle(order)
        sampled: list[Example] = []
        cursor = 0
        while len(sampled) < count:
            skill = order[cursor % len(order)]
            sampled.append(rng.choice(self.buckets[skill]))
            cursor += 1
            if cursor > count * len(order):
                break
        return sampled

    def skill_counts(self) -> dict[str, int]:
        return {str(skill): len(examples) for skill, examples in sorted(self.buckets.items())}


def _sample_support_replay(skill_supports: Sequence[Sequence[Example]],
                           seen_skills: Sequence[int], count: int,
                           rng: random.Random) -> list[Example]:
    available = [skill for skill in seen_skills if skill_supports[skill]]
    if not available or count <= 0:
        return []
    order = list(available)
    rng.shuffle(order)
    sampled: list[Example] = []
    cursor = 0
    while len(sampled) < count:
        skill = order[cursor % len(order)]
        sampled.append(rng.choice(list(skill_supports[skill])))
        cursor += 1
        if cursor > count * len(order):
            break
    return sampled


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
    if len(positives) + len(negatives) < count:
        raise ValueError(
            f"need {count} unused queries, only {len(positives) + len(negatives)} remain"
        )
    # Some semantic functions are highly imbalanced. Keep the requested sample
    # size by drawing the shortfall from the other label, and report the actual
    # label counts in the task audit rather than pretending the sample is 50/50.
    half = count // 2
    n_positive = min(half, len(positives))
    n_negative = min(count - n_positive, len(negatives))
    n_positive = min(count - n_negative, len(positives))
    points = rng.sample(positives, n_positive) + rng.sample(negatives, n_negative)
    rng.shuffle(points)
    return tuple(points), tuple((mask >> (x0 * DOMAIN_SIZE + x1)) & 1 for x0, x1 in points)


def build_retention_skills(*, n_skills: int = 20, seed: int = 430043,
                           task_class: str = "IN_SCOPE_AMBIGUOUS",
                           support_mode: str = "random",
                           min_disagreement: float = 0.40,
                           family_pool: Sequence[Expr] | None = None,
                           shuffle_seed: int | None = None,
                           exclude_families: set[str] | None = None) -> list[RetentionSkill]:
    if family_pool is None:
        catalog = build_family_catalog()
        targets = list(_targets_by_family(catalog["dev"], "dev"))
    else:
        targets = list(family_pool)
    excluded = set(exclude_families or ()) | set(EXPOSED_FAMILY_IDS)
    targets = [target for target in targets if family_id(target) not in excluded]
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(targets)
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


def _forgetting_from_matrix(matrix: Sequence[Sequence[float]]) -> dict[str, Any]:
    if not matrix:
        return {"average_forgetting": None, "worst_task_forgetting": None, "per_skill": []}
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


def recompute_retention_metrics(round_accuracy: dict[str, list[list[float]]],
                                retention_matrix: dict[str, list[list[float]]]) -> dict[str, Any]:
    """Recompute improvement and forgetting from saved per-round accuracy."""
    improvements: dict[str, Any] = {}
    for model_name, rows in round_accuracy.items():
        values = [row[3] - row[0] for row in rows]
        improvements[model_name] = {
            "mean": float(np.mean(values)) if values else None,
            "worst": float(np.min(values)) if values else None,
            "per_skill": values,
        }
    return {
        "forgetting": {
            model_name: _forgetting_from_matrix(matrix)
            for model_name, matrix in retention_matrix.items()
        },
        "improvement_round0_to_round3": improvements,
    }


def _replay_examples(*, strategy: str, skill_index: int,
                     seen_skills: Sequence[int], control_buffer: Sequence[Example],
                     balanced_buffer: BalancedReplayBuffer,
                     skill_supports: Sequence[Sequence[Example]],
                     rng: random.Random) -> tuple[list[Example], dict[str, int], int]:
    if strategy == "control":
        if not control_buffer:
            return [], {}, 0
        replay = rng.sample(list(control_buffer), min(8, len(control_buffer)))
        counts: dict[str, int] = {}
        return replay, counts, 0
    if strategy == "balanced_replay":
        replay = balanced_buffer.sample(8, rng, seen_skills)
        counts: dict[str, int] = {}
        for skill in seen_skills:
            counts[str(skill)] = 0
        for example in replay:
            # The buffer is skill-indexed; recover the source skill by identity.
            for skill in seen_skills:
                if any(example is item for item in balanced_buffer.buckets[skill]):
                    counts[str(skill)] += 1
                    break
        return replay, counts, 0
    if strategy == "balanced_support_replay":
        corrections = balanced_buffer.sample(4, rng, seen_skills)
        supports = _sample_support_replay(skill_supports, seen_skills, 4, rng)
        replay = corrections + supports
        if len(replay) < 8:
            replay.extend(
                _sample_support_replay(skill_supports, seen_skills, 8 - len(replay), rng)
            )
        if len(replay) < 8:
            replay.extend(balanced_buffer.sample(8 - len(replay), rng, seen_skills))
        counts: dict[str, int] = {str(skill): 0 for skill in seen_skills}
        for example in corrections:
            for skill in seen_skills:
                if any(example is item for item in balanced_buffer.buckets[skill]):
                    counts[str(skill)] += 1
                    break
        return replay, counts, len(supports)
    raise ValueError(f"unknown retention strategy: {strategy}")


def run_correction_retention(*, n_skills: int = 20, seed: int = 430043,
                             knn_k: int = 3, memory_mode: str = "program",
                             strategy: str = "control",
                             skills: Sequence[RetentionSkill] | None = None) -> dict[str, Any]:
    if strategy not in ("control", "balanced_replay", "balanced_support_replay"):
        raise ValueError(f"unknown retention strategy: {strategy}")
    if skills is None:
        skills = build_retention_skills(n_skills=n_skills, seed=seed)
    skills = tuple(skills)
    memories: list[list[Example]] = [[] for _ in skills]
    parameter_model = FeatureLinearLearner(seed=seed)
    control_buffer: list[Example] = []
    balanced_buffer = BalancedReplayBuffer(capacity_per_skill=20)
    skill_supports = [tuple(skill.support) for skill in skills]
    replay_rng = random.Random(seed + 900_000)
    round_accuracy = {"memory": [[0.0] * 4 for _ in skills],
                      "parameter": [[0.0] * 4 for _ in skills]}
    correction_log: list[dict[str, Any]] = []
    replay_log: list[dict[str, Any]] = []
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
            replay, replay_counts, replay_support_count = _replay_examples(
                strategy=strategy, skill_index=i, seen_skills=tuple(range(i + 1)),
                control_buffer=control_buffer, balanced_buffer=balanced_buffer,
                skill_supports=skill_supports, rng=replay_rng,
            )
            if replay:
                parameter_model.update(replay)
            control_buffer.extend(parameter_examples)
            if len(control_buffer) > 200:
                control_buffer = control_buffer[-200:]
            for correction in parameter_examples:
                balanced_buffer.add(i, correction)
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
            replay_log.append({
                "skill_index": i, "round": round_index, "strategy": strategy,
                "replay_examples": len(replay),
                "replay_corrections": len(replay) - replay_support_count,
                "replay_support_examples": replay_support_count,
                "replay_skill_counts": replay_counts,
                "control_buffer_size": len(control_buffer),
                "balanced_buffer_skill_counts": balanced_buffer.skill_counts(),
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

    recomputed = recompute_retention_metrics(round_accuracy, retention_matrix)
    memory_improvement = recomputed["improvement_round0_to_round3"]["memory"]
    parameter_improvement = recomputed["improvement_round0_to_round3"]["parameter"]
    return {
        "schema": "nucleus-pulse-v0.43-correction-retention-v1",
        "official_test_opened": False,
        "strategy": strategy,
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
        "replay_log": replay_log,
        "retention_matrix": retention_matrix,
        "forgetting": recomputed["forgetting"],
        "improvement_round0_to_round3": {
            "memory_mean": memory_improvement["mean"],
            "memory_worst": memory_improvement["worst"],
            "parameter_mean": parameter_improvement["mean"],
            "parameter_worst": parameter_improvement["worst"],
        },
        "recomputed_metrics": recomputed,
        "notes": [
            "Memory adaptation stores corrected examples per skill; it is not parameter learning.",
            "Parameter learning updates one shared SGD classifier and can forget earlier skills.",
            "Evaluation queries are disjoint from support, interaction, and correction rows.",
            "The official v0.43 test was not opened.",
        ],
    }


def build_retention_sequences(*, n_skills: int = 20,
                              seeds: Sequence[int] = (430043, 430044, 430045)
                              ) -> tuple[dict[int, list[RetentionSkill]], dict[str, Any]]:
    """Build one skill sequence per seed from disjoint development-family pools."""
    catalog = build_family_catalog()
    targets = [
        target for target in _targets_by_family(catalog["dev"], "dev")
        if family_id(target) not in EXPOSED_FAMILY_IDS
    ]
    rng = random.Random(430043)
    rng.shuffle(targets)
    pools = [targets[index::len(seeds)] for index in range(len(seeds))]
    sequences: dict[int, list[RetentionSkill]] = {}
    fallback_used: list[int] = []
    for seed, pool in zip(seeds, pools):
        try:
            sequences[int(seed)] = build_retention_skills(
                n_skills=n_skills, seed=int(seed), family_pool=pool,
                shuffle_seed=int(seed),
            )
        except RuntimeError:
            fallback_used.append(int(seed))
            sequences[int(seed)] = build_retention_skills(
                n_skills=n_skills, seed=int(seed), family_pool=targets,
                shuffle_seed=int(seed),
            )
    family_sets = {
        seed: {skill.family_id for skill in skills}
        for seed, skills in sequences.items()
    }
    overlap: dict[str, int] = {}
    seed_list = list(sequences)
    for left_index, left in enumerate(seed_list):
        for right in seed_list[left_index + 1:]:
            overlap[f"{left}:{right}"] = len(family_sets[left] & family_sets[right])
    audit = {
        "seeds": seed_list,
        "n_skills": n_skills,
        "family_counts": {str(seed): len(families) for seed, families in family_sets.items()},
        "pairwise_family_overlap": overlap,
        "independent_family_sequences": all(value == 0 for value in overlap.values()),
        "fallback_used": fallback_used,
        "exposed_family_ids_used": sorted(
            set().union(*family_sets.values()) & set(EXPOSED_FAMILY_IDS)
        ),
    }
    return sequences, audit


def run_retention_comparison(
    *, n_skills: int = 20,
    seeds: Sequence[int] = (430043, 430044, 430045),
    strategies: Sequence[str] = ("control", "balanced_replay", "balanced_support_replay"),
    memory_mode: str = "program",
    output_dir: str | Path = "results/v043_task010",
) -> dict[str, Any]:
    sequences, sequence_audit = build_retention_sequences(n_skills=n_skills, seeds=seeds)
    runs: list[dict[str, Any]] = []
    for seed, skills in sequences.items():
        for strategy in strategies:
            result = run_correction_retention(
                skills=skills, seed=int(seed), strategy=strategy,
                memory_mode=memory_mode,
            )
            runs.append({
                "seed": int(seed),
                "strategy": strategy,
                "result": result,
            })
    aggregate: dict[str, Any] = {}
    for strategy in strategies:
        rows = [row["result"] for row in runs if row["strategy"] == strategy]
        parameter_improvements = [
            row["improvement_round0_to_round3"]["parameter_mean"] for row in rows
        ]
        memory_improvements = [
            row["improvement_round0_to_round3"]["memory_mean"] for row in rows
        ]
        average_forgetting = [
            row["forgetting"]["parameter"]["average_forgetting"] for row in rows
        ]
        worst_forgetting = [
            row["forgetting"]["parameter"]["worst_task_forgetting"] for row in rows
        ]
        gate_rows = [
            (
                row["improvement_round0_to_round3"]["parameter_mean"] >= 0.10
                and row["forgetting"]["parameter"]["average_forgetting"] <= 0.05
            )
            for row in rows
        ]
        aggregate[strategy] = {
            "seed_count": len(rows),
            "parameter_improvement_mean": float(np.mean(parameter_improvements)),
            "parameter_improvement_std": (
                float(np.std(parameter_improvements, ddof=1))
                if len(parameter_improvements) > 1 else 0.0
            ),
            "parameter_improvement_worst_seed": float(np.min(parameter_improvements)),
            "average_forgetting_mean": float(np.mean(average_forgetting)),
            "average_forgetting_worst_seed": float(np.max(average_forgetting)),
            "worst_task_forgetting": float(np.max(worst_forgetting)),
            "memory_improvement_mean": float(np.mean(memory_improvements)),
            "memory_forgetting_mean": float(np.mean([
                row["forgetting"]["memory"]["average_forgetting"] for row in rows
            ])),
            "gate_passed_count": int(sum(gate_rows)),
            "all_seeds_passed": bool(all(gate_rows)),
        }
    decision = {
        "status": "PASS" if any(
            row["all_seeds_passed"] for row in aggregate.values()
        ) else "FAIL",
        "reason": (
            "At least one shared-parameter strategy met the retention gate for every seed."
            if any(row["all_seeds_passed"] for row in aggregate.values())
            else "No shared-parameter strategy met the retention gate for every seed; stop optimization."
        ),
        "memory_track_is_separate_reference": True,
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "schema": "nucleus-pulse-v0.43-retention-comparison-v1",
        "official_test_opened": False,
        "official_test_ready": False,
        "n_skills": n_skills,
        "seeds": list(seeds),
        "strategies": list(strategies),
        "sequence_audit": sequence_audit,
        "runs": runs,
        "aggregate": aggregate,
        "decision": decision,
        "notes": [
            "Development only; the official v0.43 test was not opened.",
            "The per-skill memory/program track is a separate reference, not shared-parameter consolidation.",
            "All strategies use the same skill sequence per seed, support budget, interaction budget, and independent evaluation policy.",
            "Balanced replay samples corrections across previously seen skills; support replay adds original support examples.",
            "No evaluation labels are fed back to either learner.",
        ],
    }
    (out / "retention_comparison.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return results
