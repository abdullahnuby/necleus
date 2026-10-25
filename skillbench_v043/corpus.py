"""Deterministic training corpus and development task manifests for v0.43.

The corpus is reproducible from the frozen generator settings and seed. The
manifest records counts and hashes, not raw evaluator-private test rows.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

from .dsl import Expr, ast_hash, canonical_semantic_programs, family_id, semantic_mask
from .integrity import EXPOSED_FAMILY_IDS
from .generator import (
    Example, OOD_OPERATIONS, TaskRecord, _balanced_support_and_queries,
    build_family_catalog, generate_contradictory_task, generate_ood_task,
    generate_task, split_audit, stable_family_split,
)

CORPUS_SCHEMA = "nucleus-pulse-v0.43-training-corpus-v1"
DEV_SCHEMA = "nucleus-pulse-v0.43-development-manifest-v1"
FAMILY_DEV_SCHEMA = "nucleus-pulse-v0.43-family-aware-development-manifest-v2"
FAMILY_DEV_PROTOCOL = "v0.43-family-aware-dev-v1"
GENERATOR_VERSION = "v0.43-stage2-corpus-v1"
DEFAULT_SEED = 430043
DEVELOPMENT_SUBSETS = ("fit", "calibration", "evaluation")


@dataclass(frozen=True)
class TrainingEpisode:
    episode_id: str
    family_id: str
    target_ast_hash: str
    support: tuple[Example, ...]
    query_x: tuple[tuple[int, int], ...]
    query_y: tuple[int, ...]

    def canonical_row(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "family_id": self.family_id,
            "target_ast_hash": self.target_ast_hash,
            "support": [ex.to_dict() for ex in self.support],
            "query_x": [[x[0], x[1]] for x in self.query_x],
            "query_y": list(self.query_y),
        }


def _balanced_training_queries(query_x: tuple[tuple[int, int], ...],
                               query_y: tuple[int, ...],
                               rng: random.Random,
                               count: int) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    if count != 8:
        raise ValueError("training query count is frozen at 8")
    positives = [(x, y) for x, y in zip(query_x, query_y) if y == 1]
    negatives = [(x, y) for x, y in zip(query_x, query_y) if y == 0]
    if len(positives) < 4 or len(negatives) < 4:
        raise ValueError("query stream cannot provide 4 positive and 4 negative training queries")
    selected = rng.sample(positives, 4) + rng.sample(negatives, 4)
    rng.shuffle(selected)
    return tuple(x for x, _ in selected), tuple(y for _, y in selected)


def iter_training_episodes(*, n_episodes: int = 10_000, seed: int = DEFAULT_SEED,
                           queries_per_episode: int = 8) -> Iterable[TrainingEpisode]:
    if n_episodes < 1:
        raise ValueError("n_episodes must be positive")
    train_targets = [
        expr for expr in build_family_catalog()["train"]
        if 54 <= semantic_mask(expr).bit_count() <= 10_000 - 54
    ]
    if not train_targets:
        raise RuntimeError("no balanced training targets available")
    if len({family_id(expr) for expr in train_targets}) < 100:
        raise RuntimeError("fewer than 100 balanced training families are available")
    for index in range(n_episodes):
        target = train_targets[index % len(train_targets)]
        fid = family_id(target)
        if stable_family_split(fid) != "train":
            raise AssertionError(f"training target escaped train split: {fid}")
        rng = random.Random(seed + index * 7919)
        mask = semantic_mask(target)
        support, query_x_all, query_y_all = _balanced_support_and_queries(
            mask, rng, support_mode="random"
        )
        query_x, query_y = _balanced_training_queries(
            query_x_all, query_y_all, rng, queries_per_episode
        )
        episode_id = "episode-" + hashlib.sha256(
            f"{CORPUS_SCHEMA}:{seed}:{index}".encode("utf-8")
        ).hexdigest()[:16]
        yield TrainingEpisode(
            episode_id=episode_id,
            family_id=fid,
            target_ast_hash=ast_hash(target),
            support=support,
            query_x=query_x,
            query_y=query_y,
        )


def materialize_training_corpus(*, n_episodes: int = 10_000,
                                seed: int = DEFAULT_SEED) -> tuple[TrainingEpisode, ...]:
    return tuple(iter_training_episodes(n_episodes=n_episodes, seed=seed))


def corpus_sha256(episodes: Iterable[TrainingEpisode]) -> str:
    digest = hashlib.sha256()
    for episode in episodes:
        digest.update(json.dumps(episode.canonical_row(), sort_keys=True,
                                 separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def corpus_manifest_from_episodes(episodes: Iterable[TrainingEpisode], *,
                                  requested_episodes: int,
                                  seed: int = DEFAULT_SEED) -> dict[str, Any]:
    episodes = tuple(episodes)
    families = sorted({e.family_id for e in episodes})
    targets = sorted({e.target_ast_hash for e in episodes})
    support_sizes = sorted({len(e.support) for e in episodes})
    query_sizes = sorted({len(e.query_x) for e in episodes})
    label_balances = sorted({sum(e.query_y) for e in episodes})
    return {
        "schema": CORPUS_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "requested_episodes": requested_episodes,
        "actual_episodes": len(episodes),
        "unique_families": len(families),
        "unique_targets": len(targets),
        "support_size_values": support_sizes,
        "training_query_size_values": query_sizes,
        "training_query_positive_counts": label_balances,
        "episode_sha256": corpus_sha256(episodes),
        "families_sha256": hashlib.sha256("\n".join(families).encode("utf-8")).hexdigest(),
        "targets_sha256": hashlib.sha256("\n".join(targets).encode("utf-8")).hexdigest(),
        "notes": [
            "The corpus is regenerated deterministically from this seed and generator version.",
            "Each episode has 8 support examples and 8 balanced training queries.",
            "No development or test family contributes to this corpus.",
        ],
    }


def build_corpus_manifest(*, n_episodes: int = 10_000,
                          seed: int = DEFAULT_SEED) -> dict[str, Any]:
    episodes = materialize_training_corpus(n_episodes=n_episodes, seed=seed)
    return corpus_manifest_from_episodes(
        episodes, requested_episodes=n_episodes, seed=seed
    )


def _targets_by_family(targets: Iterable[Expr], split: str) -> list[Expr]:
    output: list[Expr] = []
    seen_families: set[str] = set()
    for expr in targets:
        fid = family_id(expr)
        if fid in seen_families or stable_family_split(fid) != split:
            continue
        mask = semantic_mask(expr)
        if mask.bit_count() < 54 or (10_000 - mask.bit_count()) < 54:
            continue
        seen_families.add(fid)
        output.append(expr)
    return output


def build_development_tasks(*, seed: int = DEFAULT_SEED,
                            identifiable_count: int = 8,
                            ambiguous_count: int = 8,
                            contradictory_count: int = 8,
                            ood_count: int = 4) -> list[TaskRecord]:
    """Generate a small deterministic development set spanning all task classes."""
    catalog = build_family_catalog()
    dev_targets = _targets_by_family(catalog["dev"], "dev")
    if len(dev_targets) < 20:
        raise RuntimeError(f"not enough balanced development targets: {len(dev_targets)}")

    records: list[TaskRecord] = []
    used_families: set[str] = set()

    def take_targets(count: int) -> list[Expr]:
        selected: list[Expr] = []
        for expr in dev_targets:
            fid = family_id(expr)
            if fid in used_families:
                continue
            used_families.add(fid)
            selected.append(expr)
            if len(selected) == count:
                return selected
        raise RuntimeError(f"not enough unused development families for {count} tasks")

    for i, target in enumerate(take_targets(identifiable_count)):
        records.append(generate_task(
            target, split="dev", seed=seed + 10_000 + i,
            desired_class="IN_SCOPE_IDENTIFIABLE", support_mode="contrastive",
            max_attempts=80, task_index=i,
        ))
    for i, target in enumerate(take_targets(ambiguous_count)):
        records.append(generate_task(
            target, split="dev", seed=seed + 20_000 + i,
            desired_class="IN_SCOPE_AMBIGUOUS", support_mode="random",
            max_attempts=40, task_index=identifiable_count + i,
        ))
    for i, target in enumerate(take_targets(contradictory_count)):
        records.append(generate_contradictory_task(
            target, split="dev", seed=seed + 30_000 + i,
            task_index=identifiable_count + ambiguous_count + i,
        ))
    for i in range(ood_count):
        records.append(generate_ood_task(
            seed=seed + 40_000 + i,
            task_index=identifiable_count + ambiguous_count + contradictory_count + i,
            split="dev",
        ))
    audit = split_audit(records)
    return records, audit


def build_development_manifest(*, seed: int = DEFAULT_SEED,
                               identifiable_count: int = 8,
                               ambiguous_count: int = 8,
                               contradictory_count: int = 8,
                               ood_count: int = 4,
                               records: list[TaskRecord] | None = None,
                               audit: dict[str, Any] | None = None) -> dict[str, Any]:
    if records is None or audit is None:
        records, audit = build_development_tasks(
            seed=seed,
            identifiable_count=identifiable_count,
            ambiguous_count=ambiguous_count,
            contradictory_count=contradictory_count,
            ood_count=ood_count,
        )
    classes: dict[str, int] = {}
    for record in records:
        classes[record.classification_private] = classes.get(record.classification_private, 0) + 1
    task_rows = [
        {
            "task_id": record.task_id,
            "split": record.split,
            "family_id": record.family_id,
            "ast_sha256": record.ast_sha256,
            "semantic_sha256": record.semantic_sha256,
            "classification": record.classification_private,
            "support_count": len(record.support),
            "query_count": len(record.query_x),
            "query_positive_count": sum(record.query_y_private),
            "audit": {
                "consistent_program_count": record.audit_private.get("consistent_program_count"),
                "consistent_semantic_classes": record.audit_private.get("consistent_semantic_classes"),
                "candidate_disagreement": record.audit_private.get("candidate_disagreement"),
            },
        }
        for record in records
    ]
    manifest = {
        "schema": DEV_SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "task_count": len(records),
        "class_counts": dict(sorted(classes.items())),
        "family_count": len({r.family_id for r in records}),
        "split_audit": audit,
        "generator_settings": {
            "support_size": 8,
            "query_size": 100,
            "query_balance": "50/50",
            "grammar": "v0.43-core",
            "identifiable_support_mode": "contrastive",
            "ambiguous_support_mode": "random",
        },
        "tasks": task_rows,
        "notes": [
            "Development only; never used for final-test selection.",
            "Task rows contain hashes and audit counts, not target programs or query labels.",
        ],
    }
    return manifest


def _record_fingerprint(record: TaskRecord) -> tuple[str, str]:
    support = json.dumps(
        sorted(([ex.x[0], ex.x[1], ex.y] for ex in record.support)),
        separators=(",", ":"), sort_keys=True,
    )
    queries = json.dumps(
        sorted(([x[0], x[1]] for x in record.query_x)),
        separators=(",", ":"), sort_keys=True,
    )
    return (
        hashlib.sha256(support.encode("utf-8")).hexdigest(),
        hashlib.sha256(queries.encode("utf-8")).hexdigest(),
    )


def build_family_aware_development_tasks(
    *, seed: int = DEFAULT_SEED, identifiable_count: int = 15,
    ambiguous_count: int = 15, contradictory_count: int = 15,
    ood_count: int = 9,
) -> tuple[list[TaskRecord], dict[str, list[TaskRecord]], dict[str, Any]]:
    """Build the v2 development set with one independent family per task.

    Counts are balanced across three disjoint family subsets by
    :func:`assign_development_subsets`. OOD tasks use distinct hidden-operation
    families from ``OOD_OPERATIONS``.
    """
    catalog = build_family_catalog()
    dev_targets = _targets_by_family(catalog["dev"], "dev")
    required = identifiable_count + ambiguous_count + contradictory_count
    if len(dev_targets) < required:
        raise RuntimeError(
            f"not enough balanced development families: need {required}, found {len(dev_targets)}"
        )
    records: list[TaskRecord] = []
    used_families: set[str] = set()

    def take_class(count: int, desired_class: str, support_mode: str,
                   seed_offset: int, max_attempts: int) -> list[TaskRecord]:
        selected: list[TaskRecord] = []
        for target in dev_targets:
            fid = family_id(target)
            if fid in used_families or fid in EXPOSED_FAMILY_IDS:
                continue
            try:
                record = generate_task(
                    target, split="dev", seed=seed + seed_offset + len(selected),
                    desired_class=desired_class, support_mode=support_mode,
                    max_attempts=max_attempts, task_index=len(records) + len(selected),
                )
            except (ValueError, RuntimeError):
                continue
            if record.classification_private != desired_class:
                continue
            used_families.add(fid)
            selected.append(record)
            if len(selected) == count:
                return selected
        raise RuntimeError(
            f"could only generate {len(selected)} of {count} {desired_class} tasks "
            f"from distinct development families"
        )

    identifiable = take_class(
        identifiable_count, "IN_SCOPE_IDENTIFIABLE", "contrastive", 10_000, 80
    )
    records.extend(identifiable)
    ambiguous = take_class(
        ambiguous_count, "IN_SCOPE_AMBIGUOUS", "random", 20_000, 80
    )
    records.extend(ambiguous)
    contradictory: list[TaskRecord] = []
    for target in dev_targets:
        if len(contradictory) == contradictory_count:
            break
        fid = family_id(target)
        if fid in used_families or fid in EXPOSED_FAMILY_IDS:
            continue
        try:
            record = generate_contradictory_task(
                target, split="dev", seed=seed + 30_000 + len(contradictory),
                task_index=len(records) + len(contradictory),
            )
        except (ValueError, RuntimeError):
            continue
        used_families.add(fid)
        contradictory.append(record)
    if len(contradictory) != contradictory_count:
        raise RuntimeError(
            f"could only generate {len(contradictory)} of {contradictory_count} contradictory tasks"
        )
    records.extend(contradictory)

    ood: list[TaskRecord] = []
    for operation in OOD_OPERATIONS:
        if len(ood) == ood_count:
            break
        if operation.family_id in used_families:
            continue
        try:
            record = generate_ood_task(
                seed=seed + 40_000 + len(ood), split="dev",
                task_index=len(records) + len(ood), operation=operation,
            )
        except RuntimeError:
            continue
        used_families.add(operation.family_id)
        ood.append(record)
    if len(ood) != ood_count:
        raise RuntimeError(
            f"could only generate {len(ood)} of {ood_count} distinct OOD operation families"
        )
    records.extend(ood)
    subsets = assign_development_subsets(records, seed=seed)
    audit = family_aware_split_audit(records, subsets)
    return records, subsets, audit


def assign_development_subsets(records: Iterable[TaskRecord], *,
                               seed: int = DEFAULT_SEED) -> dict[str, list[TaskRecord]]:
    """Assign whole families to fit/calibration/evaluation, stratified by class."""
    rows = tuple(records)
    by_class: dict[str, list[TaskRecord]] = {}
    for record in rows:
        by_class.setdefault(record.classification_private, []).append(record)
    subsets: dict[str, list[TaskRecord]] = {name: [] for name in DEVELOPMENT_SUBSETS}
    rng = random.Random(seed + 700_000)
    for class_name in sorted(by_class):
        class_rows = sorted(by_class[class_name], key=lambda record: record.family_id)
        rng.shuffle(class_rows)
        for index, record in enumerate(class_rows):
            subsets[DEVELOPMENT_SUBSETS[index % len(DEVELOPMENT_SUBSETS)]].append(record)
    for name in DEVELOPMENT_SUBSETS:
        subsets[name].sort(key=lambda record: record.task_id)
    return subsets


def family_aware_split_audit(
    records: Iterable[TaskRecord], subsets: dict[str, list[TaskRecord]]
) -> dict[str, Any]:
    """Audit family, AST, semantic, task, and row separation across subsets."""
    rows = tuple(records)
    all_ids = [record.task_id for record in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("duplicate task id across development subsets")
    family_sets: dict[str, set[str]] = {}
    ast_sets: dict[str, set[str]] = {}
    semantic_sets: dict[str, set[str]] = {}
    support_sets: dict[str, set[str]] = {}
    query_sets: dict[str, set[str]] = {}
    ood_sets: dict[str, set[str]] = {}
    class_counts: dict[str, dict[str, int]] = {}
    for name, subset in subsets.items():
        family_sets[name] = {record.family_id for record in subset}
        ast_sets[name] = {record.ast_sha256 for record in subset}
        semantic_sets[name] = {record.semantic_sha256 for record in subset}
        support_sets[name] = set()
        query_sets[name] = set()
        ood_sets[name] = set()
        counts: dict[str, int] = {}
        for record in subset:
            counts[record.classification_private] = counts.get(record.classification_private, 0) + 1
            support_hash, query_hash = _record_fingerprint(record)
            if support_hash in support_sets[name] or query_hash in query_sets[name]:
                raise ValueError(f"duplicate support/query fingerprint in {name}: {record.task_id}")
            if {ex.x for ex in record.support} & set(record.query_x):
                raise ValueError(f"support/query row overlap within task: {record.task_id}")
            support_sets[name].add(support_hash)
            query_sets[name].add(query_hash)
            if record.classification_private == "OUT_OF_SCOPE":
                ood_sets[name].add(record.family_id)
        class_counts[name] = dict(sorted(counts.items()))
    names = tuple(subsets)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1:]:
            for label, sets in (
                ("family", family_sets), ("AST", ast_sets),
                ("semantic function", semantic_sets),
                ("support fingerprint", support_sets),
                ("query fingerprint", query_sets),
                ("OOD operation family", ood_sets),
            ):
                overlap = sets[left] & sets[right]
                if overlap:
                    raise ValueError(
                        f"{label} leakage between development subsets {left} and {right}: "
                        f"{sorted(overlap)[:3]}"
                    )
    return {
        "protocol_version": FAMILY_DEV_PROTOCOL,
        "task_count": len(rows),
        "family_count": len({record.family_id for record in rows}),
        "subsets": list(names),
        "task_counts": {name: len(subset) for name, subset in subsets.items()},
        "family_counts": {name: len(family_sets[name]) for name in names},
        "class_counts": class_counts,
        "ood_family_ids": {name: sorted(ood_sets[name]) for name in names},
        "family_overlap": False,
        "ast_overlap": False,
        "semantic_function_overlap": False,
        "task_id_overlap": False,
        "support_fingerprint_overlap": False,
        "query_fingerprint_overlap": False,
        "ood_operation_family_overlap": False,
        "support_query_overlap_within_task": False,
    }


def build_family_aware_development_manifest(
    *, seed: int = DEFAULT_SEED, identifiable_count: int = 15,
    ambiguous_count: int = 15, contradictory_count: int = 15,
    ood_count: int = 9, records: list[TaskRecord] | None = None,
    subsets: dict[str, list[TaskRecord]] | None = None,
    audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if records is None or subsets is None or audit is None:
        records, subsets, audit = build_family_aware_development_tasks(
            seed=seed,
            identifiable_count=identifiable_count,
            ambiguous_count=ambiguous_count,
            contradictory_count=contradictory_count,
            ood_count=ood_count,
        )
    operation_by_family = {operation.family_id: operation for operation in OOD_OPERATIONS}
    subset_of = {
        record.task_id: name for name, subset in subsets.items() for record in subset
    }
    classes: dict[str, int] = {}
    for record in records:
        classes[record.classification_private] = classes.get(record.classification_private, 0) + 1
    task_rows = [
        {
            "task_id": record.task_id,
            "subset": subset_of[record.task_id],
            "split": record.split,
            "family_id": record.family_id,
            "ast_sha256": record.ast_sha256,
            "semantic_sha256": record.semantic_sha256,
            "classification": record.classification_private,
            "support_count": len(record.support),
            "query_count": len(record.query_x),
            "query_positive_count": sum(record.query_y_private),
            "ood_operation_id": record.audit_private.get("ood_operation_id"),
            "audit": {
                "consistent_program_count": record.audit_private.get("consistent_program_count"),
                "consistent_semantic_classes": record.audit_private.get("consistent_semantic_classes"),
                "candidate_disagreement": record.audit_private.get("candidate_disagreement"),
                "known_grammar_consistent": record.audit_private.get("known_grammar_consistent"),
            },
        }
        for record in records
    ]
    manifest = {
        "schema": FAMILY_DEV_SCHEMA,
        "protocol_version": FAMILY_DEV_PROTOCOL,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "task_count": len(records),
        "class_counts": dict(sorted(classes.items())),
        "family_count": len({record.family_id for record in records}),
        "subset_counts": {name: len(subset) for name, subset in subsets.items()},
        "subset_class_counts": audit["class_counts"],
        "split_audit": audit,
        "ood_operations": [
            {
                "operation_id": operation_by_family[family].operation_id,
                "family_id": operation_by_family[family].family_id,
                "descriptor": operation_by_family[family].descriptor,
            }
            for family in sorted({
                record.family_id for record in records
                if record.classification_private == "OUT_OF_SCOPE"
            })
            if family in operation_by_family
        ],
        "generator_settings": {
            "support_size": 8,
            "query_size": 100,
            "query_balance": "50/50",
            "grammar": "v0.43-core",
            "identifiable_support_mode": "contrastive",
            "ambiguous_support_mode": "random",
            "split_unit": "family",
            "subsets": list(DEVELOPMENT_SUBSETS),
        },
        "tasks": task_rows,
        "notes": [
            "Development only; never used for final-test selection.",
            "Whole families are disjoint across fit, calibration, and evaluation.",
            "OOD tasks use distinct hidden-operation families outside v0.43-core.",
            "Task rows contain hashes and audit counts, not target programs or query labels.",
        ],
    }
    return manifest


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
