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
from .generator import (
    Example, TaskRecord, _balanced_support_and_queries, build_family_catalog,
    generate_contradictory_task, generate_ood_task, generate_task, split_audit,
    stable_family_split,
)

CORPUS_SCHEMA = "nucleus-pulse-v0.43-training-corpus-v1"
DEV_SCHEMA = "nucleus-pulse-v0.43-development-manifest-v1"
GENERATOR_VERSION = "v0.43-stage2-corpus-v1"
DEFAULT_SEED = 430043


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


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
