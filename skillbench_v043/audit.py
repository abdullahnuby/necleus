"""Preflight audits for benchmark split and label leakage."""
from __future__ import annotations

from typing import Iterable
from .dsl import Expr, ast_hash, family_id, semantic_mask
from .generator import TaskRecord, split_audit, _semantic_hash


def audit_target_splits(targets: dict[str, Iterable[Expr]]) -> dict:
    normalized = {split: tuple(exprs) for split, exprs in targets.items()}
    required = {"train", "dev", "test"}
    if set(normalized) != required:
        raise ValueError(f"splits must be exactly {sorted(required)}")
    hashes = {k: {ast_hash(x) for x in xs} for k, xs in normalized.items()}
    families = {k: {family_id(x) for x in xs} for k, xs in normalized.items()}
    semantics = {k: {_semantic_hash(semantic_mask(x)) for x in xs} for k, xs in normalized.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if hashes[a] & hashes[b]:
            raise ValueError(f"AST hash overlap: {a}/{b}")
        if families[a] & families[b]:
            raise ValueError(f"family overlap: {a}/{b}")
        if semantics[a] & semantics[b]:
            raise ValueError(f"semantic function overlap: {a}/{b}")
    return {
        "ast_hash_overlap": False,
        "family_overlap": False,
        "semantic_function_overlap": False,
        "target_count_by_split": {k: len(v) for k, v in normalized.items()},
        "family_count_by_split": {k: len(families[k]) for k in normalized},
    }


def audit_task_records(records: Iterable[TaskRecord]) -> dict:
    return split_audit(records)
