"""Deterministic tasks and support-identifiability audits for v0.43 Stage 1."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Any, Callable, Iterable, Literal

from .dsl import (
    DOMAIN_SIZE, DOMAIN_MAX, Expr, ast_depth, ast_hash, canonical_semantic_programs,
    enumerate_programs, evaluate, expr_code, family_code, family_id, mask_predict,
    point_index, semantic_mask,
)
from .integrity import assert_not_disqualified

TaskClass = Literal[
    "IN_SCOPE_IDENTIFIABLE", "IN_SCOPE_AMBIGUOUS", "OUT_OF_SCOPE", "CONTRADICTORY_SUPPORT"
]


@dataclass(frozen=True)
class OODOperation:
    """A hidden operation family reserved for development-only OOD tasks."""

    operation_id: str
    family_id: str
    descriptor: str
    mask: int


def _ood_mask(rule: Callable[[int, int], bool]) -> int:
    mask = 0
    for x0 in range(DOMAIN_SIZE):
        for x1 in range(DOMAIN_SIZE):
            if rule(x0, x1):
                mask |= 1 << (x0 * DOMAIN_SIZE + x1)
    return mask


def _ood_family_id(operation_id: str) -> str:
    return "ood-hidden-" + hashlib.sha256(
        ("nucleus-pulse-v0.43-ood-family-v2:" + operation_id).encode("utf-8")
    ).hexdigest()[:16]


OOD_OPERATIONS: tuple[OODOperation, ...] = (
    OODOperation(
        "mod7_xor_div5", "ood-hidden-mod7-family-v1",
        "x0 mod 7 in {0,1,2} XOR x1 mod 5 == 0",
        _ood_mask(lambda x0, x1: (x0 % 7 in (0, 1, 2)) != (x1 % 5 == 0)),
    ),
    OODOperation(
        "mod9_or_mod4", _ood_family_id("mod9_or_mod4"),
        "x0 mod 9 in {1,4,7} OR x1 mod 4 == 0",
        _ood_mask(lambda x0, x1: (x0 % 9 in (1, 4, 7)) or (x1 % 4 == 0)),
    ),
    OODOperation(
        "mod11_xor_mod3", _ood_family_id("mod11_xor_mod3"),
        "x0 mod 11 in {0,5,10} XOR x1 mod 3 == 0",
        _ood_mask(lambda x0, x1: (x0 % 11 in (0, 5, 10)) != (x1 % 3 == 0)),
    ),
    OODOperation(
        "mod7_or_mod7b", _ood_family_id("mod7_or_mod7b"),
        "x0 mod 7 == 0 OR x1 mod 7 in {2,3}",
        _ood_mask(lambda x0, x1: (x0 % 7 == 0) or (x1 % 7 in (2, 3))),
    ),
    OODOperation(
        "mod13_and_notdiv5", _ood_family_id("mod13_and_notdiv5"),
        "x0 mod 13 in {0,1,2,3} AND x1 mod 5 != 0",
        _ood_mask(lambda x0, x1: (x0 % 13 in (0, 1, 2, 3)) and (x1 % 5 != 0)),
    ),
    OODOperation(
        "mod8_xor_mod9", _ood_family_id("mod8_xor_mod9"),
        "x0 mod 8 in {1,3,5,7} XOR x1 mod 9 in {0,4,8}",
        _ood_mask(lambda x0, x1: (x0 % 8 in (1, 3, 5, 7)) != (x1 % 9 in (0, 4, 8))),
    ),
    OODOperation(
        "sum_mod7", _ood_family_id("sum_mod7"),
        "(x0 + x1) mod 7 in {0,1,2}",
        _ood_mask(lambda x0, x1: ((x0 + x1) % 7 in (0, 1, 2))),
    ),
    OODOperation(
        "product_mod5", _ood_family_id("product_mod5"),
        "(x0 * x1) mod 5 == 0",
        _ood_mask(lambda x0, x1: ((x0 * x1) % 5 == 0)),
    ),
    OODOperation(
        "mod6_or_mod8", _ood_family_id("mod6_or_mod8"),
        "x0 mod 6 in {0,1} OR x1 mod 8 in {3,4,5}",
        _ood_mask(lambda x0, x1: (x0 % 6 in (0, 1)) or (x1 % 8 in (3, 4, 5))),
    ),
    OODOperation(
        "mod10_xor_mod7", _ood_family_id("mod10_xor_mod7"),
        "x0 mod 10 in {0,5} XOR x1 mod 7 == 0",
        _ood_mask(lambda x0, x1: (x0 % 10 in (0, 5)) != (x1 % 7 == 0)),
    ),
    OODOperation(
        "mod9_and_mod4", _ood_family_id("mod9_and_mod4"),
        "x0 mod 9 in {0,1,2,3} AND x1 mod 4 in {0,1}",
        _ood_mask(lambda x0, x1: (x0 % 9 in (0, 1, 2, 3)) and (x1 % 4 in (0, 1))),
    ),
    OODOperation(
        "mod7_swapped", _ood_family_id("mod7_swapped"),
        "x1 mod 7 in {0,1,2} XOR x0 mod 5 == 0",
        _ood_mask(lambda x0, x1: (x1 % 7 in (0, 1, 2)) != (x0 % 5 == 0)),
    ),
)


@dataclass(frozen=True)
class Example:
    x: tuple[int, int]
    y: int

    def to_dict(self) -> dict[str, Any]:
        return {"x": [self.x[0], self.x[1]], "y": self.y}


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    split: str
    family_id: str
    ast_sha256: str
    semantic_sha256: str
    support: tuple[Example, ...]
    query_x: tuple[tuple[int, int], ...]
    query_y_private: tuple[int, ...]
    target_program_private: Any
    classification_private: TaskClass
    audit_private: dict[str, Any]

    def public_payload(self, *, include_query_labels: bool = False) -> dict[str, Any]:
        """The runtime-facing task omits gold program, family, and audit metadata."""
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "support": [ex.to_dict() for ex in self.support],
            "queries": [{"x": [x[0], x[1]]} for x in self.query_x],
        }
        if include_query_labels:
            for query, y in zip(payload["queries"], self.query_y_private):
                query["y"] = int(y)
        return payload

    def evaluator_payload(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "family_id": self.family_id,
            "ast_sha256": self.ast_sha256,
            "semantic_sha256": self.semantic_sha256,
            "target_program": self.target_program_private,
            "classification": self.classification_private,
            "support": [ex.to_dict() for ex in self.support],
            "query_x": [[x[0], x[1]] for x in self.query_x],
            "query_y": list(self.query_y_private),
            "audit": self.audit_private,
        }


def _sha_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def _semantic_hash(mask: int) -> str:
    # Variable-length integer serialization is deterministic for the finite domain.
    raw = mask.to_bytes((DOMAIN_SIZE * DOMAIN_SIZE + 7) // 8, "little")
    return hashlib.sha256(raw).hexdigest()


def stable_family_split(fid: str) -> str:
    """Predeclared 70/15/15 split based on family ID, independent of labels."""
    bucket = _sha_int("nucleus-pulse-v0.43-family-split-v1:" + fid) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "dev"
    return "test"


def build_family_catalog() -> dict[str, list[Expr]]:
    """One canonical target AST per finite-domain semantic function, grouped by family."""
    catalog: dict[str, list[Expr]] = {"train": [], "dev": [], "test": []}
    seen_family_semantics: set[tuple[str, str]] = set()
    for expr in canonical_semantic_programs():
        fid = family_id(expr)
        split = stable_family_split(fid)
        sig_hash = _semantic_hash(semantic_mask(expr))
        key = (fid, sig_hash)
        if key in seen_family_semantics:
            continue
        seen_family_semantics.add(key)
        catalog[split].append(expr)
    return catalog


def count_family_splits(programs: Iterable[Expr] | None = None) -> dict[str, int]:
    source = canonical_semantic_programs() if programs is None else tuple(programs)
    families: dict[str, str] = {}
    for expr in source:
        fid = family_id(expr)
        families[fid] = stable_family_split(fid)
    out = {name: 0 for name in ("train", "dev", "test")}
    for split in families.values():
        out[split] += 1
    return out


def _unique_semantic_masks() -> tuple[int, ...]:
    # canonical_semantic_programs returns one representative per truth function.
    return tuple(semantic_mask(expr) for expr in canonical_semantic_programs())


def _contrastive_support(mask: int, rng: random.Random, used: set[int] | None = None) -> tuple[Example, ...]:
    """Greedily choose four positive/four negative examples that split hypotheses.

    This creates a deliberately informative support set for the identifiable
    stratum. The sampling policy is fixed and visible; it is not used for the
    ambiguous stratum, whose support remains random.
    """
    used_indices = set() if used is None else set(used)
    hypotheses = list(_unique_semantic_masks())
    target_pos = [i for i in range(DOMAIN_SIZE * DOMAIN_SIZE) if (mask >> i) & 1]
    target_neg = [i for i in range(DOMAIN_SIZE * DOMAIN_SIZE) if not ((mask >> i) & 1)]
    if len(target_pos) < 54 or len(target_neg) < 54:
        raise ValueError("target function is too imbalanced for the registered support/query design")
    selected: list[Example] = []
    selected_by_label = {0: 0, 1: 0}
    # Each iteration considers a seeded sample of points from each target class,
    # selecting the point that removes the most currently consistent functions.
    for _ in range(8):
        candidate_points: list[tuple[int, int]] = []
        for y, pool in ((1, target_pos), (0, target_neg)):
            if selected_by_label[y] >= 4:
                continue
            available = [idx for idx in pool if idx not in used_indices]
            if not available:
                continue
            sample = rng.sample(available, min(512, len(available)))
            candidate_points.extend((idx, y) for idx in sample)
        if not candidate_points:
            raise RuntimeError("unable to complete contrastive support")
        scored: list[tuple[int, float, int, int]] = []
        total = len(hypotheses)
        for idx, y in candidate_points:
            agreeing = sum(((hypothesis >> idx) & 1) == y for hypothesis in hypotheses)
            # Minimize the surviving version space. Tie-break towards balanced
            # partitions (more information) and then with a deterministic random key.
            balance = abs(total - 2 * agreeing)
            scored.append((agreeing, -balance, rng.randrange(1 << 30), idx * 2 + y))
        scored.sort()
        _, _, _, encoded = scored[0]
        idx, y = divmod(encoded, 2)
        x = (idx // DOMAIN_SIZE, idx % DOMAIN_SIZE)
        selected.append(Example(x, y))
        selected_by_label[y] += 1
        used_indices.add(idx)
        hypotheses = [h for h in hypotheses if ((h >> idx) & 1) == y]
        if not hypotheses:
            raise AssertionError("contrastive support eliminated the target function")
    if selected_by_label != {0: 4, 1: 4}:
        raise AssertionError(f"contrastive support is not 4/4 balanced: {selected_by_label}")
    rng.shuffle(selected)
    return tuple(selected)


def _balanced_support_and_queries(mask: int, rng: random.Random,
                                  support_size: int = 8, query_size: int = 100,
                                  support_mode: str = "random"
                                  ) -> tuple[tuple[Example, ...], tuple[tuple[int, int], ...], tuple[int, ...]]:
    if support_size != 8:
        raise ValueError("v0.43 preregistration fixes support_size=8")
    if query_size != 100:
        raise ValueError("v0.43 preregistration fixes query_size=100")
    positives: list[tuple[int, int]] = []
    negatives: list[tuple[int, int]] = []
    # Complete finite domain is small enough to inspect exactly.
    for x0 in range(DOMAIN_SIZE):
        base = x0 * DOMAIN_SIZE
        for x1 in range(DOMAIN_SIZE):
            point = (x0, x1)
            if (mask >> (base + x1)) & 1:
                positives.append(point)
            else:
                negatives.append(point)
    if len(positives) < 54 or len(negatives) < 54:
        raise ValueError("target function cannot provide 4 balanced support examples and 50 queries per class")

    if support_mode == "contrastive":
        support = _contrastive_support(mask, rng)
    elif support_mode == "random":
        support_pos = rng.sample(positives, 4)
        support_neg = rng.sample(negatives, 4)
        support_points = support_pos + support_neg
        rng.shuffle(support_points)
        support = tuple(Example(x, (mask >> point_index(x)) & 1) for x in support_points)
    else:
        raise ValueError("support_mode must be 'random' or 'contrastive'")

    excluded = {ex.x for ex in support}
    pos_pool = [x for x in positives if x not in excluded]
    neg_pool = [x for x in negatives if x not in excluded]
    query_points = rng.sample(pos_pool, 50) + rng.sample(neg_pool, 50)
    rng.shuffle(query_points)
    query_x = tuple(query_points)
    query_y = tuple((mask >> point_index(x)) & 1 for x in query_x)
    if len(set(query_x)) != 100 or set(query_x) & excluded:
        raise AssertionError("query sampling invariant failed")
    if sum(query_y) != 50:
        raise AssertionError("query labels must be balanced exactly 50/50")
    return support, query_x, query_y


def audit_support(support: Iterable[Example], query_x: Iterable[tuple[int, int]],
                  *, out_of_scope: bool = False,
                  programs: Iterable[Expr] | None = None) -> dict[str, Any]:
    """Count consistent ASTs and semantic classes without exposing the target."""
    support_rows = tuple(support)
    query_points = tuple(query_x)
    by_x: dict[tuple[int, int], set[int]] = {}
    for ex in support_rows:
        by_x.setdefault(ex.x, set()).add(ex.y)
    contradictory = any(len(labels) > 1 for labels in by_x.values())
    if contradictory:
        return {
            "classification": "CONTRADICTORY_SUPPORT",
            "consistent_program_count": 0,
            "consistent_semantic_classes": 0,
            "candidate_disagreement": 1.0,
            "support_identifiability": False,
            "reason": "same input has conflicting support labels",
        }
    if out_of_scope:
        return {
            "classification": "OUT_OF_SCOPE",
            "consistent_program_count": None,
            "consistent_semantic_classes": None,
            "candidate_disagreement": None,
            "support_identifiability": False,
            "reason": "private evaluator marks the required operation outside the runtime grammar",
        }

    hypotheses = tuple(enumerate_programs() if programs is None else programs)
    labels = tuple((ex.x, ex.y) for ex in support_rows)
    consistent: list[tuple[Expr, int]] = []
    for expr in hypotheses:
        mask = semantic_mask(expr)
        if all(mask_predict(mask, x) == y for x, y in labels):
            consistent.append((expr, mask))
    if not consistent:
        return {
            "classification": "CONTRADICTORY_SUPPORT",
            "consistent_program_count": 0,
            "consistent_semantic_classes": 0,
            "candidate_disagreement": 1.0,
            "support_identifiability": False,
            "reason": "no grammar program matches all support examples",
        }
    signatures = {mask for _, mask in consistent}
    disagreement_count = 0
    for x in query_points:
        preds = {(mask >> point_index(x)) & 1 for mask in signatures}
        if len(preds) > 1:
            disagreement_count += 1
    disagreement = disagreement_count / len(query_points) if query_points else 0.0
    if len(signatures) == 1:
        classification: TaskClass = "IN_SCOPE_IDENTIFIABLE"
    elif disagreement >= 0.20:
        classification = "IN_SCOPE_AMBIGUOUS"
    else:
        # The support is syntactically/semantically ambiguous, but its remaining
        # candidates do not disagree sufficiently on the chosen query distribution.
        classification = "IN_SCOPE_AMBIGUOUS"
    return {
        "classification": classification,
        "consistent_program_count": len(consistent),
        "consistent_semantic_classes": len(signatures),
        "candidate_disagreement": disagreement,
        "support_identifiability": len(signatures) == 1,
        "ambiguous_stratum_admissible": len(signatures) == 1 or disagreement >= 0.20,
        "reason": "exact enumeration over the preregistered bounded grammar",
    }


def generate_task(target: Expr, *, split: str, seed: int, desired_class: str | None = None,
                  max_attempts: int = 2000, task_index: int = 0, support_mode: str = "random") -> TaskRecord:
    """Generate one task with balanced queries and (optionally) desired identifiability class."""
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train/dev/test")
    if ast_depth(target) > 3:
        raise ValueError("target AST depth exceeds v0.43 core limit of 3")
    if split != stable_family_split(family_id(target)):
        raise ValueError("target family assigned to a different split by frozen split rule")
    mask = semantic_mask(target)
    target_sig = _semantic_hash(mask)
    target_ast_hash = ast_hash(target)
    target_family_id = family_id(target)
    for attempt in range(max_attempts):
        local_seed = seed + attempt * 104729
        rng = random.Random(local_seed)
        support, query_x, query_y = _balanced_support_and_queries(mask, rng, support_mode=support_mode)
        audit = audit_support(support, query_x)
        current = audit["classification"]
        if desired_class is not None and current != desired_class:
            continue
        if current == "IN_SCOPE_AMBIGUOUS" and audit["candidate_disagreement"] < 0.20:
            continue
        task_id = "task-" + hashlib.sha256(
            f"nucleus-pulse-v0.43-task-id-v1:{seed}:{task_index}:{attempt}".encode("utf-8")
        ).hexdigest()[:16]
        record = TaskRecord(
            task_id=task_id,
            split=split,
            family_id=target_family_id,
            ast_sha256=target_ast_hash,
            semantic_sha256=target_sig,
            support=support,
            query_x=query_x,
            query_y_private=query_y,
            target_program_private=target,
            classification_private=current,
            audit_private=audit,
        )
        assert_not_disqualified(
            task_id=record.task_id,
            family_id=record.family_id,
            support=[ex.to_dict() for ex in record.support],
            query_x=record.query_x,
            split=split,
        )
        return record
    raise RuntimeError(
        f"could not generate category={desired_class or 'any'} task for family={target_family_id} "
        f"after {max_attempts} deterministic attempts"
    )


def split_audit(records: Iterable[TaskRecord]) -> dict[str, Any]:
    rows = tuple(records)
    by_split: dict[str, list[TaskRecord]] = {"train": [], "dev": [], "test": []}
    ids: set[str] = set()
    for task in rows:
        if task.task_id in ids:
            raise ValueError(f"duplicate task id: {task.task_id}")
        ids.add(task.task_id)
        by_split[task.split].append(task)
        if task.split == "test":
            assert_not_disqualified(
                task_id=task.task_id,
                family_id=task.family_id,
                support=[ex.to_dict() for ex in task.support],
                query_x=task.query_x,
                split="test",
            )
        support_points = [ex.x for ex in task.support]
        if len(support_points) != len(set(support_points)) and task.classification_private != "CONTRADICTORY_SUPPORT":
            raise ValueError(f"duplicate support input in non-contradictory task {task.task_id}")
        if set(support_points) & set(task.query_x):
            raise ValueError(f"support/query overlap in task {task.task_id}")
        if len(task.query_x) != 100 or len(set(task.query_x)) != 100:
            raise ValueError(f"task {task.task_id} must have 100 unique query inputs")
        if sum(task.query_y_private) != 50:
            raise ValueError(f"task {task.task_id} query labels must be balanced 50/50")
        public = task.public_payload()
        forbidden = {"target_program", "target_program_private", "query_y_private", "classification", "audit", "family_id", "semantic_sha256", "ast_sha256"}
        if forbidden & set(public):
            raise ValueError(f"private evaluator fields leaked into public task {task.task_id}")
    family_sets = {split: {t.family_id for t in bucket} for split, bucket in by_split.items()}
    ast_sets = {split: {t.ast_sha256 for t in bucket} for split, bucket in by_split.items()}
    semantic_sets = {split: {t.semantic_sha256 for t in bucket} for split, bucket in by_split.items()}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if family_sets[left] & family_sets[right]:
            raise ValueError(f"family leakage between {left} and {right}")
        if ast_sets[left] & ast_sets[right]:
            raise ValueError(f"AST leakage between {left} and {right}")
        if semantic_sets[left] & semantic_sets[right]:
            raise ValueError(f"semantic-function leakage between {left} and {right}")
    return {
        "task_count": len(rows),
        "unique_task_ids": len(ids),
        "tasks_by_split": {k: len(v) for k, v in by_split.items()},
        "families_by_split": {k: len(v) for k, v in family_sets.items()},
        "ast_overlap": False,
        "family_overlap": False,
        "semantic_function_overlap": False,
        "support_query_overlap": False,
        "query_balance": "50/50 per task",
        "public_payload_private_fields": "absent",
    }


def write_jsonl(path: str, rows: Iterable[dict[str, Any]]) -> None:
    from pathlib import Path
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def generate_ood_task(*, seed: int, task_index: int = 0, split: str = "test",
                      operation: OODOperation | None = None) -> TaskRecord:
    """Create a hidden-operation task for the pre-teaching OUT_OF_SCOPE stratum.

    The default operation is the historical mod-7 XOR div-5 function. The
    family-aware development protocol passes distinct ``OOD_OPERATIONS`` so OOD
    tasks do not all share one family. Public payloads contain only labelled
    support examples and unlabelled queries; the hidden operation and labels
    remain evaluator-private.
    """
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train/dev/test")
    selected = operation or OOD_OPERATIONS[0]
    rng = random.Random(seed)
    mask = selected.mask
    # Reject support sets that a known v0.43-core program can fit. Otherwise a
    # closed-world auditor would answer an out-of-scope task by coincidence.
    support = query_x = query_y = None
    for attempt in range(500):
        candidate_support, candidate_query_x, candidate_query_y = _balanced_support_and_queries(
            mask, rng, support_mode="random"
        )
        labels = [(ex.x, ex.y) for ex in candidate_support]
        known_consistent = any(
            all(mask_predict(semantic_mask(expr), x) == y for x, y in labels)
            for expr in enumerate_programs()
        )
        if not known_consistent:
            support, query_x, query_y = candidate_support, candidate_query_x, candidate_query_y
            break
    if support is None or query_x is None or query_y is None:
        raise RuntimeError(
            f"unable to generate an out-of-scope support set outside the known grammar "
            f"for {selected.operation_id}"
        )
    audit = audit_support(support, query_x, out_of_scope=True)
    audit["known_grammar_consistent"] = False
    audit["ood_rejection_attempts"] = attempt + 1
    audit["ood_operation_id"] = selected.operation_id
    hidden_descriptor = (selected.descriptor, "x0", "x1")
    task_id = "task-" + hashlib.sha256(
        f"nucleus-pulse-v0.43-ood-task-id-v2:{selected.operation_id}:{seed}:{task_index}".encode("utf-8")
    ).hexdigest()[:16]
    record = TaskRecord(
        task_id=task_id,
        split=split,
        family_id=selected.family_id,
        ast_sha256=hashlib.sha256(repr(hidden_descriptor).encode("utf-8")).hexdigest(),
        semantic_sha256=_semantic_hash(mask),
        support=support,
        query_x=query_x,
        query_y_private=query_y,
        target_program_private=hidden_descriptor,
        classification_private="OUT_OF_SCOPE",
        audit_private=audit,
    )
    assert_not_disqualified(
        task_id=record.task_id,
        family_id=record.family_id,
        support=[ex.to_dict() for ex in record.support],
        query_x=record.query_x,
        split=split,
    )
    return record


def generate_contradictory_task(target: Expr, *, split: str, seed: int, task_index: int = 0) -> TaskRecord:
    """Create an 8-row support set with an intentional input/label contradiction."""
    if split not in ("train", "dev", "test"):
        raise ValueError("split must be train/dev/test")
    if split != stable_family_split(family_id(target)):
        raise ValueError("target family assigned to a different split by frozen split rule")
    rng = random.Random(seed)
    mask = semantic_mask(target)
    support, query_x, query_y = _balanced_support_and_queries(mask, rng, support_mode="random")
    mutable = list(support)
    source = mutable[0]
    # Replace another row with the same input and opposite label. Row count stays 8.
    mutable[1] = Example(source.x, 1 - source.y)
    contradictory_support = tuple(mutable)
    audit = audit_support(contradictory_support, query_x)
    if audit["classification"] != "CONTRADICTORY_SUPPORT":
        raise AssertionError("contradictory task generation failed its own audit")
    task_id = "task-" + hashlib.sha256(
        f"nucleus-pulse-v0.43-contradictory-task-id-v1:{seed}:{task_index}".encode("utf-8")
    ).hexdigest()[:16]
    record = TaskRecord(
        task_id=task_id,
        split=split,
        family_id=family_id(target),
        ast_sha256=ast_hash(target),
        semantic_sha256=_semantic_hash(mask),
        support=contradictory_support,
        query_x=query_x,
        query_y_private=query_y,
        target_program_private=target,
        classification_private="CONTRADICTORY_SUPPORT",
        audit_private=audit,
    )
    assert_not_disqualified(
        task_id=record.task_id,
        family_id=record.family_id,
        support=[ex.to_dict() for ex in record.support],
        query_x=record.query_x,
        split=split,
    )
    return record
