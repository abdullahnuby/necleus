"""Path B: learn a held-out operation from a declared meta-grammar.

The runtime library contains v0.43-core atoms. The held-out operation is a
modulo-residue predicate on x0 composed with a known predicate on x1. The learner
does not receive the hidden recipe or a name-to-semantics lookup table. It
searches a documented meta-grammar over modulo predicates and known atoms using
only labelled support examples.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import random
from typing import Iterable, Sequence

import numpy as np

from .baselines import Predictions, predict_program_induction_task_level, summarize
from .dsl import DOMAIN_SIZE, atom, mask_predict, semantic_mask
from .generator import Example, TaskRecord, _semantic_hash

HIDDEN_K = 7
HIDDEN_RESIDUES = (0, 1, 2)
KNOWN_K = 5
BUDGETS = (8, 16, 32)


@lru_cache(maxsize=1)
def hidden_mask() -> int:
    mask = 0
    for x0 in range(DOMAIN_SIZE):
        for x1 in range(DOMAIN_SIZE):
            hidden = x0 % HIDDEN_K in HIDDEN_RESIDUES
            known = x1 % KNOWN_K == 0
            if hidden != known:
                mask |= 1 << (x0 * DOMAIN_SIZE + x1)
    return mask


def _point_label(mask: int, x: tuple[int, int]) -> int:
    return (mask >> (x[0] * DOMAIN_SIZE + x[1])) & 1


def generate_new_operation_split(*, seed: int, n_support: int,
                                 n_eval: int = 100) -> tuple[tuple[Example, ...], tuple[tuple[int, int], ...], tuple[int, ...]]:
    if n_support not in BUDGETS:
        raise ValueError(f"support budget must be one of {BUDGETS}")
    if n_eval != 100:
        raise ValueError("evaluation query count is frozen at 100")
    rng = random.Random(seed)
    mask = hidden_mask()
    positives = [(x0, x1) for x0 in range(DOMAIN_SIZE) for x1 in range(DOMAIN_SIZE)
                 if _point_label(mask, (x0, x1)) == 1]
    negatives = [(x0, x1) for x0 in range(DOMAIN_SIZE) for x1 in range(DOMAIN_SIZE)
                 if _point_label(mask, (x0, x1)) == 0]
    half = n_support // 2
    support_points = rng.sample(positives, half) + rng.sample(negatives, half)
    rng.shuffle(support_points)
    excluded = set(support_points)
    pos_eval = [x for x in positives if x not in excluded]
    neg_eval = [x for x in negatives if x not in excluded]
    eval_points = rng.sample(pos_eval, 50) + rng.sample(neg_eval, 50)
    rng.shuffle(eval_points)
    support = tuple(Example(x, _point_label(mask, x)) for x in support_points)
    eval_x = tuple(eval_points)
    eval_y = tuple(_point_label(mask, x) for x in eval_points)
    return support, eval_x, eval_y


@lru_cache(maxsize=1)
def _hidden_candidate_masks() -> tuple[tuple[str, int], ...]:
    candidates: list[tuple[str, int]] = []
    for k in (5, 7, 9, 11):
        for residues in combinations(range(k), 3):
            mask = 0
            for x0 in range(DOMAIN_SIZE):
                if x0 % k in residues:
                    mask |= ((1 << DOMAIN_SIZE) - 1) << (x0 * DOMAIN_SIZE)
            candidates.append((f"mod{k}_in_{residues}", mask))
    return tuple(candidates)


@lru_cache(maxsize=1)
def _known_candidate_masks() -> tuple[tuple[str, int], ...]:
    atoms = [
        ("lt_x1_25", atom("lt", 1, 25)),
        ("lt_x1_50", atom("lt", 1, 50)),
        ("lt_x1_75", atom("lt", 1, 75)),
        ("even_x1", atom("even", 1)),
        ("div3_x1", atom("div3", 1)),
        ("div5_x1", atom("div5", 1)),
    ]
    candidates: list[tuple[str, int]] = []
    for name, expr in atoms:
        mask = semantic_mask(expr)
        candidates.append((name, mask))
        candidates.append((f"not_{name}", mask ^ ((1 << (DOMAIN_SIZE * DOMAIN_SIZE)) - 1)))
    return tuple(candidates)


def search_meta_grammar(support: Iterable[Example]) -> list[tuple[str, int]]:
    labels = [(ex.x, ex.y) for ex in support]
    consistent: list[tuple[str, int]] = []
    seen: set[int] = set()
    for hidden_name, hidden in _hidden_candidate_masks():
        for known_name, known in _known_candidate_masks():
            for op_name, mask in (
                ("hidden", hidden),
                ("and", hidden & known),
                ("or", hidden | known),
                ("xor", hidden ^ known),
            ):
                if mask in seen:
                    continue
                if all(mask_predict(mask, x) == y for x, y in labels):
                    seen.add(mask)
                    consistent.append((f"{hidden_name} {op_name} {known_name}", mask))
    return consistent


def predict_candidates(candidates: Sequence[tuple[str, int]],
                       query_x: Sequence[tuple[int, int]]) -> Predictions:
    if not candidates:
        n = len(query_x)
        return Predictions(np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool),
                           np.zeros(n, dtype=np.float64))
    predicted: list[int] = []
    accepted: list[bool] = []
    for x in query_x:
        votes = [mask_predict(mask, x) for _, mask in candidates]
        counts = np.bincount(votes, minlength=2)
        predicted.append(int(np.argmax(counts)))
        accepted.append(len(set(votes)) == 1)
    return Predictions(
        predicted=np.asarray(predicted, dtype=np.int64),
        accepted=np.asarray(accepted, dtype=bool),
        confidence=np.asarray(accepted, dtype=np.float64),
    )


def _task_from_support(support: Sequence[Example], eval_x: Sequence[tuple[int, int]],
                       eval_y: Sequence[int], seed: int) -> TaskRecord:
    return TaskRecord(
        task_id=f"task-newop-{seed}", split="dev",
        family_id="new-operation-mod-meta-family-v1",
        ast_sha256="0" * 64,
        semantic_sha256=_semantic_hash(hidden_mask()),
        support=tuple(support), query_x=tuple(eval_x),
        query_y_private=tuple(eval_y),
        target_program_private=("HIDDEN_MOD7_IN_0_1_2_XOR_DIV5",),
        classification_private="OUT_OF_SCOPE",
        audit_private={"meta_grammar": "mod k in residues composed with known atoms"},
    )


def run_new_operation_learning(*, seeds: Sequence[int] = (430043, 430044, 430045),
                               budgets: Sequence[int] = BUDGETS,
                               n_eval: int = 100) -> dict:
    rows: list[dict] = []
    for seed in seeds:
        for budget in budgets:
            support, eval_x, eval_y = generate_new_operation_split(
                seed=seed + budget * 1000, n_support=budget, n_eval=n_eval
            )
            task = _task_from_support(support, eval_x, eval_y, seed + budget)
            pre = predict_program_induction_task_level(task)
            pre_summary = summarize([task], [pre])["overall"]
            candidates = search_meta_grammar(support)
            post = predict_candidates(candidates, eval_x)
            post_summary = summarize([task], [post])["overall"]
            rows.append({
                "seed": int(seed),
                "budget": int(budget),
                "pre_instruction": pre_summary,
                "post_instruction": post_summary,
                "consistent_candidates": len(candidates),
                "candidate_semantic_classes": len({mask for _, mask in candidates}),
                "exact_hidden_recovered": any(
                    mask == hidden_mask() for _, mask in candidates
                ),
            })
    by_budget: dict[str, dict] = {}
    for budget in budgets:
        selected = [row for row in rows if row["budget"] == budget]
        by_budget[str(budget)] = {
            "pre_abstention_mean": float(np.mean([r["pre_instruction"]["abstention"] for r in selected])),
            "post_accuracy_mean": float(np.mean([r["post_instruction"]["accuracy"] for r in selected])),
            "post_coverage_mean": float(np.mean([r["post_instruction"]["coverage"] for r in selected])),
            "post_selective_risk_mean": float(np.mean([
                r["post_instruction"]["selective_risk"]
                for r in selected if r["post_instruction"]["selective_risk"] is not None
            ])) if any(r["post_instruction"]["selective_risk"] is not None for r in selected) else None,
            "exact_hidden_recovered_rate": float(np.mean([r["exact_hidden_recovered"] for r in selected])),
            "candidate_count_mean": float(np.mean([r["consistent_candidates"] for r in selected])),
        }
    return {
        "schema": "nucleus-pulse-v0.43-new-operation-v1",
        "official_test_opened": False,
        "hidden_operation": "x0 mod 7 in {0,1,2} XOR x1 mod 5 == 0",
        "meta_grammar": {
            "moduli": [5, 7, 9, 11],
            "residue_set_size": 3,
            "known_atoms": ["LT(x1,t)", "EVEN(x1)", "DIV3(x1)", "DIV5(x1)", "and negations"],
            "compositions": ["hidden", "and", "or", "xor"],
        },
        "budgets": list(budgets),
        "seeds": list(seeds),
        "rows": rows,
        "by_budget": by_budget,
        "notes": [
            "The learner receives only labelled support examples and unlabelled evaluation inputs.",
            "This is learning inside a declared modulo meta-grammar, not arbitrary operation discovery.",
            "Pre-instruction abstention uses the known v0.43-core grammar only.",
        ],
    }
