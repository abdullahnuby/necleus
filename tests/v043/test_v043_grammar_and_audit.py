from __future__ import annotations

from skillbench_v043.dsl import (
    ast_depth, atom, binop, canonical_semantic_programs, enumerate_programs,
    family_id, family_structure, semantic_mask,
)
from skillbench_v043.generator import (
    Example, audit_support, count_family_splits, generate_ood_task, generate_task, generate_contradictory_task,
    split_audit, stable_family_split,
)
from skillbench_v043.audit import audit_target_splits
from skillbench_v043.integrity import EXPOSED_FAMILY_IDS


def test_bounded_grammar_is_fully_enumerated_and_depth_limited():
    programs = enumerate_programs()
    assert len(programs) == 9348
    assert len({repr(p) for p in programs}) == len(programs)
    assert max(ast_depth(p) for p in programs) == 3
    assert all(ast_depth(p) <= 3 for p in programs)


def test_canonical_semantic_representatives_are_unique():
    canonical = canonical_semantic_programs()
    masks = [semantic_mask(p) for p in canonical]
    assert len(canonical) == 4570
    assert len(masks) == len(set(masks))


def test_constants_and_global_variable_renaming_do_not_create_new_families():
    a = atom("lt", 0, 25)
    b = atom("lt", 0, 75)
    assert family_id(a) == family_id(b)
    p1 = binop("xor", atom("div3", 0), atom("even", 1))
    p2 = binop("xor", atom("div3", 1), atom("even", 0))
    assert family_structure(p1) == family_structure(p2)


def test_family_split_counts_exceed_preregistered_minimums():
    counts = count_family_splits()
    assert counts["train"] >= 100
    assert counts["dev"] >= 20
    assert counts["test"] >= 30


def test_audit_separates_identifiable_and_ambiguous_support():
    target_identifiable = binop(
        "and",
        atom("even", 0),
        binop("xor", atom("div3", 1), atom("lt", 0, 25)),
    )
    split = stable_family_split(family_id(target_identifiable))
    identifiable = generate_task(
        target_identifiable,
        split=split,
        seed=45,
        desired_class="IN_SCOPE_IDENTIFIABLE",
        support_mode="contrastive",
        max_attempts=20,
    )
    assert identifiable.audit_private["consistent_semantic_classes"] == 1
    assert identifiable.classification_private == "IN_SCOPE_IDENTIFIABLE"
    assert len(identifiable.support) == 8
    assert len(identifiable.query_x) == 100
    assert sum(identifiable.query_y_private) == 50
    assert not ({ex.x for ex in identifiable.support} & set(identifiable.query_x))

    target_ambiguous = atom("lt", 0, 50)
    split_ambiguous = stable_family_split(family_id(target_ambiguous))
    ambiguous = generate_task(
        target_ambiguous,
        split=split_ambiguous,
        seed=43,
        desired_class="IN_SCOPE_AMBIGUOUS",
        support_mode="random",
        max_attempts=10,
    )
    assert ambiguous.audit_private["consistent_semantic_classes"] > 1
    assert ambiguous.audit_private["candidate_disagreement"] >= 0.20


def test_ood_and_contradictory_support_are_separate_from_coverage_gate():
    ood = generate_ood_task(seed=910)
    assert ood.classification_private == "OUT_OF_SCOPE"
    assert ood.audit_private["classification"] == "OUT_OF_SCOPE"
    contradiction = audit_support(
        [Example((1, 2), 0), Example((1, 2), 1)],
        [(3, 4), (4, 5)],
    )
    assert contradiction["classification"] == "CONTRADICTORY_SUPPORT"
    assert contradiction["consistent_program_count"] == 0
    target = atom("lt", 0, 25)
    contradictory_task = generate_contradictory_task(
        target, split=stable_family_split(family_id(target)), seed=912
    )
    assert contradictory_task.classification_private == "CONTRADICTORY_SUPPORT"
    assert len(contradictory_task.support) == 8


def test_public_payload_has_no_target_or_audit_leakage():
    task = generate_ood_task(seed=911)
    public = task.public_payload(include_query_labels=False)
    assert set(public) == {"task_id", "support", "queries"}
    assert "y" not in public["queries"][0]
    serialized = str(public)
    assert "HIDDEN_MOD7" not in serialized
    assert "target_program" not in serialized
    assert "family_id" not in serialized
    evaluator = task.evaluator_payload()
    assert "HIDDEN_MOD7" in str(evaluator["target_program"])


def test_split_audit_rejects_cross_split_semantic_or_family_leakage():
    # The same family/program assigned to two splits must be rejected before run.
    target = atom("lt", 0, 25)
    from skillbench_v043.audit import audit_target_splits
    try:
        audit_target_splits({"train": [target], "dev": [], "test": [target]})
    except ValueError as exc:
        assert "AST hash overlap" in str(exc) or "family overlap" in str(exc) or "semantic function overlap" in str(exc)
    else:
        raise AssertionError("split leakage should be rejected")


def test_generated_task_records_pass_split_audit():
    a = atom("lt", 0, 25)
    b = atom("lt", 1, 75)
    c = binop("xor", atom("even", 0), atom("div3", 1))
    targets = [(a, stable_family_split(family_id(a))), (b, stable_family_split(family_id(b))), (c, stable_family_split(family_id(c)))]
    # Keep one unique target per split; semantic/family duplicates in a given
    # split are not a leakage issue, while cross-split collision is.
    chosen = {}
    for expr, split in targets:
        if split == "test" and family_id(expr) in EXPOSED_FAMILY_IDS:
            continue
        chosen.setdefault(split, expr)
    if set(chosen) != {"train", "dev", "test"}:
        catalog = canonical_semantic_programs()
        for expr in catalog:
            split = stable_family_split(family_id(expr))
            if split == "test" and family_id(expr) in EXPOSED_FAMILY_IDS:
                continue
            if split not in chosen and semantic_mask(expr) not in {semantic_mask(x) for x in chosen.values()}:
                chosen[split] = expr
            if len(chosen) == 3:
                break
    records = []
    for i, split in enumerate(("train", "dev", "test")):
        expr = chosen[split]
        records.append(generate_task(expr, split=split, seed=1200 + i, task_index=i))
    report = split_audit(records)
    assert report["ast_overlap"] is False
    assert report["family_overlap"] is False
    assert report["semantic_function_overlap"] is False
