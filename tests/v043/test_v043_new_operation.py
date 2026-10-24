from __future__ import annotations

from skillbench_v043.new_operation import (
    generate_new_operation_split,
    hidden_mask,
    run_new_operation_learning,
    search_meta_grammar,
)


def test_new_operation_split_is_disjoint_and_balanced() -> None:
    support, eval_x, eval_y = generate_new_operation_split(seed=7, n_support=16)
    assert len(support) == 16
    assert len(eval_x) == 100
    assert sum(ex.y for ex in support) == 8
    assert sum(eval_y) == 50
    assert {ex.x for ex in support}.isdisjoint(set(eval_x))


def test_meta_grammar_search_recovers_hidden_operation_at_16_examples() -> None:
    support, eval_x, eval_y = generate_new_operation_split(seed=11, n_support=16)
    candidates = search_meta_grammar(support)
    assert candidates
    assert any(mask == hidden_mask() for _, mask in candidates)
    assert all(
        all(((mask >> (x[0] * 100 + x[1])) & 1) == y for x, y in
            [(ex.x, ex.y) for ex in support])
        for _, mask in candidates
    )
    assert len(eval_x) == 100


def test_new_operation_evaluation_reports_budgeted_learning() -> None:
    result = run_new_operation_learning(seeds=(430043,), budgets=(16,))
    assert result["official_test_opened"] is False
    assert result["by_budget"]["16"]["pre_abstention_mean"] == 1.0
    assert result["by_budget"]["16"]["post_coverage_mean"] == 1.0
    assert result["by_budget"]["16"]["post_accuracy_mean"] == 1.0
    assert result["by_budget"]["16"]["exact_hidden_recovered_rate"] == 1.0
