from __future__ import annotations

import random

import pytest

from skillbench_v043.continual import (
    BalancedReplayBuffer,
    build_retention_sequences,
    build_retention_skills,
    recompute_retention_metrics,
    run_retention_comparison,
)
from skillbench_v043.generator import Example


def test_recompute_retention_metrics_from_saved_accuracy() -> None:
    round_accuracy = {
        "memory": [[0.2, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.7]],
        "parameter": [[0.2, 0.4, 0.5, 0.6], [0.3, 0.4, 0.5, 0.7]],
    }
    retention_matrix = {
        "memory": [[0.2], [0.2, 0.3]],
        "parameter": [[0.5], [0.4, 0.6]],
    }
    recomputed = recompute_retention_metrics(round_accuracy, retention_matrix)
    assert recomputed["improvement_round0_to_round3"]["memory"]["mean"] == pytest.approx(0.4)
    assert recomputed["improvement_round0_to_round3"]["parameter"]["mean"] == pytest.approx(0.4)
    assert recomputed["forgetting"]["memory"]["average_forgetting"] == pytest.approx(0.0)
    assert recomputed["forgetting"]["parameter"]["average_forgetting"] == pytest.approx(0.05)
    assert recomputed["forgetting"]["parameter"]["worst_task_forgetting"] == pytest.approx(0.1)
    assert recomputed["forgetting"]["parameter"]["per_skill"] == pytest.approx([0.1, 0.0])


def test_balanced_replay_samples_every_seen_skill() -> None:
    buffer = BalancedReplayBuffer(capacity_per_skill=2)
    for skill in range(3):
        buffer.add(skill, Example((skill, skill), skill % 2))
    sampled = buffer.sample(3, random.Random(7), (0, 1, 2))
    sources = set()
    for example in sampled:
        for skill in range(3):
            if any(example is item for item in buffer.buckets[skill]):
                sources.add(skill)
    assert sources == {0, 1, 2}


def test_retention_eval_is_disjoint_from_support_and_interactions() -> None:
    skills = build_retention_skills(n_skills=1, seed=430043)
    skill = skills[0]
    interaction = {
        point for round_points in skill.interaction_rounds for point in round_points
    }
    support = {example.x for example in skill.support}
    assert len(skill.eval_x) == 100
    assert len(set(skill.eval_x)) == 100
    assert not (set(skill.eval_x) & support)
    assert not (set(skill.eval_x) & interaction)


def test_retention_sequences_use_disjoint_family_pools() -> None:
    sequences, audit = build_retention_sequences(
        n_skills=2, seeds=(430043, 430044, 430045)
    )
    assert len(sequences) == 3
    assert audit["independent_family_sequences"] is True
    assert all(value == 0 for value in audit["pairwise_family_overlap"].values())
    assert audit["exposed_family_ids_used"] == []
    assert audit["fallback_used"] == []


def test_retention_comparison_small_is_machine_readable(tmp_path) -> None:
    result = run_retention_comparison(
        n_skills=2,
        seeds=(430043,),
        strategies=("control", "balanced_replay", "balanced_support_replay"),
        memory_mode="knn",
        output_dir=tmp_path,
    )
    assert (tmp_path / "retention_comparison.json").exists()
    assert result["official_test_opened"] is False
    assert result["decision"]["memory_track_is_separate_reference"] is True
    assert len(result["runs"]) == 3
    for strategy in ("control", "balanced_replay", "balanced_support_replay"):
        assert strategy in result["aggregate"]
        assert result["aggregate"][strategy]["seed_count"] == 1
