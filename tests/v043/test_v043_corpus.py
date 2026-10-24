from __future__ import annotations

import json

from skillbench_v043.corpus import (
    build_corpus_manifest,
    build_development_manifest,
    corpus_sha256,
    iter_training_episodes,
    materialize_training_corpus,
)
from skillbench_v043.generator import stable_family_split


def test_training_corpus_is_deterministic_and_train_only() -> None:
    first = materialize_training_corpus(n_episodes=64, seed=777)
    second = materialize_training_corpus(n_episodes=64, seed=777)
    assert len(first) == 64
    assert corpus_sha256(first) == corpus_sha256(second)
    for episode in first:
        assert stable_family_split(episode.family_id) == "train"
        assert len(episode.support) == 8
        assert len(episode.query_x) == 8
        assert sum(episode.query_y) == 4
        assert not ({ex.x for ex in episode.support} & set(episode.query_x))


def test_corpus_manifest_reports_actual_counts() -> None:
    manifest = build_corpus_manifest(n_episodes=64, seed=778)
    assert manifest["requested_episodes"] == 64
    assert manifest["actual_episodes"] == 64
    assert manifest["support_size_values"] == [8]
    assert manifest["training_query_size_values"] == [8]
    assert manifest["training_query_positive_counts"] == [4]
    assert len(manifest["episode_sha256"]) == 64
    assert manifest["unique_families"] >= 1
    assert manifest["unique_targets"] >= 1


def test_development_manifest_covers_all_classes_without_private_rows() -> None:
    manifest = build_development_manifest(
        seed=779, identifiable_count=2, ambiguous_count=2,
        contradictory_count=2, ood_count=1,
    )
    assert manifest["task_count"] == 7
    assert set(manifest["class_counts"]) == {
        "IN_SCOPE_IDENTIFIABLE", "IN_SCOPE_AMBIGUOUS",
        "CONTRADICTORY_SUPPORT", "OUT_OF_SCOPE",
    }
    serialized = json.dumps(manifest)
    assert "target_program" not in serialized
    assert "query_y" not in serialized
    assert all("support_count" in task for task in manifest["tasks"])
    assert all(task["support_count"] == 8 for task in manifest["tasks"])
    assert all(task["query_count"] == 100 for task in manifest["tasks"])
    assert manifest["split_audit"]["query_balance"] == "50/50 per task"


def test_training_iterator_rejects_zero_episodes() -> None:
    try:
        list(iter_training_episodes(n_episodes=0))
    except ValueError:
        pass
    else:
        raise AssertionError("zero-episode corpus must be rejected")
