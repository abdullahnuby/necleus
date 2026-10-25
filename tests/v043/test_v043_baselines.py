from __future__ import annotations

import numpy as np

from skillbench_v043.baselines import (
    Predictions,
    deepsets_predictions,
    predict_constant,
    predict_linear,
    predict_program_induction,
    predict_program_induction_task_level,
    predict_retrieval,
    predict_tree,
    select_threshold,
    summarize,
)
from skillbench_v043.corpus import materialize_training_corpus
from skillbench_v043.deepsets import DeepSetsConfig, DeepSetsNumpy, stack_episodes
from skillbench_v043.generator import Example, TaskRecord


def _task() -> TaskRecord:
    support = (
        Example((0, 0), 0), Example((1, 1), 1), Example((2, 0), 0),
        Example((3, 1), 1), Example((4, 0), 0), Example((5, 1), 1),
        Example((6, 0), 0), Example((7, 1), 1),
    )
    query_x = tuple((i, i % 2) for i in range(100))
    query_y = tuple(i % 2 for i in range(100))
    return TaskRecord(
        task_id="task-test", split="dev", family_id="family-test",
        ast_sha256="a" * 64, semantic_sha256="b" * 64,
        support=support, query_x=query_x, query_y_private=query_y,
        target_program_private=("even", 0),
        classification_private="IN_SCOPE_IDENTIFIABLE",
        audit_private={"consistent_program_count": 1,
                       "consistent_semantic_classes": 1,
                       "candidate_disagreement": 0.0},
    )


def test_deepsets_forward_shapes_and_checkpoint_roundtrip(tmp_path) -> None:
    episodes = materialize_training_corpus(n_episodes=16, seed=99)
    config = DeepSetsConfig(hidden=8, epochs=2, batch_size=4, seed=7)
    model = DeepSetsNumpy(config)
    sx, sy, qx, qy = stack_episodes(episodes)
    proba = model.predict_proba(sx, sy, qx)
    assert proba.shape == qy.shape
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)
    history = model.fit(episodes)
    assert len(history) == 2
    checkpoint = tmp_path / "model.npz"
    digest = model.save(checkpoint, metadata={"seed": 7})
    assert len(digest) == 64
    after = model.predict_proba(sx, sy, qx)
    loaded = DeepSetsNumpy.load(checkpoint, config)
    reloaded = loaded.predict_proba(sx, sy, qx)
    assert np.allclose(reloaded, after, atol=1e-9)
    assert reloaded.shape == proba.shape


def test_baselines_return_one_prediction_per_query() -> None:
    task = _task()
    for predictor in (predict_constant, predict_retrieval, predict_linear,
                      predict_tree, predict_program_induction):
        predictions = predictor(task)
        assert predictions.predicted.shape == (100,)
        assert predictions.accepted.shape == (100,)
        assert set(np.unique(predictions.predicted)) <= {0, 1}


def test_task_level_program_induction_abstains_on_ambiguous_support() -> None:
    task = _task()
    predictions = predict_program_induction_task_level(task)
    assert predictions.predicted.shape == (100,)
    assert predictions.accepted.sum() == 0
    assert predictions.confidence.sum() == 0


def _classified_task(*, task_id: str, family_id: str, classification: str,
                     query_y: tuple[int, ...]) -> TaskRecord:
    base = _task()
    return TaskRecord(
        task_id=task_id,
        split="dev",
        family_id=family_id,
        ast_sha256=("a" if classification == "IN_SCOPE_IDENTIFIABLE" else "c") * 64,
        semantic_sha256=("b" if classification == "IN_SCOPE_IDENTIFIABLE" else "d") * 64,
        support=base.support,
        query_x=base.query_x,
        query_y_private=query_y,
        target_program_private=base.target_program_private,
        classification_private=classification,
        audit_private=base.audit_private,
    )


def _threshold_fixture(*, identifiable_probability: float = 0.99,
                       identifiable_y: tuple[int, ...] | None = None):
    y = identifiable_y if identifiable_y is not None else tuple(1 for _ in range(100))
    identifiable = _classified_task(
        task_id="task-ident", family_id="family-ident",
        classification="IN_SCOPE_IDENTIFIABLE", query_y=y,
    )
    ambiguous = _classified_task(
        task_id="task-amb", family_id="family-amb",
        classification="IN_SCOPE_AMBIGUOUS", query_y=tuple(1 for _ in range(100)),
    )
    return [identifiable, ambiguous], [
        np.full(100, identifiable_probability),
        np.full(100, 0.50),
    ]


def test_metrics_and_threshold_selection_are_well_formed() -> None:
    task = _task()
    predictions = predict_retrieval(task)
    summary = summarize([task], [predictions])
    assert summary["tasks"] == 1
    assert summary["overall"]["queries"] == 100
    assert 0.0 <= summary["overall"]["accuracy"] <= 1.0
    probabilities = [np.linspace(0.5, 0.99, 100)]
    selection = select_threshold([task], probabilities, thresholds=(0.5, 0.9))
    # A single identifiable task cannot satisfy the required ambiguous-abstention
    # criterion, so this is explicitly not a successful calibration.
    assert selection["feasible"] is False
    assert selection["chosen"] is None
    assert selection["chosen_threshold"] is None
    assert selection["diagnostic_fallback"] is not None
    assert len(selection["sweep"]) == 2


def test_threshold_no_predictions_accepted_keeps_risk_undefined() -> None:
    records, probabilities = _threshold_fixture(identifiable_probability=0.50)
    selection = select_threshold(records, probabilities, thresholds=(0.95,))
    row = selection["sweep"][0]
    assert row["identifiable_accepted"] == 0
    assert row["identifiable_coverage"] == 0.0
    assert row["identifiable_selective_risk"] is None
    assert row["violations"]
    assert "identifiable_selective_risk_undefined" in row["violations"]
    assert selection["feasible"] is False
    assert selection["chosen"] is None


def test_threshold_full_coverage_with_excessive_risk_is_infeasible() -> None:
    records, probabilities = _threshold_fixture(
        identifiable_probability=0.99, identifiable_y=tuple(0 for _ in range(100))
    )
    selection = select_threshold(records, probabilities, thresholds=(0.50,))
    row = selection["sweep"][0]
    assert row["identifiable_coverage"] == 1.0
    assert row["identifiable_selective_risk"] == 1.0
    assert row["feasible"] is False
    assert selection["chosen"] is None


def test_threshold_high_abstention_with_near_zero_coverage_is_infeasible() -> None:
    records, probabilities = _threshold_fixture(identifiable_probability=0.50)
    selection = select_threshold(records, probabilities, thresholds=(0.95,))
    row = selection["sweep"][0]
    assert row["ambiguous_abstention"] == 1.0
    assert row["identifiable_coverage"] == 0.0
    assert "identifiable_coverage<0.90" in row["violations"]
    assert selection["chosen"] is None


def test_threshold_genuinely_feasible_candidate() -> None:
    records, probabilities = _threshold_fixture(identifiable_probability=0.99)
    selection = select_threshold(records, probabilities, thresholds=(0.90,))
    row = selection["sweep"][0]
    assert row["feasible"] is True
    assert row["identifiable_coverage"] == 1.0
    assert row["identifiable_selective_risk"] == 0.0
    assert row["ambiguous_abstention"] == 1.0
    assert selection["feasible"] is True
    assert selection["chosen_threshold"] == 0.90


def test_threshold_tie_break_is_deterministic_and_prefers_lower_threshold() -> None:
    records, probabilities = _threshold_fixture(identifiable_probability=0.99)
    first = select_threshold(records, probabilities, thresholds=(0.90, 0.95))
    second = select_threshold(records, probabilities, thresholds=(0.95, 0.90))
    assert first["chosen_threshold"] == 0.90
    assert second["chosen_threshold"] == 0.90
    assert first["chosen"] == second["chosen"]


def test_threshold_family_clustered_bootstrap_reports_family_units() -> None:
    records, probabilities = _threshold_fixture(identifiable_probability=0.99)
    predictions = [deepsets_predictions(p, 0.90) for p in probabilities]
    summary = summarize(records, predictions, cluster_bootstrap=50, bootstrap_seed=11)
    ident = summary["by_class"]["IN_SCOPE_IDENTIFIABLE"]
    assert ident["n_families"] == 1
    assert ident["family_clustered"]["n_families"] == 1
    assert ident["family_clustered"]["coverage_ci95"] == [1.0, 1.0]
    assert ident["accepted"] == 100
    assert len(summary["per_family"]) == 2


def test_deepsets_abstention_threshold_monotonicity() -> None:
    proba = np.asarray([0.51, 0.8, 0.99])
    low = deepsets_predictions(proba, 0.5)
    high = deepsets_predictions(proba, 0.95)
    assert low.accepted.sum() >= high.accepted.sum()
    assert high.accepted.tolist() == [False, False, True]


def test_predictions_dataclass_does_not_accept_target_program() -> None:
    prediction = Predictions(
        predicted=np.zeros(2, dtype=np.int64),
        accepted=np.ones(2, dtype=bool),
        confidence=np.ones(2),
    )
    assert not hasattr(prediction, "target_program")
