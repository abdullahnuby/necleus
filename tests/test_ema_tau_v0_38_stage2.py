import json
import numpy as np
import pytest
from experiments import tune_ema_tau_v0_38 as tune


def test_development_protocol_is_separate_from_official_seed_range():
    assert set(tune.DEV_SEEDS).isdisjoint(tune.OFFICIAL_SEEDS)
    assert tune.TAU_CANDIDATES == (2.0, 4.0, 8.0)
    assert tune.LOCAL_SLOPE == 2.0
    assert "validation balanced_accuracy at epoch 10" in tune.protocol_config()["primary_selection_metric"]
    assert tune.protocol_config()["final_test_created_or_accessed"] is False


def test_development_split_is_balanced_unique_and_disjoint():
    train, val, y_train, y_val, diag = tune.make_development_splits(tune.DEV_SEEDS[0], "sync")
    assert len(train) == tune.v35.N_TRAIN
    assert len(val) == tune.v35.N_VALIDATION
    assert diag["train_validation_overlap"] == 0
    assert diag["train_unique"] == len(train)
    assert diag["validation_unique"] == len(val)
    assert np.bincount(y_train, minlength=2).tolist() == [len(y_train)//2, len(y_train)//2]
    assert np.bincount(y_val, minlength=2).tolist() == [len(y_val)//2, len(y_val)//2]


def test_selection_rule_prefers_highest_validation_score_and_ties_smaller_tau(monkeypatch):
    # Isolate the deterministic selection rule with a tiny synthetic result matrix.
    rows = []
    for seed in tune.DEV_SEEDS:
        for task in tune.TASKS:
            for tau, score in ((2.0, 0.75), (4.0, 0.75), (8.0, 0.74)):
                for epoch in tune.CHECKPOINTS:
                    rows.append({"seed": seed, "task": task, "tau_ticks": tau, "epochs": epoch,
                                 "validation_accuracy": score, "validation_balanced_accuracy": score,
                                 "validation_class_0_recall": score, "validation_class_1_recall": score,
                                 "validation_predicted_positive_rate": 0.5})
    activity = []
    summary, paired, scores, selected = tune.summarize(rows, activity)
    assert selected["tau_ticks"] == 2.0
    assert len(paired) == len(tune.TASKS) * len(tune.CHECKPOINTS) * 2


def test_atomic_json_serializes_strict_json(tmp_path):
    path = tmp_path / "data.json"
    tune.atomic_json(path, {"ok": True, "none": None, "values": [1, 2]})
    assert json.loads(path.read_text()) == {"ok": True, "none": None, "values": [1, 2]}
