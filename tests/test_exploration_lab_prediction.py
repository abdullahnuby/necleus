from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exploration_lab.agent import AgentConfig, SpikingExplorer
from exploration_lab.city import CityRuntime, CityRuntimeConfig
from exploration_lab.games import FEATURE_DIM
from exploration_lab.prediction import LocalPredictor, PredictionConfig, PredictionMetrics

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TMP = ROOT / ".local_private" / "pytest_prediction"


class LocalPredictorTests(unittest.TestCase):
    def setUp(self) -> None:
        LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    def _tmpdir(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(dir=LOCAL_TMP)

    def test_prediction_shapes_clipping_and_state_roundtrip(self) -> None:
        predictor = LocalPredictor(
            PredictionConfig(hidden_dim=8, action_dim=4, input_dim=FEATURE_DIM),
            seed=3,
        )
        features = np.linspace(0.0, 1.0, FEATURE_DIM)
        hidden = np.linspace(0.0, 1.0, 8)
        prediction = predictor.predict(features, hidden, 2)
        self.assertEqual(prediction.shape, (FEATURE_DIM,))
        self.assertTrue(np.all(np.isfinite(prediction)))
        self.assertGreaterEqual(float(prediction.min()), 0.0)
        self.assertLessEqual(float(prediction.max()), 1.0)
        restored = LocalPredictor.from_state_dict(predictor.state_dict())
        self.assertTrue(np.array_equal(predictor.weights, restored.weights))
        self.assertTrue(np.array_equal(predictor.bias, restored.bias))
        self.assertEqual(restored.config, predictor.config)

    def test_prediction_metrics_have_no_reward_fields(self) -> None:
        self.assertEqual(
            set(PredictionMetrics.__dataclass_fields__),
            {"mae", "mse", "prediction_updates"},
        )
        self.assertEqual(
            set(PredictionConfig.__dataclass_fields__),
            {
                "hidden_dim",
                "action_dim",
                "input_dim",
                "learning_rate",
                "weight_max",
                "weight_decay",
                "init_scale",
            },
        )

    def test_hidden_activity_is_saved_and_restored(self) -> None:
        agent = SpikingExplorer(AgentConfig(hidden_dim=16), seed=11)
        agent.act([0.5] * FEATURE_DIM, (0, 1, 2, 3))
        restored = SpikingExplorer.from_state_dict(agent.state_dict())
        self.assertTrue(np.array_equal(agent.last_hidden_activity, restored.last_hidden_activity))
        self.assertEqual(restored.last_hidden_activity.shape, (16,))
        self.assertTrue(np.all(restored.last_hidden_activity >= 0.0))
        self.assertTrue(np.all(restored.last_hidden_activity <= 1.0))

    def test_local_delta_rule_reduces_error_on_constant_target(self) -> None:
        predictor = LocalPredictor(
            PredictionConfig(
                hidden_dim=8,
                action_dim=4,
                input_dim=FEATURE_DIM,
                learning_rate=0.05,
                weight_max=2.0,
                weight_decay=0.0,
            ),
            seed=4,
        )
        features = np.ones(FEATURE_DIM)
        hidden = np.zeros(8)
        target = np.full(FEATURE_DIM, 0.7)
        errors: list[float] = []
        initial_weights = predictor.weights.copy()
        for _ in range(200):
            predictor.predict(features, hidden, 0)
            metrics = predictor.observe_next(target)
            errors.append(metrics.mae)
        self.assertLess(float(np.mean(errors[-20:])), float(np.mean(errors[:20])))
        self.assertLess(errors[-1], errors[0])
        self.assertFalse(np.array_equal(initial_weights, predictor.weights))
        self.assertGreater(predictor.total_updates, 0)
        self.assertTrue(np.all(np.isfinite(predictor.weights)))

    def test_zero_learning_rate_keeps_predictor_fixed(self) -> None:
        predictor = LocalPredictor(
            PredictionConfig(hidden_dim=8, action_dim=4, input_dim=FEATURE_DIM, learning_rate=0.0),
            seed=5,
        )
        weights = predictor.weights.copy()
        bias = predictor.bias.copy()
        predictor.predict(np.zeros(FEATURE_DIM), np.zeros(8), 1)
        metrics = predictor.observe_next(np.ones(FEATURE_DIM))
        self.assertGreater(metrics.mae, 0.0)
        self.assertTrue(np.array_equal(weights, predictor.weights))
        self.assertTrue(np.array_equal(bias, predictor.bias))
        self.assertEqual(predictor.total_updates, 0)
        self.assertEqual(predictor.steps, 1)

    def test_prediction_update_does_not_touch_agent_weights(self) -> None:
        agent = SpikingExplorer(AgentConfig(), seed=7)
        agent.act([0.5] * FEATURE_DIM, (0, 1, 2, 3))
        input_weights = agent.w_input_hidden.copy()
        hidden_weights = agent.w_hidden_hidden.copy()
        action_weights = agent.w_hidden_action.copy()
        predictor = LocalPredictor(
            PredictionConfig(hidden_dim=agent.config.hidden_dim, action_dim=agent.config.action_dim),
            seed=0,
        )
        predictor.predict([0.5] * FEATURE_DIM, agent.last_hidden_activity, 0)
        predictor.observe_next([0.25] * FEATURE_DIM)
        self.assertTrue(np.array_equal(input_weights, agent.w_input_hidden))
        self.assertTrue(np.array_equal(hidden_weights, agent.w_hidden_hidden))
        self.assertTrue(np.array_equal(action_weights, agent.w_hidden_action))
        self.assertEqual(agent.last_hidden_activity.shape, (agent.config.hidden_dim,))
        self.assertTrue(np.all(agent.last_hidden_activity >= 0.0))
        self.assertTrue(np.all(agent.last_hidden_activity <= 1.0))


class CityPredictionIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    def _tmpdir(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(dir=LOCAL_TMP)

    def _runtime(
        self,
        root: Path,
        *,
        seed: int,
        steps: int,
        prediction_learning_rate: float,
        checkpoint: str,
        log: str,
        summary: str,
        resume: Path | None = None,
    ) -> CityRuntime:
        return CityRuntime(
            CityRuntimeConfig(
                seed=seed,
                width=16,
                height=16,
                agent_count=3,
                max_steps=steps,
                delay=0.0,
                checkpoint_every=0,
                render=False,
                progress_every=0,
                prediction_learning_rate=prediction_learning_rate,
                prediction_window=3,
            ),
            checkpoint_path=root / checkpoint,
            log_path=root / log,
            summary_path=root / summary,
            agent_config=AgentConfig(hidden_dim=16),
            resume_path=resume,
        )

    def test_runtime_logs_prediction_and_resume_preserves_predictor(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            first = self._runtime(
                root,
                seed=23,
                steps=6,
                prediction_learning_rate=0.001,
                checkpoint="first.json",
                log="first.jsonl",
                summary="first_summary.json",
            )
            first_summary = first.run()
            first_weights = first.predictors[0].weights.copy()
            self.assertEqual(first_summary["prediction"]["enabled"], True)
            self.assertEqual(first_summary["prediction"]["steps"], 6)
            self.assertEqual(len(first_summary["prediction"]["per_agent"]), 3)
            records = [
                json.loads(line)
                for line in (root / "first.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            step_records = [record for record in records if record["event"] == "city_step"]
            self.assertEqual(len(step_records), 6)
            for record in step_records:
                self.assertEqual(len(record["prediction_mae"]), 3)
                self.assertEqual(len(record["prediction_mse"]), 3)
                self.assertTrue(all(np.isfinite(record["prediction_mae"])))
                self.assertTrue(all(np.isfinite(record["prediction_mse"])))
                for forbidden in ("reward", "score", "penalty", "correctness"):
                    self.assertNotIn(forbidden, record)

            checkpoint = json.loads((root / "first.json").read_text(encoding="utf-8"))
            self.assertIn("predictors", checkpoint)
            self.assertIn("prediction_metrics", checkpoint)
            self.assertEqual(len(checkpoint["predictors"]), 3)

            second = self._runtime(
                root,
                seed=23,
                steps=8,
                prediction_learning_rate=0.001,
                checkpoint="second.json",
                log="second.jsonl",
                summary="second_summary.json",
                resume=root / "first.json",
            )
            self.assertTrue(np.array_equal(first_weights, second.predictors[0].weights))
            second_summary = second.run()
            self.assertEqual(second_summary["steps"], 8)
            self.assertEqual(second_summary["prediction"]["steps"], 8)
            self.assertEqual(second_summary["prediction"]["prediction_updates"], [8, 8, 8])

    def test_control_and_predictive_city_trajectories_are_identical(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            control = self._runtime(
                root,
                seed=29,
                steps=5,
                prediction_learning_rate=0.0,
                checkpoint="control.json",
                log="control.jsonl",
                summary="control_summary.json",
            )
            predictive = self._runtime(
                root,
                seed=29,
                steps=5,
                prediction_learning_rate=0.001,
                checkpoint="predictive.json",
                log="predictive.jsonl",
                summary="predictive_summary.json",
            )
            control_weights = control.predictors[0].weights.copy()
            control_summary = control.run()
            predictive_summary = predictive.run()

            def step_records(path: Path) -> list[dict[str, object]]:
                return [
                    record
                    for record in (
                        json.loads(line)
                        for line in path.read_text(encoding="utf-8").splitlines()
                    )
                    if record["event"] == "city_step"
                ]

            control_steps = step_records(root / "control.jsonl")
            predictive_steps = step_records(root / "predictive.jsonl")
            self.assertEqual(len(control_steps), len(predictive_steps))
            for left, right in zip(control_steps, predictive_steps):
                for key in (
                    "positions",
                    "actions",
                    "moved",
                    "blocked_reasons",
                    "collision_agents",
                    "interaction_pairs",
                    "hidden_spikes",
                ):
                    self.assertEqual(left[key], right[key])
            self.assertTrue(np.array_equal(control_weights, control.predictors[0].weights))
            self.assertFalse(np.array_equal(control_weights, predictive.predictors[0].weights))
            self.assertEqual(control_summary["prediction"]["prediction_updates"], [0, 0, 0])
            self.assertEqual(predictive_summary["prediction"]["prediction_updates"], [5, 5, 5])
            self.assertTrue(control_summary["prediction"]["current_weights_identical"])
            self.assertFalse(predictive_summary["prediction"]["current_weights_identical"])
            self.assertIsNotNone(control_summary["prediction"]["mean_mae"])
            self.assertIsNotNone(predictive_summary["prediction"]["mean_mae"])


if __name__ == "__main__":
    unittest.main()
