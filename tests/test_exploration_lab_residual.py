from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exploration_lab.agent import AgentConfig, SpikingExplorer
from exploration_lab.city import CityRuntime, CityRuntimeConfig
from exploration_lab.games import FEATURE_DIM
from exploration_lab.prediction import LocalPredictor, PredictionConfig
from exploration_lab.residual import ResidualConfig, residual_summary, split_residual
from exploration_lab.runtime import JsonlLogger

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TMP = ROOT / ".local_private" / "pytest_residual"


def _features(offset: float = 0.0) -> np.ndarray:
    return np.clip(np.linspace(0.0, 1.0, FEATURE_DIM) + offset, 0.0, 1.0)


def _residual(scale: float = 0.5) -> np.ndarray:
    vector = np.zeros(FEATURE_DIM, dtype=np.float64)
    vector[0] = scale
    vector[1] = -scale
    vector[2] = 2.0 * scale
    vector[3] = -2.0 * scale
    return vector


def _runtime(
    tmp: Path,
    *,
    residual_feedback: bool,
    hidden: int = 8,
    agents: int = 2,
    width: int = 12,
    height: int = 12,
    seed: int = 41,
    prediction_learning_rate: float = 0.001,
    prediction_window: int = 3,
    log_max_bytes: int = 0,
    log_backup_count: int = 2,
    name: str = "run",
) -> CityRuntime:
    directory = tmp / name
    directory.mkdir(parents=True, exist_ok=True)
    config = CityRuntimeConfig(
        seed=seed,
        width=width,
        height=height,
        agent_count=agents,
        max_steps=0,
        delay=0.0,
        checkpoint_every=0,
        render=False,
        progress_every=0,
        prediction_learning_rate=prediction_learning_rate,
        prediction_window=prediction_window,
        residual_feedback=residual_feedback,
        log_max_bytes=log_max_bytes,
        log_backup_count=log_backup_count,
    )
    return CityRuntime(
        config,
        checkpoint_path=directory / "checkpoint.json",
        log_path=directory / "session.jsonl",
        agent_config=AgentConfig(
            hidden_dim=hidden,
            ticks_per_step=3,
            residual_enabled=residual_feedback,
        ),
    )


class ResidualEncodingTests(unittest.TestCase):
    def test_signed_channels_preserve_direction_and_clip(self) -> None:
        residual = _residual(0.5)
        positive, negative = split_residual(residual)
        self.assertTrue(np.all(positive >= 0.0))
        self.assertTrue(np.all(negative >= 0.0))
        np.testing.assert_allclose(positive[0], 0.5)
        np.testing.assert_allclose(negative[0], 0.0)
        np.testing.assert_allclose(positive[1], 0.0)
        np.testing.assert_allclose(negative[1], 0.5)
        np.testing.assert_allclose(positive[2], 1.0)
        np.testing.assert_allclose(negative[3], 1.0)
        np.testing.assert_allclose(positive - negative, np.clip(residual, -1.0, 1.0))

        summary = residual_summary(residual)
        self.assertGreater(summary["positive_mean"], 0.0)
        self.assertGreater(summary["negative_mean"], 0.0)
        self.assertGreater(summary["max_abs"], 0.0)

    def test_invalid_residual_shape_and_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            split_residual(np.zeros(FEATURE_DIM - 1))
        bad = np.zeros(FEATURE_DIM)
        bad[0] = np.nan
        with self.assertRaises(ValueError):
            split_residual(bad)


class ResidualAgentTests(unittest.TestCase):
    def test_residual_is_delivered_once_to_the_next_act(self) -> None:
        agent = SpikingExplorer(
            AgentConfig(hidden_dim=8, ticks_per_step=3, residual_enabled=True),
            seed=7,
        )
        self.assertIsNone(agent.pending_residual)
        self.assertIsNone(agent.last_residual_input)

        agent.set_prediction_residual(_residual(0.5))
        self.assertIsNotNone(agent.pending_residual)
        agent.act(_features(), (0, 1, 2, 3))
        self.assertIsNone(agent.pending_residual)
        np.testing.assert_allclose(
            agent.last_residual_input,
            np.clip(_residual(0.5), -1.0, 1.0),
        )
        self.assertGreater(agent.last_residual_current_abs_mean, 0.0)

        agent.act(_features(), (0, 1, 2, 3))
        np.testing.assert_allclose(agent.last_residual_input, np.zeros(FEATURE_DIM))
        self.assertEqual(agent.last_residual_current_abs_mean, 0.0)

    def test_legacy_agent_rejects_residual_input(self) -> None:
        agent = SpikingExplorer(AgentConfig(hidden_dim=8), seed=7)
        self.assertFalse(agent.config.residual_enabled)
        self.assertFalse(np.any(agent.w_residual_pos_hidden))
        self.assertFalse(np.any(agent.w_residual_neg_hidden))
        with self.assertRaises(ValueError):
            agent.set_prediction_residual(_residual())
        agent.act(_features(), (0, 1, 2, 3))
        self.assertIsNone(agent.last_residual_input)

    def test_residual_pathway_changes_hidden_activity(self) -> None:
        first = SpikingExplorer(
            AgentConfig(hidden_dim=16, ticks_per_step=3, residual_enabled=True),
            seed=11,
        )
        second = SpikingExplorer.from_state_dict(first.state_dict())
        original_input = first.w_input_hidden.copy()
        original_recurrent = first.w_hidden_hidden.copy()
        original_action = first.w_hidden_action.copy()

        first.set_prediction_residual(np.ones(FEATURE_DIM, dtype=np.float64))
        first.act(_features(), (0, 1, 2, 3))
        second.act(_features(), (0, 1, 2, 3))

        self.assertFalse(np.array_equal(first.last_hidden_activity, second.last_hidden_activity))
        # The dedicated residual matrices exist separately from the observation,
        # recurrent, and action pathways.
        self.assertEqual(first.w_residual_pos_hidden.shape, (FEATURE_DIM, 16))
        self.assertEqual(first.w_residual_neg_hidden.shape, (FEATURE_DIM, 16))
        self.assertGreater(float(np.abs(first.w_residual_pos_hidden).mean()), 0.0)
        self.assertLess(float(first.w_residual_neg_hidden.max()), 0.0)
        self.assertEqual(first.w_input_hidden.shape, original_input.shape)
        self.assertEqual(first.w_hidden_hidden.shape, original_recurrent.shape)
        self.assertEqual(first.w_hidden_action.shape, original_action.shape)

    def test_residual_pathway_state_roundtrip_is_exact(self) -> None:
        agent = SpikingExplorer(
            AgentConfig(hidden_dim=8, ticks_per_step=3, residual_enabled=True),
            seed=13,
        )
        agent.act(_features(), (0, 1, 2, 3))
        agent.set_prediction_residual(_residual(0.3))
        state = agent.state_dict()
        self.assertIn("w_residual_pos_hidden", state)
        self.assertIn("w_residual_neg_hidden", state)
        self.assertIn("pending_residual", state)
        self.assertIn("last_residual_input", state)

        restored = SpikingExplorer.from_state_dict(state)
        np.testing.assert_array_equal(agent.w_residual_pos_hidden, restored.w_residual_pos_hidden)
        np.testing.assert_array_equal(agent.w_residual_neg_hidden, restored.w_residual_neg_hidden)
        np.testing.assert_array_equal(agent.pending_residual, restored.pending_residual)
        np.testing.assert_array_equal(agent.last_residual_input, restored.last_residual_input)

        decision_a = agent.act(_features(), (0, 1, 2, 3))
        decision_b = restored.act(_features(), (0, 1, 2, 3))
        self.assertEqual(decision_a, decision_b)
        np.testing.assert_array_equal(agent.last_hidden_activity, restored.last_hidden_activity)

    def test_legacy_v1_checkpoint_requires_explicit_migration(self) -> None:
        agent = SpikingExplorer(AgentConfig(hidden_dim=8), seed=17)
        agent.act(_features(), (0, 1, 2, 3))
        legacy = agent.state_dict()
        legacy["schema"] = "nucleus-exploration-agent-v1"
        for key in (
            "w_residual_pos_hidden",
            "w_residual_neg_hidden",
            "pending_residual",
            "last_residual_input",
            "migration_note",
        ):
            legacy.pop(key, None)

        with self.assertRaises(ValueError):
            SpikingExplorer.from_state_dict(legacy)
        migrated = SpikingExplorer.from_state_dict(legacy, allow_legacy_migration=True)
        np.testing.assert_array_equal(agent.w_input_hidden, migrated.w_input_hidden)
        np.testing.assert_array_equal(agent.w_hidden_action, migrated.w_hidden_action)
        self.assertFalse(migrated.config.residual_enabled)
        self.assertFalse(np.any(migrated.w_residual_pos_hidden))
        self.assertIsNotNone(migrated.migration_note)


class PredictorIsolationTests(unittest.TestCase):
    def test_prediction_update_does_not_touch_agent_weights(self) -> None:
        agent = SpikingExplorer(
            AgentConfig(hidden_dim=8, ticks_per_step=2, residual_enabled=True),
            seed=19,
        )
        predictor = LocalPredictor(
            PredictionConfig(hidden_dim=8, action_dim=4, input_dim=FEATURE_DIM),
            seed=3,
        )
        agent.act(_features(), (0, 1, 2, 3))
        snapshot = {
            "input": agent.w_input_hidden.copy(),
            "recurrent": agent.w_hidden_hidden.copy(),
            "action": agent.w_hidden_action.copy(),
            "residual_pos": agent.w_residual_pos_hidden.copy(),
            "residual_neg": agent.w_residual_neg_hidden.copy(),
        }
        predictor.predict(_features(), agent.last_hidden_activity, 1)
        predictor.observe_next(_features(0.1))
        np.testing.assert_array_equal(agent.w_input_hidden, snapshot["input"])
        np.testing.assert_array_equal(agent.w_hidden_hidden, snapshot["recurrent"])
        np.testing.assert_array_equal(agent.w_hidden_action, snapshot["action"])
        np.testing.assert_array_equal(agent.w_residual_pos_hidden, snapshot["residual_pos"])
        np.testing.assert_array_equal(agent.w_residual_neg_hidden, snapshot["residual_neg"])

    def test_predictor_residual_roundtrip_and_validation(self) -> None:
        predictor = LocalPredictor(
            PredictionConfig(hidden_dim=8, action_dim=4, input_dim=FEATURE_DIM),
            seed=5,
        )
        predictor.predict(_features(), np.zeros(8), 0)
        predictor.observe_next(_features(0.2))
        self.assertIsNotNone(predictor.last_residual)
        state = predictor.state_dict()
        restored = LocalPredictor.from_state_dict(state)
        np.testing.assert_array_equal(predictor.last_residual, restored.last_residual)
        np.testing.assert_array_equal(predictor.weights, restored.weights)

        bad_shape = dict(state)
        bad_shape["last_residual"] = [0.0, 1.0]
        with self.assertRaises(ValueError):
            LocalPredictor.from_state_dict(bad_shape)

        bad_value = dict(state)
        residual = np.zeros(FEATURE_DIM)
        residual[0] = np.nan
        bad_value["last_residual"] = residual.tolist()
        with self.assertRaises(ValueError):
            LocalPredictor.from_state_dict(bad_value)

        out_of_range = dict(state)
        residual = np.zeros(FEATURE_DIM)
        residual[0] = 2.0
        out_of_range["last_residual"] = residual.tolist()
        with self.assertRaises(ValueError):
            LocalPredictor.from_state_dict(out_of_range)

        bad_weights = dict(state)
        weights = np.asarray(state["weights"], dtype=np.float64).copy()
        weights[0, 0] = predictor.config.weight_max + 1.0
        bad_weights["weights"] = weights.tolist()
        with self.assertRaises(ValueError):
            LocalPredictor.from_state_dict(bad_weights)

        bad_counters = dict(state)
        bad_counters["steps"] = -1
        with self.assertRaises(ValueError):
            LocalPredictor.from_state_dict(bad_counters)


class CityResidualTests(unittest.TestCase):
    def setUp(self) -> None:
        LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    def _tmpdir(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(dir=LOCAL_TMP)

    def test_first_step_has_no_residual_and_second_step_uses_previous(self) -> None:
        with self._tmpdir() as name:
            runtime = _runtime(Path(name), residual_feedback=True)
            self.assertIsNone(runtime.agents[0].pending_residual)
            runtime.step_once()
            first = runtime.last_record
            self.assertIsNotNone(first)
            self.assertEqual(first["residual_input_abs_mean"], [0.0, 0.0])
            self.assertEqual(first["residual_current_abs_mean"], [0.0, 0.0])
            pending = [agent.pending_residual.copy() for agent in runtime.agents]
            self.assertTrue(all(value is not None for value in pending))
            runtime.step_once()
            second = runtime.last_record
            self.assertIsNotNone(second)
            expected = [float(np.abs(value).mean()) for value in pending]
            np.testing.assert_allclose(second["residual_input_abs_mean"], expected)
            for agent, value in zip(runtime.agents, pending):
                np.testing.assert_allclose(agent.last_residual_input, value)
            runtime.logger.close()

    def test_adaptive_and_legacy_checkpoint_resume_are_exact(self) -> None:
        for mode in (False, True):
            with self.subTest(residual_feedback=mode), self._tmpdir() as name:
                root = Path(name)
                runtime = _runtime(root, residual_feedback=mode, name="original")
                for _ in range(4):
                    runtime.step_once()
                runtime.checkpoint()
                checkpoint_path = runtime.checkpoint_path
                self.assertTrue(checkpoint_path.is_file())

                resumed_config = CityRuntimeConfig(
                    seed=runtime.config.seed,
                    width=runtime.config.width,
                    height=runtime.config.height,
                    agent_count=runtime.config.agent_count,
                    max_steps=0,
                    delay=0.0,
                    checkpoint_every=0,
                    render=False,
                    progress_every=0,
                    prediction_learning_rate=runtime.config.prediction_learning_rate,
                    prediction_weight_max=runtime.config.prediction_weight_max,
                    prediction_weight_decay=runtime.config.prediction_weight_decay,
                    prediction_window=runtime.config.prediction_window,
                    residual_feedback=mode,
                    log_max_bytes=0,
                    log_backup_count=2,
                )
                resumed = CityRuntime(
                    resumed_config,
                    checkpoint_path=root / "resumed_checkpoint.json",
                    log_path=root / "resumed_session.jsonl",
                    resume_path=checkpoint_path,
                )
                self.assertEqual(resumed.step_count, runtime.step_count)
                self.assertEqual(resumed.config.residual_feedback, mode)
                self.assertEqual(
                    resumed.agents[0].config.residual_enabled,
                    mode,
                )

                for _ in range(3):
                    runtime.step_once()
                    resumed.step_once()
                    self.assertEqual(
                        runtime.city.snapshot(),
                        resumed.city.snapshot(),
                    )
                    for left, right in zip(runtime.agents, resumed.agents):
                        self.assertEqual(left.state_dict(), right.state_dict())
                    for left, right in zip(runtime.predictors, resumed.predictors):
                        self.assertEqual(left.state_dict(), right.state_dict())
                    left_record = dict(runtime.last_record or {})
                    right_record = dict(resumed.last_record or {})
                    left_record.pop("ts_utc", None)
                    right_record.pop("ts_utc", None)
                    left_record.pop("session_id", None)
                    right_record.pop("session_id", None)
                    self.assertEqual(left_record, right_record)
                runtime.logger.close()
                resumed.logger.close()

    def test_resume_config_consistency_and_actual_config_reporting(self) -> None:
        with self._tmpdir() as name:
            root = Path(name)
            runtime = _runtime(
                root,
                residual_feedback=True,
                hidden=10,
                agents=2,
                width=14,
                height=14,
                seed=29,
                prediction_window=4,
                name="original",
            )
            runtime.step_once()
            runtime.checkpoint()
            checkpoint_path = runtime.checkpoint_path

            mismatched = CityRuntimeConfig(
                seed=runtime.config.seed,
                width=16,
                height=runtime.config.height,
                agent_count=runtime.config.agent_count,
                max_steps=0,
                delay=0.0,
                checkpoint_every=0,
                render=False,
                progress_every=0,
                prediction_learning_rate=runtime.config.prediction_learning_rate,
                prediction_weight_max=runtime.config.prediction_weight_max,
                prediction_weight_decay=runtime.config.prediction_weight_decay,
                prediction_window=runtime.config.prediction_window,
                residual_feedback=True,
                log_max_bytes=0,
            )
            with self.assertRaises(ValueError) as context:
                CityRuntime(
                    mismatched,
                    checkpoint_path=root / "bad_checkpoint.json",
                    log_path=root / "bad_session.jsonl",
                    resume_path=checkpoint_path,
                )
            self.assertIn("resume configuration mismatch", str(context.exception))
            self.assertIn("width", str(context.exception))

            matching = CityRuntimeConfig(
                seed=runtime.config.seed,
                width=runtime.config.width,
                height=runtime.config.height,
                agent_count=runtime.config.agent_count,
                max_steps=0,
                delay=0.0,
                checkpoint_every=0,
                render=False,
                progress_every=0,
                prediction_learning_rate=runtime.config.prediction_learning_rate,
                prediction_weight_max=runtime.config.prediction_weight_max,
                prediction_weight_decay=runtime.config.prediction_weight_decay,
                prediction_window=runtime.config.prediction_window,
                residual_feedback=True,
                log_max_bytes=0,
            )
            resumed = CityRuntime(
                matching,
                checkpoint_path=root / "ok_checkpoint.json",
                log_path=root / "ok_session.jsonl",
                resume_path=checkpoint_path,
            )
            self.assertEqual(resumed.config.width, 14)
            self.assertEqual(resumed.config.height, 14)
            self.assertEqual(resumed.config.agent_count, 2)
            self.assertTrue(resumed.config.residual_feedback)
            summary = resumed.summary()
            self.assertEqual(summary["runtime_config"]["width"], 14)
            self.assertEqual(summary["runtime_config"]["agent_count"], 2)
            self.assertEqual(summary["mode"], "adaptive")
            self.assertTrue(summary["residual"]["enabled"])
            runtime.logger.close()
            resumed.logger.close()

    def test_malformed_predictor_list_is_rejected_and_missing_list_requires_migration(self) -> None:
        with self._tmpdir() as name:
            root = Path(name)
            runtime = _runtime(root, residual_feedback=True, name="original")
            runtime.step_once()
            runtime.checkpoint()
            payload = json.loads(runtime.checkpoint_path.read_text(encoding="utf-8"))

            malformed = dict(payload)
            malformed["predictors"] = payload["predictors"][:1]
            malformed_path = root / "malformed.json"
            malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                CityRuntime(
                    runtime.config,
                    checkpoint_path=root / "malformed_checkpoint.json",
                    log_path=root / "malformed_session.jsonl",
                    resume_path=malformed_path,
                )
            self.assertIn("predictor count", str(context.exception))

            missing = dict(payload)
            missing.pop("predictors", None)
            missing.pop("prediction_metrics", None)
            missing_path = root / "missing.json"
            missing_path.write_text(json.dumps(missing), encoding="utf-8")
            with self.assertRaises(ValueError) as context:
                CityRuntime(
                    runtime.config,
                    checkpoint_path=root / "missing_checkpoint.json",
                    log_path=root / "missing_session.jsonl",
                    resume_path=missing_path,
                )
            self.assertIn("no predictor list", str(context.exception))

            migration_config = CityRuntimeConfig(
                **{**runtime.config.__dict__, "allow_legacy_migration": True}
            )
            migrated = CityRuntime(
                migration_config,
                checkpoint_path=root / "migrated_checkpoint.json",
                log_path=root / "migrated_session.jsonl",
                resume_path=missing_path,
            )
            self.assertIsNotNone(migrated.prediction_migration_note)
            self.assertEqual(len(migrated.predictors), 2)
            runtime.logger.close()
            migrated.logger.close()

    def test_log_rotation_preserves_jsonl_lines_and_session_continuity(self) -> None:
        with self._tmpdir() as name:
            path = Path(name) / "rotating.jsonl"
            logger = JsonlLogger(
                path,
                max_bytes=1000,
                backup_count=10,
                session_id="session-test",
            )
            for index in range(30):
                logger.write({"event": "record", "sequence": index, "payload": "x" * 40})
            logger.close()

            segments = logger.segment_paths()
            self.assertGreaterEqual(len(segments), 2)
            sequences: list[int] = []
            saw_rotation = False
            for segment in segments:
                text = segment.read_text(encoding="utf-8")
                self.assertTrue(text.endswith("\n"))
                for line in text.splitlines():
                    record = json.loads(line)
                    self.assertEqual(record["session_id"], "session-test")
                    self.assertIn("log_segment", record)
                    if record["event"] == "log_rotation":
                        saw_rotation = True
                    else:
                        sequences.append(int(record["sequence"]))
            self.assertTrue(saw_rotation)
            self.assertEqual(sequences, sorted(sequences))
            self.assertEqual(sequences, list(range(30)))

    def test_city_runtime_rotates_logs(self) -> None:
        with self._tmpdir() as name:
            runtime = _runtime(
                Path(name),
                residual_feedback=True,
                log_max_bytes=900,
                log_backup_count=2,
            )
            runtime.config = CityRuntimeConfig(**{**runtime.config.__dict__, "max_steps": 8})
            summary = runtime.run()
            self.assertGreaterEqual(len(summary["log_segments"]), 2)
            for segment in summary["log_segments"]:
                path = Path(segment["path"])
                self.assertTrue(path.is_file())
                for line in path.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    self.assertEqual(record["session_id"], runtime.session_id)

    def test_legacy_mode_is_deterministic_and_delivers_no_residual(self) -> None:
        with self._tmpdir() as name:
            root = Path(name)
            first = _runtime(root, residual_feedback=False, name="first")
            second = _runtime(root, residual_feedback=False, name="second")
            for _ in range(4):
                first.step_once()
                second.step_once()
                left = dict(first.last_record or {})
                right = dict(second.last_record or {})
                left.pop("ts_utc", None)
                right.pop("ts_utc", None)
                left.pop("session_id", None)
                right.pop("session_id", None)
                self.assertEqual(left, right)
                self.assertFalse(left["residual_feedback"])
                self.assertEqual(left["residual_input_abs_mean"], [0.0, 0.0])
                self.assertEqual(left["residual_current_abs_mean"], [0.0, 0.0])
            for agent in first.agents:
                self.assertFalse(agent.config.residual_enabled)
                self.assertIsNone(agent.last_residual_input)
                self.assertIsNone(agent.pending_residual)
            summary = first.summary()
            self.assertEqual(summary["mode"], "legacy")
            self.assertFalse(summary["residual"]["enabled"])
            first.logger.close()
            second.logger.close()

    def test_fixed_predictor_control_keeps_predictor_weights_fixed(self) -> None:
        with self._tmpdir() as name:
            runtime = _runtime(
                Path(name),
                residual_feedback=True,
                prediction_learning_rate=0.0,
            )
            before = [predictor.weights.copy() for predictor in runtime.predictors]
            for _ in range(4):
                runtime.step_once()
            for predictor, weights in zip(runtime.predictors, before):
                np.testing.assert_array_equal(predictor.weights, weights)
                self.assertEqual(predictor.total_updates, 0)
            self.assertGreater(
                float(np.abs(runtime.predictors[0].last_residual).mean()),
                0.0,
            )
            self.assertGreater(runtime.agents[0].last_residual_current_abs_mean, 0.0)
            runtime.logger.close()

    def test_residual_and_plasticity_values_remain_bounded(self) -> None:
        with self._tmpdir() as name:
            runtime = _runtime(Path(name), residual_feedback=True)
            for _ in range(20):
                runtime.step_once()
            for agent in runtime.agents:
                cfg = agent.config
                for matrix in (
                    agent.w_input_hidden,
                    agent.w_hidden_hidden,
                    agent.w_hidden_action,
                    agent.w_residual_pos_hidden,
                    agent.w_residual_neg_hidden,
                    agent.v_hidden,
                    agent.i_hidden,
                    agent.trace_hidden,
                ):
                    self.assertTrue(np.all(np.isfinite(matrix)))
                self.assertGreaterEqual(float(agent.w_input_hidden.min()), 0.0)
                self.assertLessEqual(float(agent.w_input_hidden.max()), cfg.weight_max + 1e-9)
                self.assertGreaterEqual(float(agent.w_hidden_action.min()), 0.0)
                self.assertLessEqual(float(agent.w_hidden_action.max()), cfg.weight_max + 1e-9)
                self.assertGreaterEqual(float(agent.w_residual_pos_hidden.min()), 0.0)
                self.assertLessEqual(
                    float(agent.w_residual_pos_hidden.max()),
                    cfg.residual_weight_scale + 1e-9,
                )
                self.assertLessEqual(float(agent.w_residual_neg_hidden.max()), 0.0)
                self.assertGreaterEqual(
                    float(agent.w_residual_neg_hidden.min()),
                    -cfg.residual_weight_scale - 1e-9,
                )
                if agent.pending_residual is not None:
                    self.assertLessEqual(float(np.abs(agent.pending_residual).max()), 1.0 + 1e-9)
            for predictor in runtime.predictors:
                self.assertLessEqual(
                    float(np.abs(predictor.weights).max()),
                    predictor.config.weight_max + 1e-9,
                )
                self.assertLessEqual(float(np.abs(predictor.bias).max()), 1.0 + 1e-9)
            runtime.logger.close()


if __name__ == "__main__":
    unittest.main()
