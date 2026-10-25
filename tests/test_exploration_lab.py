from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exploration_lab.agent import AgentConfig, SpikingExplorer
from exploration_lab.encoding import FEATURE_DIM, feature_digest, spikes_from_features
from exploration_lab.games import GAME_REGISTRY, GridWorld, PongLite, Snake, make_game
from exploration_lab.runtime import ExplorationRuntime, RuntimeConfig

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TMP = ROOT / ".local_private" / "pytest_exploration"


class ExplorationLabTests(unittest.TestCase):
    def setUp(self) -> None:
        LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    def _tmpdir(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(dir=LOCAL_TMP)

    def test_all_games_expose_fixed_observations_and_render(self) -> None:
        for name in GAME_REGISTRY:
            game = make_game(name, seed=7)
            observation = game.reset()
            self.assertEqual(len(observation.features), FEATURE_DIM)
            self.assertTrue(np.all(np.isfinite(observation.features)))
            self.assertTrue(observation.summary)
            self.assertTrue(observation.render)
            self.assertTrue(game.legal_actions)
            for action in game.legal_actions:
                game.reset(seed=7)
                result = game.step(action)
                self.assertIsInstance(result.done, bool)
                self.assertEqual(len(result.observation.features), FEATURE_DIM)
                self.assertIn(result.reason, {
                    "moved", "blocked", "goal_reached", "timeout",
                    "wall_collision", "self_collision", "food", "grid_filled",
                    "paddle_hit", "missed_paddle",
                })

    def test_illegal_actions_are_rejected(self) -> None:
        for name in GAME_REGISTRY:
            game = make_game(name, seed=1)
            with self.assertRaises(ValueError):
                game.step(99)
        pong = PongLite(seed=1)
        with self.assertRaises(ValueError):
            pong.step(3)

    def test_snapshot_restore_round_trip_and_next_step_match(self) -> None:
        for name in GAME_REGISTRY:
            game = make_game(name, seed=3)
            game.reset(seed=3)
            for _ in range(3):
                game.step(game.legal_actions[0])
            snapshot = game.snapshot()
            clone = make_game(name, seed=99)
            clone.restore(snapshot)
            self.assertEqual(game.observe().features, clone.observe().features)
            self.assertEqual(game.observe().summary, clone.observe().summary)
            self.assertEqual(game.snapshot(), clone.snapshot())
            left = game.step(game.legal_actions[1])
            right = clone.step(clone.legal_actions[1])
            self.assertEqual(left.observation.features, right.observation.features)
            self.assertEqual(left.done, right.done)
            self.assertEqual(left.reason, right.reason)

    def test_reset_with_same_seed_is_deterministic(self) -> None:
        for name in GAME_REGISTRY:
            first = make_game(name, seed=11).observe()
            second = make_game(name, seed=11).observe()
            self.assertEqual(first.features, second.features)
            self.assertEqual(first.summary, second.summary)

    def test_gridworld_action_moves_agent(self) -> None:
        game = GridWorld(seed=1)
        game.agent_pos = (3, 3)
        game.goal_pos = (6, 6)
        game.walls = {(0, 0)}
        game.done = False
        game.episode_steps = 0
        game.step(3)
        self.assertEqual(game.agent_pos, (4, 3))
        game.step(0)
        self.assertEqual(game.agent_pos, (4, 2))

    def test_snake_action_changes_head(self) -> None:
        game = Snake(seed=1)
        game.snake = [(4, 4), (3, 4), (2, 4)]
        game.direction = 3
        game.food = (0, 0)
        game.done = False
        game.episode_steps = 0
        game.step(0)
        self.assertEqual(game.snake[0], (4, 3))

    def test_pong_action_moves_paddle(self) -> None:
        game = PongLite(seed=1)
        game.paddle_y = 4
        game.ball_x = 8.0
        game.ball_y = 4.0
        game.ball_vx = 0.1
        game.ball_vy = 0.0
        game.done = False
        game.episode_steps = 0
        game.step(0)
        self.assertEqual(game.paddle_y, 3)
        game.step(1)
        self.assertEqual(game.paddle_y, 4)

    def test_encoding_is_binary_and_rejects_wrong_shape(self) -> None:
        spikes = spikes_from_features([0.0] * FEATURE_DIM)
        self.assertEqual(float(spikes.sum()), 0.0)
        spikes = spikes_from_features([1.0] * FEATURE_DIM)
        self.assertEqual(float(spikes.sum()), float(FEATURE_DIM))
        with self.assertRaises(ValueError):
            spikes_from_features([0.0] * (FEATURE_DIM - 1))
        self.assertEqual(feature_digest([0.0] * FEATURE_DIM), feature_digest([0.0] * FEATURE_DIM))

    def test_agent_falls_back_when_no_activity_signal(self) -> None:
        agent = SpikingExplorer(seed=1)
        decision = agent.act([0.0] * FEATURE_DIM, (0, 1, 2, 3))
        self.assertTrue(decision.random_fallback)
        self.assertEqual(decision.selection_reason, "no_activity_signal_uniform_random")
        self.assertIn(decision.action, (0, 1, 2, 3))
        self.assertAlmostEqual(sum(decision.probabilities), 1.0)

    def test_agent_api_has_no_reward_or_outcome_parameter(self) -> None:
        signature = inspect.signature(SpikingExplorer.act)
        self.assertEqual(list(signature.parameters), ["self", "features", "legal_actions"])
        source = inspect.getsource(SpikingExplorer._apply_stdp).lower()
        for forbidden in ("reward", "score", "target", "done", "label", "winner"):
            self.assertNotIn(forbidden, source)

    def test_agent_weights_are_bounded_and_persist_across_transient_reset(self) -> None:
        agent = SpikingExplorer(seed=2)
        initial = agent.w_input_hidden.copy()
        for _ in range(100):
            agent.act([1.0] * FEATURE_DIM, (0, 1, 2, 3))
        self.assertEqual(agent.steps, 100)
        self.assertGreater(agent.total_weight_updates, 0)
        self.assertTrue(np.all(agent.w_input_hidden >= 0.0))
        self.assertTrue(np.all(agent.w_input_hidden <= agent.config.weight_max + 1e-12))
        self.assertTrue(np.all(agent.w_hidden_action >= 0.0))
        self.assertTrue(np.all(agent.w_hidden_action <= agent.config.weight_max + 1e-12))
        self.assertFalse(np.allclose(initial, agent.w_input_hidden))
        before = agent.w_input_hidden.copy()
        steps = agent.steps
        agent.reset_transient()
        self.assertTrue(np.array_equal(before, agent.w_input_hidden))
        self.assertEqual(agent.steps, steps)

    def test_agent_checkpoint_round_trip_reproduces_decision(self) -> None:
        agent = SpikingExplorer(seed=5)
        for _ in range(20):
            agent.act([1.0] * FEATURE_DIM, (0, 1, 2, 3))
        restored = SpikingExplorer.from_state_dict(agent.state_dict())
        self.assertTrue(np.array_equal(agent.w_input_hidden, restored.w_input_hidden))
        self.assertTrue(np.array_equal(agent.w_hidden_action, restored.w_hidden_action))
        self.assertTrue(np.array_equal(agent.w_hidden_hidden, restored.w_hidden_hidden))
        features = [0.0] * FEATURE_DIM
        features[0] = 1.0
        features[5] = 1.0
        first = agent.act(features, (0, 1, 2, 3))
        second = restored.act(features, (0, 1, 2, 3))
        self.assertEqual(first, second)

    def test_runtime_short_run_checkpoint_and_resume(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            config = RuntimeConfig(
                game_names=("gridworld", "snake", "pong"),
                seed=17,
                delay=0.0,
                checkpoint_every=5,
                max_steps=12,
                render=False,
                progress_every=0,
            )
            checkpoint = root / "checkpoint.json"
            log = root / "session.jsonl"
            summary_path = root / "summary.json"
            runtime = ExplorationRuntime(
                config,
                checkpoint_path=checkpoint,
                log_path=log,
                summary_path=summary_path,
                agent_config=AgentConfig(hidden_dim=16, ticks_per_step=2),
            )
            summary = runtime.run()
            self.assertEqual(summary["steps"], 12)
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(log.is_file())
            self.assertTrue(summary_path.is_file())
            records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            step_records = [record for record in records if record["event"] == "step"]
            self.assertEqual(len(step_records), 12)
            self.assertTrue(all("reward" not in record for record in step_records))
            self.assertTrue(all("score" not in record for record in step_records))
            self.assertTrue(any(record["event"] == "checkpoint" for record in records))

            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.assertEqual(checkpoint_payload["step"], 12)
            self.assertEqual(checkpoint_payload["game_names"], ["gridworld", "snake", "pong"])

            resumed_config = RuntimeConfig(
                game_names=(),
                seed=17,
                delay=0.0,
                checkpoint_every=0,
                max_steps=17,
                render=False,
                progress_every=0,
            )
            resumed = ExplorationRuntime(
                resumed_config,
                checkpoint_path=root / "resumed.json",
                log_path=root / "resumed.jsonl",
                summary_path=root / "resumed_summary.json",
                resume_path=checkpoint,
            )
            self.assertEqual(resumed.step_count, 12)
            resumed_summary = resumed.run()
            self.assertEqual(resumed_summary["steps"], 17)
            self.assertTrue((root / "resumed.json").is_file())

    def test_runtime_request_stop_saves_checkpoint(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            config = RuntimeConfig(
                game_names=("gridworld",),
                seed=3,
                delay=0.0,
                checkpoint_every=0,
                max_steps=0,
                render=False,
                progress_every=0,
            )
            runtime = ExplorationRuntime(
                config,
                checkpoint_path=root / "stop.json",
                log_path=root / "stop.jsonl",
                summary_path=root / "stop_summary.json",
                agent_config=AgentConfig(hidden_dim=8, ticks_per_step=2),
            )
            counter = {"steps": 0}

            def stop_after_three(active: ExplorationRuntime) -> None:
                counter["steps"] += 1
                if counter["steps"] >= 3:
                    active.request_stop()

            runtime.on_step = stop_after_three
            summary = runtime.run()
            self.assertEqual(summary["steps"], 3)
            self.assertTrue((root / "stop.json").is_file())
            payload = json.loads((root / "stop.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["step"], 3)

    def test_restored_runtime_continues_identically_from_same_checkpoint(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            base_config = RuntimeConfig(
                game_names=("gridworld",),
                seed=23,
                delay=0.0,
                checkpoint_every=0,
                max_steps=4,
                render=False,
                progress_every=0,
            )
            base = ExplorationRuntime(
                base_config,
                checkpoint_path=root / "base.json",
                log_path=root / "base.jsonl",
                summary_path=root / "base_summary.json",
                agent_config=AgentConfig(hidden_dim=8, ticks_per_step=2),
            )
            base.run()
            records = []
            for index in (1, 2):
                resumed = ExplorationRuntime(
                    RuntimeConfig(
                        game_names=(),
                        seed=23,
                        delay=0.0,
                        checkpoint_every=0,
                        max_steps=5,
                        render=False,
                        progress_every=0,
                    ),
                    checkpoint_path=root / f"resume_{index}.json",
                    log_path=root / f"resume_{index}.jsonl",
                    summary_path=root / f"resume_{index}_summary.json",
                    resume_path=root / "base.json",
                )
                resumed.run()
                lines = (root / f"resume_{index}.jsonl").read_text(encoding="utf-8").splitlines()
                step = [json.loads(line) for line in lines if json.loads(line)["event"] == "step"][0]
                records.append(step)
            self.assertEqual(records[0]["action"], records[1]["action"])
            self.assertEqual(records[0]["observation_digest"], records[1]["observation_digest"])
            self.assertEqual(records[0]["game"], records[1]["game"])

    def test_keyboard_interrupt_saves_checkpoint(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            runtime = ExplorationRuntime(
                RuntimeConfig(
                    game_names=("gridworld",),
                    seed=3,
                    delay=0.0,
                    checkpoint_every=0,
                    max_steps=0,
                    render=False,
                    progress_every=0,
                ),
                checkpoint_path=root / "interrupt.json",
                log_path=root / "interrupt.jsonl",
                summary_path=root / "interrupt_summary.json",
                agent_config=AgentConfig(hidden_dim=8, ticks_per_step=2),
            )

            def interrupt() -> None:
                raise KeyboardInterrupt

            runtime._step_once = interrupt  # type: ignore[method-assign]
            summary = runtime.run()
            self.assertTrue(summary["interrupted"])
            self.assertTrue((root / "interrupt.json").is_file())
            self.assertTrue((root / "interrupt_summary.json").is_file())

    def test_step_result_has_no_reward_fields(self) -> None:
        from exploration_lab.games import StepResult
        self.assertEqual(tuple(StepResult.__dataclass_fields__), ("observation", "done", "reason"))


if __name__ == "__main__":
    unittest.main()
