from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exploration_lab.agent import AgentConfig
from exploration_lab.city import (
    CITY_ACTIONS,
    CITY_ACTION_NAMES,
    City,
    CityConfig,
    CityRuntime,
    CityRuntimeConfig,
)
from exploration_lab.games import FEATURE_DIM

ROOT = Path(__file__).resolve().parents[1]
LOCAL_TMP = ROOT / ".local_private" / "pytest_city"


class CityTests(unittest.TestCase):
    def setUp(self) -> None:
        LOCAL_TMP.mkdir(parents=True, exist_ok=True)

    def _tmpdir(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(dir=LOCAL_TMP)

    def test_observation_is_fixed_bounded_and_renderable(self) -> None:
        city = City(CityConfig(width=16, height=16, agent_count=4), seed=7)
        observations = city.observe_all()
        self.assertEqual(len(observations), 4)
        for observation in observations:
            self.assertEqual(len(observation.features), FEATURE_DIM)
            self.assertTrue(np.all(np.isfinite(observation.features)))
            self.assertGreaterEqual(min(observation.features), 0.0)
            self.assertLessEqual(max(observation.features), 1.0)
            self.assertTrue(observation.summary)
            self.assertEqual(len(observation.render), 16)
            self.assertTrue(all(len(row) == 16 for row in observation.render))

    def test_snapshot_restore_and_next_step_match(self) -> None:
        city = City(CityConfig(width=16, height=16, agent_count=4), seed=3)
        for action_set in ([0, 1, 2, 3], [3, 2, 1, 0], [1, 1, 1, 1]):
            city.step(action_set)
        snapshot = city.snapshot()
        restored = City(CityConfig(width=16, height=16, agent_count=4), seed=99)
        restored.restore(json.loads(json.dumps(snapshot)))
        self.assertEqual(city.positions, restored.positions)
        self.assertEqual(city.walls, restored.walls)
        self.assertEqual(city.observe_all()[0].features, restored.observe_all()[0].features)
        self.assertEqual(city.snapshot(), restored.snapshot())
        left = city.step([0, 1, 2, 3])
        right = restored.step([0, 1, 2, 3])
        self.assertEqual(left.positions, right.positions)
        self.assertEqual(left.moved, right.moved)
        self.assertEqual(left.blocked_reasons, right.blocked_reasons)
        self.assertEqual(left.interaction_pairs, right.interaction_pairs)

    def test_simultaneous_collision_is_logged_without_reward(self) -> None:
        city = City(CityConfig(width=16, height=16, agent_count=2), seed=5)
        city.positions = [(8, 8), (10, 8)]
        city.visited = [{(8, 8)}, {(10, 8)}]
        self.assertNotIn((8, 8), city.walls)
        self.assertNotIn((10, 8), city.walls)
        self.assertNotIn((9, 8), city.walls)
        result = city.step([3, 2])
        self.assertEqual(result.collision_agents, (0, 1))
        self.assertEqual(result.moved, (False, False))
        self.assertEqual(result.blocked_reasons, ("collision", "collision"))
        self.assertEqual(city.collision_events, 2)
        self.assertFalse(hasattr(result, "reward"))
        self.assertFalse(hasattr(result, "score"))
        self.assertFalse(hasattr(result, "penalty"))
        self.assertFalse(hasattr(result, "error"))

    def test_illegal_action_is_rejected(self) -> None:
        city = City(CityConfig(width=16, height=16, agent_count=2), seed=1)
        with self.assertRaises(ValueError):
            city.step([4, 0])
        with self.assertRaises(ValueError):
            city.step([0])

    def test_city_step_result_has_no_reward_fields(self) -> None:
        fields = inspect.signature(City.step).parameters
        self.assertEqual(tuple(fields), ("self", "actions"))
        self.assertTrue(all(name in CITY_ACTION_NAMES for name in ("up", "down", "left", "right")))
        self.assertEqual(CITY_ACTIONS, (0, 1, 2, 3))

    def test_clones_start_with_identical_weights_and_independent_rng(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            runtime = CityRuntime(
                CityRuntimeConfig(
                    seed=11,
                    width=16,
                    height=16,
                    agent_count=4,
                    max_steps=2,
                    delay=0.0,
                    checkpoint_every=0,
                    render=False,
                    progress_every=0,
                ),
                checkpoint_path=root / "checkpoint.json",
                log_path=root / "session.jsonl",
                summary_path=root / "summary.json",
                agent_config=AgentConfig(),
            )
            self.assertTrue(runtime.initial_weights_identical)
            reference = runtime.agents[0].w_input_hidden
            for agent in runtime.agents[1:]:
                self.assertTrue(np.array_equal(reference, agent.w_input_hidden))
            rng_states = [json.dumps(agent.state_dict()["rng"], sort_keys=True) for agent in runtime.agents]
            self.assertGreater(len(set(rng_states)), 1)
            runtime.logger.close()

    def test_runtime_short_run_writes_log_checkpoint_and_summary(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            runtime = CityRuntime(
                CityRuntimeConfig(
                    seed=13,
                    width=16,
                    height=16,
                    agent_count=4,
                    max_steps=6,
                    delay=0.0,
                    checkpoint_every=3,
                    render=False,
                    progress_every=0,
                ),
                checkpoint_path=root / "checkpoint.json",
                log_path=root / "session.jsonl",
                summary_path=root / "summary.json",
                agent_config=AgentConfig(),
            )
            summary = runtime.run()
            self.assertEqual(summary["steps"], 6)
            self.assertEqual(summary["agent_count"], 4)
            self.assertEqual(summary["total_agent_steps"], 24)
            self.assertTrue((root / "checkpoint.json").is_file())
            self.assertTrue((root / "summary.json").is_file())
            lines = (root / "session.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertGreaterEqual(len(lines), 6)
            records = [json.loads(line) for line in lines]
            step_records = [record for record in records if record["event"] == "city_step"]
            self.assertEqual(len(step_records), 6)
            for record in step_records:
                for forbidden in ("reward", "score", "penalty", "error", "correctness"):
                    self.assertNotIn(forbidden, record)
            self.assertEqual(summary["fallback_actions"], runtime.fallback_count)
            self.assertIn("note", summary)

    def test_runtime_resume_preserves_agents_and_continues_counter(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)
            first = CityRuntime(
                CityRuntimeConfig(
                    seed=17,
                    width=16,
                    height=16,
                    agent_count=3,
                    max_steps=5,
                    delay=0.0,
                    checkpoint_every=0,
                    render=False,
                    progress_every=0,
                ),
                checkpoint_path=root / "first.json",
                log_path=root / "first.jsonl",
                summary_path=root / "first_summary.json",
                agent_config=AgentConfig(),
            )
            first_summary = first.run()
            first_weights = first.agents[0].w_input_hidden.copy()
            second = CityRuntime(
                CityRuntimeConfig(
                    seed=17,
                    width=16,
                    height=16,
                    agent_count=3,
                    max_steps=9,
                    delay=0.0,
                    checkpoint_every=0,
                    render=False,
                    progress_every=0,
                ),
                checkpoint_path=root / "second.json",
                log_path=root / "second.jsonl",
                summary_path=root / "second_summary.json",
                resume_path=root / "first.json",
            )
            second_summary = second.run()
            self.assertEqual(second_summary["steps"], 9)
            self.assertEqual(second_summary["resumed_from"], str(root / "first.json"))
            self.assertFalse(np.array_equal(first_weights, second.agents[0].w_input_hidden))
            self.assertEqual(second_summary["agent_count"], 3)
            self.assertGreaterEqual(second_summary["total_agent_steps"], 27)

    def test_keyboard_interrupt_saves_checkpoint_and_summary(self) -> None:
        with self._tmpdir() as directory:
            root = Path(directory)

            def interrupt(_runtime: CityRuntime) -> None:
                raise KeyboardInterrupt

            runtime = CityRuntime(
                CityRuntimeConfig(
                    seed=19,
                    width=16,
                    height=16,
                    agent_count=3,
                    max_steps=0,
                    delay=0.0,
                    checkpoint_every=0,
                    render=False,
                    progress_every=0,
                ),
                checkpoint_path=root / "checkpoint.json",
                log_path=root / "session.jsonl",
                summary_path=root / "summary.json",
                agent_config=AgentConfig(),
                on_step=interrupt,
            )
            summary = runtime.run()
            self.assertEqual(summary["steps"], 1)
            self.assertTrue((root / "checkpoint.json").is_file())
            self.assertTrue((root / "summary.json").is_file())
            records = [
                json.loads(line)
                for line in (root / "session.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(any(record["event"] == "keyboard_interrupt" for record in records))

    def test_render_contains_agent_digits(self) -> None:
        city = City(CityConfig(width=16, height=16, agent_count=3), seed=2)
        rendered = "\n".join(city.render())
        for index in range(3):
            self.assertIn(str(index), rendered)


if __name__ == "__main__":
    unittest.main()
