"""Shared multi-agent city for the reward-free exploration lab.

The city is a small grid with buildings, roads, a central plaza, and several
clones of :class:`exploration_lab.agent.SpikingExplorer`.  Each clone has its own
persistent LIF state and independent RNG stream, but all clones start from the
same weights.  They share one environment and move simultaneously.

Important scope note: blocked moves, collisions, proximity events, and action
counts are *observed and logged only*.  They are not rewards, penalties,
correctness labels, or error signals, and they are never passed to the learning
rule.  Weight changes still come only from local spike-timing-dependent
plasticity inside each clone.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .agent import ActionDecision, AgentConfig, SpikingExplorer
from .games import FEATURE_DIM, Observation
from .runtime import JsonlLogger, format_duration, sha256_file, utc_now

CITY_SCHEMA = "nucleus-exploration-city-v1"
CITY_CHECKPOINT_SCHEMA = "nucleus-exploration-city-runtime-v1"
CITY_ACTION_NAMES = ("up", "down", "left", "right")
CITY_ACTIONS = (0, 1, 2, 3)
_MOVES = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}


def _rng_to_json(rng: random.Random) -> dict[str, Any]:
    version, state, gauss_next = rng.getstate()
    return {"version": version, "state": list(state), "gauss_next": gauss_next}


def _rng_from_json(data: dict[str, Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((int(data["version"]), tuple(int(x) for x in data["state"]), data["gauss_next"]))
    return rng


def _cell_key(cell: tuple[int, int]) -> list[int]:
    return [int(cell[0]), int(cell[1])]


@dataclass(frozen=True)
class CityConfig:
    width: int = 32
    height: int = 32
    agent_count: int = 8
    block_size: int = 5
    block_stride: int = 8
    plaza_radius: int = 3
    wall_erosion: float = 0.06


@dataclass(frozen=True)
class CityStepResult:
    observations: tuple[Observation, ...]
    actions: tuple[int, ...]
    positions: tuple[tuple[int, int], ...]
    moved: tuple[bool, ...]
    blocked_reasons: tuple[str, ...]
    collision_agents: tuple[int, ...]
    interaction_pairs: tuple[tuple[int, int], ...]


class City:
    """A deterministic shared grid city with simultaneous agent movement."""

    def __init__(self, config: CityConfig | None = None, seed: int | None = None) -> None:
        self.config = config or CityConfig()
        self._validate_config()
        self._rng = random.Random(seed)
        self.step_count = 0
        self.blocked_moves = 0
        self.collision_events = 0
        self.interaction_events = 0
        self.agent_blocked_counts = [0 for _ in range(self.config.agent_count)]
        self.agent_collision_counts = [0 for _ in range(self.config.agent_count)]
        self.interaction_pair_counts: Counter[tuple[int, int]] = Counter()
        self._build_map()
        self.positions = self._spawn_positions()
        self.visited: list[set[tuple[int, int]]] = [{position} for position in self.positions]

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.width < 12 or cfg.height < 12:
            raise ValueError("city must be at least 12x12")
        if cfg.agent_count <= 0:
            raise ValueError("agent_count must be positive")
        if cfg.block_size <= 0 or cfg.block_stride <= 1:
            raise ValueError("block dimensions must be positive")
        if cfg.plaza_radius < 1:
            raise ValueError("plaza_radius must be positive")
        if not 0.0 <= cfg.wall_erosion < 1.0:
            raise ValueError("wall_erosion must be in [0, 1)")

    def _build_map(self) -> None:
        cfg = self.config
        self.border: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        for x in range(cfg.width):
            self.border.add((x, 0))
            self.border.add((x, cfg.height - 1))
        for y in range(cfg.height):
            self.border.add((0, y))
            self.border.add((cfg.width - 1, y))
        self.walls.update(self.border)

        for x0 in range(2, cfg.width - 2, cfg.block_stride):
            for y0 in range(2, cfg.height - 2, cfg.block_stride):
                for dx in range(cfg.block_size):
                    for dy in range(cfg.block_size):
                        cell = (x0 + dx, y0 + dy)
                        if cell in self.border:
                            continue
                        if cell[0] >= cfg.width - 1 or cell[1] >= cfg.height - 1:
                            continue
                        self.walls.add(cell)

        # Carve a few small alleys so the city is not a perfectly regular maze.
        for cell in sorted(self.walls):
            if cell in self.border:
                continue
            if self._rng.random() < cfg.wall_erosion:
                self.walls.remove(cell)

        center_x = cfg.width // 2
        center_y = cfg.height // 2
        self.plaza: set[tuple[int, int]] = set()
        for y in range(max(1, center_y - cfg.plaza_radius), min(cfg.height - 1, center_y + cfg.plaza_radius + 1)):
            for x in range(max(1, center_x - cfg.plaza_radius), min(cfg.width - 1, center_x + cfg.plaza_radius + 1)):
                cell = (x, y)
                self.walls.discard(cell)
                self.plaza.add(cell)

        self.open_cells = {
            (x, y)
            for y in range(1, cfg.height - 1)
            for x in range(1, cfg.width - 1)
            if (x, y) not in self.walls
        }
        if len(self.open_cells) < cfg.agent_count:
            raise ValueError("city does not have enough open cells for the requested agents")

    def _spawn_positions(self) -> list[tuple[int, int]]:
        cfg = self.config
        center_x = cfg.width // 2
        center_y = cfg.height // 2
        candidates = [
            (x, y)
            for y in range(max(1, center_y - 3), min(cfg.height - 1, center_y + 4))
            for x in range(max(1, center_x - 3), min(cfg.width - 1, center_x + 4))
            if (x, y) in self.open_cells
        ]
        if len(candidates) < cfg.agent_count:
            candidates = sorted(self.open_cells)
        return self._rng.sample(candidates, cfg.agent_count)

    def _is_wall(self, cell: tuple[int, int]) -> bool:
        x, y = cell
        if x < 0 or y < 0 or x >= self.config.width or y >= self.config.height:
            return True
        return cell in self.walls

    def _cell_type(self, cell: tuple[int, int]) -> list[float]:
        if cell in self.border:
            return [0.0, 0.0, 0.0, 1.0]
        if cell in self.walls:
            return [0.0, 0.0, 1.0, 0.0]
        if cell in self.plaza:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]

    def observe(self, agent_index: int, render: tuple[str, ...] | None = None) -> Observation:
        cfg = self.config
        if agent_index < 0 or agent_index >= len(self.positions):
            raise IndexError("agent_index out of range")
        x, y = self.positions[agent_index]
        others = [
            position
            for index, position in enumerate(self.positions)
            if index != agent_index
        ]

        wall_patch: list[float] = []
        agent_patch: list[float] = []
        other_set = set(others)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                cell = (x + dx, y + dy)
                wall_patch.append(1.0 if self._is_wall(cell) else 0.0)
                agent_patch.append(1.0 if cell in other_set else 0.0)

        if others:
            nearest = min(
                others,
                key=lambda position: (position[0] - x) ** 2 + (position[1] - y) ** 2,
            )
            raw_dx = nearest[0] - x
            raw_dy = nearest[1] - y
            distance = (raw_dx * raw_dx + raw_dy * raw_dy) ** 0.5
            scale = max(1, cfg.width - 1, cfg.height - 1)
            dx_norm = raw_dx / scale
            dy_norm = raw_dy / scale
            distance_norm = min(1.0, distance / (scale * (2 ** 0.5)))
            neighbors = sum(
                1
                for position in others
                if max(abs(position[0] - x), abs(position[1] - y)) <= 2
            )
        else:
            dx_norm = 0.0
            dy_norm = 0.0
            distance_norm = 0.0
            neighbors = 0

        neighbor_bin = min(neighbors, 3)
        neighbor_one_hot = [1.0 if neighbor_bin == index else 0.0 for index in range(4)]
        wall_adjacency = [
            1.0 if self._is_wall((x + dx, y + dy)) else 0.0
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
        ]
        open_directions = [
            1.0 if not self._is_wall((x + dx, y + dy)) else 0.0
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))
        ]

        features: list[float] = [
            1.0,
            0.0,
            0.0,
            0.0,
            *wall_patch,
            *agent_patch,
            x / max(1, cfg.width - 1),
            y / max(1, cfg.height - 1),
            abs(dx_norm),
            abs(dy_norm),
            max(0.0, dx_norm),
            max(0.0, -dx_norm),
            max(0.0, dy_norm),
            max(0.0, -dy_norm),
            *neighbor_one_hot,
            *wall_adjacency,
            *open_directions,
            distance_norm,
            min(1.0, neighbors / 24.0),
            *self._cell_type((x, y)),
        ]
        if len(features) != FEATURE_DIM:
            raise AssertionError(f"city observation has {len(features)} features, expected {FEATURE_DIM}")
        summary = (
            f"agent={agent_index} pos=({x},{y}) neighbors={neighbors} "
            f"nearest={distance_norm:.2f} blocked_dirs={int(sum(wall_adjacency))}"
        )
        return Observation(tuple(features), summary, render if render is not None else self.render())

    def observe_all(self) -> list[Observation]:
        render = self.render()
        return [self.observe(index, render) for index in range(len(self.positions))]

    def step(self, actions: Sequence[int]) -> CityStepResult:
        if len(actions) != len(self.positions):
            raise ValueError("actions length must equal agent count")
        actions_tuple = tuple(int(action) for action in actions)
        for action in actions_tuple:
            if action not in CITY_ACTIONS:
                raise ValueError(f"illegal city action {action}")

        targets = [
            (x + _MOVES[action][0], y + _MOVES[action][1])
            for (x, y), action in zip(self.positions, actions_tuple)
        ]
        target_counts = Counter(targets)
        occupancy = set(self.positions)
        new_positions: list[tuple[int, int]] = []
        moved: list[bool] = []
        blocked_reasons: list[str] = []
        collision_agents: list[int] = []

        for index, (position, target) in enumerate(zip(self.positions, targets)):
            if self._is_wall(target):
                new_positions.append(position)
                moved.append(False)
                blocked_reasons.append("wall")
                self.agent_blocked_counts[index] += 1
                continue
            if target_counts[target] > 1:
                new_positions.append(position)
                moved.append(False)
                blocked_reasons.append("collision")
                collision_agents.append(index)
                self.agent_blocked_counts[index] += 1
                self.agent_collision_counts[index] += 1
                continue
            if target in occupancy and target != position:
                new_positions.append(position)
                moved.append(False)
                blocked_reasons.append("occupied")
                self.agent_blocked_counts[index] += 1
                continue
            new_positions.append(target)
            moved.append(True)
            blocked_reasons.append("")

        self.positions = new_positions
        for index, position in enumerate(self.positions):
            self.visited[index].add(position)

        interaction_pairs: list[tuple[int, int]] = []
        for first in range(len(self.positions)):
            for second in range(first + 1, len(self.positions)):
                x1, y1 = self.positions[first]
                x2, y2 = self.positions[second]
                if max(abs(x1 - x2), abs(y1 - y2)) <= 2:
                    interaction_pairs.append((first, second))
                    self.interaction_pair_counts[(first, second)] += 1

        self.step_count += 1
        self.blocked_moves += sum(1 for value in moved if not value)
        self.collision_events += len(collision_agents)
        self.interaction_events += len(interaction_pairs)
        return CityStepResult(
            observations=tuple(self.observe_all()),
            actions=actions_tuple,
            positions=tuple(self.positions),
            moved=tuple(moved),
            blocked_reasons=tuple(blocked_reasons),
            collision_agents=tuple(collision_agents),
            interaction_pairs=tuple(interaction_pairs),
        )

    def render(self) -> tuple[str, ...]:
        rows: list[list[str]] = []
        for y in range(self.config.height):
            row: list[str] = []
            for x in range(self.config.width):
                cell = (x, y)
                if cell in self.walls:
                    row.append("#")
                elif cell in self.plaza:
                    row.append(".")
                else:
                    row.append(" ")
            rows.append(row)
        for index, (x, y) in enumerate(self.positions):
            rows[y][x] = str(index % 10)
        return tuple("".join(row) for row in rows)

    def stats(self) -> dict[str, Any]:
        return {
            "width": self.config.width,
            "height": self.config.height,
            "agent_count": self.config.agent_count,
            "open_cells": len(self.open_cells),
            "wall_cells": len(self.walls),
            "plaza_cells": len(self.plaza),
            "step_count": self.step_count,
            "blocked_moves": self.blocked_moves,
            "collision_events": self.collision_events,
            "interaction_events": self.interaction_events,
            "agent_blocked_counts": list(self.agent_blocked_counts),
            "agent_collision_counts": list(self.agent_collision_counts),
            "unique_cells_per_agent": [len(cells) for cells in self.visited],
            "union_unique_cells": len(set().union(*self.visited)) if self.visited else 0,
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": CITY_SCHEMA,
            "config": asdict(self.config),
            "rng": _rng_to_json(self._rng),
            "step_count": self.step_count,
            "blocked_moves": self.blocked_moves,
            "collision_events": self.collision_events,
            "interaction_events": self.interaction_events,
            "agent_blocked_counts": list(self.agent_blocked_counts),
            "agent_collision_counts": list(self.agent_collision_counts),
            "interaction_pair_counts": [
                [first, second, count]
                for (first, second), count in sorted(self.interaction_pair_counts.items())
            ],
            "walls": [_cell_key(cell) for cell in sorted(self.walls)],
            "plaza": [_cell_key(cell) for cell in sorted(self.plaza)],
            "border": [_cell_key(cell) for cell in sorted(self.border)],
            "positions": [_cell_key(position) for position in self.positions],
            "visited": [sorted(_cell_key(cell) for cell in cells) for cells in self.visited],
        }

    def restore(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != CITY_SCHEMA:
            raise ValueError("unsupported city snapshot schema")
        incoming = CityConfig(**snapshot["config"])
        if (
            incoming.width != self.config.width
            or incoming.height != self.config.height
            or incoming.agent_count != self.config.agent_count
        ):
            raise ValueError("city snapshot dimensions do not match the current city")
        self._rng = _rng_from_json(snapshot["rng"])
        self.step_count = int(snapshot["step_count"])
        self.blocked_moves = int(snapshot["blocked_moves"])
        self.collision_events = int(snapshot["collision_events"])
        self.interaction_events = int(snapshot["interaction_events"])
        self.agent_blocked_counts = [int(value) for value in snapshot["agent_blocked_counts"]]
        self.agent_collision_counts = [int(value) for value in snapshot["agent_collision_counts"]]
        self.interaction_pair_counts = Counter(
            {
                (int(first), int(second)): int(count)
                for first, second, count in snapshot["interaction_pair_counts"]
            }
        )
        self.walls = {tuple(int(value) for value in cell) for cell in snapshot["walls"]}
        self.plaza = {tuple(int(value) for value in cell) for cell in snapshot["plaza"]}
        self.border = {tuple(int(value) for value in cell) for cell in snapshot["border"]}
        self.positions = [tuple(int(value) for value in cell) for cell in snapshot["positions"]]
        self.visited = [
            {tuple(int(value) for value in cell) for cell in cells}
            for cells in snapshot["visited"]
        ]
        if len(self.positions) != self.config.agent_count:
            raise ValueError("snapshot positions do not match agent_count")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("snapshot contains duplicate agent positions")
        for position in self.positions:
            if self._is_wall(position):
                raise ValueError("snapshot contains an agent on a wall")
        self.open_cells = {
            (x, y)
            for y in range(1, self.config.height - 1)
            for x in range(1, self.config.width - 1)
            if (x, y) not in self.walls
        }


@dataclass(frozen=True)
class CityRuntimeConfig:
    seed: int = 12012
    width: int = 32
    height: int = 32
    agent_count: int = 8
    max_steps: int = 0
    delay: float = 0.05
    checkpoint_every: int = 500
    render: bool = True
    progress_every: int = 500
    clone_seed_stride: int = 1009


class CityTerminalRenderer:
    def __init__(self, ansi: bool = True) -> None:
        self.ansi = bool(ansi and sys.stdout.isatty())
        if self.ansi:
            os.system("")

    def render(self, runtime: "CityRuntime") -> None:
        if self.ansi:
            print("\x1b[2J\x1b[H", end="")
        print(self.frame(runtime))

    def frame(self, runtime: "CityRuntime") -> str:
        city_stats = runtime.city.stats()
        lines = [
            "NUCLEUS EXPLORATION CITY - shared, reward-free multi-agent loop",
            "=" * 78,
            f"step: {runtime.step_count:<7} agents: {runtime.config.agent_count:<3} "
            f"elapsed: {format_duration(runtime.elapsed_seconds)}   "
            f"speed: {runtime.config.delay:.3f}s/step",
            f"open cells: {city_stats['open_cells']}   interactions: "
            f"{city_stats['interaction_events']}   collisions: "
            f"{city_stats['collision_events']}   blocked: {city_stats['blocked_moves']}",
        ]
        if runtime.last_decisions:
            actions = " ".join(
                f"{index}:{CITY_ACTION_NAMES[decision.action]}"
                for index, decision in enumerate(runtime.last_decisions)
            )
            fallback = sum(1 for decision in runtime.last_decisions if decision.random_fallback)
            lines.append(f"actions: {actions}   fallback: {fallback}")
        divergence = runtime.weight_divergence()
        lines.append(
            "clone weight divergence: "
            f"input->hidden mean={divergence['input_hidden_mean']:.6f} "
            f"hidden->action mean={divergence['hidden_action_mean']:.6f}"
        )
        lines.append(f"checkpoint: {runtime.checkpoint_path}")
        if runtime.last_checkpoint_message:
            lines.append(f"checkpoint event: {runtime.last_checkpoint_message}")
        lines.append("Ctrl+C: save checkpoint and stop")
        lines.append("-" * 78)
        lines.extend(runtime.city.render())
        return "\n".join(lines)


class CityRuntime:
    """Owns the city, clones, chronological log, checkpoint, and display."""

    def __init__(
        self,
        config: CityRuntimeConfig,
        *,
        checkpoint_path: Path,
        log_path: Path,
        summary_path: Path | None = None,
        agent_config: AgentConfig | None = None,
        resume_path: Path | None = None,
        on_step: Callable[["CityRuntime"], None] | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path) if summary_path else self.checkpoint_path.parent / "city_summary.json"
        self.on_step = on_step
        self._stop = threading.Event()
        self._start_monotonic = time.monotonic()
        self.events: list[str] = []
        self.last_decisions: list[ActionDecision] = []
        self.last_record: dict[str, Any] | None = None
        self.last_checkpoint_message = ""
        self.checkpoints_saved = 0
        self.fallback_count = 0
        self.step_count = 0
        self.action_counts: Counter[str] = Counter()
        self.per_agent_action_counts: list[Counter[str]] = [
            Counter() for _ in range(config.agent_count)
        ]
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(3).hex()
        self._resumed_from: str | None = None
        self.initial_weights_identical = False

        if resume_path is not None:
            payload = json.loads(Path(resume_path).read_text(encoding="utf-8"))
            if payload.get("schema") != CITY_CHECKPOINT_SCHEMA:
                raise ValueError("unsupported city runtime checkpoint schema")
            city_payload = payload["city"]
            city_config = CityConfig(**city_payload["config"])
            self.city = City(city_config, seed=0)
            self.city.restore(city_payload)
            self.agents = [SpikingExplorer.from_state_dict(state) for state in payload["agents"]]
            if len(self.agents) != self.city.config.agent_count:
                raise ValueError("checkpoint agent count does not match city agent count")
            self.step_count = int(payload["step"])
            self.fallback_count = int(payload.get("fallback_count", 0))
            self.checkpoints_saved = int(payload.get("checkpoints_saved", 0))
            counters = payload.get("counters", {})
            self.action_counts = Counter({str(key): int(value) for key, value in counters.get("action_counts", {}).items()})
            self.per_agent_action_counts = [
                Counter({str(key): int(value) for key, value in counts.items()})
                for counts in counters.get("per_agent_action_counts", [])
            ][: len(self.agents)]
            while len(self.per_agent_action_counts) < len(self.agents):
                self.per_agent_action_counts.append(Counter())
            self._resumed_from = str(resume_path)
            self.initial_weights_identical = bool(payload.get("initial_weights_identical", False))
        else:
            city_config = CityConfig(
                width=config.width,
                height=config.height,
                agent_count=config.agent_count,
            )
            self.city = City(city_config, seed=config.seed)
            prototype = SpikingExplorer(agent_config or AgentConfig(), seed=config.seed)
            base_state = prototype.state_dict()
            self.agents = []
            for index in range(config.agent_count):
                clone = SpikingExplorer.from_state_dict(base_state)
                clone._rng.seed(config.seed + config.clone_seed_stride * (index + 1))
                self.agents.append(clone)
            self.initial_weights_identical = all(
                np.array_equal(self.agents[0].w_input_hidden, agent.w_input_hidden)
                and np.array_equal(self.agents[0].w_hidden_hidden, agent.w_hidden_hidden)
                and np.array_equal(self.agents[0].w_hidden_action, agent.w_hidden_action)
                for agent in self.agents[1:]
            )

        self.logger = JsonlLogger(self.log_path)
        self.renderer = CityTerminalRenderer(ansi=config.render)

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    def request_stop(self) -> None:
        self._stop.set()

    def _event(self, event: str, **fields: Any) -> None:
        record = {"ts_utc": utc_now(), "session_id": self.session_id, "event": event, **fields}
        self.logger.write(record)
        text = event
        if fields:
            text += " " + " ".join(f"{key}={value}" for key, value in fields.items())
        self.events.append(text)
        if len(self.events) > 8:
            self.events = self.events[-8:]

    def step_once(self) -> CityStepResult:
        observations = self.city.observe_all()
        decisions = [
            agent.act(observation.features, CITY_ACTIONS)
            for agent, observation in zip(self.agents, observations)
        ]
        result = self.city.step([decision.action for decision in decisions])
        self.step_count = self.city.step_count
        self.fallback_count += sum(1 for decision in decisions if decision.random_fallback)
        for index, decision in enumerate(decisions):
            name = CITY_ACTION_NAMES[decision.action]
            self.action_counts[name] += 1
            self.per_agent_action_counts[index][name] += 1
        self.last_decisions = decisions
        record = {
            "ts_utc": utc_now(),
            "session_id": self.session_id,
            "event": "city_step",
            "step": self.step_count,
            "positions": [list(position) for position in result.positions],
            "actions": [CITY_ACTION_NAMES[action] for action in result.actions],
            "moved": list(result.moved),
            "blocked_reasons": list(result.blocked_reasons),
            "collision_agents": list(result.collision_agents),
            "interaction_pairs": [list(pair) for pair in result.interaction_pairs],
            "fallback": [decision.random_fallback for decision in decisions],
            "hidden_spikes": [decision.hidden_spikes for decision in decisions],
            "action_spikes": [list(decision.action_spikes) for decision in decisions],
            "observation_summaries": [observation.summary for observation in observations],
            "interaction_events_total": self.city.interaction_events,
            "collision_events_total": self.city.collision_events,
            "blocked_moves_total": self.city.blocked_moves,
        }
        self.logger.write(record)
        self.last_record = record
        if self.on_step is not None:
            self.on_step(self)
        return result

    def checkpoint(self) -> dict[str, Any]:
        self.checkpoints_saved += 1
        payload = {
            "schema": CITY_CHECKPOINT_SCHEMA,
            "created_utc": utc_now(),
            "config": asdict(self.config),
            "city": self.city.snapshot(),
            "agents": [agent.state_dict() for agent in self.agents],
            "step": self.step_count,
            "fallback_count": self.fallback_count,
            "checkpoints_saved": self.checkpoints_saved,
            "counters": {
                "action_counts": dict(self.action_counts),
                "per_agent_action_counts": [dict(counts) for counts in self.per_agent_action_counts],
            },
            "initial_weights_identical": self.initial_weights_identical,
            "resumed_from": self._resumed_from,
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.checkpoint_path)
        self.last_checkpoint_message = f"saved at step {self.step_count}"
        self._event("checkpoint", step=self.step_count, path=str(self.checkpoint_path))
        return payload

    def weight_divergence(self) -> dict[str, float]:
        if len(self.agents) < 2:
            return {"input_hidden_mean": 0.0, "hidden_action_mean": 0.0, "max_pair_mean": 0.0}
        input_differences: list[float] = []
        action_differences: list[float] = []
        for first in range(len(self.agents)):
            for second in range(first + 1, len(self.agents)):
                input_differences.append(
                    float(np.abs(self.agents[first].w_input_hidden - self.agents[second].w_input_hidden).mean())
                )
                action_differences.append(
                    float(np.abs(self.agents[first].w_hidden_action - self.agents[second].w_hidden_action).mean())
                )
        return {
            "input_hidden_mean": float(np.mean(input_differences)),
            "hidden_action_mean": float(np.mean(action_differences)),
            "max_pair_mean": float(max(input_differences + action_differences)),
        }

    def summary(self) -> dict[str, Any]:
        city_stats = self.city.stats()
        unique_cells = [len(cells) for cells in self.city.visited]
        union_unique = len(set().union(*self.city.visited)) if self.city.visited else 0
        divergence = self.weight_divergence()
        pairwise = []
        for first in range(len(self.agents)):
            for second in range(first + 1, len(self.agents)):
                pairwise.append(
                    {
                        "pair": [first, second],
                        "input_hidden_mean_abs_diff": float(
                            np.abs(self.agents[first].w_input_hidden - self.agents[second].w_input_hidden).mean()
                        ),
                        "hidden_action_mean_abs_diff": float(
                            np.abs(self.agents[first].w_hidden_action - self.agents[second].w_hidden_action).mean()
                        ),
                    }
                )
        return {
            "session_id": self.session_id,
            "resumed_from": self._resumed_from,
            "steps": self.step_count,
            "agent_count": len(self.agents),
            "total_agent_steps": self.step_count * len(self.agents),
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "runtime_config": asdict(self.config),
            "agent_config": asdict(self.agents[0].config) if self.agents else None,
            "city": city_stats,
            "fallback_actions": self.fallback_count,
            "action_counts": dict(sorted(self.action_counts.items())),
            "per_agent_action_counts": [dict(sorted(counts.items())) for counts in self.per_agent_action_counts],
            "interaction_pair_counts": [
                {"pair": [first, second], "count": count}
                for (first, second), count in sorted(self.city.interaction_pair_counts.items())
            ],
            "unique_cells_per_agent": unique_cells,
            "union_unique_cells": union_unique,
            "coverage_ratio": union_unique / max(1, city_stats["open_cells"]),
            "final_positions": [list(position) for position in self.city.positions],
            "initial_weights_identical": self.initial_weights_identical,
            "weight_divergence": divergence,
            "pairwise_weight_divergence": pairwise,
            "weight_stats": [agent.weight_stats() for agent in self.agents],
            "checkpoints_saved": self.checkpoints_saved,
            "log_path": str(self.log_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path) if self.checkpoint_path.is_file() else None,
            "log_sha256": sha256_file(self.log_path) if self.log_path.is_file() else None,
            "note": (
                "Operational multi-agent counters only. No reward, score, penalty, "
                "correctness signal, or error signal is used. This is not evidence of "
                "learning, intelligence, cooperation, competition, or skill."
            ),
        }

    def run(self) -> dict[str, Any]:
        self._event(
            "session_start",
            city=self.city.stats(),
            agents=len(self.agents),
            seed=self.config.seed,
            delay=self.config.delay,
            checkpoint_every=self.config.checkpoint_every,
            resumed_from=self._resumed_from,
            step=self.step_count,
        )
        try:
            while not self._stop.is_set():
                if self.config.max_steps and self.step_count >= self.config.max_steps:
                    break
                self.step_once()
                if self.config.render:
                    self.renderer.render(self)
                elif self.config.progress_every and self.step_count % self.config.progress_every == 0:
                    print(
                        f"step={self.step_count} interactions={self.city.interaction_events} "
                        f"collisions={self.city.collision_events} blocked={self.city.blocked_moves} "
                        f"fallback={self.fallback_count}",
                        flush=True,
                    )
                if self.config.checkpoint_every and self.step_count % self.config.checkpoint_every == 0:
                    self.checkpoint()
                if self.config.delay:
                    time.sleep(self.config.delay)
        except KeyboardInterrupt:
            self._event("keyboard_interrupt", step=self.step_count)
        finally:
            self.checkpoint()
            self._event("session_end", step=self.step_count)
            self.logger.close()
        summary = self.summary()
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m exploration_lab.city",
        description=(
            "Shared multi-agent city with cloned reward-free spiking agents. "
            "Collisions and interactions are logged, never used as learning signals."
        ),
    )
    parser.add_argument("--agents", type=int, default=8, help="number of clones (default: 8)")
    parser.add_argument("--width", type=int, default=32, help="city width (default: 32)")
    parser.add_argument("--height", type=int, default=32, help="city height (default: 32)")
    parser.add_argument("--seed", type=int, default=12012, help="initial RNG seed (default: 12012)")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between steps; 0 = maximum speed")
    parser.add_argument("--max-steps", type=int, default=0, help="stop after N steps; 0 = run until Ctrl+C")
    parser.add_argument("--checkpoint-every", type=int, default=500, help="checkpoint interval in steps")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(".local_private/exploration/city/checkpoint.json"),
                        help="checkpoint path")
    parser.add_argument("--log", type=Path,
                        default=Path(".local_private/exploration/city/session.jsonl"),
                        help="chronological JSONL log")
    parser.add_argument("--summary", type=Path, default=None, help="session summary path")
    parser.add_argument("--resume", type=Path, default=None, help="restore a city checkpoint before running")
    parser.add_argument("--no-render", action="store_true", help="disable the live terminal display")
    parser.add_argument("--progress-every", type=int, default=500, help="headless progress interval")
    parser.add_argument("--hidden", type=int, default=64, help="hidden neuron count for a new run")
    parser.add_argument("--ticks-per-step", type=int, default=4, help="internal ticks per step")
    parser.add_argument("--learning-rate", type=float, default=0.0020, help="STDP potentiation rate")
    parser.add_argument("--depression-rate", type=float, default=0.0021, help="STDP depression rate")
    parser.add_argument("--temperature", type=float, default=0.6, help="softmax temperature")
    parser.add_argument("--weight-max", type=float, default=1.0, help="maximum plastic weight")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.agents <= 0:
        raise SystemExit("--agents must be positive")
    if args.max_steps < 0 or args.checkpoint_every < 0 or args.delay < 0 or args.progress_every < 0:
        raise SystemExit("step, checkpoint, delay, and progress values must be non-negative")

    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"checkpoint not found: {args.resume}")
        agent_config = None
    else:
        agent_config = AgentConfig(
            hidden_dim=args.hidden,
            ticks_per_step=args.ticks_per_step,
            lr_ltp=args.learning_rate,
            lr_ltd=args.depression_rate,
            temperature=args.temperature,
            weight_max=args.weight_max,
        )

    config = CityRuntimeConfig(
        seed=args.seed,
        width=args.width,
        height=args.height,
        agent_count=args.agents,
        max_steps=args.max_steps,
        delay=args.delay,
        checkpoint_every=args.checkpoint_every,
        render=not args.no_render,
        progress_every=args.progress_every,
    )
    runtime = CityRuntime(
        config,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        summary_path=args.summary,
        agent_config=agent_config,
        resume_path=args.resume,
    )
    print("Nucleus Exploration City")
    print(f"agents: {len(runtime.agents)}   city: {runtime.city.config.width}x{runtime.city.config.height}")
    print(f"log: {runtime.log_path}")
    print(f"checkpoint: {runtime.checkpoint_path}")
    if runtime._resumed_from:
        print(f"resumed: {runtime._resumed_from} at step {runtime.step_count}")
    print("No reward, score, penalty, correctness signal, or error signal is sent to the learning rule.")
    print("Ctrl+C saves a checkpoint and exits.")
    print()
    summary = runtime.run()
    print()
    print("CITY SESSION SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
