"""Shared multi-agent city for the reward-free exploration lab.

The city is a small grid with buildings, roads, a central plaza, and several
clones of :class:`exploration_lab.agent.SpikingExplorer`.  Each clone has its own
persistent LIF state and independent RNG stream, but all clones start from the
same weights.  They share one environment and move simultaneously.

Scope note: blocked moves, collisions, proximity events, and action counts are
*observed and logged only*.  They are not rewards, penalties, or correctness
labels and are never passed to the action-learning rule.  Action weights still
change only through local spike-timing-dependent plasticity inside each clone.

TASK-013 adds a separate local prediction head per clone.  It is trained by a
self-supervised next-observation prediction error, using only its own prediction
error and presynaptic activity.  Prediction error trains only that local forward
model; it never updates the action STDP pathway and is not a reward or a city
outcome.  In legacy mode the predictor has no path to action selection.

TASK-014 optionally feeds the signed feature-level prediction residual from the
previous transition into a dedicated positive/negative residual input pathway of
the same clone on the next perception-action cycle.  Legacy mode keeps TASK-013
behavior exactly; adaptive mode enables the pathway.  No clone shares residuals or
predictor weights, and no reward or action-correctness signal is introduced.  The
residual is never a direct action-probability bonus.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, replace
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

from .agent import (
    LEGACY_CHECKPOINT_SCHEMA,
    ActionDecision,
    AgentConfig,
    SpikingExplorer,
)
from .games import FEATURE_DIM, Observation
from .prediction import LocalPredictor, PredictionConfig, PredictionMetrics
from .residual import ResidualConfig, residual_summary
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
    prediction_learning_rate: float = 0.001
    prediction_weight_max: float = 2.0
    prediction_weight_decay: float = 1e-6
    prediction_window: int = 500
    residual_feedback: bool = False
    allow_legacy_migration: bool = False
    log_max_bytes: int = 16 * 1024 * 1024
    log_backup_count: int = 3


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
        mode = "adaptive residual feedback" if runtime.config.residual_feedback else "legacy (no residual feedback)"
        lines = [
            "NUCLEUS EXPLORATION CITY - shared, reward-free multi-agent loop",
            "=" * 78,
            f"step: {runtime.step_count:<7} agents: {len(runtime.agents):<3} "
            f"elapsed: {format_duration(runtime.elapsed_seconds)}   "
            f"speed: {runtime.config.delay:.3f}s/step",
            f"mode: {mode}   predictor: local next-observation",
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
        if runtime.predictors:
            recent_errors = [
                predictor.last_metrics.mae
                for predictor in runtime.predictors
                if predictor.last_metrics is not None
            ]
            recent_residuals = [
                float(np.abs(predictor.last_residual).mean())
                for predictor in runtime.predictors
                if predictor.last_residual is not None
            ]
            if recent_errors:
                lines.append(
                    "local prediction MAE: "
                    f"mean={float(np.mean(recent_errors)):.6f} "
                    f"min={min(recent_errors):.6f} max={max(recent_errors):.6f}"
                )
            if recent_residuals:
                lines.append(
                    "signed prediction residual: "
                    f"abs_mean={float(np.mean(recent_residuals)):.6f} "
                    f"max_abs={max(float(np.abs(p.last_residual).max()) for p in runtime.predictors if p.last_residual is not None):.6f}"
                )
        if runtime.config.residual_feedback:
            current = [
                agent.last_residual_current_abs_mean for agent in runtime.agents
            ]
            lines.append(
                "residual pathway: positive/negative -> hidden   "
                f"last current abs mean={float(np.mean(current)):.6f}"
            )
        else:
            lines.append("residual pathway: disabled (legacy)")
        lines.append(f"checkpoint: {runtime.checkpoint_path}")
        if runtime.last_checkpoint_message:
            lines.append(f"checkpoint event: {runtime.last_checkpoint_message}")
        lines.append("Ctrl+C: save checkpoint and stop")
        lines.append("-" * 78)
        lines.extend(runtime.city.render())
        return "\n".join(lines)


class CityRuntime:
    """Owns the city, clones, chronological log, checkpoint, and display."""

    _AUTHORITATIVE_FIELDS = (
        "seed",
        "width",
        "height",
        "agent_count",
        "clone_seed_stride",
        "prediction_learning_rate",
        "prediction_weight_max",
        "prediction_weight_decay",
        "prediction_window",
        "residual_feedback",
    )
    _OVERRIDE_FIELDS = (
        "max_steps",
        "delay",
        "render",
        "checkpoint_every",
        "progress_every",
        "log_max_bytes",
        "log_backup_count",
        "allow_legacy_migration",
    )

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
        self._validate_runtime_config(config)
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
        self.predictors: list[LocalPredictor] = []
        self.prediction_config: PredictionConfig | None = None
        self.prediction_window = max(1, int(config.prediction_window))
        self.prediction_initial_weights_identical = False
        self.prediction_steps = 0
        self.prediction_initial_steps = 0
        self.prediction_initial_mae_sum: list[float] = []
        self.prediction_mae_total: list[float] = []
        self.prediction_mse_total: list[float] = []
        self.prediction_recent_mae: list[deque[float]] = []
        self.prediction_migration_note: str | None = None
        resume_payload: dict[str, Any] | None = None

        if resume_path is not None:
            payload = json.loads(Path(resume_path).read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("city checkpoint must be a JSON object")
            resume_payload = payload
            if payload.get("schema") != CITY_CHECKPOINT_SCHEMA:
                raise ValueError("unsupported city runtime checkpoint schema")
            checkpoint_config = self._checkpoint_config(payload)
            self.config = self._merge_resume_config(checkpoint_config, config)
            self.prediction_window = max(1, int(self.config.prediction_window))
            city_payload = payload.get("city")
            if not isinstance(city_payload, dict):
                raise ValueError("checkpoint is missing the city snapshot")
            try:
                city_config = CityConfig(**city_payload["config"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid checkpoint city config: {exc}") from exc
            if (
                city_config.width != self.config.width
                or city_config.height != self.config.height
                or city_config.agent_count != self.config.agent_count
            ):
                raise ValueError(
                    "checkpoint city dimensions/agent count do not match the runtime config"
                )
            self.city = City(city_config, seed=0)
            self.city.restore(city_payload)
            agent_states = payload.get("agents")
            if not isinstance(agent_states, list):
                raise ValueError("checkpoint is missing the agent list")
            if len(agent_states) != self.city.config.agent_count:
                raise ValueError(
                    "checkpoint agent list length does not match city agent count"
                )
            if self.config.residual_feedback:
                for index, state in enumerate(agent_states):
                    if not isinstance(state, dict):
                        raise ValueError(f"agent checkpoint {index} is not an object")
                    if state.get("schema") == LEGACY_CHECKPOINT_SCHEMA:
                        raise ValueError(
                            "adaptive residual feedback requires agent checkpoint v2; "
                            f"agent {index} is legacy v1 and has no residual pathway. "
                            "Resume in legacy mode or start a new adaptive session."
                        )
            self.agents = [
                SpikingExplorer.from_state_dict(
                    state,
                    allow_legacy_migration=self.config.allow_legacy_migration,
                )
                for state in agent_states
            ]
            try:
                self.step_count = int(payload["step"])
                self.fallback_count = int(payload.get("fallback_count", 0))
                self.checkpoints_saved = int(payload.get("checkpoints_saved", 0))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid checkpoint counters: {exc}") from exc
            if self.step_count < 0:
                raise ValueError("checkpoint step count must be non-negative")
            if self.step_count != self.city.step_count:
                raise ValueError(
                    "checkpoint step count does not match the restored city step count"
                )
            if self.fallback_count < 0:
                raise ValueError("checkpoint fallback_count must be non-negative")
            if self.checkpoints_saved < 0:
                raise ValueError("checkpoint checkpoints_saved must be non-negative")
            self._restore_counters(payload, len(self.agents))
            self._resumed_from = str(resume_path)
            self.initial_weights_identical = bool(payload.get("initial_weights_identical", False))
        else:
            city_config = CityConfig(
                width=config.width,
                height=config.height,
                agent_count=config.agent_count,
            )
            self.city = City(city_config, seed=config.seed)
            base_agent_config = agent_config or AgentConfig()
            if base_agent_config.residual_enabled != config.residual_feedback:
                base_agent_config = replace(
                    base_agent_config,
                    residual_enabled=config.residual_feedback,
                )
            prototype = SpikingExplorer(base_agent_config, seed=config.seed)
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
                and np.array_equal(
                    self.agents[0].w_residual_pos_hidden,
                    agent.w_residual_pos_hidden,
                )
                and np.array_equal(
                    self.agents[0].w_residual_neg_hidden,
                    agent.w_residual_neg_hidden,
                )
                for agent in self.agents[1:]
            )

        self._validate_agent_configs()
        self._validate_runtime_config(self.config)
        self._setup_prediction(resume_payload)
        self.logger = JsonlLogger(
            self.log_path,
            max_bytes=self.config.log_max_bytes,
            backup_count=self.config.log_backup_count,
            session_id=self.session_id,
        )
        self.renderer = CityTerminalRenderer(ansi=self.config.render)

    @staticmethod
    def _validate_runtime_config(config: CityRuntimeConfig) -> None:
        if config.width < 12 or config.height < 12:
            raise ValueError("city must be at least 12x12")
        if config.agent_count <= 0:
            raise ValueError("agent_count must be positive")
        if config.max_steps < 0 or config.checkpoint_every < 0 or config.progress_every < 0:
            raise ValueError("step/checkpoint/progress values must be non-negative")
        if not np.isfinite(config.delay) or config.delay < 0.0:
            raise ValueError("delay must be non-negative and finite")
        if not np.isfinite(config.prediction_learning_rate) or config.prediction_learning_rate < 0.0:
            raise ValueError("prediction_learning_rate must be non-negative and finite")
        if not np.isfinite(config.prediction_weight_max) or config.prediction_weight_max <= 0.0:
            raise ValueError("prediction_weight_max must be positive and finite")
        if not np.isfinite(config.prediction_weight_decay) or config.prediction_weight_decay < 0.0:
            raise ValueError("prediction_weight_decay must be non-negative and finite")
        if config.prediction_window <= 0:
            raise ValueError("prediction_window must be positive")
        if config.log_max_bytes < 0 or config.log_backup_count < 0:
            raise ValueError("log rotation values must be non-negative")
        if config.log_max_bytes > 0 and config.log_backup_count < 1:
            raise ValueError("log rotation requires log_backup_count >= 1")

    @staticmethod
    def _checkpoint_config(payload: dict[str, Any]) -> CityRuntimeConfig:
        if not isinstance(payload, dict):
            raise ValueError("city checkpoint must be a JSON object")
        data = payload.get("config")
        if not isinstance(data, dict):
            raise ValueError("checkpoint is missing the runtime configuration")
        try:
            return CityRuntimeConfig(**data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"checkpoint runtime configuration is incompatible: {exc}") from exc

    def _merge_resume_config(
        self,
        checkpoint_config: CityRuntimeConfig,
        requested: CityRuntimeConfig,
    ) -> CityRuntimeConfig:
        mismatches: list[str] = []
        for field in self._AUTHORITATIVE_FIELDS:
            requested_value = getattr(requested, field)
            saved_value = getattr(checkpoint_config, field)
            if requested_value != saved_value:
                mismatches.append(
                    f"{field}: requested={requested_value!r} checkpoint={saved_value!r}"
                )
        if mismatches:
            raise ValueError(
                "resume configuration mismatch; checkpoint architecture is authoritative: "
                + "; ".join(mismatches)
                + ". Only display/session controls may be overridden: "
                + ", ".join(self._OVERRIDE_FIELDS)
            )
        return replace(
            checkpoint_config,
            max_steps=requested.max_steps,
            delay=requested.delay,
            render=requested.render,
            checkpoint_every=requested.checkpoint_every,
            progress_every=requested.progress_every,
            log_max_bytes=requested.log_max_bytes,
            log_backup_count=requested.log_backup_count,
            allow_legacy_migration=requested.allow_legacy_migration,
        )

    def _restore_counters(self, payload: dict[str, Any], count: int) -> None:
        counters = payload.get("counters", {})
        if counters is None:
            counters = {}
        if not isinstance(counters, dict):
            raise ValueError("checkpoint counters must be an object")
        action_counts = counters.get("action_counts", {})
        if not isinstance(action_counts, dict):
            raise ValueError("checkpoint action_counts must be an object")
        self.action_counts = Counter()
        for key, value in action_counts.items():
            number = int(value)
            if number < 0:
                raise ValueError("checkpoint action_counts must be non-negative")
            self.action_counts[str(key)] = number
        per_agent = counters.get("per_agent_action_counts")
        if per_agent is None:
            self.per_agent_action_counts = [Counter() for _ in range(count)]
            return
        if not isinstance(per_agent, list) or len(per_agent) != count:
            raise ValueError(
                "checkpoint per_agent_action_counts must contain one object per agent"
            )
        restored: list[Counter[str]] = []
        for index, counts in enumerate(per_agent):
            if not isinstance(counts, dict):
                raise ValueError(
                    f"checkpoint per_agent_action_counts[{index}] must be an object"
                )
            counter: Counter[str] = Counter()
            for key, value in counts.items():
                number = int(value)
                if number < 0:
                    raise ValueError("checkpoint per-agent action counts must be non-negative")
                counter[str(key)] = number
            restored.append(counter)
        self.per_agent_action_counts = restored

    def _validate_agent_configs(self) -> None:
        if not self.agents:
            raise ValueError("city runtime requires at least one agent")
        if len(self.agents) != self.city.config.agent_count:
            raise ValueError("agent count does not match the city configuration")
        base = self.agents[0].config
        for index, agent in enumerate(self.agents):
            if agent.config != base:
                raise ValueError(
                    f"agent {index} configuration differs from agent 0; "
                    "refusing to resume a heterogeneous city"
                )
        if base.residual_enabled != self.config.residual_feedback:
            raise ValueError(
                "residual feedback mode does not match the agent configuration: "
                f"runtime={self.config.residual_feedback} agent={base.residual_enabled}"
            )

    def _expected_prediction_config(self) -> PredictionConfig:
        base = self.agents[0].config
        return PredictionConfig(
            hidden_dim=base.hidden_dim,
            action_dim=base.action_dim,
            input_dim=FEATURE_DIM,
            learning_rate=self.config.prediction_learning_rate,
            weight_max=self.config.prediction_weight_max,
            weight_decay=self.config.prediction_weight_decay,
        )

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

    def _new_predictors(self) -> list[LocalPredictor]:
        if not self.agents:
            return []
        config = self._expected_prediction_config()
        prototype = LocalPredictor(config, seed=0)
        state = prototype.state_dict()
        return [LocalPredictor.from_state_dict(state) for _ in self.agents]

    def _prediction_weights_identical(self) -> bool:
        if len(self.predictors) < 2:
            return True
        first = self.predictors[0]
        return all(
            np.array_equal(first.weights, predictor.weights)
            and np.array_equal(first.bias, predictor.bias)
            for predictor in self.predictors[1:]
        )

    def _setup_prediction(self, resume_payload: dict[str, Any] | None) -> None:
        count = len(self.agents)
        self.prediction_initial_mae_sum = [0.0 for _ in range(count)]
        self.prediction_mae_total = [0.0 for _ in range(count)]
        self.prediction_mse_total = [0.0 for _ in range(count)]
        self.prediction_recent_mae = [
            deque(maxlen=self.prediction_window) for _ in range(count)
        ]
        expected = self._expected_prediction_config()
        if resume_payload is not None:
            predictors_payload = resume_payload.get("predictors")
            if predictors_payload is None:
                if not self.config.allow_legacy_migration:
                    raise ValueError(
                        "checkpoint has no predictor list; pass "
                        "allow_legacy_migration=True to initialize predictors "
                        "explicitly without discarding the agent state"
                    )
                self.predictors = self._new_predictors()
                self.prediction_migration_note = (
                    "checkpoint had no predictor list; new local predictors were "
                    "initialized explicitly because allow_legacy_migration=True"
                )
            else:
                if not isinstance(predictors_payload, list):
                    raise ValueError("checkpoint predictors must be a list")
                if len(predictors_payload) != count:
                    raise ValueError(
                        "checkpoint predictor count does not match agent count: "
                        f"{len(predictors_payload)} != {count}"
                    )
                self.predictors = []
                for index, state in enumerate(predictors_payload):
                    try:
                        self.predictors.append(LocalPredictor.from_state_dict(state))
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError(f"predictor {index} is invalid: {exc}") from exc
            for index, predictor in enumerate(self.predictors):
                if predictor.config != expected:
                    raise ValueError(
                        f"predictor {index} config does not match checkpoint runtime "
                        f"config: {predictor.config} != {expected}"
                    )
        else:
            self.predictors = self._new_predictors()
        self.prediction_config = self.predictors[0].config if self.predictors else None

        metrics_present = bool(
            resume_payload is not None and "prediction_metrics" in resume_payload
        )
        if (
            resume_payload is not None
            and resume_payload.get("predictors") is not None
            and not metrics_present
        ):
            raise ValueError(
                "checkpoint has predictors but no prediction_metrics; refusing to "
                "continue with incomplete predictor statistics"
            )
        metrics = (resume_payload or {}).get("prediction_metrics", {})
        if metrics is None:
            metrics = {}
        if not isinstance(metrics, dict):
            raise ValueError("checkpoint prediction_metrics must be an object")
        if metrics_present:
            self._restore_prediction_metrics(metrics, count)
        if resume_payload is not None:
            note = resume_payload.get("prediction_migration_note")
            if note is not None and not isinstance(note, str):
                raise ValueError("checkpoint prediction_migration_note must be a string or null")
            if self.prediction_migration_note is None:
                self.prediction_migration_note = note
        self.prediction_initial_weights_identical = self._prediction_weights_identical()

    def _restore_prediction_metrics(self, metrics: dict[str, Any], count: int) -> None:
        try:
            steps = int(metrics.get("steps", 0))
            initial_steps = int(metrics.get("initial_steps", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction step counters: {exc}") from exc
        if steps < 0 or initial_steps < 0:
            raise ValueError("prediction step counters must be non-negative")
        if steps != self.step_count:
            raise ValueError(
                "prediction step counter does not match the restored city step count: "
                f"{steps} != {self.step_count}"
            )
        if initial_steps > steps:
            raise ValueError("prediction initial_steps cannot exceed prediction steps")
        if initial_steps > self.prediction_window:
            raise ValueError(
                "prediction initial_steps cannot exceed the restored prediction window"
            )
        self.prediction_steps = steps
        self.prediction_initial_steps = initial_steps

        def numeric_list(key: str) -> list[float]:
            values = metrics.get(key, [])
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(
                    f"prediction {key} must contain one numeric value per agent"
                )
            result: list[float] = []
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"prediction {key} contains a non-numeric value") from exc
                if not np.isfinite(number) or number < 0.0:
                    raise ValueError(
                        f"prediction {key} must contain finite non-negative values"
                    )
                result.append(number)
            return result

        self.prediction_initial_mae_sum = numeric_list("initial_mae_sum")
        self.prediction_mae_total = numeric_list("mae_total")
        self.prediction_mse_total = numeric_list("mse_total")
        recent = metrics.get("recent_mae", [])
        if not isinstance(recent, list) or len(recent) != count:
            raise ValueError("prediction recent_mae must contain one list per agent")
        for index, values in enumerate(recent):
            if not isinstance(values, list) or len(values) > self.prediction_window:
                raise ValueError(
                    f"prediction recent_mae[{index}] is not a bounded list"
                )
            restored: list[float] = []
            for value in values:
                try:
                    number = float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"prediction recent_mae[{index}] contains a non-numeric value"
                    ) from exc
                if not np.isfinite(number) or number < 0.0:
                    raise ValueError(
                        f"prediction recent_mae[{index}] must be finite and non-negative"
                    )
                restored.append(number)
            self.prediction_recent_mae[index].extend(restored)
        for index, predictor in enumerate(self.predictors):
            if predictor.steps != self.prediction_steps:
                raise ValueError(
                    f"predictor {index} step counter does not match prediction_metrics"
                )

    def _record_prediction_metrics(self, metrics: Sequence[PredictionMetrics]) -> None:
        if not metrics:
            return
        self.prediction_steps += 1
        for index, metric in enumerate(metrics[: len(self.prediction_mae_total)]):
            self.prediction_mae_total[index] += metric.mae
            self.prediction_mse_total[index] += metric.mse
            self.prediction_recent_mae[index].append(metric.mae)
            if self.prediction_initial_steps < self.prediction_window:
                self.prediction_initial_mae_sum[index] += metric.mae
        if self.prediction_initial_steps < self.prediction_window:
            self.prediction_initial_steps += 1

    def prediction_summary(self) -> dict[str, Any]:
        if not self.predictors:
            return {"enabled": False, "note": "No prediction head is active."}
        per_agent: list[dict[str, Any]] = []
        for index, predictor in enumerate(self.predictors):
            steps = self.prediction_steps
            mean_mae = self.prediction_mae_total[index] / steps if steps else None
            mean_mse = self.prediction_mse_total[index] / steps if steps else None
            initial_mae = (
                self.prediction_initial_mae_sum[index] / self.prediction_initial_steps
                if self.prediction_initial_steps
                else None
            )
            recent = self.prediction_recent_mae[index]
            final_mae = float(np.mean(recent)) if recent else None
            change = (
                final_mae - initial_mae
                if final_mae is not None and initial_mae is not None
                else None
            )
            reduction = (
                (initial_mae - final_mae) / initial_mae
                if initial_mae is not None and initial_mae > 0.0 and final_mae is not None
                else None
            )
            per_agent.append(
                {
                    "agent": index,
                    "mae_mean": mean_mae,
                    "mse_mean": mean_mse,
                    "initial_mae": initial_mae,
                    "final_mae": final_mae,
                    "mae_change": change,
                    "mae_reduction_ratio": reduction,
                    "prediction_updates": predictor.total_updates,
                    "last_residual": residual_summary(
                        predictor.last_residual,
                        ResidualConfig(input_dim=FEATURE_DIM),
                    ),
                    "weight_stats": predictor.weight_stats(),
                }
            )

        def aggregate(key: str) -> float | None:
            values = [entry[key] for entry in per_agent if entry[key] is not None]
            return float(np.mean(values)) if values else None

        mean_initial = aggregate("initial_mae")
        mean_final = aggregate("final_mae")
        return {
            "enabled": True,
            "kind": "self-supervised local next-observation prediction",
            "config": asdict(self.prediction_config) if self.prediction_config else None,
            "steps": self.prediction_steps,
            "initial_window": self.prediction_window,
            "initial_steps": self.prediction_initial_steps,
            "per_agent": per_agent,
            "mean_mae": aggregate("mae_mean"),
            "mean_mse": aggregate("mse_mean"),
            "initial_mae": mean_initial,
            "final_mae": mean_final,
            "mae_change": (
                mean_final - mean_initial
                if mean_final is not None and mean_initial is not None
                else None
            ),
            "mae_reduction_ratio": (
                (mean_initial - mean_final) / mean_initial
                if mean_initial is not None and mean_initial > 0.0 and mean_final is not None
                else None
            ),
            "prediction_updates": [predictor.total_updates for predictor in self.predictors],
            "weight_stats": [predictor.weight_stats() for predictor in self.predictors],
            "initial_weights_identical": self.prediction_initial_weights_identical,
            "current_weights_identical": self._prediction_weights_identical(),
            "migration_note": self.prediction_migration_note,
            "initial_window_note": (
                "The initial window is the first N steps. In the predictive condition "
                "it already contains predictor updates; use the fixed-predictor control "
                "for a no-learning baseline."
            ),
            "note": (
                "Self-supervised local next-observation prediction error only. "
                "In legacy mode it never selects actions or updates action STDP. "
                "In adaptive mode its signed residual enters the same clone's "
                "separate fixed pathway on the next step; it still never updates "
                "action STDP and is not a reward, score, penalty, or city outcome."
            ),
        }

    def step_once(self) -> CityStepResult:
        if len(self.predictors) != len(self.agents):
            raise RuntimeError(
                "predictor count does not match agent count: "
                f"{len(self.predictors)} != {len(self.agents)}"
            )
        observations = self.city.observe_all()
        if len(observations) != len(self.agents):
            raise RuntimeError("city observation count does not match agent count")
        decisions = [
            agent.act(observation.features, CITY_ACTIONS)
            for agent, observation in zip(self.agents, observations)
        ]
        for predictor, observation, agent, decision in zip(
            self.predictors, observations, self.agents, decisions
        ):
            predictor.predict(
                observation.features,
                agent.last_hidden_activity,
                decision.action,
            )
        result = self.city.step([decision.action for decision in decisions])
        prediction_metrics = [
            predictor.observe_next(observation.features)
            for predictor, observation in zip(self.predictors, result.observations)
        ]
        if self.config.residual_feedback:
            for index, (agent, predictor) in enumerate(zip(self.agents, self.predictors)):
                if predictor.last_residual is None:
                    raise RuntimeError(
                        f"predictor {index} did not produce a residual after the transition"
                    )
                agent.set_prediction_residual(predictor.last_residual)
        self._record_prediction_metrics(prediction_metrics)
        self.step_count = self.city.step_count
        self.fallback_count += sum(1 for decision in decisions if decision.random_fallback)
        for index, decision in enumerate(decisions):
            name = CITY_ACTION_NAMES[decision.action]
            self.action_counts[name] += 1
            self.per_agent_action_counts[index][name] += 1
        self.last_decisions = decisions
        residual_inputs = [
            residual_summary(agent.last_residual_input, ResidualConfig(input_dim=FEATURE_DIM))
            for agent in self.agents
        ]
        prediction_residuals = [
            residual_summary(predictor.last_residual, ResidualConfig(input_dim=FEATURE_DIM))
            for predictor in self.predictors
        ]
        record = {
            "ts_utc": utc_now(),
            "session_id": self.session_id,
            "event": "city_step",
            "step": self.step_count,
            "residual_feedback": self.config.residual_feedback,
            "positions": [list(position) for position in result.positions],
            "actions": [CITY_ACTION_NAMES[action] for action in result.actions],
            "moved": list(result.moved),
            "blocked_reasons": list(result.blocked_reasons),
            "collision_agents": list(result.collision_agents),
            "interaction_pairs": [list(pair) for pair in result.interaction_pairs],
            "fallback": [decision.random_fallback for decision in decisions],
            "hidden_spikes": [decision.hidden_spikes for decision in decisions],
            "action_spikes": [list(decision.action_spikes) for decision in decisions],
            "prediction_mae": [metric.mae for metric in prediction_metrics],
            "prediction_mse": [metric.mse for metric in prediction_metrics],
            "prediction_updates": [metric.prediction_updates for metric in prediction_metrics],
            "prediction_residual_abs_mean": [
                summary["abs_mean"] for summary in prediction_residuals
            ],
            "prediction_residual_signed_mean": [
                summary["signed_mean"] for summary in prediction_residuals
            ],
            "prediction_residual_positive_mean": [
                summary["positive_mean"] for summary in prediction_residuals
            ],
            "prediction_residual_negative_mean": [
                summary["negative_mean"] for summary in prediction_residuals
            ],
            "prediction_residual_max_abs": [
                summary["max_abs"] for summary in prediction_residuals
            ],
            "residual_input_abs_mean": [
                summary["abs_mean"] for summary in residual_inputs
            ],
            "residual_input_signed_mean": [
                summary["signed_mean"] for summary in residual_inputs
            ],
            "residual_input_max_abs": [
                summary["max_abs"] for summary in residual_inputs
            ],
            "residual_current_abs_mean": [
                float(agent.last_residual_current_abs_mean) for agent in self.agents
            ],
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
            "agent_config": asdict(self.agents[0].config) if self.agents else None,
            "prediction_config": (
                asdict(self.prediction_config) if self.prediction_config else None
            ),
            "prediction_migration_note": self.prediction_migration_note,
            "city": self.city.snapshot(),
            "agents": [agent.state_dict() for agent in self.agents],
            "predictors": [predictor.state_dict() for predictor in self.predictors],
            "prediction_metrics": {
                "steps": self.prediction_steps,
                "initial_steps": self.prediction_initial_steps,
                "initial_mae_sum": self.prediction_initial_mae_sum,
                "mae_total": self.prediction_mae_total,
                "mse_total": self.prediction_mse_total,
                "recent_mae": [list(values) for values in self.prediction_recent_mae],
            },
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
            return {
                "input_hidden_mean": 0.0,
                "hidden_action_mean": 0.0,
                "residual_pos_hidden_mean": 0.0,
                "residual_neg_hidden_mean": 0.0,
                "max_pair_mean": 0.0,
            }
        input_differences: list[float] = []
        action_differences: list[float] = []
        residual_pos_differences: list[float] = []
        residual_neg_differences: list[float] = []
        for first in range(len(self.agents)):
            for second in range(first + 1, len(self.agents)):
                input_differences.append(
                    float(np.abs(self.agents[first].w_input_hidden - self.agents[second].w_input_hidden).mean())
                )
                action_differences.append(
                    float(np.abs(self.agents[first].w_hidden_action - self.agents[second].w_hidden_action).mean())
                )
                residual_pos_differences.append(
                    float(
                        np.abs(
                            self.agents[first].w_residual_pos_hidden
                            - self.agents[second].w_residual_pos_hidden
                        ).mean()
                    )
                )
                residual_neg_differences.append(
                    float(
                        np.abs(
                            self.agents[first].w_residual_neg_hidden
                            - self.agents[second].w_residual_neg_hidden
                        ).mean()
                    )
                )
        return {
            "input_hidden_mean": float(np.mean(input_differences)),
            "hidden_action_mean": float(np.mean(action_differences)),
            "residual_pos_hidden_mean": float(np.mean(residual_pos_differences)),
            "residual_neg_hidden_mean": float(np.mean(residual_neg_differences)),
            "max_pair_mean": float(
                max(
                    input_differences
                    + action_differences
                    + residual_pos_differences
                    + residual_neg_differences
                )
            ),
        }

    def residual_feedback_summary(self) -> dict[str, Any]:
        per_agent: list[dict[str, Any]] = []
        for index, agent in enumerate(self.agents):
            stats = agent.weight_stats()
            per_agent.append(
                {
                    "agent": index,
                    "enabled": bool(agent.config.residual_enabled),
                    "pending_residual": residual_summary(
                        agent.pending_residual,
                        ResidualConfig(input_dim=FEATURE_DIM),
                    ),
                    "last_residual_input": residual_summary(
                        agent.last_residual_input,
                        ResidualConfig(input_dim=FEATURE_DIM),
                    ),
                    "last_residual_current_abs_mean": float(
                        agent.last_residual_current_abs_mean
                    ),
                    "residual_pos_abs_mean": stats["residual_pos_abs_mean"],
                    "residual_neg_abs_mean": stats["residual_neg_abs_mean"],
                }
            )
        pending = [
            entry["pending_residual"]["abs_mean"]
            for entry in per_agent
            if entry["enabled"]
        ]
        used = [
            entry["last_residual_input"]["abs_mean"]
            for entry in per_agent
            if entry["enabled"]
        ]
        current = [
            entry["last_residual_current_abs_mean"]
            for entry in per_agent
            if entry["enabled"]
        ]
        return {
            "enabled": bool(self.config.residual_feedback),
            "mode": "adaptive" if self.config.residual_feedback else "legacy",
            "channels": ["positive", "negative"],
            "projection": "separate fixed non-plastic residual matrices into hidden neurons",
            "timing": (
                "residual from the previous transition is consumed once at the next "
                "act(); no residual is delivered before that transition"
            ),
            "per_agent": per_agent,
            "pending_residual_abs_mean": float(np.mean(pending)) if pending else 0.0,
            "last_residual_input_abs_mean": float(np.mean(used)) if used else 0.0,
            "last_residual_current_abs_mean": float(np.mean(current)) if current else 0.0,
            "note": (
                "Signed feature-level prediction residual only. It is not a reward, "
                "score, penalty, correctness label, target action, or action bonus."
            ),
        }

    def _log_segment_info(self) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for path in self.logger.segment_paths():
            if not path.is_file():
                continue
            segments.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        return segments

    def close(self) -> None:
        """Close the append-only log handle without saving a checkpoint."""
        self.logger.close()

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
                        "residual_pos_hidden_mean_abs_diff": float(
                            np.abs(
                                self.agents[first].w_residual_pos_hidden
                                - self.agents[second].w_residual_pos_hidden
                            ).mean()
                        ),
                        "residual_neg_hidden_mean_abs_diff": float(
                            np.abs(
                                self.agents[first].w_residual_neg_hidden
                                - self.agents[second].w_residual_neg_hidden
                            ).mean()
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
            "mode": "adaptive" if self.config.residual_feedback else "legacy",
            "residual_feedback": bool(self.config.residual_feedback),
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
            "prediction": self.prediction_summary(),
            "residual": self.residual_feedback_summary(),
            "checkpoints_saved": self.checkpoints_saved,
            "log_path": str(self.log_path),
            "log_segments": self._log_segment_info(),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path) if self.checkpoint_path.is_file() else None,
            "log_sha256": sha256_file(self.log_path) if self.log_path.is_file() else None,
            "note": (
                "Action STDP uses no reward, score, penalty, correctness signal, or "
                "city outcome. A separate local predictor is trained by self-supervised "
                "next-observation prediction error. In adaptive mode its signed residual "
                "enters the same clone's separate residual pathway on the next step; "
                "in legacy mode that pathway is disabled. This is not evidence of "
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
            prediction_learning_rate=self.config.prediction_learning_rate,
            prediction_window=self.prediction_window,
            residual_feedback=self.config.residual_feedback,
            log_max_bytes=self.config.log_max_bytes,
            log_backup_count=self.config.log_backup_count,
            allow_legacy_migration=self.config.allow_legacy_migration,
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
                    recent_errors = [
                        predictor.last_metrics.mae
                        for predictor in self.predictors
                        if predictor.last_metrics is not None
                    ]
                    recent_residuals = [
                        float(np.abs(predictor.last_residual).mean())
                        for predictor in self.predictors
                        if predictor.last_residual is not None
                    ]
                    prediction_text = (
                        f" pred_mae={float(np.mean(recent_errors)):.6f}"
                        if recent_errors
                        else ""
                    )
                    residual_text = (
                        f" residual_abs={float(np.mean(recent_residuals)):.6f}"
                        if recent_residuals
                        else ""
                    )
                    print(
                        f"step={self.step_count} mode="
                        f"{'adaptive' if self.config.residual_feedback else 'legacy'} "
                        f"interactions={self.city.interaction_events} "
                        f"collisions={self.city.collision_events} blocked={self.city.blocked_moves} "
                        f"fallback={self.fallback_count}{prediction_text}{residual_text}",
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
            "Shared multi-agent city with cloned spiking agents. Action STDP is "
            "reward-free; a separate local predictor learns from self-supervised "
            "next-observation prediction error. Adaptive mode optionally feeds the "
            "signed residual into a separate internal pathway."
        ),
    )
    parser.add_argument("--agents", type=int, default=argparse.SUPPRESS,
                        help="number of clones for a new run (default: 8)")
    parser.add_argument("--width", type=int, default=argparse.SUPPRESS,
                        help="city width for a new run (default: 32)")
    parser.add_argument("--height", type=int, default=argparse.SUPPRESS,
                        help="city height for a new run (default: 32)")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS,
                        help="initial RNG seed for a new run (default: 12012)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="seconds between steps; 0 = maximum speed")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stop after N steps; 0 = run until Ctrl+C")
    parser.add_argument("--checkpoint-every", type=int, default=500,
                        help="checkpoint interval in steps")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(".local_private/exploration/city/checkpoint.json"),
                        help="checkpoint path")
    parser.add_argument("--log", type=Path,
                        default=Path(".local_private/exploration/city/session.jsonl"),
                        help="chronological JSONL log")
    parser.add_argument("--summary", type=Path, default=None, help="session summary path")
    parser.add_argument("--resume", type=Path, default=None,
                        help="restore a city checkpoint before running")
    parser.add_argument("--no-render", action="store_true", help="disable the live terminal display")
    parser.add_argument("--progress-every", type=int, default=500,
                        help="headless progress interval")
    parser.add_argument("--hidden", type=int, default=argparse.SUPPRESS,
                        help="hidden neuron count for a new run (default: 64)")
    parser.add_argument("--ticks-per-step", type=int, default=argparse.SUPPRESS,
                        help="internal ticks per step for a new run (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=argparse.SUPPRESS,
                        help="STDP potentiation rate for a new run (default: 0.0020)")
    parser.add_argument("--depression-rate", type=float, default=argparse.SUPPRESS,
                        help="STDP depression rate for a new run (default: 0.0021)")
    parser.add_argument("--temperature", type=float, default=argparse.SUPPRESS,
                        help="softmax temperature for a new run (default: 0.6)")
    parser.add_argument("--weight-max", type=float, default=argparse.SUPPRESS,
                        help="maximum plastic weight for a new run (default: 1.0)")
    parser.add_argument("--prediction-learning-rate", type=float, default=argparse.SUPPRESS,
                        help="local prediction-head learning rate for a new run; "
                             "0 fixes the predictor (default: 0.001)")
    parser.add_argument("--prediction-weight-max", type=float, default=argparse.SUPPRESS,
                        help="maximum absolute prediction weight for a new run (default: 2.0)")
    parser.add_argument("--prediction-weight-decay", type=float, default=argparse.SUPPRESS,
                        help="prediction weight decay per update for a new run (default: 1e-6)")
    parser.add_argument("--prediction-window", type=int, default=argparse.SUPPRESS,
                        help="initial/final MAE window in steps for a new run (default: 500)")
    parser.add_argument(
        "--residual-feedback",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help=(
            "enable adaptive signed prediction-residual feedback into the agent's "
            "separate internal pathway (default: disabled legacy mode)"
        ),
    )
    parser.add_argument(
        "--allow-legacy-migration",
        action="store_true",
        help=(
            "explicitly allow legacy v1 agent checkpoints to initialize zero "
            "residual weights, or old checkpoints without a predictor list to "
            "initialize predictors; never silently replaces learned state"
        ),
    )
    parser.add_argument(
        "--log-max-bytes",
        type=int,
        default=16 * 1024 * 1024,
        help="rotate JSONL logs after this many bytes; 0 disables rotation",
    )
    parser.add_argument(
        "--log-backup-count",
        type=int,
        default=3,
        help="number of rotated JSONL backups to retain",
    )
    return parser


def _arg(args: argparse.Namespace, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"checkpoint not found: {args.resume}")
        try:
            payload = json.loads(args.resume.read_text(encoding="utf-8"))
            checkpoint_config = CityRuntime._checkpoint_config(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"cannot read checkpoint configuration: {exc}") from exc

        explicit_architecture = {
            "agents": "agent_count",
            "width": "width",
            "height": "height",
            "seed": "seed",
            "prediction_learning_rate": "prediction_learning_rate",
            "prediction_weight_max": "prediction_weight_max",
            "prediction_weight_decay": "prediction_weight_decay",
            "prediction_window": "prediction_window",
            "residual_feedback": "residual_feedback",
        }
        for flag, field in explicit_architecture.items():
            if hasattr(args, flag):
                requested = getattr(args, flag)
                saved = getattr(checkpoint_config, field)
                if requested != saved:
                    raise SystemExit(
                        f"--{flag.replace('_', '-')}={requested!r} conflicts with "
                        f"checkpoint {field}={saved!r}; checkpoint architecture is "
                        "authoritative"
                    )
        for flag in (
            "hidden",
            "ticks_per_step",
            "learning_rate",
            "depression_rate",
            "temperature",
            "weight_max",
        ):
            if hasattr(args, flag):
                raise SystemExit(
                    f"--{flag.replace('_', '-')} cannot override agent architecture "
                    "when resuming; the checkpoint is authoritative"
                )
        config = replace(
            checkpoint_config,
            max_steps=args.max_steps,
            delay=args.delay,
            render=not args.no_render,
            checkpoint_every=args.checkpoint_every,
            progress_every=args.progress_every,
            log_max_bytes=args.log_max_bytes,
            log_backup_count=args.log_backup_count,
            allow_legacy_migration=args.allow_legacy_migration,
        )
        agent_config = None
    else:
        config = CityRuntimeConfig(
            seed=_arg(args, "seed", 12012),
            width=_arg(args, "width", 32),
            height=_arg(args, "height", 32),
            agent_count=_arg(args, "agents", 8),
            max_steps=args.max_steps,
            delay=args.delay,
            checkpoint_every=args.checkpoint_every,
            render=not args.no_render,
            progress_every=args.progress_every,
            prediction_learning_rate=_arg(args, "prediction_learning_rate", 0.001),
            prediction_weight_max=_arg(args, "prediction_weight_max", 2.0),
            prediction_weight_decay=_arg(args, "prediction_weight_decay", 1e-6),
            prediction_window=_arg(args, "prediction_window", 500),
            residual_feedback=_arg(args, "residual_feedback", False),
            allow_legacy_migration=args.allow_legacy_migration,
            log_max_bytes=args.log_max_bytes,
            log_backup_count=args.log_backup_count,
        )
        agent_config = AgentConfig(
            hidden_dim=_arg(args, "hidden", 64),
            ticks_per_step=_arg(args, "ticks_per_step", 4),
            lr_ltp=_arg(args, "learning_rate", 0.0020),
            lr_ltd=_arg(args, "depression_rate", 0.0021),
            temperature=_arg(args, "temperature", 0.6),
            weight_max=_arg(args, "weight_max", 1.0),
            residual_enabled=config.residual_feedback,
        )

    if config.agent_count <= 0:
        raise SystemExit("--agents must be positive")
    if config.max_steps < 0 or config.checkpoint_every < 0 or config.delay < 0 or config.progress_every < 0:
        raise SystemExit("step, checkpoint, delay, and progress values must be non-negative")
    if (
        config.prediction_learning_rate < 0.0
        or config.prediction_weight_max <= 0.0
        or config.prediction_weight_decay < 0.0
        or config.prediction_window <= 0
    ):
        raise SystemExit(
            "prediction learning rate/decay must be non-negative, weight max positive, "
            "and prediction window positive"
        )
    if config.log_max_bytes < 0 or config.log_backup_count < 0:
        raise SystemExit("log rotation values must be non-negative")
    if config.log_max_bytes > 0 and config.log_backup_count < 1:
        raise SystemExit("--log-backup-count must be at least 1 when rotation is enabled")

    try:
        runtime = CityRuntime(
            config,
            checkpoint_path=args.checkpoint,
            log_path=args.log,
            summary_path=args.summary,
            agent_config=agent_config,
            resume_path=args.resume,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot start city runtime: {exc}") from exc
    print("Nucleus Exploration City")
    print(f"agents: {len(runtime.agents)}   city: {runtime.city.config.width}x{runtime.city.config.height}")
    print(
        "mode: "
        + (
            "adaptive signed residual feedback"
            if runtime.config.residual_feedback
            else "legacy (residual feedback disabled)"
        )
    )
    print(f"log: {runtime.log_path}   rotation: {runtime.config.log_max_bytes} bytes/"
          f"{runtime.config.log_backup_count} backups")
    print(f"checkpoint: {runtime.checkpoint_path}")
    if runtime._resumed_from:
        print(f"resumed: {runtime._resumed_from} at step {runtime.step_count}")
    if runtime.prediction_migration_note:
        print(f"migration: {runtime.prediction_migration_note}")
    print("Action STDP receives no reward, score, penalty, correctness signal, or city outcome.")
    print(
        "A separate local predictor learns from next-observation prediction error; "
        "it never updates action STDP and is not a reward."
    )
    if runtime.config.residual_feedback:
        print(
            "Adaptive mode routes the signed residual into a separate fixed "
            "internal pathway on the next step."
        )
    print("Ctrl+C saves a checkpoint and exits.")
    print()
    summary = runtime.run()
    print()
    print("CITY SESSION SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
