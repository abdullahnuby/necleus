"""Three small local games for the reward-free exploration lab.

The games expose a deliberately small common interface:

    reset(seed=None) -> Observation
    observe() -> Observation
    step(action) -> StepResult

An observation is a fixed-length numeric vector plus a human-readable summary and
terminal render lines.  A step result carries only the next observation, a
``done`` flag, and a short environmental reason.  There is no reward, score,
target action, or correctness field anywhere in this interface.

Snapshots are JSON-safe dictionaries so a checkpoint can resume in the middle of
an episode without resetting the game.
"""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, ClassVar

FEATURE_DIM = 128


def _pad(values: list[float]) -> tuple[float, ...]:
    if len(values) > FEATURE_DIM:
        raise ValueError(f"observation uses {len(values)} features, max is {FEATURE_DIM}")
    return tuple(values + [0.0] * (FEATURE_DIM - len(values)))


def _rng_to_json(rng: random.Random) -> dict[str, Any]:
    version, state, gauss_next = rng.getstate()
    return {"version": version, "state": list(state), "gauss_next": gauss_next}


def _rng_from_json(data: dict[str, Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((int(data["version"]), tuple(int(x) for x in data["state"]), data["gauss_next"]))
    return rng


@dataclass(frozen=True)
class Observation:
    features: tuple[float, ...]
    summary: str
    render: tuple[str, ...]


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    done: bool
    reason: str


class BaseGame:
    """Common game lifecycle.  Subclasses own the actual rules."""

    name: ClassVar[str]
    action_names: ClassVar[tuple[str, ...]]
    legal_actions: ClassVar[tuple[int, ...]]

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)
        self.episode_steps = 0
        self.episode_index = 0
        self.done = False
        self._reset_episode()

    def reset(self, seed: int | None = None) -> Observation:
        if seed is not None:
            self._rng.seed(seed)
        self.episode_steps = 0
        self.episode_index += 1
        self.done = False
        self._reset_episode()
        return self.observe()

    def observe(self) -> Observation:
        raise NotImplementedError

    def step(self, action: int) -> StepResult:
        raise NotImplementedError

    def _reset_episode(self) -> None:
        raise NotImplementedError

    def _state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _load_state(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema": "nucleus-exploration-game-v1",
            "name": self.name,
            "rng": _rng_to_json(self._rng),
            "episode_steps": self.episode_steps,
            "episode_index": self.episode_index,
            "done": self.done,
            "state": self._state(),
        }

    def restore(self, snapshot: dict[str, Any]) -> Observation:
        if snapshot.get("name") != self.name:
            raise ValueError(f"snapshot is for {snapshot.get('name')!r}, not {self.name!r}")
        self._rng = _rng_from_json(snapshot["rng"])
        self.episode_steps = int(snapshot["episode_steps"])
        self.episode_index = int(snapshot["episode_index"])
        self.done = bool(snapshot["done"])
        self._load_state(snapshot["state"])
        return self.observe()

    def _validate_action(self, action: int) -> None:
        if action not in self.legal_actions:
            raise ValueError(f"{self.name} received illegal action {action}")


class GridWorld(BaseGame):
    """7x7 grid with fixed border walls, random interior walls, and one goal."""

    name = "gridworld"
    action_names = ("up", "down", "left", "right")
    legal_actions = (0, 1, 2, 3)
    SIZE = 7
    MAX_STEPS = 80
    DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def _reset_episode(self) -> None:
        interior = [(x, y) for x in range(1, self.SIZE - 1) for y in range(1, self.SIZE - 1)]
        start, goal = self._rng.sample(interior, 2)
        walls = {(x, y) for x in range(self.SIZE) for y in range(self.SIZE)
                 if x in (0, self.SIZE - 1) or y in (0, self.SIZE - 1)}
        free = [cell for cell in interior if cell not in (start, goal)]
        self._rng.shuffle(free)
        walls.update(free[: self._rng.randint(5, 9)])
        self.agent_pos = start
        self.goal_pos = goal
        self.walls = walls

    def _state(self) -> dict[str, Any]:
        return {
            "agent_pos": list(self.agent_pos),
            "goal_pos": list(self.goal_pos),
            "walls": sorted([list(cell) for cell in self.walls]),
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        self.agent_pos = tuple(int(v) for v in state["agent_pos"])
        self.goal_pos = tuple(int(v) for v in state["goal_pos"])
        self.walls = {tuple(int(v) for v in cell) for cell in state["walls"]}

    def step(self, action: int) -> StepResult:
        self._validate_action(action)
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        dx, dy = self.DELTAS[action]
        candidate = (self.agent_pos[0] + dx, self.agent_pos[1] + dy)
        reason = "moved"
        if candidate not in self.walls:
            self.agent_pos = candidate
        else:
            reason = "blocked"
        self.episode_steps += 1
        if self.agent_pos == self.goal_pos:
            self.done = True
            reason = "goal_reached"
        elif self.episode_steps >= self.MAX_STEPS:
            self.done = True
            reason = "timeout"
        return StepResult(self.observe(), self.done, reason)

    def observe(self) -> Observation:
        values = [1.0, 0.0, 0.0, 0.0]
        ax, ay = self.agent_pos
        for py in range(ay - 2, ay + 3):
            for px in range(ax - 2, ax + 3):
                wall = 1.0 if (px, py) in self.walls or not (0 <= px < self.SIZE and 0 <= py < self.SIZE) else 0.0
                goal = 1.0 if (px, py) == self.goal_pos else 0.0
                body = 1.0 if (px, py) == self.agent_pos else 0.0
                values.extend((wall, goal, body))
        goal_dx = (self.goal_pos[0] - ax) / (self.SIZE - 1)
        goal_dy = (self.goal_pos[1] - ay) / (self.SIZE - 1)
        values.extend((
            ax / (self.SIZE - 1),
            ay / (self.SIZE - 1),
            goal_dx,
            goal_dy,
            max(0.0, goal_dx),
            max(0.0, -goal_dx),
            max(0.0, goal_dy),
            max(0.0, -goal_dy),
            self.episode_steps / self.MAX_STEPS,
            1.0,
        ))
        rows = []
        for y in range(self.SIZE):
            row = []
            for x in range(self.SIZE):
                cell = (x, y)
                if cell == self.agent_pos:
                    row.append("A")
                elif cell == self.goal_pos:
                    row.append("G")
                elif cell in self.walls:
                    row.append("#")
                else:
                    row.append(".")
            rows.append("".join(row))
        summary = (
            f"agent={self.agent_pos} goal={self.goal_pos} "
            f"walls={len(self.walls)} step={self.episode_steps}/{self.MAX_STEPS}"
        )
        return Observation(_pad(values), summary, tuple(rows))


class Snake(BaseGame):
    """8x8 snake.  Absolute direction actions; collision ends the episode."""

    name = "snake"
    action_names = ("up", "down", "left", "right")
    legal_actions = (0, 1, 2, 3)
    SIZE = 8
    MAX_STEPS = 200
    DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def _reset_episode(self) -> None:
        mid = self.SIZE // 2
        self.snake = [(mid, mid), (mid - 1, mid), (mid - 2, mid)]
        self.direction = 3
        self.food = self._spawn_food()

    def _spawn_food(self) -> tuple[int, int]:
        free = [(x, y) for x in range(self.SIZE) for y in range(self.SIZE)
                if (x, y) not in set(self.snake)]
        return self._rng.choice(free)

    def _state(self) -> dict[str, Any]:
        return {
            "snake": [list(cell) for cell in self.snake],
            "direction": self.direction,
            "food": list(self.food),
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        self.snake = [tuple(int(v) for v in cell) for cell in state["snake"]]
        self.direction = int(state["direction"])
        self.food = tuple(int(v) for v in state["food"])

    def step(self, action: int) -> StepResult:
        self._validate_action(action)
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        self.direction = action
        dx, dy = self.DELTAS[action]
        head = (self.snake[0][0] + dx, self.snake[0][1] + dy)
        eating = head == self.food
        body = self.snake if eating else self.snake[:-1]
        reason = "moved"
        if not (0 <= head[0] < self.SIZE and 0 <= head[1] < self.SIZE):
            self.done = True
            reason = "wall_collision"
        elif head in body:
            self.done = True
            reason = "self_collision"
        else:
            self.snake.insert(0, head)
            if eating:
                reason = "food"
                if len(self.snake) == self.SIZE * self.SIZE:
                    self.done = True
                    reason = "grid_filled"
                else:
                    self.food = self._spawn_food()
            else:
                self.snake.pop()
        self.episode_steps += 1
        if not self.done and self.episode_steps >= self.MAX_STEPS:
            self.done = True
            reason = "timeout"
        return StepResult(self.observe(), self.done, reason)

    def observe(self) -> Observation:
        values = [0.0, 1.0, 0.0, 0.0]
        direction = [0.0] * 4
        direction[self.direction] = 1.0
        values.extend(direction)
        body = set(self.snake)
        hx, hy = self.snake[0]
        for py in range(hy - 2, hy + 3):
            for px in range(hx - 2, hx + 3):
                outside = not (0 <= px < self.SIZE and 0 <= py < self.SIZE)
                values.extend((
                    1.0 if (px, py) in body else 0.0,
                    1.0 if (px, py) == self.food else 0.0,
                    1.0 if outside else 0.0,
                ))
        food_dx = (self.food[0] - hx) / (self.SIZE - 1)
        food_dy = (self.food[1] - hy) / (self.SIZE - 1)
        values.extend((
            len(self.snake) / (self.SIZE * self.SIZE),
            food_dx,
            food_dy,
            max(0.0, food_dx),
            max(0.0, -food_dx),
            max(0.0, food_dy),
            max(0.0, -food_dy),
            self.episode_steps / self.MAX_STEPS,
            1.0,
        ))
        rows = []
        for y in range(self.SIZE):
            row = []
            for x in range(self.SIZE):
                cell = (x, y)
                if cell == self.snake[0]:
                    row.append("S")
                elif cell in body:
                    row.append("s")
                elif cell == self.food:
                    row.append("F")
                else:
                    row.append(".")
            rows.append("".join(row))
        summary = (
            f"head={self.snake[0]} length={len(self.snake)} food={self.food} "
            f"direction={self.action_names[self.direction]} step={self.episode_steps}/{self.MAX_STEPS}"
        )
        return Observation(_pad(values), summary, tuple(rows))


class PongLite(BaseGame):
    """Vertical paddle, bouncing ball, no score.  A miss ends the episode."""

    name = "pong"
    action_names = ("up", "down", "stay")
    legal_actions = (0, 1, 2)
    WIDTH = 16
    HEIGHT = 9
    PADDLE_HEIGHT = 2
    PADDLE_X = 1
    MAX_STEPS = 600

    def _reset_episode(self) -> None:
        self.paddle_y = (self.HEIGHT - self.PADDLE_HEIGHT) // 2
        self.ball_x = (self.WIDTH - 1) / 2
        self.ball_y = (self.HEIGHT - 1) / 2
        self.ball_vx = self._rng.choice((-0.22, 0.22))
        self.ball_vy = self._rng.uniform(-0.10, 0.10)

    def _state(self) -> dict[str, Any]:
        return {
            "paddle_y": self.paddle_y,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_vx": self.ball_vx,
            "ball_vy": self.ball_vy,
        }

    def _load_state(self, state: dict[str, Any]) -> None:
        self.paddle_y = int(state["paddle_y"])
        self.ball_x = float(state["ball_x"])
        self.ball_y = float(state["ball_y"])
        self.ball_vx = float(state["ball_vx"])
        self.ball_vy = float(state["ball_vy"])

    def step(self, action: int) -> StepResult:
        self._validate_action(action)
        if self.done:
            raise RuntimeError("episode is over; call reset()")
        if action == 0:
            self.paddle_y = max(0, self.paddle_y - 1)
        elif action == 1:
            self.paddle_y = min(self.HEIGHT - self.PADDLE_HEIGHT, self.paddle_y + 1)

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        if self.ball_y < 0.0:
            self.ball_y = -self.ball_y
            self.ball_vy = abs(self.ball_vy)
        elif self.ball_y > self.HEIGHT - 1:
            self.ball_y = 2 * (self.HEIGHT - 1) - self.ball_y
            self.ball_vy = -abs(self.ball_vy)
        if self.ball_x > self.WIDTH - 1:
            self.ball_x = 2 * (self.WIDTH - 1) - self.ball_x
            self.ball_vx = -abs(self.ball_vx)

        reason = "moved"
        paddle_center = self.paddle_y + self.PADDLE_HEIGHT / 2
        if (self.ball_vx < 0 and self.PADDLE_X - 0.5 <= self.ball_x <= self.PADDLE_X + 1.0
                and self.paddle_y <= self.ball_y <= self.paddle_y + self.PADDLE_HEIGHT - 1):
            self.ball_x = self.PADDLE_X + 1.0
            self.ball_vx = abs(self.ball_vx)
            self.ball_vy = max(-0.25, min(0.25, self.ball_vy + (self.ball_y - paddle_center) * 0.08))
            reason = "paddle_hit"
        elif self.ball_x < -0.5:
            self.done = True
            reason = "missed_paddle"

        self.episode_steps += 1
        if not self.done and self.episode_steps >= self.MAX_STEPS:
            self.done = True
            reason = "timeout"
        return StepResult(self.observe(), self.done, reason)

    def observe(self) -> Observation:
        vx = self.ball_vx / 0.3
        vy = self.ball_vy / 0.3
        values = [0.0, 0.0, 1.0, 0.0,
                  self.paddle_y / (self.HEIGHT - self.PADDLE_HEIGHT),
                  self.ball_x / (self.WIDTH - 1),
                  self.ball_y / (self.HEIGHT - 1),
                  vx,
                  vy,
                  max(0.0, vx),
                  max(0.0, -vx),
                  max(0.0, vy),
                  max(0.0, -vy),
                  (self.ball_y - (self.paddle_y + self.PADDLE_HEIGHT / 2)) / (self.HEIGHT - 1),
                  self.episode_steps / self.MAX_STEPS,
                  1.0]
        rows = []
        for y in range(self.HEIGHT):
            row = []
            for x in range(self.WIDTH):
                if abs(self.ball_x - x) < 0.5 and abs(self.ball_y - y) < 0.5:
                    row.append("O")
                elif x == self.PADDLE_X and self.paddle_y <= y < self.paddle_y + self.PADDLE_HEIGHT:
                    row.append("|")
                else:
                    row.append(".")
            rows.append("".join(row))
        summary = (
            f"paddle_y={self.paddle_y} ball=({self.ball_x:.2f},{self.ball_y:.2f}) "
            f"v=({self.ball_vx:.2f},{self.ball_vy:.2f}) step={self.episode_steps}/{self.MAX_STEPS}"
        )
        return Observation(_pad(values), summary, tuple(rows))


GAME_REGISTRY: dict[str, type[BaseGame]] = {
    GridWorld.name: GridWorld,
    Snake.name: Snake,
    PongLite.name: PongLite,
}


def make_game(name: str, seed: int | None = None) -> BaseGame:
    try:
        return GAME_REGISTRY[name](seed=seed)
    except KeyError as exc:
        raise ValueError(f"unknown game {name!r}; choose from {sorted(GAME_REGISTRY)}") from exc
