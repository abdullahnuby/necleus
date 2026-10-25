"""Continuous exploration runtime: loop, logging, checkpointing, live display.

The runtime is the only component that sees an episode-ending flag.  It uses that
flag solely to reset the game and advance to the next game.  The flag is never
passed to the agent and is not a learning signal.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Sequence

from .agent import ActionDecision, AgentConfig, SpikingExplorer
from .encoding import feature_digest
from .games import GAME_REGISTRY, BaseGame, make_game

CHECKPOINT_SCHEMA = "nucleus-exploration-runtime-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


@dataclass(frozen=True)
class RuntimeConfig:
    game_names: tuple[str, ...] = ("gridworld", "snake", "pong")
    seed: int = 11011
    delay: float = 0.05
    checkpoint_every: int = 250
    max_steps: int = 0
    render: bool = True
    progress_every: int = 100


class TerminalRenderer:
    def __init__(self, ansi: bool = True) -> None:
        self.ansi = bool(ansi and sys.stdout.isatty())
        if self.ansi:
            # Ask Windows terminals to interpret ANSI control sequences.
            os.system("")

    def render(self, runtime: "ExplorationRuntime") -> None:
        if self.ansi:
            print("\x1b[2J\x1b[H", end="")
        print(self.frame(runtime))

    def frame(self, runtime: "ExplorationRuntime") -> str:
        game = runtime.current_game
        decision = runtime.last_decision
        record = runtime.last_record
        elapsed = runtime.elapsed_seconds
        lines = [
            "NUCLEUS EXPLORATION LAB - reward-free continuous interaction",
            "=" * 72,
            f"game: {game.name:<12} episode: {runtime.episode_count:<6} "
            f"step: {runtime.step_count:<8} elapsed: {format_duration(elapsed)}",
            f"speed: {runtime.config.delay:.3f}s/step   checkpoint every: "
            f"{runtime.config.checkpoint_every or 'disabled'} steps",
        ]
        if decision is not None:
            action_name = game.action_names[decision.action]
            lines.append(
                f"action: {action_name} ({decision.action})   "
                f"selection: {decision.selection_reason}   "
                f"random_fallback: {'yes' if decision.random_fallback else 'no'}"
            )
            scores = " ".join(f"{s:.2f}" for s in decision.scores)
            probs = " ".join(f"{p:.2f}" for p in decision.probabilities)
            lines.append(f"action activity: [{scores}]   probabilities: [{probs}]")
            lines.append(
                f"hidden spikes: {decision.hidden_spikes:<3}   "
                f"action spikes: {list(decision.action_spikes)}"
            )
        if record is not None:
            lines.append(f"observation: {record['observation_summary']}")
            lines.append(f"last outcome flag (game reset only): done={record['done']} reason={record['done_reason']}")
        stats = runtime.agent.weight_stats()
        lines.append(
            "weights: input->hidden mean={:.4f} max={:.4f} | hidden->action mean={:.4f} max={:.4f} | updates={}".format(
                stats["input_hidden"]["mean"], stats["input_hidden"]["max"],
                stats["hidden_action"]["mean"], stats["hidden_action"]["max"],
                int(stats["total_weight_updates"]),
            )
        )
        lines.append(f"checkpoint: {runtime.checkpoint_path}")
        if runtime.last_checkpoint_message:
            lines.append(f"checkpoint event: {runtime.last_checkpoint_message}")
        for event in list(runtime.events)[-3:]:
            lines.append(f"event: {event}")
        lines.append("Ctrl+C: save checkpoint and stop")
        lines.append("-" * 72)
        lines.extend(game.observe().render)
        return "\n".join(lines)


def format_duration(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ExplorationRuntime:
    """Owns the games, agent, chronological log, checkpoint, and display."""

    def __init__(
        self,
        config: RuntimeConfig,
        *,
        checkpoint_path: Path,
        log_path: Path,
        summary_path: Path | None = None,
        agent_config: AgentConfig | None = None,
        resume_path: Path | None = None,
        on_step: Callable[["ExplorationRuntime"], None] | None = None,
    ) -> None:
        self.config = config
        self.checkpoint_path = Path(checkpoint_path)
        self.log_path = Path(log_path)
        self.summary_path = Path(summary_path) if summary_path else self.checkpoint_path.parent / "last_session_summary.json"
        self.on_step = on_step
        self._stop = threading.Event()
        self._start_monotonic = time.monotonic()
        self.events: deque[str] = deque(maxlen=8)
        self.last_decision: ActionDecision | None = None
        self.last_record: dict[str, Any] | None = None
        self.last_checkpoint_message = ""
        self.checkpoints_saved = 0
        self.fallback_count = 0
        self.episode_count = 0
        self.step_count = 0
        self.game_index = 0
        self.session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + os.urandom(3).hex()
        self._resumed_from: str | None = None

        if resume_path is not None:
            payload = json.loads(Path(resume_path).read_text(encoding="utf-8"))
            if payload.get("schema") != CHECKPOINT_SCHEMA:
                raise ValueError("unsupported runtime checkpoint schema")
            checkpoint_games = tuple(payload["game_names"])
            if config.game_names and tuple(config.game_names) != checkpoint_games:
                raise ValueError(
                    f"--games {tuple(config.game_names)} does not match checkpoint games {checkpoint_games}"
                )
            self.game_names = checkpoint_games
            self.agent = SpikingExplorer.from_state_dict(payload["agent"])
            self.step_count = int(payload["step"])
            self.episode_count = int(payload["episode"])
            self.game_index = int(payload["game_index"]) % len(self.game_names)
            self.fallback_count = int(payload.get("fallback_count", 0))
            self.checkpoints_saved = int(payload.get("checkpoints_saved", 0))
            self._resumed_from = str(resume_path)
            self.games = {name: make_game(name) for name in self.game_names}
            for name, snapshot in payload["game_snapshots"].items():
                self.games[name].restore(snapshot)
        else:
            self.game_names = tuple(config.game_names)
            if not self.game_names:
                raise ValueError("at least one game is required")
            self.agent = SpikingExplorer(agent_config or AgentConfig(), seed=config.seed)
            self.games = {
                name: make_game(name, seed=config.seed + 1009 * index)
                for index, name in enumerate(self.game_names)
            }

        for name in self.game_names:
            if name not in GAME_REGISTRY:
                raise ValueError(f"unknown game {name!r}")
        self.logger = JsonlLogger(self.log_path)
        self.renderer = TerminalRenderer(ansi=config.render)
        self._ensure_playable()

    @property
    def current_game(self) -> BaseGame:
        return self.games[self.game_names[self.game_index]]

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_monotonic

    def request_stop(self) -> None:
        self._stop.set()

    def _ensure_playable(self) -> None:
        game = self.current_game
        if game.done:
            game.reset()
            self._advance_game()

    def _advance_game(self) -> None:
        previous = self.current_game.name
        self.game_index = (self.game_index + 1) % len(self.game_names)
        game = self.current_game
        if game.done:
            game.reset()
        self._event("game_switch", **{"from": previous, "to": game.name})

    def _event(self, event: str, **fields: Any) -> None:
        record = {"ts_utc": utc_now(), "session_id": self.session_id, "event": event, **fields}
        self.logger.write(record)
        text = event
        if fields:
            text += " " + " ".join(f"{key}={value}" for key, value in fields.items())
        self.events.append(text)

    def run(self) -> dict[str, Any]:
        self._event(
            "session_start",
            games=list(self.game_names),
            seed=self.config.seed,
            delay=self.config.delay,
            checkpoint_every=self.config.checkpoint_every,
            resumed_from=self._resumed_from,
            step=self.step_count,
            episode=self.episode_count,
        )
        interrupted = False
        try:
            while not self._stop.is_set():
                if self.config.max_steps and self.step_count >= self.config.max_steps:
                    self._event("max_steps_reached", step=self.step_count)
                    break
                self._step_once()
                if self.on_step is not None:
                    self.on_step(self)
                if self.config.checkpoint_every and self.step_count % self.config.checkpoint_every == 0:
                    self.save_checkpoint(reason="periodic")
                if self.config.render:
                    self.renderer.render(self)
                elif self.config.progress_every and self.step_count % self.config.progress_every == 0:
                    print(self._progress_line(), flush=True)
                if self.config.delay > 0 and self._stop.wait(self.config.delay):
                    break
        except KeyboardInterrupt:
            interrupted = True
            self._stop.set()
            self._event("interrupt", step=self.step_count)
        finally:
            self.save_checkpoint(reason="stop")
            summary = self.summary(interrupted=interrupted)
            self._event("session_end", **{k: summary[k] for k in (
                "steps", "episodes", "fallback_actions", "checkpoints_saved", "elapsed_seconds",
            )})
            self.logger.close()
            self.summary_path.parent.mkdir(parents=True, exist_ok=True)
            self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    def _step_once(self) -> None:
        game = self.current_game
        observation = game.observe()
        decision = self.agent.act(observation.features, game.legal_actions)
        result = game.step(decision.action)
        self.step_count += 1
        self.fallback_count += int(decision.random_fallback)
        record = {
            "ts_utc": utc_now(),
            "session_id": self.session_id,
            "event": "step",
            "step": self.step_count,
            "episode": self.episode_count,
            "game": game.name,
            "game_episode": game.episode_index,
            "action": decision.action,
            "action_name": game.action_names[decision.action],
            "legal_actions": list(game.legal_actions),
            "random_fallback": decision.random_fallback,
            "selection_reason": decision.selection_reason,
            "action_scores": list(decision.scores),
            "action_probabilities": list(decision.probabilities),
            "hidden_spikes": decision.hidden_spikes,
            "action_spikes": list(decision.action_spikes),
            "weight_stats": self.agent.weight_stats(),
            "observation": [round(float(value), 6) for value in observation.features],
            "observation_summary": observation.summary,
            "observation_digest": feature_digest(observation.features),
            "done": result.done,
            "done_reason": result.reason,
        }
        self.logger.write(record)
        self.last_record = record
        self.last_decision = decision

        if result.done:
            self.episode_count += 1
            self._event("episode_end", game=game.name, reason=result.reason, step=self.step_count,
                        episode=self.episode_count)
            # The outcome flag stops this game only.  It is never passed to the agent.
            # Agent activity, traces, and weights stay continuous across game resets.
            game.reset()
            self._advance_game()

    def _progress_line(self) -> str:
        stats = self.agent.weight_stats()
        return (
            f"step={self.step_count} episodes={self.episode_count} "
            f"game={self.current_game.name} fallback={self.fallback_count} "
            f"w_in_hidden_mean={stats['input_hidden']['mean']:.5f} "
            f"w_hidden_action_mean={stats['hidden_action']['mean']:.5f} "
            f"elapsed={format_duration(self.elapsed_seconds)}"
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "created_utc": utc_now(),
            "session_id": self.session_id,
            "game_names": list(self.game_names),
            "game_index": self.game_index,
            "step": self.step_count,
            "episode": self.episode_count,
            "fallback_count": self.fallback_count,
            "checkpoints_saved": self.checkpoints_saved,
            "agent": self.agent.state_dict(),
            "game_snapshots": {name: game.snapshot() for name, game in self.games.items()},
        }

    def save_checkpoint(self, reason: str) -> Path:
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.checkpoints_saved += 1
        payload = self.checkpoint_payload()
        temporary = self.checkpoint_path.with_suffix(self.checkpoint_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self.checkpoint_path)
        digest = sha256_file(self.checkpoint_path)
        self.last_checkpoint_message = (
            f"saved step={self.step_count} reason={reason} sha256={digest[:16]}... -> {self.checkpoint_path}"
        )
        self._event("checkpoint", step=self.step_count, reason=reason,
                    path=str(self.checkpoint_path), sha256=digest)
        if not self.config.render:
            print(f"[checkpoint] {self.last_checkpoint_message}", flush=True)
        return self.checkpoint_path

    def summary(self, *, interrupted: bool) -> dict[str, Any]:
        stats = self.agent.weight_stats()
        return {
            "schema": "nucleus-exploration-session-summary-v1",
            "session_id": self.session_id,
            "resumed_from": self._resumed_from,
            "interrupted": interrupted,
            "games": list(self.game_names),
            "steps": self.step_count,
            "episodes": self.episode_count,
            "fallback_actions": self.fallback_count,
            "checkpoints_saved": self.checkpoints_saved,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "log_path": str(self.log_path),
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_sha256": sha256_file(self.checkpoint_path) if self.checkpoint_path.is_file() else None,
            "agent_steps": self.agent.steps,
            "agent_ticks": self.agent.ticks,
            "agent_weight_updates": self.agent.total_weight_updates,
            "weight_stats": stats,
            "note": (
                "Operational counters only. This summary is not a measure of learning, "
                "intelligence, skill, or game performance."
            ),
        }
