"""Command-line interface for continuous local exploration.

Examples
--------
Watch all three games at a readable speed:

    py -3.14 -m exploration_lab

Fast headless smoke run:

    py -3.14 -m exploration_lab --no-render --delay 0 --max-steps 300 \
        --checkpoint-every 100

Resume the latest checkpoint:

    py -3.14 -m exploration_lab --resume .local_private/exploration/checkpoint.json

All operational output defaults to ``.local_private/exploration/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import AgentConfig
from .games import GAME_REGISTRY
from .runtime import ExplorationRuntime, RuntimeConfig


def parse_games(value: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in value.split(",") if part.strip())
    if not names:
        raise argparse.ArgumentTypeError("at least one game name is required")
    unknown = [name for name in names if name not in GAME_REGISTRY]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown games {unknown}; choose from {sorted(GAME_REGISTRY)}"
        )
    if len(set(names)) != len(names):
        raise argparse.ArgumentTypeError("game names must be unique")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m exploration_lab",
        description="Continuous reward-free exploration in local games. No reward, score, or target is used.",
    )
    parser.add_argument("--games", type=parse_games, default=None,
                        help="comma-separated games: gridworld,snake,pong (default: all)")
    parser.add_argument("--seed", type=int, default=11011, help="initial RNG seed (default: 11011)")
    parser.add_argument("--delay", type=float, default=0.05,
                        help="seconds between environment steps; 0 = maximum speed (default: 0.05)")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="stop after N steps; 0 = run until Ctrl+C (default: 0)")
    parser.add_argument("--checkpoint-every", type=int, default=250,
                        help="save a checkpoint every N steps; 0 = only on stop (default: 250)")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(".local_private/exploration/checkpoint.json"),
                        help="checkpoint path (default: .local_private/exploration/checkpoint.json)")
    parser.add_argument("--log", type=Path,
                        default=Path(".local_private/exploration/session.jsonl"),
                        help="chronological JSONL log (default: .local_private/exploration/session.jsonl)")
    parser.add_argument("--summary", type=Path, default=None,
                        help="session summary path (default: beside the checkpoint)")
    parser.add_argument("--resume", type=Path, default=None,
                        help="restore agent and game state from a checkpoint before running")
    parser.add_argument("--no-render", action="store_true",
                        help="disable the live terminal display; print periodic progress lines instead")
    parser.add_argument("--progress-every", type=int, default=100,
                        help="headless progress interval in steps (default: 100)")
    parser.add_argument("--hidden", type=int, default=64,
                        help="hidden neuron count for a new run (default: 64)")
    parser.add_argument("--ticks-per-step", type=int, default=4,
                        help="internal simulation ticks per environment step (default: 4)")
    parser.add_argument("--learning-rate", type=float, default=0.0020,
                        help="STDP potentiation rate for a new run (default: 0.0020)")
    parser.add_argument("--depression-rate", type=float, default=0.0021,
                        help="STDP depression rate for a new run (default: 0.0021)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="softmax exploration temperature (default: 0.6)")
    parser.add_argument("--weight-max", type=float, default=1.0,
                        help="maximum plastic weight for a new run (default: 1.0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be >= 0")
    if args.checkpoint_every < 0:
        raise SystemExit("--checkpoint-every must be >= 0")
    if args.delay < 0:
        raise SystemExit("--delay must be >= 0")
    if args.progress_every < 0:
        raise SystemExit("--progress-every must be >= 0")

    if args.resume is not None:
        if not args.resume.is_file():
            raise SystemExit(f"checkpoint not found: {args.resume}")
        game_names = args.games if args.games is not None else ()
        agent_config = None
    else:
        game_names = args.games if args.games is not None else ("gridworld", "snake", "pong")
        agent_config = AgentConfig(
            hidden_dim=args.hidden,
            ticks_per_step=args.ticks_per_step,
            lr_ltp=args.learning_rate,
            lr_ltd=args.depression_rate,
            temperature=args.temperature,
            weight_max=args.weight_max,
        )

    config = RuntimeConfig(
        game_names=tuple(game_names),
        seed=args.seed,
        delay=args.delay,
        checkpoint_every=args.checkpoint_every,
        max_steps=args.max_steps,
        render=not args.no_render,
        progress_every=args.progress_every,
    )
    runtime = ExplorationRuntime(
        config,
        checkpoint_path=args.checkpoint,
        log_path=args.log,
        summary_path=args.summary,
        agent_config=agent_config,
        resume_path=args.resume,
    )

    print("Nucleus Exploration Lab")
    print(f"games: {', '.join(runtime.game_names)}")
    print(f"log: {runtime.log_path}")
    print(f"checkpoint: {runtime.checkpoint_path}")
    if runtime._resumed_from:
        print(f"resumed: {runtime._resumed_from} at step {runtime.step_count}")
    print("No reward, score, target action, or game outcome is sent to the learning rule.")
    print("Ctrl+C saves a checkpoint and exits.")
    print()

    summary = runtime.run()
    print()
    print("SESSION SUMMARY")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
