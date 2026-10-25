# TASK-011 execution plan

## Scope

Build `exploration_lab/`: three local games, a persistent step-based spiking agent with
reward-free local STDP, checkpoint/resume, JSONL logging, and an interactive terminal
runtime. No changes to historical results, official v0.43 test, or existing experiment
gates.

## Planned files

- `exploration_lab/games.py` — unified game interface, GridWorld, Snake, Pong Lite.
- `exploration_lab/encoding.py` — fixed-size observation encoding and spike conversion.
- `exploration_lab/agent.py` — persistent LIF state, bounded pair-based STDP, exploratory
  action selection, serialization.
- `exploration_lab/runtime.py` — continuous loop, game cycling, JSONL log, checkpointing,
  terminal display, Ctrl+C handling.
- `exploration_lab/cli.py`, `exploration_lab/__main__.py` — runnable CLI.
- `tests/test_exploration_lab.py` — environment, action, persistence, loop, and
  reward-independence tests.
- `docs/EXPLORATION_LAB.md` — architecture, persistence, limitations, run commands.
- `docs/CHANGELOG.md`, `docs/handoff/LATEST_REPORT.md`,
  `docs/handoff/history/2026-10-26-TASK-011.md`, handoff logs.

## Acceptance conditions

1. All three games implement `reset()`, `observe()`, `step(action)`, episode-end signal,
   terminal rendering, and JSON-safe snapshot/restore.
2. Agent state persists across environment steps and episodes; weights and RNG state
   round-trip through a checkpoint.
3. Learning path receives only observations and legal actions. It never receives done,
   score, reward, target action, or game outcome.
4. Runtime runs until stopped, cycles games, logs chronologically, checkpoints
   periodically, displays live state, and saves on Ctrl+C / stop.
5. Tests and full suite pass; leakage scan and handoff validator pass.
6. No gameplay data or checkpoints are committed. Private artifacts stay under
   `.local_private/`.

## Explicit non-goals

- No scientific success criterion, accuracy target, win rate, or reward schedule.
- No hand-written game policy and no pretrained game model.
- No hyperparameter selection using gameplay outcomes.
- No claim that weight changes demonstrate learning or intelligence.
