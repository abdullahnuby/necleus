# Nucleus Exploration City

**Task:** 2026-10-27-TASK-012  
**Status:** engineering pilot complete; no scientific success criterion.

## 1. What this is

`exploration_lab/city.py` adds a shared multi-agent city to the exploration lab.
Eight clones of `SpikingExplorer` start from identical weights and independent
RNG streams, then move simultaneously in one 32x32 city. Each clone keeps its
own persistent LIF state, STDP traces, and weights across every step.

The city has buildings, roads, a central plaza, and no goal, score, resource, or
episode. Agents can block each other, collide when two target the same cell, and
come close enough to be logged as an interaction. Those events are **observed and
logged only**. They are not rewards, penalties, correctness labels, or error
signals, and they are never passed to the learning rule.

This is an interaction-observation environment, not a benchmark, a game, or a
claim that clones learn to cooperate, compete, or navigate.

## 2. What this is not

- No reward, score, penalty, target action, label, correctness signal, or
  error-driven weight update.
- No hand-written navigation, cooperation, competition, or city strategy.
- No pretrained model and no external service.
- No selection of hyperparameters or checkpoints based on city outcomes.
- No ranking of clones and no claim that one clone is better.
- No modification of historical v0.42/v0.43 results or the official v0.43 test.
- No claim that movement, collisions, coverage, or weight change demonstrates
  learning, intelligence, cooperation, competition, or skill.

The user decision for this pilot was explicitly **"without an error signal"**.
The earlier question about error-driven learning is therefore not implemented
here; blocked moves and collisions remain observations.

## 3. City layout

The default city is 32x32 with:

- a wall border;
- regular building blocks with small random alleys;
- a 7x7 central plaza;
- 532 open cells, 492 wall cells, and 49 plaza cells in the pilot seed.

The city is generated deterministically from a seed and supports JSON-safe
snapshot/restore. The final map is derived from the checkpoint and included in
`docs/handoff/logs/2026-10-27-TASK-012/city_run_summary.txt`.

## 4. Observation vector

Every clone receives exactly 128 features in `[0, 1]`:

| Slice | Content |
|---|---|
| 0-3 | city context one-hot |
| 4-52 | 7x7 local wall patch |
| 53-101 | 7x7 local other-agent patch |
| 102-103 | normalized self position |
| 104-105 | absolute horizontal/vertical offset to nearest other clone |
| 106-109 | signed nearest-clone direction channels |
| 110-113 | neighbor-count bins (0, 1, 2, 3+) |
| 114-117 | wall adjacency in four directions |
| 118-121 | open directions in four directions |
| 122 | normalized nearest-clone distance |
| 123 | local neighbor density |
| 124-127 | local cell-type one-hot (plaza, road, building, border) |

The encoding is fixed and deterministic. There is no learned embedding and no
game-specific target.

## 5. Movement and collisions

Actions are `up`, `down`, `left`, `right` (0-3). All clones choose an action from
their current observation, then the city applies all moves simultaneously:

- a move into a wall or outside the city is blocked;
- two clones targeting the same cell are both blocked and logged as a collision;
- a move into a cell currently occupied by another clone is blocked;
- otherwise the clone moves.

The city records blocked reasons, collision agents, and interaction pairs. It
does not return a reward or a success flag. `CityStepResult` has no reward,
score, penalty, or error field.

## 6. Clones and persistence

All clones are created from one prototype state, so their initial
`input->hidden`, `hidden->hidden`, and `hidden->action` weights are identical.
Each clone then gets an independent RNG stream. The pilot verifies
`initial_weights_identical=true`.

Every clone retains its own membrane potentials, currents, refractory counters,
STDP traces, previous hidden spikes, counters, and weights between steps. There
is no episode reset. Learning remains pair-based STDP from the existing
`SpikingExplorer`; the city only supplies observations and receives actions.

## 7. Runtime

Run the default 8-clone city until `Ctrl+C`:

```powershell
py -3.14 -m exploration_lab.city
```

Run the exact 5,000-step pilot headlessly:

```powershell
py -3.14 -m exploration_lab.city --no-render --delay 0 --max-steps 5000 ^
  --checkpoint-every 1000 --progress-every 1000 ^
  --agents 8 --width 32 --height 32 --seed 12012 ^
  --checkpoint .local_private/exploration/city/checkpoint.json ^
  --log .local_private/exploration/city/session.jsonl ^
  --summary .local_private/exploration/city/summary.json
```

Resume from a checkpoint:

```powershell
py -3.14 -m exploration_lab.city --resume .local_private/exploration/city/checkpoint.json
```

`--max-steps` is an absolute global step counter. `Ctrl+C` saves a checkpoint
and exits. Raw JSONL rows and checkpoints stay under `.local_private/`, which is
Git-ignored; only aggregate summaries are committed.

## 8. Pilot results

The first pilot used seed 12012, eight clones, a 32x32 city, and 5,000 global
steps (40,000 clone-steps):

| Counter | Value |
|---|---|
| elapsed | 45.822 s |
| interaction pair-events | 2,986 |
| collision agents | 126 |
| blocked moves | 21,124 |
| fallback actions | 7 |
| union unique cells | 511 / 532 open cells (0.9605) |
| unique cells per clone | 263, 350, 124, 331, 205, 147, 105, 98 |
| action counts | up 11,285; down 11,375; left 6,007; right 11,333 |
| checkpoint SHA-256 | `6833206546e0e0c203a6ecddada35b82097346f3928080cbf48c4066eb19c494` |
| log SHA-256 | `a0bccffc423786f256b8957978d43e94da5eb84e06b98808640431e20d4f938e` |

All clones started with identical weights. After 5,000 steps, the mean pairwise
absolute weight difference was 0.01817 for `input->hidden` and 0.01682 for
`hidden->action`. This is a divergence observation, not a measure of learning or
quality.

The most frequent interaction pairs were `[2,7]` (882), `[4,6]` (776), `[1,4]`
(232), and `[0,3]` (159). These are pair-step counts, not friendships or
cooperation.

## 9. Limitations

- No reward or error signal means the city cannot teach a desired behavior.
- Blocked moves and collisions are not used to update weights.
- Coverage, interaction counts, and weight divergence are operational counters,
  not performance metrics.
- A longer run may drift, collapse activity, or produce different counters; this
  pilot is not a long-horizon study.
- `Ctrl+C` is covered by the runtime path and unit tests, but was not manually
  pressed in an interactive terminal during this pilot.
- The official v0.43 test remains closed.

## 10. Files

```text
exploration_lab/city.py                 City, clones, runtime, CLI
tests/test_exploration_lab_city.py      Ten city tests
docs/EXPLORATION_CITY.md                This document
docs/handoff/logs/2026-10-27-TASK-012/  Aggregate pilot evidence
```

## 11. Next step

A natural next task is a **separate, explicitly authorized** prediction-error
experiment, if desired. It would need a new task scope because it introduces a
local error signal that TASK-011 and this pilot deliberately exclude. It should
not reuse city outcomes as a reward or tune checkpoints by coverage, collisions,
or interaction counts.
