# Nucleus Exploration City

**Task:** 2026-10-27-TASK-012; TASK-013 prediction extension
**Status:** engineering pilot complete; no scientific success criterion.

TASK-013 adds a separate, local next-observation prediction head to each clone.
It is documented in `docs/EXPLORATION_PREDICTION.md`. It does not change action
STDP, action selection, or the city; its only learning signal is its own
next-observation prediction error.

## 1. What this is

`exploration_lab/city.py` adds a shared multi-agent city to the exploration lab.
Eight clones of `SpikingExplorer` start from identical weights and independent
RNG streams, then move simultaneously in one 32x32 city. Each clone keeps its
own persistent LIF state, STDP traces, and weights across every step.

The city has buildings, roads, a central plaza, and no goal, score, resource, or
episode. Agents can block each other, collide when two target the same cell, and
come close enough to be logged as an interaction. Those events are **observed and
logged only**. They are not rewards, penalties, correctness labels, or explicit
targets for the action STDP rule. The TASK-013 predictor learns only to predict
the next observation; blocked moves, collisions, and interactions are not passed
to it as rewards, penalties, labels, or target actions.

This is an interaction-observation environment, not a benchmark, a game, or a
claim that clones learn to cooperate, compete, or navigate.

## 2. What this is not

- No reward, score, penalty, target action, label, correctness signal, or
  city-outcome signal for action STDP.
- No hand-written navigation, cooperation, competition, or city strategy.
- No pretrained model and no external service.
- No selection of hyperparameters or checkpoints based on city outcomes.
- No ranking of clones and no claim that one clone is better.
- No modification of historical v0.42/v0.43 results or the official v0.43 test.
- No claim that movement, collisions, coverage, or weight change demonstrates
  learning, intelligence, cooperation, competition, or skill.

TASK-011 and TASK-012 explicitly excluded error signals. The user authorized a
new scope for TASK-013: a **separate local prediction head** may learn from its
own next-observation prediction error. That error never selects actions, never
updates action STDP, and is not a reward or a city outcome. See
`docs/EXPLORATION_PREDICTION.md`.

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
score, penalty, or error field. TASK-013 prediction metrics are computed by
`CityRuntime` after the step; they are not part of `CityStepResult` and do not
feed back into movement.

## 6. Clones and persistence

All clones are created from one prototype state, so their initial
`input->hidden`, `hidden->hidden`, and `hidden->action` weights are identical.
Each clone then gets an independent RNG stream. The pilot verifies
`initial_weights_identical=true`.

Every clone retains its own membrane potentials, currents, refractory counters,
STDP traces, previous hidden spikes, counters, and weights between steps. There
is no episode reset. Action learning remains pair-based STDP from the existing
`SpikingExplorer`; the city only supplies observations and receives actions.
TASK-013 adds one separate `LocalPredictor` per clone. It is trained after the
city step by local next-observation error and never writes to agent weights.

## 7. Runtime

Run the default 8-clone city until `Ctrl+C`:

```powershell
py -3.14 -m exploration_lab.city
```

Run the exact 5,000-step TASK-012 pilot headlessly with the predictor fixed:

```powershell
py -3.14 -m exploration_lab.city --no-render --delay 0 --max-steps 5000 ^
  --checkpoint-every 1000 --progress-every 1000 ^
  --agents 8 --width 32 --height 32 --seed 12012 ^
  --prediction-learning-rate 0 ^
  --checkpoint .local_private/exploration/city/checkpoint.json ^
  --log .local_private/exploration/city/session.jsonl ^
  --summary .local_private/exploration/city/summary.json
```

TASK-013 adds these predictor flags:

```text
--prediction-learning-rate 0.001   # 0 fixes the predictor (control)
--prediction-weight-max 2.0
--prediction-weight-decay 1e-6
--prediction-window 500
```

The default predictor learning rate is `0.001`. Use `--prediction-learning-rate 0`
when reproducing the TASK-012 fixed-predictor condition. The predictor never
changes the city/agent trajectory; see `docs/EXPLORATION_PREDICTION.md`.

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

### TASK-013 prediction extension

TASK-013 repeated the same 8-clone, 5,000-step, seed-12012 city twice: once with
a fixed predictor (`--prediction-learning-rate 0`) and once with a learning
predictor (`--prediction-learning-rate 0.001`). The city and agent trajectories
were identical; only the predictor state and prediction error differed.

| Prediction metric | Control | Predictive |
|---|---:|---:|
| predictor updates per clone | 0 | 5,000 |
| mean MAE, all steps | 0.303368 | 0.067836 |
| MAE, final 500 steps | 0.319501 | 0.045719 |
| current predictor weights identical across clones | true | false |
| elapsed | 52.607 s | 73.408 s |

The final-500 MAE difference is `0.273782`; the descriptive reduction ratio
relative to the fixed-predictor control is `0.856904`. This is a forward-model
error result only. It does not mean the clones navigate, cooperate, compete, or
learn a city task. Full design, per-clone values, hashes, and limitations are in
`docs/EXPLORATION_PREDICTION.md`; verification is in
`docs/handoff/logs/2026-10-29-TASK-013/comparison_verification.txt`.

## 9. Limitations

- Action STDP receives no reward, score, penalty, correctness signal, or city
  outcome, so the city cannot teach a desired behavior.
- Blocked moves and collisions are not used as rewards or labels. They can appear
  inside observations that the separate predictor tries to predict, but they are
  not explicit targets or action signals.
- The TASK-013 predictor learns only next-observation error. It cannot improve
  navigation, cooperation, competition, or any city outcome.
- Coverage, interaction counts, prediction error, and weight divergence are
  operational counters, not performance metrics.
- A longer run may drift, collapse activity, or produce different counters; this
  pilot is not a long-horizon study.
- `Ctrl+C` is covered by the runtime path and unit tests, but was not manually
  pressed in an interactive terminal during this pilot.
- TASK-013 used one seed and one 5,000-step run; no variance estimate is made.
- The official v0.43 test remains closed.

## 10. Files

```text
exploration_lab/city.py                        City, clones, runtime, CLI
exploration_lab/prediction.py                  Local prediction head
tests/test_exploration_lab_city.py             Ten city tests
tests/test_exploration_lab_prediction.py       Eight prediction/integration tests
docs/EXPLORATION_CITY.md                       This document
docs/EXPLORATION_PREDICTION.md                 TASK-013 design and results
docs/handoff/logs/2026-10-27-TASK-012/         TASK-012 aggregate evidence
docs/handoff/logs/2026-10-29-TASK-013/         TASK-013 evidence
```

## 11. Next step

TASK-013 implemented the separate, explicitly authorized prediction-error pilot;
see `docs/EXPLORATION_PREDICTION.md`. Any further use of prediction error to
change actions or behavior needs a new explicit task scope. Do not tune
checkpoints or hyperparameters by coverage, collisions, interaction counts, or
MAE and present that as a city result.
