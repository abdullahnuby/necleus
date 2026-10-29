# Nucleus Exploration — Local Prediction Error

**Task:** 2026-10-29-TASK-013
**Status:** engineering implementation and descriptive pilot complete; no success gate.

## 1. What changed

TASK-011 and TASK-012 deliberately excluded error signals. The user explicitly
selected **local prediction error** after TASK-012, so TASK-013 changes that
constraint for this task only.

The change is deliberately narrow:

- each city clone gets a separate **local predictive head**;
- the head predicts the next 128-feature observation;
- it learns from its own next-observation prediction error;
- it never selects actions;
- it never updates the action STDP pathway;
- no reward, score, penalty, target action, correctness label, or city outcome
  is used as a learning signal.

This is a forward-model experiment, not a navigation, cooperation, competition,
or skill-learning claim.

## 2. Architecture

For clone `i` at city step `t`:

```text
input_t = [ hidden_activity_t (64), action_one_hot_t (4), observation_t (128) ]
prediction_t = clip(W_i @ input_t + b_i, 0, 1)
error_t = observation_{t+1} - prediction_t
```

`hidden_activity_t` is the mean hidden spike activity over the four internal
ticks of the agent's step. It is exposed by `SpikingExplorer.last_hidden_activity`
and saved in agent checkpoints.

The predictor is linear and has `196 x 128` weights plus a 128-element bias. It
does not use backpropagation, recurrence, an optimizer, or a replay buffer.

## 3. Local learning rule

Only the predictive head is updated:

```text
W_i <- (1 - weight_decay) * (W_i + learning_rate * outer(input_t, error_t))
b_i <- clip(b_i + learning_rate * error_t, -1, 1)
W_i <- clip(W_i, -weight_max, weight_max)
```

Each prediction neuron updates its own incoming weights from its own error and
the presynaptic activity. The error is not propagated to hidden neurons or to the
action readout. The action STDP rule remains the same pair-based rule described
in `docs/EXPLORATION_LAB.md`.

Fixed settings for the pilot:

| Setting | Value |
|---|---|
| `learning_rate` | `0.001` (predictive); `0.0` (control) |
| `weight_max` | `2.0` |
| `weight_decay` | `1e-6` |
| `init_scale` | `0.01` |
| initial/final window | 500 steps |
| hidden activity | mean spikes over 4 ticks |
| prediction output | clipped to `[0, 1]` |

The control and predictive conditions use the same seed, city, agent weights,
predictor initialization, and step count. The predictor is initialized once and
copied identically to every clone. The predictor consumes no RNG after
initialization.

## 4. Why the city trajectories must match

The predictor is evaluated after the city step and after the actions are chosen.
It does not feed anything back into action selection, STDP, or the city. Therefore
the control and predictive runs should have identical city/agent trajectories.
This was checked rather than assumed:

- 5,000 step records in each run;
- positions, actions, moved flags, blocked reasons, collisions, interaction
  pairs, fallback flags, hidden spikes, action spikes, observation summaries, and
  cumulative counters all match;
- final city snapshots match;
- final agent state dictionaries match;
- action counts and weight statistics match;
- 0 mismatches.

Only the predictor states and prediction-error fields differ.

## 5. Pilot design

| Item | Value |
|---|---|
| Seed | `12012` |
| City | 32x32, 8 clones |
| Steps | 5,000 global steps (40,000 clone-steps) |
| Control | `--prediction-learning-rate 0` |
| Predictive | `--prediction-learning-rate 0.001` |
| Render | off |
| Delay | 0 |
| Checkpoint interval | 1,000 steps |
| Primary descriptive metric | mean absolute error (MAE) over the final 500 steps |

No success gate, checkpoint selection, or hyperparameter selection was based on
MAE. The result is descriptive.

## 6. Results

### 6.1 Prediction error

| Condition | Mean MAE, all 5,000 steps | MAE, final 500 steps | Predictor updates |
|---|---:|---:|---:|
| Control (fixed predictor) | 0.303368 | 0.319501 | 0 per clone |
| Predictive (`lr=0.001`) | 0.067836 | 0.045719 | 5,000 per clone |

The final-500 MAE difference is `0.273782` in favor of the predictive condition,
a descriptive reduction ratio of `0.856904` relative to the fixed-predictor
control. This is an error metric for the local forward model only.

Per-clone final-500 MAE:

| Clone | Control | Predictive |
|---:|---:|---:|
| 0 | 0.182286 | 0.071113 |
| 1 | 0.251579 | 0.090554 |
| 2 | 0.339602 | 0.037185 |
| 3 | 0.311043 | 0.067631 |
| 4 | 0.360684 | 0.024062 |
| 5 | 0.371032 | 0.011861 |
| 6 | 0.358589 | 0.017537 |
| 7 | 0.381196 | 0.045815 |

The predictive head's first-500 MAE was `0.147988` and its final-500 MAE was
`0.045719`. The first window is not a fixed-weight baseline: it already contains
predictor updates. The fixed-predictor control is the no-learning baseline.

### 6.2 City and agent behavior

The control and predictive runs produced the same operational city result:

| Counter | Value |
|---|---:|
| elapsed | control 52.607 s; predictive 73.408 s |
| interaction pair-events | 2,986 |
| collision agents | 126 |
| blocked moves | 21,124 |
| fallback actions | 7 |
| union unique cells | 511 / 532 (0.9605) |
| action counts | up 11,285; down 11,375; left 6,007; right 11,333 |
| mean `input->hidden` weight divergence | 0.018175 |
| mean `hidden->action` weight divergence | 0.016823 |

The extra runtime in the predictive condition is predictor computation and
checkpoint serialization, not a change in city behavior.

### 6.3 Integrity checks

- 0 city/agent trajectory mismatches across 5,000 records.
- Final city snapshots equal.
- Final agent state dictionaries equal.
- 0 non-finite prediction MAE/MSE values.
- No `reward`, `score`, `penalty`, or `correctness` keys in the step logs.
- Control predictor weights remain identical to their initialization; predictive
  predictor weights diverge and are not identical across clones.
- Checkpoint/resume tests preserve predictor state and prediction counters.
- Focused prediction + city tests: 18 passed. Full suite: 207 passed in 277.18 s.
- Private-leakage scan: PASS, 145 files, 0 findings.

Final artifact SHA-256:

| Artifact | SHA-256 |
|---|---|
| Control checkpoint | `908fbfa42533cfd17d959b07b510b5ef31668f71085f78a0b4f800c4e83b3e11` |
| Control log | `4366183aa0763a752e6d35364b7f62132a110feedae5661943735bc0ddac4d13` |
| Predictive checkpoint | `271cb741d02ed10b67f442cba1a79f337aefa368a2449c6569c29d7ff14e7e3a` |
| Predictive log | `ce9fef3d4cc1b41b12fc5148cd363c6870746ec56893202fa0aa4d6711c854a3` |

Raw logs and checkpoints remain under
`.local_private/exploration/task013_runs_final/` and are not committed. The
committed verification file is
`docs/handoff/logs/2026-10-29-TASK-013/comparison_verification.txt`.

## 7. Commands

Control:

```powershell
py -3.14 -m exploration_lab.city --agents 8 --width 32 --height 32 --seed 12012 `
  --max-steps 5000 --no-render --delay 0 --checkpoint-every 1000 `
  --progress-every 1000 --prediction-learning-rate 0 `
  --log .local_private/exploration/task013_runs_final/control/session.jsonl `
  --checkpoint .local_private/exploration/task013_runs_final/control/checkpoint.json `
  --summary .local_private/exploration/task013_runs_final/control/summary.json
```

Predictive:

```powershell
py -3.14 -m exploration_lab.city --agents 8 --width 32 --height 32 --seed 12012 `
  --max-steps 5000 --no-render --delay 0 --checkpoint-every 1000 `
  --progress-every 1000 --prediction-learning-rate 0.001 `
  --log .local_private/exploration/task013_runs_final/predictive/session.jsonl `
  --checkpoint .local_private/exploration/task013_runs_final/predictive/checkpoint.json `
  --summary .local_private/exploration/task013_runs_final/predictive/summary.json
```

Additional CLI flags:

```text
--prediction-weight-max 2.0
--prediction-weight-decay 1e-6
--prediction-window 500
```

## 8. Logs, checkpoints, and summary fields

Each `city_step` record adds:

```text
prediction_mae: [per-clone MAE]
prediction_mse: [per-clone MSE]
prediction_updates: [per-clone total predictor updates]
```

The city checkpoint adds:

```text
predictors: [per-clone predictor state]
prediction_metrics: aggregate counters and the recent-MAE window
```

The city summary adds a `prediction` object with the predictor config, per-clone
MAE/MSE, first/last window values, update counts, weight statistics, and
`current_weights_identical`. The `prediction` object explicitly states that the
error is self-supervised and never selects actions or updates action STDP.

## 9. Limitations and non-claims

- The predictor is a small linear forward model. It is not a learned policy,
  planner, world model with rollouts, or reinforcement-learning agent.
- Prediction error is not used to improve navigation, coverage, cooperation,
  competition, or any city outcome.
- The city/agent trajectories are identical between control and predictive runs;
  the result is about prediction, not behavior.
- Only one seed and one 5,000-step run were used. No variance estimate or
  long-horizon claim is made.
- The fixed-predictor control has random near-zero weights. Its error is a
  no-learning reference, not a tuned baseline.
- No hyperparameter or checkpoint was selected using MAE.
- The official v0.43 test remains closed; historical v0.42/v0.43 results were
  not modified.
- A decreasing prediction error is not evidence of intelligence, skill, or
  useful city behavior.

## 10. Files

```text
exploration_lab/prediction.py                 LocalPredictor, config, metrics
exploration_lab/agent.py                      last_hidden_activity exposure/serialization
exploration_lab/city.py                       predictor integration, checkpoint, summary, CLI
tests/test_exploration_lab_prediction.py      Predictor and integration tests
docs/EXPLORATION_PREDICTION.md                This document
docs/EXPLORATION_CITY.md                      City extension pointer
docs/handoff/logs/2026-10-29-TASK-013/        Task evidence
```
