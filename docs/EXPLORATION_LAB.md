# Nucleus Exploration Lab

**Task:** 2026-10-26-TASK-011; TASK-012 city extension; TASK-013 local prediction
head; TASK-014 signed residual feedback
**Status:** engineering implementation and checkpoint/resume validation complete;
no scientific success criterion.

## 1. What this is

`exploration_lab/` is a local, reward-free perception-action loop. A persistent
spiking agent observes one of three small games, selects an action from its own
activity, receives the next observation, and continues. Action weights change
only through local spike-timing-dependent plasticity (STDP) driven by the agent's
own spikes. The runtime can run indefinitely, display the game and internal state
in the terminal, checkpoint periodically, and resume from a checkpoint.

This is a place for the agent to interact and change. It is not a benchmark, an
accuracy experiment, or a claim that the agent learns to play well.

TASK-012 adds an optional shared multi-agent city extension. Eight clones of the
same agent share one city and move simultaneously; collisions and proximity are
logged only and are not learning signals. See `docs/EXPLORATION_CITY.md`.

TASK-013 adds a separate local next-observation prediction head per city clone.
It learns from its own prediction error, never selects actions, and never updates
action STDP. See `docs/EXPLORATION_PREDICTION.md`.

TASK-014 optionally feeds the previous transition's signed prediction residual
into a dedicated fixed positive/negative input pathway in the same clone on the
next step. It is disabled by default (legacy TASK-013 behavior) and enabled with
`--residual-feedback`. The residual is not a reward, score, penalty, correctness
label, target action, or direct action-probability bonus.

## 2. What this is not

- No reward, score, penalty, curiosity bonus, target action, label, or
  correctness signal is sent to action STDP or used to select actions.
- No hand-written game strategy and no pretrained game model.
- No hyperparameter or checkpoint selection based on wins, scores, survival, or
  any other gameplay outcome.
- No city outcome, collision, coverage, or interaction count is used to select
  actions or to update action STDP.
- TASK-013's prediction error trains only a separate local forward model. It is
  not a policy gradient, reward, or direct action-selection signal.
- TASK-014's optional residual feedback is a signed feature-level prediction
  error entering a separate fixed internal pathway on the next step. It is not a
  reward, score, penalty, correctness label, target action, or direct
  action-probability bonus, and it does not update action STDP.
- No modification of historical v0.42/v0.43 results or the official v0.43 test.
- No cloud API, LLM service, or external gameplay service.
- No claim that random movement, weight change, prediction-error reduction, or a
  passing software test demonstrates learning, intelligence, or skill.

## 3. Layout

```text
exploration_lab/
â”œâ”€â”€ games.py       Unified game interface, GridWorld, Snake, Pong Lite
â”œâ”€â”€ encoding.py    Fixed-size observation to binary spike encoding
â”œâ”€â”€ agent.py       Persistent LIF state, local STDP, action selection, serialization
â”œâ”€â”€ runtime.py     Continuous loop, game cycling, JSONL log, checkpoints, terminal display
â”œâ”€â”€ prediction.py  Separate local next-observation predictor for the city
â”œâ”€â”€ residual.py    Signed residual channels and summary helpers
â”œâ”€â”€ city.py        Shared multi-agent city, simultaneous movement, city runtime and CLI
â”œâ”€â”€ cli.py         Command-line interface
â””â”€â”€ __main__.py    `python -m exploration_lab`
tests/test_exploration_lab.py
tests/test_exploration_lab_city.py
tests/test_exploration_lab_prediction.py
tests/test_exploration_lab_residual.py
docs/EXPLORATION_LAB.md
docs/EXPLORATION_CITY.md
docs/EXPLORATION_PREDICTION.md
```

## 4. Games

All games implement the same small interface:

```python
reset(seed=None) -> Observation
observe() -> Observation
step(action) -> StepResult
snapshot() -> dict
restore(snapshot) -> Observation
```

`Observation` contains a fixed-length 128-float feature vector, a short text
summary, and terminal render lines. `StepResult` contains only the next
observation, a `done` flag, and an environmental reason string. It has no reward
or score field.

| Game | Grid | Actions | Episode end |
|---|---|---|---|
| GridWorld | 7أ—7, border/interior walls, one goal | up, down, left, right | goal reached or 80 steps |
| Snake | 8أ—8, growing body, food | up, down, left, right | wall/self collision, grid filled, or 200 steps |
| Pong Lite | 16أ—9 field, vertical paddle, bouncing ball | up, down, stay | missed paddle or 600 steps |

The games are lightweight Python/NumPy-free rule loops. No game engine or
graphics dependency is used. Each game owns a `random.Random` instance and
supports JSON-safe snapshot/restore, including its RNG state.

Game rules are not agent strategies. The agent never receives a rule such as
"move toward the ball" or "avoid the wall".

## 5. Agent

### 5.1 Persistent step engine

`SpikingExplorer` runs a fixed number of internal ticks (default 4) per
environment step. It keeps membrane potentials, synaptic currents, refractory
counters, traces, and previous hidden spikes between calls. This is deliberately
different from `pulse_core.model.PulseNetwork.run()`, which calls `reset()` at the
start of every invocation and returns a batch trace. The exploration lab needs a
continuous stepping interface, so it implements that interface in
`exploration_lab/agent.py`.

The agent reuses `pulse_core.model.LIFNeuron` for its documented membrane and
synaptic time constants and decay properties. No file in `pulse_core/` is
modified.

### 5.2 Observation encoding

Each game returns exactly 128 features. Features include a game one-hot context,
a local spatial patch, normalized positions, relative directions, and signed
direction channels. The encoder is deterministic:

```text
input_spike[j] = 1  if observation[j] >= 0.5  else 0
```

There is no learned embedding and no rate code. Signed quantities are represented
by separate positive/negative channels so a threshold encoder does not discard
direction.

### 5.3 LIF dynamics

Hidden and action neurons use the same discrete LIF update as the core model:

```text
current <- current * exp(-1/tau_syn) + input_current + bias
voltage <- voltage * exp(-1/tau_mem) + (1 - exp(-1/tau_mem)) * current
spike   <- voltage >= threshold
```

After a spike, voltage resets to zero and the neuron is refractory for one tick.
Default constants: `tau_mem=2.0`, `tau_syn=6.0`, hidden threshold `1.8`, action
threshold `0.5`. A small constant bias (`0.15` hidden, `0.02` action) keeps the
network excitable after local plasticity weakens its input weights. The bias is
not a reward or a game-specific signal.

### 5.4 Local STDP

Two synapse groups are plastic:

- observation input â†’ hidden neurons (`W_ih`)
- hidden neurons â†’ action readout neurons (`W_ha`)

Hiddenâ†’hidden recurrent synapses are fixed random sparse connections and are not
plastic.

The rule is pair-based STDP with exponential traces. At each tick, traces decay
with `exp(-1/tau_trace)`. Then:

```text
pre spike:  W -= lr_ltd * post_trace
post spike: W += lr_ltp * pre_trace
```

For `W_ih`, pre is an input spike and post is a hidden spike. For `W_ha`, pre is a
hidden spike and post is an action spike. Traces are updated after the weight
update, so same-tick pairs are ignored. This is a discrete-time implementation
choice, not a claim about biological detail.

The rule receives only spike times and the current weights. It does not receive
whether an action was correct, whether the episode ended, a score, a target, or a
game identity as a supervised label. Game identity is part of the observation
vector, not a target.

### 5.5 Action selection and fallback

Action readout activity is summarized per action:

```text
score[a] = action_spike_count[a] + peak_action_voltage[a]
```

The agent samples from a softmax over legal actions:

```text
p[a] = exp((score[a] - max_score) / temperature) / sum(...)
```

Default temperature is `0.6`. Pong uses three legal actions; the fourth action
neuron is masked out.

If the maximum legal score is below `0.05`, the network has no usable activity
signal. The agent then chooses uniformly at random over legal actions and logs:

```text
random_fallback = true
selection_reason = "no_activity_signal_uniform_random"
```

This can happen at the first step of a new process, or after the network becomes
silent. Fallback actions are visible in the log and in the terminal display.

### 5.6 Numerical bounds and stability

- Plastic weights are clipped to `[0, 1.0]`.
- A small multiplicative weight decay (`1e-5` per step) is applied.
- Every 50 steps, each postsynaptic column is scaled down if its total exceeds
  `8.0`. This is a stability constraint, not a policy.
- Membrane voltages, currents, traces, and refractory counters are finite and
  bounded by the LIF equations and the fixed tick count.
- Checkpoint loading rejects non-finite weights and plastic weights outside the allowed
  range.

These constraints are documented as numerical stability, not as a learning
objective.

### 5.7 What persists and what resets

**Persists across environment steps, episodes, and checkpoints:**

- `W_ih`, `W_hh`, `W_ha`
- membrane potentials, synaptic currents, refractory counters
- STDP traces and previous hidden spikes
- agent RNG state, step/tick counters, weight-update counter

**Resets at an episode boundary:**

- only the game state (position, snake body, ball, food, episode step counter)
- the runtime advances to the next game in a fixed cycle

The runtime does **not** call `agent.reset_transient()` on episode end. Agent
activity is continuous across game resets. `reset_transient()` exists for an
explicit cold start or for tests, but the default runtime does not use it.

At the first step of a fresh process the network has no previous hidden spikes, so
the first action may be a logged fallback. This is expected.

### 5.8 Local prediction head (city only)

Each city clone has an optional `LocalPredictor`. It receives the clone's mean
hidden activity, the chosen action one-hot, and the current 128-feature
observation, and predicts the next 128-feature observation:

```text
input_t = [hidden_activity_t, action_one_hot_t, observation_t]
prediction_t = clip(W @ input_t + b, 0, 1)
error_t = observation_{t+1} - prediction_t
W <- (1 - decay) * (W + lr * outer(input_t, error_t))
```

The update is a local delta rule. It uses no backpropagation, reward, score,
target action, correctness label, or city outcome. The predictor never writes to
`SpikingExplorer` weights, never chooses an action, and never updates STDP. Its
only role is to measure and reduce next-observation prediction error. See
`docs/EXPLORATION_PREDICTION.md` for the pilot and limitations.

### 5.9 Signed residual pathway (city, adaptive mode only)

TASK-014 keeps the predictor as the only component trained by prediction error,
but optionally lets the previous transition's signed feature residual influence
the clone's next internal activity:

```text
residual_t = observation_{t+1} - prediction_t
positive_t = max(residual_t, 0)
negative_t = max(-residual_t, 0)
residual_current = clip(W_residual_pos @ positive_t
                        + W_residual_neg @ negative_t, -1, 1)
```

`W_residual_pos` is non-negative, `W_residual_neg` is non-positive, and both are
fixed. They are initialized from a separate deterministic generator so enabling
the pathway does not perturb observation, recurrent, action, or action-sampling
RNG state. The residual current is added to hidden input on the first internal
tick of the next `act()` call only. It is then cleared. It cannot affect the
action that produced the residual, and it is never added to the observation
vector, action STDP, or action probabilities directly.

Legacy mode (the default) does not deliver the residual at all and preserves
TASK-013 behavior. Adaptive mode is explicit:

```powershell
py -3.14 -m exploration_lab.city --residual-feedback
```

The pathway is fixed, not reward-shaped, and not tuned by coverage, collisions,
blocked moves, interactions, wins/losses, MAE, action distributions, or any
gameplay outcome. The scale (`0.05`) and current clip (`[-1, 1]`) are numerical
stability bounds.

## 6. Learning-signal boundary

| Component | Receives | Never receives |
|---|---|---|
| `SpikingExplorer.act` | observation features, legal action list, optional one-shot residual current | reward, score, done, target, label, game outcome, correctness label |
| STDP update | pre/post spikes, traces, weights | reward, score, done, target, label, game outcome, prediction error |
| `LocalPredictor` (city only) | hidden activity, action, current observation, next observation | reward, score, penalty, target action, correctness label, city outcome |
| Residual pathway (city, adaptive only) | signed prediction residual from the previous transition | reward, score, penalty, target action, correctness label, city outcome |
| Runtime | game observations, `done`, environmental reason | â€” (it uses `done` only to reset/cycle games) |
| Game | action integer | agent weights or learning state |

The `done` flag is handled only by `ExplorationRuntime`. It resets the game and
moves to the next game. It is never passed to the agent or the plasticity rule.
The game's internal `reason` string (for example `goal_reached`, `missed_paddle`,
or `food`) is for rendering and logging only; it is not a reward.

In legacy mode the `LocalPredictor` error stays inside its own weight update; it
has no path to action selection, action STDP, or the city. In adaptive mode the
signed residual is delivered once to the next `act()` through the separate fixed
residual pathway. That is an explicitly authorized internal input, not a reward
or a direct action-probability bonus; it never updates STDP or the predictor's
targets.

`RewardSTDPReadout.train_trial()` and other supervised paths in
`pulse_core/sequential_stdp.py` and `pulse_core/learning.py` are not used.

## 7. Checkpoints and logs

Default private paths:

```text
.local_private/exploration/checkpoint.json
.local_private/exploration/session.jsonl
.local_private/exploration/last_session_summary.json
```

`checkpoint.json` contains:

- runtime schema, session id, step/episode counters, game index, game names
- all agent state (weights, transients, traces, RNG)
- JSON-safe snapshots of all three games

The city checkpoint additionally contains one `LocalPredictor` state per clone
and aggregate prediction counters/recent-MAE windows. Agent checkpoints include
`last_hidden_activity` so a resumed predictor can use the same input. TASK-014
bumps the city agent checkpoint schema to v2 and adds:

- the residual configuration and the two fixed residual matrices;
- the pending residual waiting for the next `act()` and the last residual input;
- predictor `last_residual`, step counters, and migration notes;
- exact-continuation state for city, agents, predictors, RNG streams, and
  counters.

A v1 checkpoint is rejected unless `--allow-legacy-migration` is supplied. The
migration preserves all old learned observation/recurrent/action state and
initializes only the new residual matrices to zero. Adaptive mode rejects a v1
agent checkpoint because it has no residual-pathway state. Malformed predictor
lists, shapes, non-finite values, out-of-range weights/biases, inconsistent step
counters, or missing prediction metrics are rejected with a precise message.

On city resume, checkpoint city dimensions, agent count, seed, clone stride,
prediction settings, and residual mode are authoritative. Only session/display
controls (`--max-steps`, `--delay`, `--no-render`, `--checkpoint-every`,
`--progress-every`, log rotation, and migration policy) may differ.

Writes are atomic: a temporary file is written and then replaced. The checkpoint
SHA-256 is logged and printed. On resume, the agent and the current game are
restored, including mid-episode game state.

`session.jsonl` is chronological and append-only. Each step record contains the
timestamp, game, episode, action, legal actions, fallback flag, selection reason,
action scores/probabilities, hidden/action spike counts, weight statistics, the
observation vector and digest, and the environment `done`/reason. City step
records add `prediction_mae`, `prediction_mse`, `prediction_updates`, residual
summary fields, and the residual input consumed on that step. Event records cover
session start/end, episode end, game switch, checkpoint, interrupt, max-step stop,
and log rotation.

City logs support safe size rotation: `--log-max-bytes` (0 disables) and
`--log-backup-count`. The base path remains the current segment; numbered backups
are retained. Rotation never splits a JSON line and records `session_id`,
`log_segment`, and a `log_rotation` event. All raw logs stay under
`.local_private/`, which is Git-ignored; they are not committed or uploaded.

## 8. Runtime commands

Watch all games at a readable speed:

```powershell
py -3.14 -m exploration_lab
```

The display shows the game name, episode, step, elapsed time, current action,
selection reason, fallback flag, action activity, spike counts, weight statistics,
last events, checkpoint notifications, and the ASCII game board. `Ctrl+C` saves a
checkpoint and exits.

Faster or slower display:

```powershell
py -3.14 -m exploration_lab --delay 0.20
py -3.14 -m exploration_lab --delay 0.01
```

Resume:

```powershell
py -3.14 -m exploration_lab --resume .local_private/exploration/checkpoint.json
```

Run the multi-agent city with TASK-014 residual feedback enabled:

```powershell
py -3.14 -m exploration_lab.city --residual-feedback
```

Without `--residual-feedback`, the city keeps legacy TASK-013 behavior. Resume a
city checkpoint with:

```powershell
py -3.14 -m exploration_lab.city --resume .local_private/exploration/city/checkpoint.json
```

Headless short run (for verification, not a performance result):

```powershell
py -3.14 -m exploration_lab --no-render --delay 0 --max-steps 300 `
    --checkpoint-every 100
```

Choose games and seed:

```powershell
py -3.14 -m exploration_lab --games gridworld,snake --seed 11011
```

`--max-steps 0` (the default) runs until stopped. `--checkpoint-every 0` disables
periodic checkpoints but still saves on stop. `--max-steps` is an absolute global
step count, so a resumed run stops when the loaded counter reaches it.

## 9. Observability

The live frame includes:

- active game and episode number
- global step counter and elapsed time
- current action name and index
- selection reason and random-fallback flag
- action scores and probabilities
- hidden and action spike counts
- inputâ†’hidden and hiddenâ†’action weight summaries
- checkpoint path and latest checkpoint event
- recent runtime events
- ASCII game board

The city runtime additionally shows the active mode (`legacy` or `adaptive`),
the last action and residual summary, internal activity, prediction diagnostics,
speed, save/resume state, and safe `Ctrl+C` behavior. The JSONL log contains the
same information per step plus the full observation vector. The session summary
contains counters, weight statistics, prediction metrics, residual summaries,
and log-segment hashes. It is explicitly labelled as operational, not as a
measure of learning or skill.

## 10. Reuse and differences from `pulse_core`

| Existing component | Reused? | Why |
|---|---|---|
| `pulse_core.model.LIFNeuron` | Yes, for time constants and decay properties | Same documented LIF kinetics |
| `pulse_core.model.PulseNetwork` | No | `run()` resets state at the start and returns a batch; unsuitable for continuous stepping |
| `pulse_core.sequential_stdp.RewardSTDPReadout` | No | It is reward-modulated and uses explicit targets; forbidden here |
| `pulse_core.learning` | No | Legacy supervised coincidence learner; not a gameplay loop |
| `pulse_core.tasks` | No | Synthetic temporal tasks, not games; used only as a design reference for input encoding |

The exploration loop is a separate execution engine. It does not alter the
benchmark, the model's historical behavior, or any v0.42/v0.43 result.

## 11. Verification

Run the focused tests:

```powershell
py -3.14 -m pytest -q tests/test_exploration_lab.py
py -3.14 -m pytest -q tests/test_exploration_lab_city.py tests/test_exploration_lab_prediction.py tests/test_exploration_lab_residual.py
```

The tests check:

- all three game interfaces, fixed observation size, terminal render, and
  episode-end signal;
- illegal action rejection;
- JSON-safe snapshot/restore and deterministic next step;
- action transmission in each game;
- binary encoding and feature digest;
- agent fallback logging, bounded weights, transient reset, and checkpoint
  round-trip;
- runtime short run, JSONL logging, periodic checkpoint, resume, stop, and
  `Ctrl+C` handling;
- absence of reward/score fields in `StepResult` and step records;
- absence of reward/outcome parameters in the agent learning API;
- city observation shape, movement, collisions, clone persistence, and
  checkpoint/resume;
- local predictor shapes, delta-rule error reduction, fixed-predictor control,
  checkpoint/resume, and identical control/predictive city trajectories;
- TASK-014 residual encoding, one-step timing, pathway isolation, state
  round-trip, bounds, predictor isolation, malformed predictor rejection,
  config-consistency checks, legacy regression, and JSONL rotation.

The TASK-014 full suite was `225 passed in 224.20s` on 2026-11-05; the focused
exploration subset was `53 passed`. Passing these tests verifies software
mechanics only. It does **not** establish that the agent learned, improved, or
plays well.

## 12. Limitations and non-claims

- The observation encoding is fixed and hand-designed; it is not learned.
- The action space is four neurons (three for Pong); there is no hierarchical or
  continuous control.
- Hidden activity can be sparse or silent for some games; fallback actions are
  logged rather than hidden.
- The STDP rule is local and pair-based. It is not a full continual-learning
  system, does not replay memories, and makes no retention guarantee.
- Weights changing is not evidence of useful learning. No gameplay metric is
  optimized or reported as success.
- The TASK-013 predictor is a small linear forward model. It is not a policy,
  planner, or reinforcement-learning agent.
- TASK-014's residual pathway is fixed and local. It can influence the next
  internal activity, but it is not a reward, target, correctness signal, or
  direct action-probability bonus. No behavioral benefit is claimed.
- TASK-013 used one seed and one 5,000-step city run. TASK-014 used deterministic
  checkpoint/resume smoke runs, not a multi-seed scientific experiment. Neither
  establishes a city performance result.
- No external network, cloud API, LLM, PyTorch training, or pretrained game model
  is used.
- The official v0.43 test remains unopened and `official_test_ready=false`.
- Defaults were chosen for numerical stability and a runnable loop, not by
  selecting the best game score.

## 13. Engineering completion

The engineering deliverable is present when:

1. all three local games run through a common interface;
2. the agent executes observation â†’ internal state â†’ action â†’ environment step â†’
   updated state continuously;
3. agent state and weights persist across episodes and can be saved/restored;
4. the runtime can run indefinitely, display behavior, checkpoint periodically,
   and stop safely;
5. the tests, leakage scan, and handoff validator pass.

The agent is not required to win, improve, or stop repeating mistakes.
