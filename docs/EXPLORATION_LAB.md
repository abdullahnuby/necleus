# Nucleus Exploration Lab

**Task:** 2026-10-26-TASK-011  
**Status:** engineering implementation complete; no scientific success criterion.

## 1. What this is

`exploration_lab/` is a local, reward-free perception-action loop. A persistent
spiking agent observes one of three small games, selects an action from its own
activity, receives the next observation, and continues. Weights change only
through local spike-timing-dependent plasticity (STDP) driven by the agent's own
spikes. The runtime can run indefinitely, display the game and internal state in
the terminal, checkpoint periodically, and resume from a checkpoint.

This is a place for the agent to interact and change. It is not a benchmark, an
accuracy experiment, or a claim that the agent learns to play well.

## 2. What this is not

- No reward, score, penalty, curiosity bonus, target action, label, or
  correctness signal is sent to the learning rule.
- No hand-written game strategy and no pretrained game model.
- No hyperparameter or checkpoint selection based on wins, scores, survival, or
  any other gameplay outcome.
- No modification of historical v0.42/v0.43 results or the official v0.43 test.
- No cloud API, LLM service, or external gameplay service.
- No claim that random movement, weight change, or a passing software test
  demonstrates learning, intelligence, or skill.

## 3. Layout

```text
exploration_lab/
├── games.py       Unified game interface, GridWorld, Snake, Pong Lite
├── encoding.py    Fixed-size observation to binary spike encoding
├── agent.py       Persistent LIF state, local STDP, action selection, serialization
├── runtime.py     Continuous loop, game cycling, JSONL log, checkpoints, terminal display
├── cli.py         Command-line interface
└── __main__.py    `python -m exploration_lab`
tests/test_exploration_lab.py
docs/EXPLORATION_LAB.md
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
| GridWorld | 7×7, border/interior walls, one goal | up, down, left, right | goal reached or 80 steps |
| Snake | 8×8, growing body, food | up, down, left, right | wall/self collision, grid filled, or 200 steps |
| Pong Lite | 16×9 field, vertical paddle, bouncing ball | up, down, stay | missed paddle or 600 steps |

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

- observation input → hidden neurons (`W_ih`)
- hidden neurons → action readout neurons (`W_ha`)

Hidden→hidden recurrent synapses are fixed random sparse connections and are not
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

## 6. Learning-signal boundary

| Component | Receives | Never receives |
|---|---|---|
| `SpikingExplorer.act` | observation features, legal action list | reward, score, done, target, label, game outcome |
| STDP update | pre/post spikes, traces, weights | reward, score, done, target, label, game outcome |
| Runtime | game observations, `done`, environmental reason | — (it uses `done` only to reset/cycle games) |
| Game | action integer | agent weights or learning state |

The `done` flag is handled only by `ExplorationRuntime`. It resets the game and
moves to the next game. It is never passed to the agent or the plasticity rule.
The game's internal `reason` string (for example `goal_reached`, `missed_paddle`,
or `food`) is for rendering and logging only; it is not a reward.

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

Writes are atomic: a temporary file is written and then replaced. The checkpoint
SHA-256 is logged and printed. On resume, the agent and the current game are
restored, including mid-episode game state.

`session.jsonl` is chronological and append-only. Each step record contains the
timestamp, game, episode, action, legal actions, fallback flag, selection reason,
action scores/probabilities, hidden/action spike counts, weight statistics, the
observation vector and digest, and the environment `done`/reason. Event records
cover session start/end, episode end, game switch, checkpoint, interrupt, and
max-step stop.

`.local_private/` is Git-ignored. Gameplay logs and checkpoints are not committed
or uploaded.

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
- input→hidden and hidden→action weight summaries
- checkpoint path and latest checkpoint event
- recent runtime events
- ASCII game board

The JSONL log contains the same information per step plus the full observation
vector. The session summary contains counters and weight statistics only. It is
explicitly labelled as operational, not as a measure of learning or skill.

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
- absence of reward/outcome parameters in the agent learning API.

Passing these tests verifies software mechanics only. It does **not** establish
that the agent learned, improved, or plays well.

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
- No external network, cloud API, LLM, PyTorch training, or pretrained game model
  is used.
- The official v0.43 test remains unopened and `official_test_ready=false`.
- Defaults were chosen for numerical stability and a runnable loop, not by
  selecting the best game score.

## 13. Engineering completion

The engineering deliverable is present when:

1. all three local games run through a common interface;
2. the agent executes observation → internal state → action → environment step →
   updated state continuously;
3. agent state and weights persist across episodes and can be saved/restored;
4. the runtime can run indefinitely, display behavior, checkpoint periodically,
   and stop safely;
5. the tests, leakage scan, and handoff validator pass.

The agent is not required to win, improve, or stop repeating mistakes.
