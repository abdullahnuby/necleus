# TASK-014 Implementation Plan

- **Task:** `2026-11-04-TASK-014`
- **Starting commit:** `5b55af21442985cf7f09bf84f031352792d0a8ee`
- **Scope:** `exploration_lab/`, its tests, and the listed documentation/handoff files only.
- **Constraint:** preserve TASK-011/012/013 legacy behavior; do not touch historical v0.42/v0.43 results or open the official v0.43 test.

## Architecture decision

The signed next-observation residual is kept feature-level:

```text
residual_t = clip(observation_(t+1) - prediction_t, -1, 1)
positive_t = max(residual_t, 0)
negative_t = max(-residual_t, 0)
```

The agent gets two separate residual input channels and two separate fixed projection
matrices:

```text
w_residual_pos_hidden : (128, hidden)  >= 0
w_residual_neg_hidden : (128, hidden)  <= 0
residual_current = clip(w_pos.T @ positive + w_neg.T @ negative,
                        -residual_current_max, +residual_current_max)
```

The residual current is added only at tick 0 of the next `act()` call, beside the
observation input, and never merged into the observation vector. The projection is
fixed and deterministic (not reward-shaped, not tuned against gameplay). The only
predictive component that learns remains `LocalPredictor`, trained by its own delta
rule. This keeps the residual a sensory signal rather than a reward or action bonus.

## Temporal order in `CityRuntime.step_once()`

1. Read current observations.
2. Each agent consumes only the residual stored at the end of the previous step.
3. Agent selects an action from its own observation + residual + recurrent dynamics.
4. Predictor predicts the next observation from current observation, hidden activity,
   and chosen action.
5. City executes all actions simultaneously.
6. Read next observations.
7. Each predictor computes and stores its own signed residual and updates only its own
   weights.
8. In adaptive mode, that residual is stored on the same clone for the next step.
9. In legacy mode, no residual is delivered to the agent and no residual pathway is used.

No future observation is visible before the transition that produces it. No clone
shares residuals, predictor weights, or prediction errors with another clone.

## Modes and configuration

- `residual_feedback=False` (default): exact TASK-013 behavior. Residual weights are
  zero and unused; predictor learning and diagnostics remain.
- `residual_feedback=True`: adaptive mode. Agent config has `residual_enabled=True`,
  residual projection weights are initialized deterministically, and the previous
  residual enters the next action cycle.
- `prediction_learning_rate=0` remains a fixed-predictor control in either mode.

Checkpoint resume rules:

- City dimensions, seed, agent count, clone stride, prediction settings, residual
  mode, and agent architecture are authoritative from the checkpoint.
- Only display/session controls may be overridden: `max_steps`, `delay`, `render`,
  `checkpoint_every`, `progress_every`, log rotation settings, and migration policy.
- Incompatible requested architecture is rejected with field names and values.
- The renderer and summary use the actual restored config.

## Checkpoint migration and validation

- Bump the agent checkpoint schema to v2.
- v2 restores observation, residual, recurrent, action, membrane, current, refractory,
  STDP traces, RNG, counters, pending residual, and last residual input.
- v1 agent checkpoints are accepted only with explicit `allow_legacy_migration=True`;
  original learned weights are preserved and zero residual weights are created with a
  recorded migration note. Adaptive mode rejects v1 checkpoints because no residual
  pathway exists to reconstruct.
- Missing predictor lists are accepted only with explicit migration; malformed or
  wrong-length predictor lists are always rejected, never silently replaced.
- `LocalPredictor.from_state_dict()` validates schema, shapes, finite values, weight
  and bias bounds, non-negative counters, and residual shape/range.
- City resume validates step/counter consistency, predictor configs, agent configs,
  and prediction metric shapes.

## Logging

- `JsonlLogger` gains optional size-based rotation with numbered backups
  (`.1` newest backup, increasing suffix older), line-safe writes, flush after each
  record, and a `log_rotation` record in the new segment.
- Every city record carries a stable `session_id` and `log_segment`; rotation never
  splits a JSONL line.
- Raw logs/checkpoints stay under `.local_private/`; summaries include segment paths
  and SHA-256 hashes, not private file contents.

## Files

- `exploration_lab/residual.py` (new): residual clipping/splitting/summary helpers.
- `exploration_lab/agent.py`: residual config/state/pathway, v2 checkpoint, migration.
- `exploration_lab/prediction.py`: residual persistence and strict validation.
- `exploration_lab/city.py`: timing, adaptive/legacy modes, config consistency,
  predictor validation, rotation flags, diagnostics, renderer/summary/checkpoint.
- `exploration_lab/runtime.py`: `JsonlLogger` rotation and continuity.
- `exploration_lab/cli.py`: mode, migration, log flags; checkpoint-authoritative resume.
- `tests/test_exploration_lab_residual.py` (new): required architecture/data-flow tests.
- Existing tests updated only where the new schema/config requires it.
- `docs/EXPLORATION_LAB.md`, `docs/EXPLORATION_CITY.md`,
  `docs/EXPLORATION_PREDICTION.md`, `README.md`, `docs/README.md`,
  `docs/CHANGELOG.md`, handoff report/history, and `.gitattributes`.

## Acceptance checks

- Syntax/import checks.
- Targeted residual/prediction/city/exploration tests.
- Full project suite under Python 3.14.
- `scripts/check_private_leakage.py` PASS.
- `scripts/verify_handoff.py` PASS.
- Deterministic legacy and adaptive smoke sessions.
- Checkpoint restore equivalence in both modes.
- Historical v0.42/v0.43 files unchanged; official v0.43 test not run.
- Clean commit, push to `origin/main`, remote SHA verification, clean tree.

## Explicit non-goals

No reward, score, penalty, correctness label, target action, desired action, action
probability bonus, benchmark, success gate, or behavior-quality comparison. No tuning
against coverage, collisions, blocked moves, interactions, wins/losses, MAE, action
distributions, or gameplay outcomes.
