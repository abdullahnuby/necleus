# TASK-013 — Local prediction-error learning in the shared city

## Authorization
- The user explicitly selected "خطأ تنبؤي محلي" after TASK-012.
- This task changes the TASK-011/TASK-012 "no error signal" constraint for this task only.
- No reward, score, penalty, target action, correctness label, or city outcome is used as a learning signal.

## Scope
- Add a per-clone local predictive head that predicts the next 128-feature observation from:
  - the clone's current hidden activity,
  - the chosen action one-hot,
  - the current observation.
- Train only the predictive head with a local delta rule: prediction error times presynaptic activity.
- Do not use prediction error to choose actions or to update the action STDP pathway.
- Add a control run with prediction learning rate 0 (fixed predictor) using the same seed and city.
- Run 8 clones x 5,000 steps for both predictive and control conditions.
- Log prediction error and aggregate summaries; raw logs/checkpoints stay under .local_private/.

## Files allowed to change
- exploration_lab/prediction.py (new)
- exploration_lab/agent.py (expose last hidden activity; optional checkpoint field)
- exploration_lab/city.py (integrate predictor, checkpoint/resume, CLI, logs, summary)
- tests/test_exploration_lab_prediction.py (new)
- tests/test_exploration_lab_city.py (small integration test)
- docs/EXPLORATION_PREDICTION.md (new)
- docs/EXPLORATION_CITY.md (pointer)
- docs/EXPLORATION_LAB.md (pointer)
- docs/CHANGELOG.md
- README.md
- .gitattributes (freeze TASK-013 history/logs)
- docs/handoff/LATEST_REPORT.md
- docs/handoff/history/2026-10-29-TASK-013.md
- docs/handoff/logs/2026-10-29-TASK-013/**

## Pre-registered operational comparison
- Primary descriptive metric: mean absolute prediction error (MAE) over the final 500 global steps.
- Control: identical city, seed, agents, and predictor initialization; prediction learning rate = 0.
- Predictive: same setup; prediction learning rate fixed before the run.
- The predictor never feeds actions, so the city/agent trajectories should be identical between conditions; verify this from positions/actions.
- No success gate and no checkpoint selection based on MAE. Results are descriptive only.

## Acceptance
- Prediction head shapes and finite outputs; local update changes only predictor weights.
- Prediction error decreases on a deterministic synthetic sequence.
- City runtime logs prediction MAE/MSE per clone and writes aggregate summary.
- Checkpoint/resume preserves predictor state.
- City tests, prediction tests, full suite, leakage scan, and handoff validator pass.
- Predictive and control runs complete with identical agent trajectories and no reward fields.
- Changes are committed and pushed; working tree is clean.

## Non-goals
- No claim that prediction error improves navigation, cooperation, competition, or skill.
- No reward or city-outcome feedback.
- No official v0.43 test opening or historical result changes.
