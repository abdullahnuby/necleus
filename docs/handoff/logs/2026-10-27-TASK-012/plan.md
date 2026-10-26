# TASK-012 — Multi-agent city observation pilot

## Scope
- Add a shared city environment with eight clones of the existing spiking explorer.
- Clones start from identical weights and independent RNG streams; they share one city.
- No reward, score, penalty, correctness signal, or error-driven weight update is added.
- Blocked moves, collisions, proximity events, and action counts are logged only as observations.
- Learning remains local timing-based STDP from the existing SpikingExplorer.
- Run a first pilot: 8 clones x 5,000 global steps, headless, logs under .local_private/.
- Add tests, docs, handoff report, hashes, leakage scan, validator, commit, and push.

## Files allowed to change
- exploration_lab/city.py (new)
- tests/test_exploration_lab_city.py (new)
- docs/EXPLORATION_CITY.md (new)
- docs/EXPLORATION_LAB.md (small pointer)
- docs/CHANGELOG.md
- README.md
- .gitattributes (freeze TASK-012 logs/history)
- docs/handoff/LATEST_REPORT.md
- docs/handoff/history/2026-10-27-TASK-012.md
- docs/handoff/logs/2026-10-27-TASK-012/**

## Acceptance
- City observations are 128 finite features in [0, 1].
- Simultaneous movement has deterministic collision/blocked handling and no reward fields.
- Snapshot/restore and runtime checkpoint/resume preserve city and clone state.
- Short runtime test writes JSONL, checkpoint, and aggregate summary.
- 8-clone x 5,000-step pilot completes and produces an aggregate summary.
- Full pytest suite, leakage scan, and handoff validator pass.
- Changes are committed and pushed; working tree is clean.

## Explicit non-goals
- No claim of learning, intelligence, skill, cooperation, or competition.
- No performance ranking of clones.
- No tuning based on city outcomes.
- No official v0.43 test access or historical result changes.
