# TASK-010 Execution Plan — Development Evaluation Integrity and Remediation

**Task:** `2026-10-25-TASK-010`
**Start commit:** `d888772b3f4a4b4e8e8142e9073aaa90b8f056c6`
**Official v0.43 test:** **NOT OPENED — OUT OF SCOPE**
**Policy:** preserve all historical results; write new artifacts under new versioned
paths; never reinterpret a failed gate as a pass.

## Scope

1. Fix threshold selection so undefined selective risk is never treated as zero
   and no-feasible-threshold is explicit.
2. Build a family-aware development protocol with at least 50 families and
   disjoint fit/calibration/evaluation family subsets.
3. Add family-clustered uncertainty estimates while retaining query-level
   descriptive summaries.
4. Diagnose the existing DeepSets checkpoints before retraining.
5. Run one bounded, preregistered DeepSets remediation cycle.
6. Audit and compare bounded continual-learning/replay strategies.
7. Integrate the new protocol into the CLI and rerun the full verification.

## Allowed files

- `skillbench_v043/baselines.py`, `corpus.py`, `generator.py`, `pipeline.py`,
  `abstention.py`, `deepsets.py`, `continual.py`, `cli.py`
- New `skillbench_v043/diagnostics.py` and `skillbench_v043/remediation.py`
- New tests under `tests/v043/`
- New result artifacts under `results/v043_task010/`
- New local checkpoints under `.local_private/v043_task010/`
- New docs and handoff files for TASK-010

## Frozen inputs

- Existing `results/v043_development/*` remain byte-identical.
- Existing checkpoints under `.local_private/v043_development/checkpoints/`
  remain the unchanged control.
- Preregistered gates remain: identifiable coverage ≥ 0.90 with selective risk
  ≤ 0.05; ambiguous/OOD/contradictory abstention ≥ 0.90 with false acceptance
  ≤ 0.10; retention improvement ≥ 10 pp, average forgetting ≤ 5 pp.
- No official test family, task, query, label, or evaluator artifact is opened.

## Phase acceptance criteria

| Phase | Acceptance |
|---|---|
| B | Undefined risk is `None`; no-feasible returns `feasible=false` and `chosen_threshold=null`; complete sweep; tests cover all requested cases |
| C | ≥50 unique dev families; ≥3 disjoint family subsets; OOD families distinct; family-cluster bootstrap reported |
| D | Diagnostic report measures curves, distributions, calibration, per-task/family metrics, seed variation, discrimination, and whether gates are attainable |
| E | Preregistered plan hash exists before execution; ≥3 seeds; checkpoints hashed; same family-aware protocol; gates unchanged |
| F | Retention audit/tests; ≥3 independent sequences; control + balanced replay + one additional strategy + memory reference; per-round/per-skill records |
| G | CLI exposes manifest, split validation, baselines, training, diagnostics, new-operation, retention, machine-readable summary; fresh run metadata |
| H | Evidence-based decision, updated roadmap/readiness, 11-section handoff, hashes, full tests, scanner, validator, commit/push |

## Verification commands

```powershell
py -3.14 scripts/verify_handoff.py
.\.venv\Scripts\python.exe scripts\check_private_leakage.py
.\.venv\Scripts\python.exe -m pytest -q --no-header -p no:cacheprovider
git status --porcelain=v1
git ls-remote origin refs/heads/main
```

## Stop conditions

- If the grammar cannot provide 50 eligible development families, report the
  actual number and mark the criterion BLOCKED rather than renaming families.
- If a bounded experiment does not pass the frozen gates, keep `FAIL` and stop
  optimizing.
- If any step would require the official test, stop and report `BLOCKED`.
