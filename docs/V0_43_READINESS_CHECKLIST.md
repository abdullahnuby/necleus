# v0.43 Official-Test Readiness Checklist

**Decision:** **BLOCKED — DO NOT OPEN THE OFFICIAL TEST**
**`official_test_ready`:** `false`
**`official_test_opened`:** `false`
**Evaluation date:** 2026-10-24
**Starting commit:** `f7bd8fd888d41654c62e983d27f76c077c00d5a8`

This checklist is deliberately conservative. A software test passing is not
learning evidence, and a development result is not an official-test result.

## Mandatory checklist

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | Evaluator-private smoke files removed from the current tracked tree | PASS | `docs/BENCHMARK_INTEGRITY_V0_43.md`; `git ls-files results/v043_stage1_smoke` excludes the three private files |
| 2 | Local forensic copies preserved byte-identically outside Git | PASS | `docs/handoff/logs/2026-10-24-TASK-006/private_preservation.txt`; `.local_private/` |
| 3 | Exposed task IDs, families, and fingerprints disqualified in code | PASS | `skillbench_v043/integrity.py`; `tests/test_private_leakage_guard.py` |
| 4 | Recurrence guard scans public artifacts | PASS | `scripts/check_private_leakage.py`: 137 data files, 0 findings |
| 5 | Grammar and split minimums met | PASS | 649 train / 140 dev / 134 test families; after exposed-family exclusion, 133 eligible test families |
| 6 | Deterministic corpus ≥10,000 episodes from ≥100 train families | PASS | 10,000 episodes, 648 families, 3,285 targets; SHA-256 `cf78f7b28c0b549eefd01e171b8f475df06b8df24bd6b40e23cd31c33fd2a478` |
| 7 | All required baselines execute with equal information | PASS | `results/v043_development/development_results.json` |
| 8 | DeepSets checkpoints reload and are multi-seed | PASS | 3 seeds; hashes in `development_results.json` |
| 9 | Symbolic task-level abstention gate | PASS | identifiable coverage 1.0, selective risk 0.0; ambiguous/OOD/contradictory abstention 1.0 |
| 10 | Learned DeepSets abstention gate | **FAIL** | identifiable coverage 0.0042, selective risk 0.6000; aggregate coverage 0.0224, selective risk 0.3560 |
| 11 | New-operation pre-instruction abstention | PASS | abstention 1.0 before instruction |
| 12 | New-operation post-instruction at 8/16/32 examples | **PARTIAL** | 16/32: accuracy 1.0, coverage 1.0; 8: accuracy 0.6467, coverage 0.0900 |
| 13 | Correction loop and 20-skill retention execution | PASS | 20 skills, 4 rounds, 100 independent queries per skill |
| 14 | Retention improvement ≥10 percentage points | **FAIL** | memory +2.4 pp; parameter +5.95 pp |
| 15 | Average forgetting ≤5 percentage points | **FAIL** | memory 0.0; parameter 19.2% average, 42% worst |
| 16 | Integrated CLI for all development phases | PASS | `skillbench_v043/cli.py`; `tests/v043/test_v043_cli.py` |
| 17 | Full test suite | PASS | 155 passed, 0 failed, 47.03 s; final re-run 155/155 in 55.29 s |
| 18 | Official test not opened | PASS | no official-test command exists; `official_test_opened=false` everywhere |
| 19 | External technical review | **BLOCKED** | not requested or completed |
| 20 | Official-test commitment hash issued before opening | **NOT ISSUED** | `docs/V0_43_OFFICIAL_TEST_COMMITMENT_PLAN.md` |

## Blocking failures

1. **DeepSets abstention:** the learned model cannot meet the identifiable
   coverage/risk gate on development data. A threshold that answers enough
   identifiable queries also accepts too many errors; the selected thresholds
   answer almost nothing.
2. **Retention:** the in-context track does not reach the +10 pp improvement
   target; the parameter track forgets 19.2% on average and 42% on the worst
   skill, far above the 5% target.
3. **New-operation 8-example budget:** the 8-example result is partial. Only the
   16- and 32-example budgets reach full coverage and accuracy.
4. **External review and commitment:** no independent technical review has
   accepted the protocol, and no official-test commitment hash has been issued.

## What may be claimed

- The development pipeline is implemented and reproducible.
- The integrity exposure is documented, the current tree is clean, and exposed
  tasks are disqualified.
- The symbolic task-level abstention policy meets its development gate.
- New-operation learning inside the declared meta-grammar reaches full accuracy
  at 16 and 32 examples.
- The full software suite passes.

## What may not be claimed

- That v0.43 is ready for the official test.
- That DeepSets learns a useful selective policy.
- That retention meets the preregistered target.
- That the system invents arbitrary new operations outside the declared
  meta-grammar.
- That any development result is an official-test result.

## Required next action

Do not open the official test. Either improve the failing scientific gates and
rerun development evaluation, or preregister a narrower, explicitly weaker claim
for external review. After review, issue the commitment hash and only then open
a newly generated official test whose evaluator artifacts remain private.
