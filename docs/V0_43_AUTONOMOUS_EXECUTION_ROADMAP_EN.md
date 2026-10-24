# v0.43 Autonomous Execution Roadmap

**Status:** Execution checkpoint after Phases 0–7. The development pipeline is
implemented and tested, but the official v0.43 test remains **NOT OPENED** and
`official_test_ready=false`.

**Repository:** `https://github.com/abdullahnuby/necleus`
**Local root:** `D:\PROGRAMING\BAND\engineer_nucleus\necleus`
**Starting commit:** `f7bd8fd888d41654c62e983d27f76c077c00d5a8`
**Official v0.43 test:** **NOT OPENED**. It may be opened only after the
readiness checklist passes and an external technical review accepts it.

## Execution rules

1. Phases run in order. A failed gate is recorded, not hidden by starting a
   later phase that depends on it.
2. Development data may select configurations; final-test data may never do so.
3. Every completed task gets `LATEST_REPORT.md`, an immutable history report,
   logs, hashes, a changelog entry, a passing handoff validator, and a verified
   push to `origin/main`.
4. `NOT RUN` is used for work that did not execute. Software tests are not
   learning evidence.
5. The pulse/STDP layer remains out of scope.
6. Evaluator-private artifacts are local-only. `.local_private/` is ignored.

## Task sequence

| Task | Scope | Status | Evidence |
|---|---|---|---|
| `2026-10-24-TASK-006` | Phase 0 revalidation, Phase 1 integrity, Phase 2 generator/auditor | COMPLETE (local gates) | `docs/handoff/logs/2026-10-24-TASK-006/` |
| `2026-10-24-TASK-007` | Phase 3 corpus, baselines, DeepSets training, development evaluation | COMPLETE (implementation); scientific gate partial | `results/v043_development/development_results.json` |
| `2026-10-24-TASK-008` | Phase 4 abstention/new-operation, Phase 5 correction/retention | COMPLETE (execution); retention gate failed | `results/v043_development/abstention_results.json`, `new_operation_results.json`, `correction_retention_results.json` |
| `2026-10-24-TASK-009` | Phase 6 integration/CLI, Phase 7 readiness freeze | COMPLETE (implementation); official test blocked | `docs/V0_43_READINESS_CHECKLIST.md` |

---

## Phase 0 — Revalidate the starting point

**Objective:** Confirm the repository state, dependencies, test baseline, and
prior claims before changing code.

**Current status:** **PASS**.

**Evidence:** `docs/handoff/logs/2026-10-24-TASK-006/git_start.txt` and
`pytest_integrity.txt`. The local and remote starting commit was `f7bd8fd`;
the tree was clean. The pre-change full suite passed 131 tests. PyTorch is not
installed in `.venv`, so the neural baseline is a documented NumPy DeepSets
implementation.

**Known limitations:** CPU-only local execution; no PyTorch; no cloud training.

---

## Phase 1 — Protect benchmark integrity

**Objective:** Remove evaluator-private smoke artifacts from the current public
tree, preserve forensic copies locally, permanently disqualify exposed tasks and
families, and prevent recurrence.

**Current status:** **PASS**.

**Evidence:** `docs/BENCHMARK_INTEGRITY_V0_43.md`,
`docs/V0_43_EXPOSED_SMOKE_DISQUALIFICATION.json`,
`docs/handoff/logs/2026-10-24-TASK-006/private_preservation.txt`, and
`leakage_scan.txt`.

**What passed:** six private files were removed from the current Git index
(three from `v043_stage1_smoke`, three from `v043_stage1_recheck`); byte-identical
copies remain under `.local_private/`; exposed task IDs, families, and
fingerprints are rejected; the scanner passes with 137 data files scanned and no
findings; CLI smoke writes private payloads only under `.local_private/`.

**Known limitations:** The old blobs remain in public Git history. No history
rewrite or force-push was performed. Exposed tasks are disqualified, not secret.

---

## Phase 2 — Complete task generation and hypothesis auditing

**Objective:** Make task creation deterministic, inspectable, balanced, and
leakage-safe for all four task classes.

**Current status:** **PASS**.

**Evidence:** `results/v043_development/dev_manifest.json`;
`skillbench_v043/generator.py`, `audit.py`, `corpus.py`, `integrity.py`;
`tests/v043/test_v043_grammar_and_audit.py`, `test_v043_corpus.py`.

**What passed:** all four classes are generated; 8 support examples; 100 unique
queries balanced 50/50; support/query disjointness; split isolation; public
payloads omit private fields; deterministic regeneration. Out-of-scope support
is rejection-sampled until no known v0.43-core program fits, so the closed-world
auditor cannot answer by coincidence. Development manifest: 28 tasks, 25
families, 8 identifiable / 8 ambiguous / 8 contradictory / 4 OOD.

**Known limitations:** The grammar is deliberately bounded; results do not
generalize to wider program languages.

---

## Phase 3 — Meta-training and fair baselines

**Objective:** Build a reproducible development corpus and evaluate all required
baselines with equal information.

**Current status:** **COMPLETE for implementation; scientific performance gate
PARTIAL**.

**Evidence:** `results/v043_development/corpus_manifest.json`,
`development_results.json`, `.local_private/v043_development/checkpoints/`,
`docs/handoff/logs/2026-10-24-TASK-006/development_run_final.txt`.

**Corpus:** 10,000 actual episodes from 648 unique train families and 3,285
unique targets; episode SHA-256
`cf78f7b28c0b549eefd01e171b8f475df06b8df24bd6b40e23cd31c33fd2a478`.

**Training:** 9,500 train / 500 validation episodes, three seeds
(`430043`, `430044`, `430045`), 10 epochs, hidden 128, NumPy float64 on local CPU,
runtime 498.9 s, peak Python allocation 80.06 MB. Checkpoint SHA-256:
`a9834f87e9af8126ae95de66442c09d0d9d91934054ad4a965688a9e68666054`,
`f42e97c8c4c8671f9df00d6f580ee89a1ba7be9eb3b57a2c40f6293242e28098`,
`e273e6be91468c9b48eb7be77e5d89d2ddee655ed95d8d2d18d586f578507f29`.

**Evaluation (14 held-out development tasks, 1,400 queries):**

| Method | Accuracy | Coverage | Selective risk |
|---|---:|---:|---:|
| Constant | 0.5000 | 1.0000 | 0.5000 |
| Retrieval | 0.5686 | 1.0000 | 0.4314 |
| Linear | 0.5557 | 1.0000 | 0.4443 |
| Decision tree | 0.5714 | 1.0000 | 0.4286 |
| Program induction (query-level) | 0.7343 | 0.3750 | 0.0000 |
| Program induction (task-level) | 0.6429 | 0.2857 | 0.0000 |
| DeepSets (3-seed aggregate) | 0.5729 | 0.0224 | 0.3560 |

**What passed:** every required baseline executes; checkpoints reload; three
seeds; uncertainty intervals and confusion matrices are reported; no evaluator
private inputs are used; tests confirm information separation.

**What failed:** DeepSets is not a competitive or useful selective predictor on
development data. Its identifiable-class coverage is 0.0042 and selective risk
is 0.6000 in the seed where accepted queries exist. This is a scientific failure,
not a plumbing failure.

**Known limitations:** Compute budget is small; the corpus uses random support,
which makes many episodes intrinsically ambiguous. More capable neural training
or a different corpus curriculum is future work, not evidence of success.

---

## Phase 4 — Abstention and new-operation learning

**Objective:** Implement separate Path A and Path B evaluation tracks with
calibrated abstention and a genuinely held-out operation.

**Current status:** **Path A symbolic task-level PASS; DeepSets FAIL; Path B
PARTIAL at 8 examples, PASS at 16/32**.

**Evidence:** `results/v043_development/abstention_results.json`,
`new_operation_results.json`, `docs/handoff/logs/2026-10-24-TASK-006/abstention_run.txt`,
`phase45_evidence.txt`.

**Path A symbolic task-level gate:** identifiable coverage 1.0 with selective
risk 0.0; ambiguous, out-of-scope, and contradictory abstention all 1.0 with
false acceptance 0.0. This policy abstains on the whole task unless exactly one
semantic function is consistent with support; it uses only runtime-available
support consistency, not private class labels.

**Path A query-level symbolic comparison:** identifiable gate passes, but
ambiguous abstention is 0.6875 and false acceptance 0.3125, so it does not meet
the task-level gate. The conservative task-level policy is the reported one.

**Path A DeepSets gate:** fails. Development-selected thresholds produce
aggregate coverage 0.0224, selective risk 0.3560; identifiable-class coverage
0.0042 and selective risk 0.6000. No threshold sweep met the preregistered
coverage/risk gate.

**Path B new operation:** hidden operation is `x0 mod 7 in {0,1,2} XOR
x1 mod 5 == 0`, learned inside a declared modulo meta-grammar. Pre-instruction
abstention is 1.0. With 8 examples, accuracy 0.6467, coverage 0.0900, selective
risk 0.0; with 16 or 32 examples, accuracy 1.0, coverage 1.0, selective risk 0.0,
and the exact hidden function is recovered in 3/3 seeds. The 8-example budget is
a partial success, not a full pass.

**Known limitations:** This is learning inside a declared meta-grammar, not
arbitrary operation discovery. The meta-grammar includes the hidden modulus, so
the result does not establish open-ended invention.

---

## Phase 5 — Corrective feedback and continual learning

**Objective:** Execute the three-round correction loop and measure retention
across at least 20 sequential skills.

**Current status:** **COMPLETE for execution; retention targets FAIL**.

**Evidence:** `results/v043_development/correction_retention_results.json`,
`docs/handoff/logs/2026-10-24-TASK-006/retention_full_program_memory_hard.txt`.

**Loop:** 20 development skills, 8 support examples, Round 0 plus 3 correction
rounds of 50 new interaction queries each, corrections only after errors, and a
fixed independent 100-query evaluation set per skill. Evaluation queries are
disjoint from support, interaction, and correction rows.

**In-context memory track:** Round 0→3 improvement +2.4 percentage points
(target ≥10), average forgetting 0.0, worst-task forgetting 0.0. The memory
track does not forget because each skill keeps its own corrected context, but it
does not meet the improvement target.

**Continual parameter track:** Round 0→3 improvement +5.95 percentage points
(target ≥10), average forgetting 19.2% (target ≤5%), worst-task forgetting 42%.
The declared replay policy reduces forgetting relative to no replay but does not
reach the target.

**Known limitations:** The in-context and parameter tracks are reported
separately. A negative result is not redefined as success.

---

## Phase 6 — Integrated development pipeline

**Objective:** Expose the full workflow through a reproducible CLI and run
integration tests with small deterministic fixtures.

**Current status:** **PASS**.

**Evidence:** `skillbench_v043/cli.py`, `tests/v043/test_v043_cli.py`,
`docs/handoff/logs/2026-10-24-TASK-006/pytest_full.txt`.

**Commands:** `inspect`, `smoke`, `corpus`, `dev`, `train`, `abstention`,
`new-operation`, `correction-retention`. The smoke command writes public payloads
under `results/v043_cli_smoke/` and evaluator-private payloads only under
`.local_private/v043_cli_smoke/`; it refuses exposed task IDs.

**Tests:** full suite 155 passed, 0 failed, in 47.03 s; a final post-documentation
re-run passed 155/155 in 55.29 s. Focused v0.43 and integrity suite passed 32/33
before the CLI smoke seed fix, then 4/4 CLI tests passed; the final full suite
passed 155/155.

**Known limitations:** The CLI does not open the official test. There is no
command that can do so.

---

## Phase 7 — Final-test readiness and development freeze

**Objective:** Prepare the official evaluation without opening it.

**Current status:** **BLOCKED**.

**Evidence:** `docs/V0_43_READINESS_CHECKLIST.md`,
`docs/V0_43_OFFICIAL_TEST_COMMITMENT_PLAN.md`.

**What passed:** family separation; exposed smoke exclusion; deterministic
development artifacts; three-seed checkpoints; CLI; full suite; private
artifacts outside Git; official test not opened.

**What blocks readiness:** DeepSets abstention gate failure; retention
improvement/forgetting failure; 8-example new-operation partial result; no
external technical review; no issued official-test commitment.

**Decision:** `official_test_ready=false`. The official v0.43 test must not be
opened or run. A future task must either improve the failing scientific gates or
explicitly preregister a narrower claim before external review.

---

## Current checkpoint

- **Latest verified commit at task start:** `f7bd8fd888d41654c62e983d27f76c077c00d5a8`.
- **Working tree at task start:** clean.
- **Current task:** `2026-10-24-TASK-006` through `2026-10-24-TASK-009`, executed
  in one continuous session with separate reports.
- **Current phase:** Phase 7 readiness freeze.
- **Exact next action:** commit and push the development evidence, then request
  external review of `docs/V0_43_READINESS_CHECKLIST.md`.
- **Do not do next:** open the official v0.43 test, resume STDP, add pulse
  mechanisms, or reinterpret the failed retention/DeepSets gates as successes.
