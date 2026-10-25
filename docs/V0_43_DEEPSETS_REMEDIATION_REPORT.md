# v0.43 DeepSets Bounded Remediation Report

**Task:** `2026-10-25-TASK-010`
**Protocol:** `v0.43-family-aware-dev-v1`
**Preregistered plan:** `docs/V0_43_DEEPSETS_REMEDIATION_PLAN.md`
**Preregistered plan SHA-256:** `25bff6d781855180a7a156a295746faedca14b5c81d9bb8ae0730162b4b0f393`
**Frozen manifest:** `results/v043_task010/dev_manifest.json`
**Frozen manifest SHA-256:** `2b73a94d95513c3316f331f05352689c9557dc90eac2fe6ee8b9de77191a1706`
**Machine-readable result:** `results/v043_task010/remediation_results.json`
**Result SHA-256:** `3e240966b3d6b86920747f92b59bc35babd51a2889af8195d81d15a97fe1e19c`
**Official v0.43 test:** **NOT OPENED**
**`official_test_ready`:** `false`
**Decision:** **FAIL — stop optimization**

## 1. Purpose

This report closes the bounded DeepSets remediation cycle that was preregistered
in `docs/V0_43_DEEPSETS_REMEDIATION_PLAN.md`. The cycle tests two single-factor
training changes against the unchanged control:

- **E1:** a longer schedule (20 epochs, checkpoint selected by validation loss).
- **E2:** a larger meta-training corpus (20,000 episodes, 10 epochs).

No architecture, objective, gate, or calibration change was permitted. The
purpose was to decide whether the failed abstention gate is explained by
insufficient training before making a stronger architecture claim.

## 2. Fixed conditions actually used

| Item | Value |
|---|---|
| Development families | 54, family-disjoint across `fit` / `calibration` / `evaluation` |
| Calibration tasks | 18 (5 identifiable, 5 ambiguous, 5 contradictory, 3 OOD) |
| Evaluation tasks | 18 (same class composition) |
| Model seeds | `430043`, `430044`, `430045` |
| Architecture | `DeepSetsNumpy` (phi 3→128→128, mean pooling, rho 130→128→128→1) |
| Optimizer | Adam, learning rate `1e-3`, batch `32`, L2 `1e-4`, gradient clip `5.0` |
| Corpus seed | `430043` |
| Bootstrap | 500 family-cluster resamples, seed `430043` |
| Device | local CPU, NumPy float64 |
| Checkpoint directory | `.local_private/v043_task010/remediation/checkpoints/` |

The manifest hash and split audit in `remediation_results.json` match the frozen
manifest. The result file records the preregistered plan hash, and that hash
matches the current plan file. The plan was written before this run; the result
file does not contain an independent external timestamp.

## 3. Results

### 3.1 Decision table

| Experiment | Seeds with a feasible threshold | Seeds passing all gates | Aggregate decision |
|---|---:|---:|---|
| E0 — unchanged control | 0 / 3 | 0 / 3 | FAIL |
| E1 — longer schedule | 0 / 3 | 0 / 3 | FAIL |
| E2 — more episodes | 0 / 3 | 0 / 3 | FAIL |

No candidate produced a feasible calibration threshold on any seed. Under the
frozen rule, the primary evaluation therefore abstains on every query for every
candidate, so identifiable coverage is `0.0` and selective risk is undefined
(`null`). The abstention gates for ambiguous, OOD, and contradictory classes are
trivially satisfied only because nothing is accepted; they are not evidence of a
useful selective policy.

### 3.2 Aggregate accuracy (all queries, before abstention)

| Experiment | Identifiable | Ambiguous | OOD | Contradictory | Overall |
|---|---:|---:|---:|---:|---:|
| E0 control | 0.5660 | 0.5813 | 0.4778 | 0.6600 | 0.5817 |
| E1 longer schedule | 0.5533 | 0.5840 | 0.4967 | 0.6613 | 0.5824 |
| E2 more episodes | 0.5747 | 0.5733 | 0.4967 | 0.6780 | 0.5900 |

The differences are small. E2 raises overall accuracy by about 0.8 percentage
points and contradictory-support accuracy by about 1.8 points, but identifiable
accuracy remains near chance for a binary task, and the score ordering problem
documented in the diagnostic report is not corrected.

### 3.3 Training behavior

| Experiment | Selected epoch(s) | Final train loss | Final validation loss | Validation accuracy |
|---|---|---|---:|---:|---:|
| E0 control | 10 | — | 0.651–0.660 | 0.609–0.616 |
| E1 seed 430043 | 20 | 0.6216 | 0.6488 | 0.6120 |
| E1 seed 430044 | 20 | 0.6234 | 0.6557 | 0.6120 |
| E1 seed 430045 | 10 | 0.6230 | 0.6530 | 0.6130 |
| E2 all seeds | 10 | 0.6245–0.6250 | 0.6224–0.6238 | 0.6366–0.6426 |

E1 selected epoch 10 for seed `430045` because epoch 10 had lower validation
loss; its checkpoint is byte-identical to the control checkpoint
`e273e6be91468c9b48eb7be77e5d89d2ddee655ed95d8d2d18d586f578507f29`.
For the other two E1 seeds, epoch 20 was selected and the validation loss
improved by roughly 0.002–0.004. E2 improved validation loss and validation
accuracy more visibly, but the evaluation score ordering did not improve enough
to create a feasible threshold.

### 3.4 Runtime

| Experiment | Runtime (s) |
|---|---:|
| E0 | 30.6 |
| E1 | 375.3 |
| E2 | 360.2 |
| Total cycle | 851.2 |

Peak `tracemalloc` was about 109 MB for E1 and 83 MB for E2. These are local
process-memory observations, not a hardware benchmark.

## 4. Diagnostic fallback (not a pass)

For each seed with no feasible threshold, the closest-to-feasible sweep row was
reported separately:

| Experiment | Seed | Diagnostic fallback threshold |
|---|---:|---:|
| E0 | 430043 / 430044 / 430045 | 0.70 / 0.75 / 0.65 |
| E1 | 430043 / 430044 / 430045 | 0.50 / 0.75 / 0.65 |
| E2 | 430043 / 430044 / 430045 | 0.50 / 0.70 / 0.75 |

These rows still violate at least one frozen criterion. They are diagnostic
information only and are never counted as a passing calibration.

## 5. Interpretation

The bounded cycle rules out the simplest explanation that the gate fails only
because the control schedule is too short or the corpus too small:

1. **Longer training does not fix the ordering.** Training and validation losses
   continue to decrease slowly, but identifiable accuracy does not improve
   materially and no calibration threshold becomes feasible.
2. **More episodes help validation but not selective behavior.** E2 reaches the
   best validation accuracy of the cycle and the best overall evaluation
   accuracy, yet its identifiable score distribution still contains no operating
   point that satisfies coverage ≥ 0.90 and risk ≤ 0.05.
3. **The failure is not a threshold-selection artifact.** The fixed selector
   returns `feasible=false` independently for every seed and every candidate.

The correct conclusion is that the current DeepSets architecture/objective is
the limiting factor for this selective-prediction gate under the frozen
protocol. That is a narrower and better-supported claim than "the model cannot
learn"; it says the tested architecture and training recipe do not produce a
usable confidence ordering for this gate.

## 6. What may and may not be claimed

**May be claimed:**

- The remediation cycle was preregistered and executed with three seeds.
- E1 and E2 changed training as specified; the control checkpoint was unchanged.
- No bounded candidate passed the frozen gates.
- The gate failure survives both a longer schedule and a larger corpus.

**May not be claimed:**

- That DeepSets is fundamentally incapable of the task.
- That a different architecture would pass.
- That the diagnostic fallback is a valid operating point.
- That any result here is an official-test result.

## 7. Consequence

Per the preregistered stop rule, DeepSets optimization stops here. The next
honest step is either an external technical review of a narrower claim or a
separately preregistered architecture/objective change; neither is part of this
task. The official v0.43 test remains closed and `official_test_ready=false`.
