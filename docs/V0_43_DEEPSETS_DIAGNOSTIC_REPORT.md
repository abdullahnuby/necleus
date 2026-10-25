# v0.43 DeepSets Diagnostic Report

**Task:** `2026-10-25-TASK-010`
**Protocol:** `v0.43-family-aware-dev-v1`
**Frozen manifest:** `results/v043_task010/dev_manifest.json`
**Manifest SHA-256:** `2b73a94d95513c3316f331f05352689c9557dc90eac2fe6ee8b9de77191a1706`
**Official v0.43 test:** **NOT OPENED**
**Status:** Diagnostic evidence collected. No gate is marked as passing.

## 1. What was measured

The development set contains 54 tasks and 54 unique families. Each of the three
subsets (`fit`, `calibration`, `evaluation`) contains 18 tasks: 5 identifiable,
5 ambiguous, 5 contradictory-support, and 3 OOD tasks. All families are disjoint
across subsets. The nine OOD tasks use nine distinct hidden-operation families
outside the `v0.43-core` grammar.

The diagnostic run used the three unchanged 10-epoch checkpoints:

| seed | checkpoint SHA-256 |
|---:|---|
| 430043 | `a9834f87e9af8126ae95de66442c09d0d9d91934054ad4a965688a9e68666054` |
| 430044 | `f42e97c8c4c8671f9df00d6f580ee89a1ba7be9eb3b57a2c40f6293242e28098` |
| 430045 | `e273e6be91468c9b48eb7be77e5d89d2ddee655ed95d8d2d18d586f578507f29` |

The new family-aware control checkpoints are byte-identical to these historical
checkpoints, so the diagnostic is a genuine control reproduction, not a new
model.

Full machine-readable evidence is in:

- `results/v043_task010/deepsets_diagnostics_old_checkpoints.json`
- `results/v043_task010/deepsets_diagnostics_v2_control.json`

## 2. Training and validation curves

| seed | epoch 1 train / val | epoch 10 train / val | val accuracy at epoch 10 |
|---:|---:|---:|---:|
| 430043 | 0.6666 / 0.6645 | 0.6291 / 0.6513 | 0.6088 |
| 430044 | 0.6639 / 0.6688 | 0.6305 / 0.6596 | 0.6148 |
| 430045 | 0.6644 / 0.6654 | 0.6297 / 0.6510 | 0.6158 |

Training and validation loss decrease slowly and then plateau around 0.63–0.66.
Validation accuracy is approximately 0.59–0.62. There is no evidence of
overfitting, but there is also no evidence that the current 10-epoch schedule
has converged to a strong task solution. This is one reason the bounded
remediation tests a longer schedule and a larger episode count.

## 3. Evaluation score distributions

Query-level evaluation distributions, averaged over the three seeds:

| class | mean confidence range | median confidence range | p95 confidence range | Brier range | ECE (10-bin) range |
|---|---|---:|---:|---:|---:|
| `IN_SCOPE_IDENTIFIABLE` | 0.582–0.597 | 0.554–0.580 | 0.727–0.742 | 0.232–0.240 | 0.035–0.062 |
| `IN_SCOPE_AMBIGUOUS` | 0.615–0.629 | 0.606–0.614 | 0.746–0.845 | 0.244–0.253 | 0.041–0.064 |
| `OUT_OF_SCOPE` | 0.557–0.585 | 0.542–0.560 | 0.652–0.751 | 0.262–0.271 | 0.088–0.108 |
| `CONTRADICTORY_SUPPORT` | 0.592–0.630 | 0.568–0.621 | 0.737–0.794 | 0.207–0.216 | 0.075–0.111 |

The distributions are close to each other, and identifiable tasks are not more
confident than ambiguous tasks. Brier scores around 0.23–0.27 and ECE around
0.04–0.11 indicate weak but not catastrophic calibration. Calibration quality
alone is not the main problem: the ordering of confidence is.

## 4. Risk–coverage and abstention curves

The fixed selector found **no feasible threshold** on either calibration or
evaluation for any seed. The broken historical fallback also found no feasible
threshold, although it selected 0.80, 0.90, and 0.85 for seeds 430043, 430044,
and 430045 respectively.

| seed | fixed chosen | broken fallback | max eval coverage at risk ≤ 0.05 | best eval risk at coverage ≥ 0.90 |
|---:|---:|---:|---:|---:|
| 430043 | none | 0.80 | 0.018 | 0.4680 |
| 430044 | none | 0.90 | 0.014 | 0.3851 |
| 430045 | none | 0.85 | none | 0.4400 |

The evaluation curve therefore makes the preregistered joint criterion
unattainable for these checkpoints: at 90% identifiable coverage, selective risk
is roughly 39–47%, far above 5%. Conversely, keeping risk at or below 5% accepts
only about 0–2% of identifiable queries.

This result separates two failure modes:

1. The historical fallback scoring was defective: it treated `risk=None` as
   zero and could reward a near-total abstention policy. The fixed selector
   correctly returns `feasible=false`.
2. The defect was not the only cause. Even with the fixed selector, the score
   distributions do not contain a feasible operating point. The model itself is
   the limiting factor.

## 5. Identifiable vs ambiguous discrimination

Task-level discrimination uses permitted runtime scores only (model confidence
and entropy). AUC values are averaged across seeds; values near 0.5 mean no
discrimination, and values below 0.5 mean the ordering is reversed.

| comparison | score | AUC |
|---|---|---:|
| identifiable vs ambiguous | mean confidence | 0.320 |
| identifiable vs ambiguous | mean entropy | 0.680 |
| identifiable vs OOD | mean confidence | 0.667 / 0.467 / 0.667 (seed-dependent) |
| identifiable vs contradictory | mean confidence | 0.360 |

The confidence AUC of 0.32 is the central diagnostic fact: the model is
systematically less confident on identifiable tasks than on ambiguous tasks.
The entropy AUC of 0.68 is the inverse monotone transform of the same binary
confidence signal, not independent evidence of a second useful feature.

The model therefore cannot currently use its permitted runtime information to
separate identifiable from ambiguous tasks well enough to satisfy the gate.

## 6. Per-task and per-family behavior

The full per-task table is in `docs/handoff/logs/2026-10-25-TASK-010/per_task_scores.md`.
The full per-family table is in
`docs/handoff/logs/2026-10-25-TASK-010/per_family_scores.md`.

At the diagnostic fallback thresholds, per-family accuracy spans roughly
0.44–0.89. Several identifiable families are below 0.55 accuracy, while some
contradictory-support families are above 0.85. Pooled query-level accuracy
therefore hides substantial family-level variation.

## 7. Interpretation

The evidence supports a combination of causes:

- **Defective fallback scoring:** confirmed and fixed. It must never be treated
  as a valid calibration.
- **Weak and misordered confidence:** confirmed. Identifiable tasks are not more
  confident than ambiguous tasks.
- **Insufficient training/diversity is plausible but not established:** training
  and validation loss plateau, and accuracy is only around 0.59–0.62. The
  bounded remediation tests a longer schedule and more episodes before any
  architecture claim is made.
- **No evidence for a single representation defect:** the diagnostic cannot
  distinguish underfitting from an inadequate objective without the bounded
  experiments. It does establish that the current score distribution cannot
  satisfy the frozen gates.

The previous gate failure remains a valid historical result. The corrected
selector does not rewrite it; it makes the failure explicit.

## 8. Consequence for the next phase

The bounded remediation cycle is preregistered in
`docs/V0_43_DEEPSETS_REMEDIATION_PLAN.md` and its SHA-256 is recorded before execution. It keeps
the control checkpoint unchanged, uses the same family-aware calibration and
evaluation subsets, and does not change the gates. If neither the longer
schedule nor the larger episode count produces a feasible operating point, the
correct decision is to stop optimization and report the DeepSets gate as failed.
