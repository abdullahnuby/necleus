# v0.43 DeepSets Bounded Remediation Plan

**Task:** `2026-10-25-TASK-010`
**Preregistration date:** 2026-10-25
**Protocol:** `v0.43-family-aware-dev-v1`
**Frozen manifest:** `results/v043_task010/dev_manifest.json`
**Frozen manifest SHA-256:** `2b73a94d95513c3316f331f05352689c9557dc90eac2fe6ee8b9de77191a1706`
**Official v0.43 test:** **NOT OPENED**

This plan is written and hashed before any remediation training begins. The
frozen gates are not changed. The current 10-epoch checkpoints are the unchanged
control.

## 1. Fixed experimental conditions

- Development data: the frozen 54-task, 54-family manifest above.
- Calibration families and evaluation families are disjoint.
- Thresholds are selected only on calibration families.
- Evaluation-fold query labels are never used for threshold fitting or
  checkpoint selection.
- Model seeds: `430043`, `430044`, `430045`.
- Architecture: unchanged `DeepSetsNumpy` (`phi` 3→128→128, mean pooling,
  query-conditioned `rho` 130→128→128→1).
- Optimizer: unchanged Adam, learning rate `1e-3`, batch size `32`, L2 `1e-4`,
  gradient clip `5.0`.
- Corpus seed: `430043`.
- Family-cluster bootstrap: 500 resamples, seed `430043`.
- Device: local CPU, NumPy float64.
- Checkpoints: `.local_private/v043_task010/remediation/checkpoints/`.

## 2. Experiments

### E0 — unchanged control

- Training: none; load the existing 10-epoch checkpoints.
- Primary factor: none.
- Purpose: reproduce the control under the frozen family-aware protocol.

### E1 — longer training schedule

- Corpus: 10,000 training episodes, unchanged.
- Epochs: 20 instead of 10.
- Predefined checkpoints: epoch 10 and epoch 20.
- Checkpoint selection: lower validation loss on the held-out 5% training
  validation split. Ties select the earlier epoch.
- Primary factor: training schedule length.
- Secondary differences: none; same corpus, architecture, optimizer, and seed.

### E2 — increased meta-training episode count

- Corpus: 20,000 training episodes instead of 10,000.
- Epochs: 10, unchanged.
- Checkpoint selection: final epoch 10.
- Primary factor: episode count/diversity.
- Secondary differences: none; same architecture, optimizer, epochs, and seed.

No architecture or objective change is attempted in this bounded cycle. The
diagnostic report shows weak and misordered confidence, but it cannot distinguish
underfitting from an inadequate objective without first testing these two
single-factor training changes. If both fail, the correct conclusion is that the
current architecture/objective is limiting, not that another unbounded search
should begin.

## 3. Frozen gates

A candidate passes only if every seed satisfies all of the following:

- Identifiable coverage ≥ 0.90.
- Identifiable selective risk ≤ 0.05.
- Ambiguous abstention ≥ 0.90 and false acceptance ≤ 0.10.
- OOD abstention ≥ 0.90 and false acceptance ≤ 0.10.
- Contradictory-support abstention ≥ 0.90 and false acceptance ≤ 0.10.

If no feasible threshold exists on calibration, the primary evaluation abstains
on every query. The diagnostic fallback is reported separately and is never
counted as a pass.

## 4. Decision rule

- `PASS` only if at least one bounded candidate passes the frozen gates for all
  three seeds.
- Otherwise the result is `FAIL`; optimization stops and the failure is
  preserved.
- No candidate may be promoted using evaluation-fold labels.
- No official-test access is permitted under any outcome.

## 5. Required outputs

- `results/v043_task010/remediation_results.json`
- Per-seed checkpoints and SHA-256 hashes.
- Training and validation curves, runtime, and peak memory.
- Per-seed threshold selection, primary metrics, diagnostic fallback metrics,
  family-clustered confidence intervals, and gate status.
- `docs/V0_43_DEEPSETS_REMEDIATION_REPORT.md`

## 6. Stop conditions

Stop the cycle immediately if:

- the frozen manifest hash changes;
- a candidate would require evaluation-fold labels for selection;
- a run attempts to access the official test;
- the bounded experiments complete without a passing candidate.
