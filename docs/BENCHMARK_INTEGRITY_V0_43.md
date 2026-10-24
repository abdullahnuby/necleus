# v0.43 Benchmark Integrity — Exposed Smoke Artifacts

**Status:** Exposure documented; evaluator-private files removed from the current
tracked tree; exposed tasks permanently disqualified; local forensic copies
preserved outside Git.

## 1. What was exposed

The public repository previously tracked three stage-1 smoke evaluator files:

| File | SHA-256 before removal |
|---|---|
| `results/v043_stage1_smoke/train_evaluator_private.jsonl` | `26a8b05fb206b21cd5565062105a42f65cc82feeec29e04bd6aec9d3a62051ce` |
| `results/v043_stage1_smoke/dev_evaluator_private.jsonl` | `210fd10dac069bb47cf56edd7b6adb544892a390b3a8efaceb31bb6ebd91b307` |
| `results/v043_stage1_smoke/test_evaluator_private.jsonl` | `86444abd0e05107e825b63ba9d5b028716ca2b8d8777f3762c2e3bed08d62c4a` |

Each file contained a target program, family identifier, support rows, query
inputs, query labels, and private audit metadata. The files were synthetically
generated smoke artifacts, not official benchmark data, but the exposed rows and
tasks are permanently disqualified from official use.

## 2. Local forensic preservation

Byte-identical copies are stored under the Git-ignored directory:

```
.local_private/v043_stage1_smoke/
```

The copy operation and SHA-256 comparison are recorded in
`docs/handoff/logs/2026-10-24-TASK-006/private_preservation.txt`. The local
copies are for auditing only and must not be published, committed, attached to a
report, or used to tune a model.

## 3. Disqualified tasks and families

The following task IDs are permanently excluded:

- `task-84d6edfe96ffd28e`
- `task-92940f0d430a70f6`
- `task-ff56a93a34464a02`

The following family IDs are permanently excluded from the official test split:

- `001805d0b07b2fce`
- `00d96a49a204ff33`
- `00e3601b352f025e`

The public, machine-readable registry is
`docs/V0_43_EXPOSED_SMOKE_DISQUALIFICATION.json`. It contains IDs, family IDs,
and hashes/fingerprints only; it does not repeat support rows or query labels.
The code-level registry is `skillbench_v043/integrity.py`.

## 4. What was removed and what remains

The three `*_evaluator_private.jsonl` files were removed from the current Git
index with `git rm`. The public smoke files and aggregate smoke report remain as
historical evidence of the stage-1 plumbing test. They are not official data.

**Removing a file from the current commit does not remove it from public Git
history.** The old blobs remain reachable in earlier commits. No history rewrite,
filter, or force-push was performed. The correct conclusion is that the exposed
tasks are disqualified, not that the data is secret.

## 5. Recurrence prevention

- `.gitignore` excludes `.local_private/` and `*_evaluator_private.jsonl`.
- `scripts/check_private_leakage.py` scans public JSON/JSONL/CSV artifacts and
  rejects private filenames and private data keys.
- `skillbench_v043/integrity.py` rejects exposed task IDs, family IDs in the test
  split, and support/query fingerprints.
- `tests/test_private_leakage_guard.py` exercises the scanner and the
  disqualification registry.
- `skillbench_v043/cli.py smoke` writes evaluator-private payloads only under
  `.local_private/v043_cli_smoke/` and refuses to reuse an exposed task ID; the
  CLI regression test verifies that the public output tree has no private files.
- The official test generator must use `official_test_family_ids()` or an
  equivalent filter before reserving test families.

## 6. Residual risk

The historical commits remain public. The exposed support/query rows could be
memorized by a model trained on the public repository. Therefore:

1. No exposed task may be used in the official test.
2. No exposed family may be used in the official test split.
3. The official test must be newly generated and its evaluator artifacts must
   remain outside the public repository.
4. A commitment hash must be recorded before the test is opened.
5. Any model that consumed public repository data must be documented; the
   official evaluation must not rely on memorized smoke rows as evidence.

## 7. Verification commands

```powershell
.\.venv\Scripts\python.exe scripts\check_private_leakage.py
.\.venv\Scripts\python.exe -m pytest -q tests\test_private_leakage_guard.py
git ls-files results/v043_stage1_smoke
```

Expected: scanner `PASS`; tests pass; the three evaluator-private files are
absent from `git ls-files`; local copies remain under `.local_private/`.
