# v0.43 Official-Test Commitment Plan

**Status:** **NOT ISSUED**
**Reason:** `official_test_ready=false`; the development readiness checklist has
blocking failures.
**Official test:** **NOT OPENED**.

## Purpose

This document defines the commitment that must be issued *before* the official
v0.43 test is opened. It is a plan, not a completed commitment. No official-test
task, support row, query label, evaluator file, or private test family list is
present in this document or in the public repository.

## Commitment contents

When every mandatory readiness item passes and external review accepts the
protocol, the commitment must contain:

1. The exact Git commit SHA of the frozen implementation.
2. SHA-256 hashes of the frozen model configuration, seeds, thresholds, and
   analysis plan.
3. SHA-256 hashes of the official-test generator and evaluator code.
4. A private manifest of reserved test families, stored outside Git.
5. A private evaluator manifest with support/query rows and labels, stored
   outside Git.
6. The predeclared success criteria and the exact analysis commands.
7. A statement that no final-test result was used for model selection,
   threshold selection, or feature development.
8. The date and reviewer identity for the external technical review.

## Storage rules

- The commitment and all official-test artifacts must live under a private,
  Git-ignored location, for example `.local_private/v043_official_test/`.
- Only the commitment *hash* and a public-safe status record may be committed.
- No support rows, query labels, target programs, private family lists, or
  evaluator metadata may be committed to the public repository.
- The commitment must be recorded before the test is generated or opened.

## Current blockers

- DeepSets abstention gate failure.
- Retention improvement/forgetting failure.
- Partial 8-example new-operation result.
- No external technical review.
- No issued commitment hash.

## Decision

The official v0.43 test remains closed. This plan must not be interpreted as a
commitment to run it, and the existence of a commitment plan does not make the
benchmark ready.
