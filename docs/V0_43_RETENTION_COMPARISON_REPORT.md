# v0.43 Retention Strategy Comparison Report

**Task:** `2026-10-25-TASK-010`
**Protocol:** development-only correction/retention loop
**Machine-readable result:** `results/v043_task010/retention_comparison.json`
**Result SHA-256:** `11cdd53bb1b859a8cf502269c9f74720cc2f6c2d4a9119c9b598f136f852e1fb`
**Official v0.43 test:** **NOT OPENED**
**`official_test_ready`:** `false`
**Decision:** **FAIL — no shared-parameter strategy met the retention gate**

## 1. Purpose

This comparison audits the historical retention claim and tests whether a
stronger replay policy can meet the frozen retention gate. It compares:

1. **`control`** — FIFO replay of recent corrections (the historical behavior).
2. **`balanced_replay`** — equalized sampling across all previously seen skills.
3. **`balanced_support_replay`** — half balanced corrections, half original
   support examples, also balanced across seen skills.
4. **Per-skill memory/program track** — a separate reference, not a
   shared-parameter consolidation method.

## 2. Design

| Item | Value |
|---|---|
| Independent sequences | 3, one per seed (`430043`, `430044`, `430045`) |
| Skills per sequence | 20 |
| Family overlap between sequences | 0 for every seed pair |
| Rounds per skill | 4 |
| Interaction queries per round | 50 |
| Independent evaluation queries per skill | 100 |
| Evaluation disjointness | evaluation queries are disjoint from support, interaction, and correction rows |
| Exposed families used | none |
| Fallback sequences used | none |
| Shared learner | one `SGDClassifier` over 15 deterministic features |
| Memory reference | per-skill program-induction memory |
| Official test | not opened |

Each seed uses a disjoint development-family pool. All strategies for a seed use
the identical skill order, support budget, interaction budget, and evaluation
queries, so the comparison is paired.

## 3. Gate

A shared-parameter strategy passes only if, for **every** seed:

- parameter improvement from round 0 to round 3 is at least **+10 percentage
  points**, and
- average forgetting is at most **5 percentage points**.

The memory/program track is reported separately and is not counted as a
shared-parameter pass.

## 4. Results

### 4.1 Aggregate across three seeds

| Strategy | Parameter improvement (mean) | Worst seed | Average forgetting (mean) | Worst-task forgetting | Seeds passing |
|---|---:|---:|---:|---:|---:|
| control | +2.57 pp | +0.80 pp | 20.92% | 50% | 0 / 3 |
| balanced replay | +7.02 pp | +2.00 pp | 22.48% | 73% | 0 / 3 |
| balanced support replay | +5.87 pp | −1.30 pp | 20.38% | 59% | 0 / 3 |
| memory/program reference | +4.55 pp | +2.45 pp | 0.00% | 0% | separate reference |

### 4.2 Per-seed detail

| Seed | Strategy | Parameter improvement | Average forgetting | Worst-task forgetting |
|---:|---|---:|---:|---:|
| 430043 | control | +0.80 pp | 22.15% | 50% |
| 430043 | balanced replay | +11.55 pp | 25.30% | 73% |
| 430043 | balanced support replay | −1.30 pp | 20.10% | 58% |
| 430044 | control | +1.95 pp | 21.30% | 48% |
| 430044 | balanced replay | +7.50 pp | 24.15% | 43% |
| 430044 | balanced support replay | +9.85 pp | 18.75% | 52% |
| 430045 | control | +4.95 pp | 19.30% | 46% |
| 430045 | balanced replay | +2.00 pp | 18.00% | 49% |
| 430045 | balanced support replay | +9.05 pp | 22.30% | 59% |

The per-seed worst-skill parameter improvement is negative for every strategy
and seed (between −1.3 and −3.7 percentage points). Pooled improvement
therefore hides skills that get worse.

## 5. Interpretation

1. **Balanced replay improves the average more than FIFO replay, but not
   reliably.** The mean rises from +2.57 pp to +7.02 pp, yet the worst seed is
   only +2.00 pp and forgetting rises to 22.48%.
2. **Support replay does not solve forgetting.** It has the lowest mean
   forgetting (20.38%) but still four times the 5% gate, and one seed has
   negative improvement.
3. **The one apparent success is not a pass.** Seed `430043` with balanced
   replay reaches +11.55 pp, but its average forgetting is 25.30%, far above the
   gate. A strategy that learns faster while forgetting more does not satisfy
   the joint criterion.
4. **The memory reference has no forgetting but also does not reach the
   improvement target.** It is a legitimate per-skill reference, not evidence
   that the shared model retains skills.
5. **The historical failure is reproduced, not repaired.** The control result
   here is in the same range as the earlier TASK-008 result, and the two new
   strategies do not change the conclusion.

The correct conclusion is that replay balancing alone is insufficient under the
current feature learner and gate. This is a narrower claim than "continual
learning is impossible"; it says the tested shared-parameter strategies and
budgets do not meet the frozen retention criterion.

## 6. What may and may not be claimed

**May be claimed:**

- Three independent, family-disjoint sequences were evaluated with paired
  strategies.
- Balanced replay raises mean parameter improvement relative to FIFO replay.
- No shared-parameter strategy met the joint retention gate on all seeds.
- The memory/program reference has zero measured forgetting but does not reach
  the improvement target.

**May not be claimed:**

- That balanced replay is a validated retention solution.
- That the memory track demonstrates shared-parameter retention.
- That the official v0.43 test supports any of these numbers.

## 7. Consequence

Per the task rule, retention optimization stops here. The official v0.43 test
remains closed, `official_test_ready=false`, and the retention gate remains
**FAIL** in the readiness checklist.
