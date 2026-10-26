# Nucleus Pulse

**A research program on spiking computation, compositional program induction, calibrated abstention, and continual learning.**

Nucleus Pulse is an incremental, evidence-driven project built around a strict separation between two questions:

1. **Does the implementation work?** This is an engineering question, answered by unit tests, reference comparisons, integrity checks, and reproducible runs.
2. **Is there evidence that a learning claim is true?** This is a scientific question, answered only by preregistered experiments, frozen test sets, equal-information baselines, uncertainty intervals, and explicit failure reporting.

The repository contains a discrete-time pulse simulator and a benchmark for compositional skill learning. They are evaluated separately. Neither is presented as a product, a brain simulation, or a completed model of cognition.

---

## Status at a glance

| Item | State |
|---|---|
| Latest official learning run | **v0.42** (40 seeds, two temporal tasks) |
| Latest benchmark specification | **v0.43** (design frozen in documentation; not executed) |
| Latest implemented component | **v0.43-stage1** (task grammar, hypothesis auditor, leakage checks) |
| Official v0.43 test | **Not open** |
| Pulse simulator | Implemented; mechanism-level validation only |
| Meta-learner, feedback loop, retention tests | **Not implemented** |
| `tests/v043` | **9 passed** |
| Full `tests/` suite | **Not runnable in the authoring environment** without optional dependencies (`matplotlib`) |

The latest committed state and the complete audit trail are in [`docs/handoff/LATEST_REPORT.md`](docs/handoff/LATEST_REPORT.md).

---

## Contents

- [1. Overview](#1-overview)
- [2. Research question and claim boundary](#2-research-question-and-claim-boundary)
- [3. Epistemic contract](#3-epistemic-contract)
- [4. Current status](#4-current-status)
- [5. Results to date](#5-results-to-date)
- [6. Architecture](#6-architecture)
- [7. The v0.43 benchmark design](#7-the-v043-benchmark-design)
- [8. Repository layout](#8-repository-layout)
- [9. Installation and reproducibility](#9-installation-and-reproducibility)
- [10. Testing and verification](#10-testing-and-verification)
- [11. Data and artifact integrity](#11-data-and-artifact-integrity)
- [12. Limitations and non-claims](#12-limitations-and-non-claims)
- [13. Roadmap](#13-roadmap)
- [14. Documentation](#14-documentation)
- [15. License and citation](#15-license-and-citation)
- [16. Contact](#16-contact)

---

## 1. Overview

Nucleus Pulse began as an exploration of event-driven spiking computation: signals travel as discrete pulses through individual synapses rather than as dense vectors through a layered network. The simulator implements per-synapse currents, leaky integrate-and-fire cells, synaptic saturation, inhibition, conduction delays, and a temporal coincidence (AND) cell.

The project then shifted its center of gravity. A series of learning experiments (v0.28–v0.42) compared pulse-based readers and reservoir representations against strong supervised baselines. Those experiments produced useful negative and diagnostic results, but they did not establish a general learning advantage for the pulse layer. The project therefore moved to a more fundamental question:

> Can a system induce a compositional program from a few examples, recognize when the examples are insufficient, detect operations outside its library, learn a new operation from additional examples, and improve through corrective feedback without forgetting earlier skills?

That question is the subject of the v0.43 benchmark. The current implementation covers only the first layer: a bounded, fully enumerable task grammar, an exact hypothesis auditor, family-level splits, and leakage checks. It is a tooling milestone, not a benchmark result.

### What this repository is

- A reproducible research codebase with frozen protocols, seeds, hashes, and archived results.
- A pulse simulator whose mechanics are tested independently of any learning claim.
- A benchmark specification that separates answer quality from abstention quality.
- A public audit trail that records failures, corrections, and unexecuted checks.

### What this repository is not

- It is not a claim that the pulse layer is generally superior or inferior to conventional models.
- It is not a claim that spiking hardware saves energy; no hardware or energy measurement is included.
- It is not a brain simulation.
- It is not a completed benchmark; the official v0.43 test has not been opened.
- It is not evidence that STDP is universally ineffective. The v0.42 comparison is confounded by differences in standardization and optimization budget.

---

## 2. Research question and claim boundary

### 2.1 Primary question (v0.43)

Given a small support set of input-output examples drawn from a declared operation library, can a system:

1. **Induce** a compositional program when the support set identifies it?
2. **Abstain** when multiple semantically distinct programs remain consistent with the support set?
3. **Detect** when a required operation is absent from its library and abstain before instruction?
4. **Learn** that operation from additional input-output examples within a declared meta-grammar?
5. **Improve** through selective corrective feedback without losing previously learned skills?

### 2.2 Two separate paths

The benchmark deliberately separates two claims that are often conflated:

**Path A: Closed-world program induction.** The target program is generated from a declared grammar and is present in the hypothesis space. Success measures whether the support examples are sufficient to identify the program or function. This is inference within a bounded language, not proof of general skill acquisition.

**Path B: Learning an operation outside the library.** An operation type is removed from the runtime library and the base grammar. Before instruction, the system should classify the query as out of scope and abstain. It then receives 8, 16, and 32 input-output examples from a teacher interface. Success measures generalization to unseen inputs within a declared meta-grammar. This is not arbitrary function induction from finite data.

### 2.3 Claim boundary

The strongest evidence for reusable learning must come from cross-task meta-learning, new-operation learning, feedback-driven improvement, and retention on queries that never enter any update. A closed-world induction result alone is reported as program inference, not as general learning.

---

## 3. Epistemic contract

### 3.1 Operating rules

Every experiment in this repository is expected to satisfy the following rules:

1. **One primary question per experiment.** Secondary observations are labeled as exploratory.
2. **Preregistration before test.** Seeds, thresholds, baselines, exclusions, and analysis units are frozen before the final test is examined.
3. **Development and test separation.** Hyperparameters and abstention thresholds are selected on development data only. The final test is evaluated once.
4. **Family-level splits.** Program families, not individual examples, define the train/dev/test boundary. Confidence intervals use family-level cluster bootstrap, not thousands of correlated queries as if they were independent.
5. **Abstention is not coverage.** Answer quality and abstention quality have separate gates. A high coverage threshold is never applied to cases where abstention is the correct behavior.
6. **Equal-information baselines.** Every baseline receives the same inputs, support examples, and evaluation queries. A weak baseline is not evidence of a strong method.
7. **Failures are reported.** Negative results, excluded seeds, aborted runs, and unexecuted checks remain visible. The repository uses `NOT RUN` rather than silently omitting work.
8. **Past results are immutable.** Corrections are appended as new documents. The v0.42.1 interpretation note is an example: it changes the interpretation, not the raw results or the protocol.
9. **Engineering correctness is not learning evidence.** A passing unit test proves that code behaves as specified; it does not prove that a model learned.

### 3.2 Five evidence levels

The repository uses a five-level claim ladder. A claim is only as strong as the highest level it has actually reached.

| Level | Name | What it requires | Example in this repository |
|---|---|---|---|
| **L1** | Implementation evidence | Unit and integration tests show that the specified mechanics execute correctly. | `tests/v043`: grammar enumeration, split audit, leakage checks |
| **L2** | Mechanism evidence | A component behaves as designed under controlled inputs, without a trained reader. | v0.27 temporal AND cell; reference recurrence comparison |
| **L3** | Diagnostic evidence | A manipulation changes activity or features in a measured way, but no reader is trained and no accuracy claim is made. | v0.40 and v0.41 input-gain diagnostics |
| **L4** | Learning evidence | A reader or learner is trained and evaluated on held-out data with uncertainty intervals and a preregistered gate. | v0.29, v0.37, v0.39, v0.42 |
| **L5** | Benchmark evidence | Preregistered, frozen test, equal-information baselines, family-level intervals, all gates reported, and independent replication or re-execution. | **None yet** |

Current placement:

- v0.43-stage1 is **L1**.
- v0.27 is **L2**.
- v0.40 and v0.41 are **L3**.
- v0.29, v0.37, v0.39, and v0.42 are **L4**, each with explicit scope limits.
- No claim in this repository has reached **L5**.

### 3.3 What a negative result means here

A failed gate is reported as a failure of the specified protocol, not as proof that a broad class of methods is impossible. For example, the v0.42 result does not show that all STDP rules fail. It shows that one frozen one-pass STDP-style reader, trained on a fixed reservoir representation, did not outperform a standardized logistic reader under the tested budget, and that the comparison is additionally confounded by differences in standardization and optimization.

---

## 4. Current status

### 4.1 Version map

| Version | Question or change | Evidence status |
|---|---|---|
| v0.27 | Can a saturated pulse circuit implement a temporal AND coincidence? | L2 mechanism test; parameters set manually, not learned |
| v0.28.1 | Sequential STDP with a pulse-based reader | L4 learning run |
| v0.29 | Sequential learning and forgetting versus a linear reader | L4; no seed met the 85% eligibility gate, so the forgetting comparison is inconclusive |
| v0.30 | Learning curve: decodability and sample efficiency across feature sets | L4 learning run |
| v0.31 | Incremental learning with 4, 8, and 16 temporal windows | L4 learning run |
| v0.32 | Effect of repeated passes over the same training data | L4 learning run |
| v0.33 | Heterogeneous recurrent delays (1–4 ticks) versus a one-tick delay | L4 learning run |
| v0.34 | Dale-style inhibitory structure (25% inhibitory sources) | L4 learning run |
| v0.35 | Activity-adaptive inhibitory gain | L4; the first run was discarded after an EMA ordering bug and all seeds were rerun |
| v0.36 | Official local adaptive inhibition run (slope 6 selected on development) | L4 learning run |
| v0.37 | Official local adaptive inhibition comparison | L4; small descriptive effects, no strong advantage |
| v0.38 | EMA time-constant tuning | Development only |
| v0.39 | Official confirmation of `tau=8` versus `tau=4` | L4; no confirmed advantage |
| v0.40 | Input-gain diagnostic, one seed | L3 component diagnostic |
| v0.41 | Input-gain diagnostic, eight development seeds | L3; confidence intervals include zero |
| v0.42 | One-pass reward-modulated STDP reader versus a linear reader | L4; large negative difference, confounded comparison |
| v0.42.1 | Interpretation correction for v0.42 | Documentation only |
| v0.42.2 | v0.43 specification after pre-build review | Documentation only |
| v0.43-stage1 | Bounded grammar, task generator, exact auditor, family splits, leakage checks | L1 tooling only; no learning result |

### 4.2 What is implemented

- `pulse_core/`: event-driven LIF simulation, per-synapse currents, saturation, inhibition, reference recurrence, temporal tasks, and legacy learning code.
- `skillbench_v043/`: finite typed Boolean DSL, full enumeration, family definition, deterministic splits, support/query generation, identifiability classification, leakage audit, and a CLI.
- `exploration_lab/`: a separate local, reward-free exploration loop with three small games, a persistent step-based spiking agent, local STDP, checkpoint/resume, and a live terminal display. TASK-012 adds a shared 8-clone city extension with simultaneous movement and collision/interaction logging. See `docs/EXPLORATION_LAB.md` and `docs/EXPLORATION_CITY.md`. It is not a benchmark and makes no learning or skill claim.
- `experiments/`: runnable scripts for v0.27 through v0.42.
- `tests/`: unit and integration tests, including the nine v0.43 tests.
- `results/`: archived outputs, reports, CSV/JSON artifacts, and plots.
- `docs/`: specifications, protocols, architecture notes, changelog, and the mandatory handoff system.
- `scripts/`: handoff validator, optional Brian2 comparison, and post-run analysis helpers.

### 4.3 What is not implemented

- The v0.43 meta-learner and the 10,000-episode meta-training corpus.
- The full baseline suite, including the DeepSets meta-learning baseline.
- Calibrated abstention thresholds and the abstention evaluation pipeline.
- New-operation learning at 8, 16, and 32 examples.
- The three-round error-correction loop.
- Sequential retention tests across at least 20 skills.
- Preregistration hashes, frozen test manifest, and the official v0.43 test run.

The CLI reports this directly:

```json
{
  "family_minimums_met": true,
  "official_test_ready": false,
  "blocked_until": [
    "10,000 meta-learning episodes",
    "100 train/20 dev/30 test families",
    "all baselines",
    "feedback loop",
    "retention tests",
    "preregistration and hashes"
  ]
}
```

The family counts currently meet the numeric minimums, but the splits are candidate splits and have not been frozen in a final preregistration manifest. The other blockers remain open.

---

## 5. Results to date

This section reports the results that most directly shaped the project. Each subsection states what the result supports and what it does not support. Full reports, per-seed data, and plots are in `results/`.

### 5.1 v0.42: one-pass STDP reader versus a linear reader

**Question.** Can a reward-modulated STDP-style reader on a fixed 16-window reservoir representation outperform a standardized linear reader with 50, 100, or 200 training examples?

**Protocol.**

- 40 independent seeds: `5200..5239`.
- Two tasks: `sync` (synchrony) and `order` (A before B by 2–4 spikes).
- 200 balanced training examples per task and seed, with nested prefixes at 50 and 100.
- 1,000 independent balanced test examples per task and seed.
- Both readers receive the same frozen reservoir features: spike rates over 16 windows, 384 features.
- The STDP reader makes one pass per example with frozen settings.
- The linear reader is `StandardScaler + LogisticRegression(C=1, max_iter=250)` with settings fixed before the run.
- Paired seed-cluster bootstrap with 40 resampling units; the two tasks are summarized into one difference per seed.

**Result at 200 examples.**

| Training examples | STDP accuracy (95% CI) | Linear accuracy (95% CI) | Difference, STDP − linear (pp, 95% CI) |
|---:|---:|---:|---:|
| 50 | 50.96% [50.56, 51.38] | 63.17% [62.35, 63.97] | −12.20 [−13.04, −11.35] |
| 100 | 51.40% [51.02, 51.77] | 66.76% [65.90, 67.58] | −15.36 [−16.23, −14.51] |
| 200 | 52.26% [51.71, 52.82] | 71.06% [70.31, 71.81] | −18.81 [−19.71, −17.91] |

At 200 examples by task: `sync` 51.56% versus 73.74% (−22.18 pp); `order` 52.95% versus 68.39% (−15.44 pp).

**Preregistered decision.** The gate required STDP to exceed the linear reader by at least +2 pp with a confidence interval excluding zero. It did not. The STDP reader stayed close to chance and the gap widened with more examples. The project stopped the STDP ablation series.

**What this does not show.** The comparison is not a clean causal estimate of the STDP rule. The STDP path used one pass per example, while the linear path used iterative LBFGS optimization and standardization. The −18.81 pp difference belongs to these two training pipelines, not to STDP in isolation. The v0.42.1 note records this correction explicitly.

### 5.2 v0.41: input-gain sensitivity across development seeds

**Question.** Does scaling input weights by 0.75× or 1.25× change reservoir activity, and does local adaptive inhibition reduce that sensitivity relative to fixed inhibition?

**Protocol.**

- Eight development seeds: `14111, 14129, 14147, 14165, 14183, 14201, 14219, 14237`.
- Two tasks: `sync` and `order`.
- 32 balanced sequences per seed and task from the training pool only.
- Input scales: 0.75×, 1.0×, 1.25×.
- Two conditions: fixed inhibition and local adaptive inhibition (slope 2.0, EMA tau 4.0, target rate 0.13, gain bounds 0.5–2.0).
- 96 records total. No reader was trained; no validation or test set was created.

**Result.** Local adaptive inhibition reduced the spike-count sensitivity metric by −0.35 pp for `sync` (95% CI [−1.18, +0.43]) and −1.48 pp for `order` (95% CI [−3.82, +0.11]). Both intervals include zero. Feature variability did not improve: the relative L2 change increased slightly under local inhibition (`sync` +0.0158 [0.0065, 0.0281]; `order` +0.0121 [−0.0005, 0.0228]).

**Interpretation.** This is a component diagnostic, not a learning result. It does not establish better robustness, accuracy, or energy efficiency. It does show that reducing spike-count variation does not automatically make the temporal representation more stable.

### 5.3 v0.39: EMA time constant, official confirmation

**Question.** Does `tau=8` outperform `tau=4` for local adaptive inhibition on new official seeds?

**Protocol.** 40 seeds (`12000..12039`), two conditions, 6,000 unique training examples, 1,000 validation, and 1,000 independent test examples per task and seed. The primary endpoint was fixed at 10 passes. The test set was used for evaluation only.

**Result.** The balanced test accuracy difference (`tau=8 − tau=4`, averaged over tasks within each seed) was **−0.124 pp**, with a 95% CI of **[−0.301, +0.043]**. Seed differences were 20 positive, 19 negative, and 1 zero. The interval includes zero.

**Interpretation.** No confirmed advantage for `tau=8`. The development-set preference did not replicate as a test-set advantage.

### 5.4 v0.37: local adaptive inhibition, official comparison

**Protocol.** 40 official seeds (`11000..11039`), 24 cells, 16 temporal windows, 6,000 training examples, 1,000 validation, and 1,000 test examples per task and seed. The local slope was selected on separate development seeds (`9200..9207`) and frozen before the official run.

**Result at 10 passes (local adaptive minus fixed inhibition).**

- `sync`: **+0.27 pp** [0.03, 0.52].
- `order`: **+0.15 pp** [−0.14, +0.43].

The 85% validation threshold was reached by 7/40 `sync` seeds with fixed inhibition and 8/40 with local inhibition; no `order` seed reached it under any condition. Activity was similar across conditions (about 96–97 spikes per example).

**Interpretation.** The local mechanism produced a small descriptive improvement in `sync` and an inconclusive difference in `order`. It did not produce a broadly robust learning advantage.

### 5.5 v0.29: sequential learning and the limits of a forgetting claim

**Protocol.** 40 seeds, two tasks, 500 independent IID test examples per task, and a 300-example validation set for task 1. The preregistered eligibility threshold for a meaningful forgetting comparison was 85% validation accuracy on task 1 before learning task 2. No replay of task 1 was allowed during task 2.

**Result.** Zero of 40 seeds reached the 85% validation threshold for the pulse reader, and zero of 40 for the linear reader. There were no jointly eligible seeds. The forgetting comparison was therefore not available.

**Interpretation.** This is a methodological result: the experiment did not create a valid forgetting comparison. The correct conclusion is that the base task was not learned well enough to make the forgetting measurement interpretable, not that one method forgets less than the other.

### 5.6 Earlier work and corrections

The v0.30–v0.35 runs examined learning curves, incremental windows, repeated passes, heterogeneous delays, Dale-style inhibition, and activity-adaptive inhibition. Their reports are the authoritative source for their numbers. Two integrity notes are worth highlighting:

- **v0.35:** The first run was excluded. The EMA was updated before the first tick's gain was computed, making the initial gain slightly less than 1.0. The order was corrected, hashes were refrozen, and all 40 seeds were rerun from scratch. Only corrected results are reported.
- **v0.42.1:** The interpretation of v0.42 was corrected without changing the raw results or the protocol. The original report describes a difference between two pipelines; it does not isolate the causal effect of STDP.

---

## 6. Architecture

### 6.1 Pulse simulator (`pulse_core/`)

The simulator is a discrete-time, event-driven software model. It is not a physical circuit simulator and does not model membrane chemistry or detailed channel kinetics.

**Synapse state.** Each synapse has a source, target, weight, delay, optional polarity, optional synaptic time constant, and optional saturation cap. Each synapse keeps an independent current state:

```text
I_syn <- I_syn * exp(-dt / tau_syn)
```

When a pulse arrives:

```text
I_syn <- clip(I_syn + polarity * weight, -cap, +cap)   # if cap is defined
```

The cell's input current is the sum of its incoming synaptic currents. Independent per-synapse state means that a repeated train on one saturated synapse cannot increase its current without bound, while a second input can still contribute independently.

**Cell dynamics.** Each leaky integrate-and-fire cell follows:

```text
V <- V_reset + (V - V_reset) * exp(-dt / tau_mem)
     + (1 - exp(-dt / tau_mem)) * I_total
```

If `V` crosses the threshold, the cell emits a pulse and resets. If `clear_synaptic_current_on_spike=True`, incoming currents are also cleared. Every internal pulse is delayed by at least one time step to preserve causality.

**Inhibition.** A synapse with `polarity=-1` contributes a negative current. When it coincides with excitatory input, it can suppress the cell's response. This behavior was tested in v0.27.

**Temporal AND cell.** The v0.27 circuit uses one LIF cell with two input synapses:

- weights 1.0, per-synapse current cap 1.0, `tau_syn=0.8`;
- `tau_mem=1.0`, threshold 1.02, refractory period 2 steps.

Two pulses in the same step, or one step apart, exceed the threshold. A single pulse or a train on one channel does not. A two-step separation did not fire in the test set. These parameters were set manually; the cell did not learn the AND operation from data.

**Reference validation.** The numerical path is compared against an independent implementation of the discrete recurrence. In the reference test set, the current and voltage differences were numerically zero. This is an independent code check, not an external simulator validation.

**Optional Brian2 comparison.** `scripts/compare_brian2_optional.py` is included, but it was **not run** in the authoring environment because Brian2 was not installed and package download was unavailable. No external Brian2 agreement is claimed.

**Performance limits.** The Python event loop is slower than a single-sample MLP inference and much slower than batched MLP inference in the measured environment. The absence of dense matrix multiplication is not, by itself, a performance benefit. Any energy or latency advantage would depend on implementation and hardware; neither is measured here.

### 6.2 Readout and learning

The learning experiments use a frozen reservoir and a readout trained on spike-rate features over 16 temporal windows (384 features). Two readouts appear in the record:

- a one-pass reward-modulated STDP-style reader with frozen settings;
- a supervised linear reader (`StandardScaler + LogisticRegression`) with frozen settings.

The tasks are synthetic temporal classification tasks:

- `sync`: detect synchronous input events;
- `order`: detect whether event A precedes event B by 2–4 spikes.

The reservoir itself does not learn from labels. This isolates the readout comparison from representation learning, but it also means the experiments do not test a fully plastic spiking network.

### 6.3 v0.43 skill benchmark (`skillbench_v043/`)

The benchmark is a finite, typed Boolean DSL over two integer inputs. It has four components:

1. **`dsl.py`**: atoms, literals, expressions, normalization, AST hashing, family identity, semantic masks, and full enumeration.
2. **`generator.py`**: deterministic family splits, support/query generation, ambiguous and out-of-scope tasks, contradictory support, and split audits.
3. **`audit.py`**: leakage and overlap checks across families, ASTs, semantic functions, and splits.
4. **`cli.py`**: `inspect` and `smoke` commands for mechanics checks.

The public task payload does not contain the target AST, the family identity, evaluator metadata, or final query labels. The evaluator record is kept separate.

---

## 7. The v0.43 benchmark design

### 7.1 Grammar and domain

- Input domain: `x0, x1` each in `0..99`, for 10,000 distinct input points.
- Atoms: `LT(x_i, t)` with `t ∈ {25, 50, 75}`, plus `EVEN(x_i)`, `DIV3(x_i)`, and `DIV5(x_i)`, for `i ∈ {0,1}`. Twelve atoms in total.
- Literal: an atom or `NOT(atom)`.
- Expression: a literal, a binary operation (`AND`, `OR`, `XOR`) over two literals, or a nested shape with two binary nodes over atoms.
- Maximum AST depth: 3. At most two binary nodes. `NOT` is not allowed over a compound expression.
- Binary operands are normalized because `AND`, `OR`, and `XOR` are commutative.

This grammar is deliberately bounded and fully enumerable. It is **not** all Boolean trees of depth 3. Any future claim applies only to this declared language unless the hypothesis space is expanded and preregistered again.

Full enumeration yields:

- **9,348** syntactic programs;
- **4,570** distinct semantic functions on the complete 10,000-point domain.

### 7.2 Families and splits

A **family** is a normalized structural schema, not a single example and not a simple renaming of symbols or change of constant values. The family identity preserves atom type, variable repetition pattern, and operator structure while ignoring absolute variable names and constant values.

Families are assigned to splits by a fixed SHA-256 bucket that is independent of labels, in a 70/15/15 proportion. The current candidate split contains:

| Split | Families | Canonical targets |
|---|---:|---:|
| Train | 649 | 3,286 |
| Development | 140 | 665 |
| Test | 134 | 619 |

These counts meet the numeric family minimums, but the split is not yet frozen in a final preregistration manifest. The official test remains closed.

### 7.3 Support sets and queries

- Each Path A task begins with **8 support examples**, balanced 4 positive and 4 negative.
- Each task has **100 unique queries**, exactly 50 positive and 50 negative.
- Support inputs and query inputs do not overlap.
- At least 30% of queries should be discriminative relative to near-competitor programs when possible.
- For an ambiguous task, consistent programs must disagree on at least 20% of the discriminative query set; otherwise the task is resampled.
- Label balance prevents a constant "all false" baseline from reaching 80% through distribution bias.

### 7.4 Task classes

Every task is assigned to one of four preregistered classes before evaluation:

| Class | Definition | Correct behavior | Gate |
|---|---|---|---|
| `IN_SCOPE_IDENTIFIABLE` | Exactly one semantic class remains consistent with the support set within the query scope. | Answer | Coverage ≥ 90%; selective risk ≤ 5% |
| `IN_SCOPE_AMBIGUOUS` | Two or more semantic classes remain consistent and disagree on at least 20% of discriminative queries. | Abstain | Abstention ≥ 90%; false acceptance ≤ 10% |
| `OUT_OF_SCOPE` | The query requires an operation absent from the current library. | Abstain before instruction | Abstention ≥ 90%; false acceptance ≤ 10% |
| `CONTRADICTORY_SUPPORT` | No program in the grammar explains all support examples. | Abstain | Abstention ≥ 90% |

The auditor computes, for each task:

- `consistent_program_count`: syntactic programs matching all support examples;
- `consistent_semantic_classes`: distinct functions on the query domain;
- `candidate_disagreement`: fraction of queries where consistent candidates disagree;
- `support_identifiability`: whether the support set identifies one function in scope.

These evaluator-side statistics never enter the model.

### 7.5 Baselines

The benchmark requires equal-information baselines:

1. **Majority/constant** reference; with 50/50 queries it should be near 50%, not 80%.
2. **Support retrieval**: nearest support example, measuring how far memorization goes.
3. **Per-task linear or tree model** fitted on the support set only; a low-data reference, not the only learning baseline.
4. **Compositional program induction**: search or inference within the declared grammar using the task support.
5. **Neural meta-learning baseline (DeepSets)**: each support example is encoded by a two-hidden-layer MLP `φ` (128 units, ReLU) that receives the input and label; the eight encodings are mean-pooled; the pooled context and query representation pass through a two-hidden-layer MLP `ρ` (128 units, ReLU) to a binary logit. The network trains on at least 10,000 task episodes from at least 100 training families. It never receives the task ID, target AST, or evaluation query labels.
6. **Oracle evaluator**: knows the true program only to audit generator and interpreter correctness; it is not a competitor or a learning result.

### 7.6 Error correction and retention

The static "8 examples, then 100 questions" protocol is not sufficient to measure learning from mistakes. A separate protocol adds:

1. **Round 0:** 8 support examples, then 50 balanced interaction queries. A new evaluation set is held by the evaluator and never shown to the system.
2. **Selective feedback:** after each interaction query, the system receives the correct label only if its answer was wrong. Correct answers reveal nothing new.
3. **Rounds 1–3:** 50 new interaction queries per round, with the same selective correction. The system may update its state or parameters according to a frozen algorithm.
4. **Evaluation after each round:** a fixed set of 100 new queries that never enters any update. The same set is used across rounds so that differences are paired.
5. **Sequential skills:** at least 20 skills are learned in sequence from train/continual families separate from test. After each new skill, frozen evaluation queries for earlier skills are re-administered without replaying their support examples.
6. **New operation:** before instruction, the system must abstain according to the out-of-scope gate. It then receives 8, 16, and 32 input-output examples and is evaluated on new fixed queries after each level.
7. **Leakage prevention:** final evaluation sets do not overlap with support, interaction, or correction data. No final label enters any update. Every correction is logged with round, task ID, and hash.

Preregistered gates for this protocol:

- At least **+10 pp** improvement from Round 0 to Round 3 on the fixed new-query set, with a paired family-level 95% interval above zero.
- Coverage increases from 8 to 16 to 32 new-operation examples while selective risk stays at or below 5% on accepted answers.
- Average forgetting of earlier skills is at most **5 pp** below the best post-learning performance for each skill, with worst-task forgetting also reported.

If a gate fails, the result is reported as a failure or as inconclusive evidence for that protocol. The threshold is not changed after seeing the test.

### 7.7 Readiness gate

The official v0.43 test does not open until all of the following exist:

- a frozen task generator with a documented grammar, depth limits, and types;
- automatic enumeration of consistent programs and semantic equivalence classes after support;
- a balanced 50/50 query generator with hypothesis-discrimination checks;
- at least 30 reserved test families (target 50+), at least 100 train families, at least 20 development families, and at least 10,000 meta-learning episodes;
- separation of Path B hidden operations from the runtime library, with a documented prior/meta-grammar;
- a leakage auditor for ASTs, inputs, labels, task IDs, and library contents;
- the complete baseline suite, including the meta-learning network, with equal settings;
- an abstention evaluator that tests each class independently and does not mix thresholds;
- a feedback loop separated from final evaluation, with a correction log;
- retention tests after every sequential skill;
- preregistration, seeds, SHA-256 hashes, and reproducible run commands.

Any missing item means the status is **not ready for an official test**. The pulse layer is not expanded and STDP tuning is not resumed before this gate is passed and a reproducible practical benefit is shown.

---

## 8. Repository layout

```text
.
├── pulse_core/                 Event-driven LIF simulator, tasks, learning, reference recurrence
│   ├── model.py                Synapses, cells, network, per-synapse currents
│   ├── reference.py            Independent discrete-recurrence reference
│   ├── tasks.py                Synthetic temporal tasks (sync, order)
│   ├── learning.py             Legacy supervised coincidence learner
│   └── sequential_stdp.py      Sequential STDP experiments
├── skillbench_v043/            v0.43 finite DSL benchmark
│   ├── dsl.py                  Grammar, enumeration, normalization, family identity
│   ├── generator.py            Splits, support/query generation, OOD and contradictory tasks
│   ├── audit.py                Leakage and overlap audits
│   └── cli.py                  `inspect` and `smoke` commands
├── exploration_lab/            Local reward-free games + persistent spiking exploration loop + shared city
├── experiments/                Runnable experiment scripts for v0.27–v0.42
├── tests/                      Unit and integration tests
│   └── v043/                   Nine v0.43 grammar/audit/leakage tests
├── results/                    Frozen outputs, reports, CSV/JSON, plots
├── docs/                       Specifications, protocols, architecture, changelog, handoff
│   ├── handoff/                LATEST_REPORT.md, history, logs, template
│   ├── V0_43_BENCHMARK_SPEC_AR.md
│   ├── NEXT_STAGE_V0_43_PLAN_AR.md
│   ├── IMPLEMENTATION_STAGE1_V0_43_AR.md
│   └── architecture_ar.md
├── scripts/                    Handoff validator, optional Brian2 comparison, analysis helpers
├── AGENTS.md                   OpenCode entry-point instructions
├── OPENCODE_RULES.md           Detailed mandatory workflow and sync rules
├── .gitattributes              LF policy for byte-identical clones
└── .gitignore                  Cache, environment, and editor exclusions
```

---

## 9. Installation and reproducibility

### 9.1 Requirements

The core simulator (`pulse_core/model.py`, `pulse_core/learning.py`, `pulse_core/tasks.py`) and the v0.43 benchmark (`skillbench_v043/`) use the Python standard library only. `pulse_core/sequential_stdp.py` requires NumPy.

The full experiment suite additionally uses:

- `numpy>=1.24`
- `scikit-learn>=1.3`
- `matplotlib>=3.8`
- `threadpoolctl>=3.1`
- `pytest>=8.0` for tests
- `torch>=2.0` for the experiment requirements file (not required by the v0.43 stage-1 tests)

The repository was developed and tested with **Python 3.14.6** on Windows. Other versions are not claimed to be verified.

Install the development and benchmark dependencies:

```bash
python -m pip install -r docs/requirements-dev.txt
python -m pip install -r docs/requirements-benchmark.txt
```

On Windows, the authoring environment uses the launcher:

```powershell
py -3.14 -m pip install -r docs/requirements-dev.txt
py -3.14 -m pip install -r docs/requirements-benchmark.txt
```

### 9.2 Inspect the v0.43 grammar

```bash
python -m skillbench_v043.cli inspect
```

Expected structure (values are deterministic):

```json
{
  "stage": "v0.43-stage1",
  "syntactic_programs": 9348,
  "unique_semantic_functions": 4570,
  "family_count_by_split": {"train": 649, "dev": 140, "test": 134},
  "family_minimums_met": true,
  "official_test_ready": false
}
```

### 9.3 Run the mechanics smoke test

```bash
python -m skillbench_v043.cli smoke --output results/v043_stage1_smoke
```

This writes a three-task smoke run. It checks generator mechanics, classification, and leakage separation. **It is not a performance result and must not be cited as one.** Running it again overwrites the archived output unless a different output directory is used; preserve the committed artifacts.

### 9.4 Run the tests

```bash
python -m pytest -q tests/v043
python scripts/verify_handoff.py
```

On Windows:

```powershell
py -3.14 -m pytest -q tests/v043
py -3.14 scripts/verify_handoff.py
```

### 9.5 Reproducing an experiment

Experiment scripts are in `experiments/` and write to `results/`. Each script records its seeds, protocol, and integrity checks. Before rerunning an official experiment, read its report and protocol file; do not overwrite archived results, and do not reuse a test set that has already been opened.

Example:

```bash
python experiments/run_stdp_vs_linear_v0_42.py
```

The exact command, seed list, and output paths are documented in the corresponding report and protocol files. Results are committed so that a reader can audit them without rerunning.

### 9.6 Run the exploration lab

The exploration lab is separate from the benchmark and has no reward, score, or
success criterion. It runs locally and writes checkpoints and gameplay logs under
the Git-ignored `.local_private/` tree.

```powershell
py -3.14 -m exploration_lab
```

`Ctrl+C` saves a checkpoint and exits. Use `--delay` to change the display speed,
`--resume` to continue from a checkpoint, and `--no-render` for a headless run.
Full commands, persistence rules, and non-claims are in `docs/EXPLORATION_LAB.md`.

### 9.7 Run the multi-agent city

`exploration_lab.city` places eight clones of the same spiking agent in one
32×32 city. The clones start with identical weights and independent RNG streams,
move simultaneously, and keep their own persistent LIF/STDP state. Blocked
moves, collisions, proximity events, and action counts are logged only; they are
not rewards, penalties, correctness labels, or error signals.

```powershell
py -3.14 -m exploration_lab.city
```

Headless 5,000-step pilot:

```powershell
py -3.14 -m exploration_lab.city --no-render --delay 0 --max-steps 5000 `
  --checkpoint-every 1000 --progress-every 1000 `
  --agents 8 --width 32 --height 32 --seed 12012
```

Raw JSONL rows and checkpoints stay under `.local_private/`. Aggregate pilot
evidence and limitations are in `docs/EXPLORATION_CITY.md`. The city makes no
claim of learning, cooperation, competition, or skill.

---

## 10. Testing and verification

### 10.1 Current test status

| Check | Command | Result |
|---|---|---|
| v0.43 tests | `python -m pytest -q tests/v043` | **43 passed** (last verified 2026-10-27; 267.86 s) |
| Full suite | `python -m pytest -q tests` | **199 passed** (last verified 2026-10-27; 278.25 s) |
| Handoff report structure | `python scripts/verify_handoff.py` | **PASS** |

The full suite passes in the current authoring environment. Install
`docs/requirements-dev.txt` before rerunning so optional test dependencies are
present. The city pilot adds ten focused tests; they are included in the 199.
The 43-test v0.43 suite is a supporting test set, not the official v0.43 test;
the official test remains closed.

### 10.2 What the v0.43 tests cover

- Full enumeration counts and semantic deduplication.
- Deterministic family assignment and split reproducibility.
- Family, AST, and semantic-function separation across splits.
- Support/query disjointness and 50/50 query balance.
- Exact consistency counting and semantic equivalence classes.
- Identifiability classification, including ambiguous, out-of-scope, and contradictory cases.
- Public payload separation from evaluator-private metadata.
- Leakage checks for target ASTs and task identities.

They do not cover a meta-learner, abstention calibration, feedback, retention, or benchmark performance, because those components do not exist yet.

### 10.3 What the handoff validator covers

`scripts/verify_handoff.py` checks that:

- `docs/handoff/LATEST_REPORT.md` exists and is non-empty;
- `docs/handoff/history/` contains valid `YYYY-MM-DD-TASK-NNN.md` reports;
- task IDs are unique;
- every report contains the eleven required sections in the required order;
- no section is a placeholder;
- the latest report mentions the newest task ID.

A failed validation means the task is incomplete, regardless of test results.

---

## 11. Data and artifact integrity

### 11.1 Frozen protocols and seeds

Official experiments record their seed lists, hyperparameters, and success criteria before execution. The v0.42 run used seeds `5200..5239`; v0.39 used `12000..12039`; v0.37 used `11000..11039`. Development seeds are kept separate from official seeds.

### 11.2 Hashes and manifests

The repository includes SHA-256 manifests for several stages (`docs/MANIFEST_SHA256*.txt`) and per-task handoff hashes under `docs/handoff/logs/`. The `.gitattributes` file enforces LF line endings, and the local Git configuration disables CRLF conversion, so text files remain byte-identical across clones. This matters because a checksum that changes on checkout is not a usable integrity check.

### 11.3 Immutable results and appended corrections

Raw results and reports are not rewritten to fit a later interpretation. When an interpretation changes, a new document is added. The v0.42.1 note is the clearest example: the numbers stay the same, but the causal claim is narrowed.

### 11.4 Excluded runs

A run that violates its protocol is excluded and documented. In v0.35, the first run updated the EMA before computing the first tick's gain. The implementation was corrected, hashes were refrozen, and all 40 seeds were rerun. Only corrected results appear in the report.

### 11.5 Public repository caveat

This repository is public and includes the stage-1 smoke artifacts, including `results/v043_stage1_smoke/*_evaluator_private.jsonl` and `test_public.jsonl`. These are synthetic smoke-test artifacts, not official benchmark data. The smoke task must not be reused as an official test task, and evaluator-private outputs should be reconsidered before a real test is opened. This caveat is also recorded in the TASK-002 handoff report.

### 11.6 Reproducibility limits

- The optional Brian2 comparison was **not run**.
- The full test suite was **not completed** in the authoring environment.
- Some historical experiment code (v0.16–v0.21) is not included; the v0.42 report states this explicitly.
- No external replication of any result has been performed.

---

## 12. Limitations and non-claims

1. **The pulse layer has not shown a general learning advantage.** The v0.42 comparison was negative for the tested STDP reader, and the comparison is confounded. This is not a universal verdict on spiking networks or STDP.
2. **No hardware or energy claim.** No neuromorphic hardware, latency benchmark, or energy measurement is included. The Python event loop is slower than dense MLP inference in the measured environment.
3. **No brain simulation claim.** The model is a software abstraction with simplified cells and synapses.
4. **No general program-induction claim.** v0.43-stage1 covers a deliberately bounded grammar. Results will not automatically transfer to richer languages.
5. **No learning result for v0.43.** Stage 1 tests tools, splits, and leakage checks. It does not train a meta-learner or evaluate accuracy.
6. **No confirmed benefit for local adaptive inhibition.** v0.37 and v0.41 show small descriptive effects, often with intervals that include zero.
7. **No confirmed benefit for `tau=8`.** v0.39 found a difference of −0.124 pp with an interval including zero.
8. **The v0.29 forgetting comparison is inconclusive.** No seed met the eligibility gate.
9. **The full test suite does not pass in the authoring environment.** The failure is a missing optional dependency, not a completed test run.
10. **No external replication.** No claim has reached the highest evidence level.

---

## 13. Roadmap

The binding build order for v0.43 is:

1. **Freeze the task definition and grammar.** Lock the grammar, depth limits, operation library, task classes, family minimums, sampling plan, development seeds, and comparison targets.
2. **Complete the generator and splits.** Produce at least 100 training families, 20 development families, and 30 reserved test families (target 50+), with at least 10,000 meta-learning episodes.
3. **Complete hypothesis enumeration and task auditing.** Compute consistent programs, semantic classes, and candidate disagreement for every task. Refuse to run if the ambiguity classification cannot be justified.
4. **Run leakage audits.** Check ASTs, inputs, labels, task IDs, library contents, and feedback streams before and after the run.
5. **Implement the full baseline suite.** Constant, support retrieval, per-task linear/tree, compositional program induction, and the DeepSets meta-learning baseline, all with equal information.
6. **Calibrate abstention on development only.** Separate gates for identifiable, ambiguous, out-of-scope, and contradictory cases.
7. **Implement new-operation learning.** 8, 16, and 32 examples, with evaluation on fixed unseen queries after each level.
8. **Implement error correction and retention.** Three correction rounds, a fixed unseen evaluation set, at least 20 sequential skills, and average plus worst-task forgetting.
9. **Preregister and freeze.** Record seeds, hashes, thresholds, baselines, and analysis code before opening the test.
10. **Run the official test once.** Publish all results, including failures and exclusions, and all baselines.

Only after a reproducible practical benefit is shown should the pulse layer be reconsidered as an optional component. It should enter as an ablation with a clear question, not as an assumed advantage.

---

## 14. Documentation

### 14.1 English documents

- [`README.md`](README.md): this document.
- [`docs/README.md`](docs/README.md): English documentation index.
- [`docs/handoff/LATEST_REPORT.md`](docs/handoff/LATEST_REPORT.md): latest task report and audit trail.
- [`AGENTS.md`](AGENTS.md): entry-point instructions for the OpenCode agent.
- [`OPENCODE_RULES.md`](OPENCODE_RULES.md): detailed mandatory workflow, handoff, and repository-sync rules.

### 14.2 Detailed research documents (Arabic)

The detailed specifications and historical reports were written in Arabic to preserve their original provenance. They are not retroactively translated. Key documents:

| Document | Content |
|---|---|
| `docs/PROJECT_SUMMARY_AR.md` | Project decision and status after v0.42.2 |
| `docs/architecture_ar.md` | Pulse simulator architecture and performance limits |
| `docs/V0_43_BENCHMARK_SPEC_AR.md` | Full v0.43 benchmark specification |
| `docs/NEXT_STAGE_V0_43_PLAN_AR.md` | Build order and decision gates |
| `docs/IMPLEMENTATION_STAGE1_V0_43_AR.md` | Stage-1 scope and limits |
| `docs/V0_43_ERROR_CORRECTION_AND_RETENTION_PROTOCOL_AR.md` | Feedback and retention protocol |
| `docs/V0_43_PREREGISTRATION_CHECKLIST_AR.md` | Official-test readiness checklist |
| `results/stdp_vs_linear_v0_42/report_ar.md` | v0.42 official report |
| `results/stdp_vs_linear_v0_42/INTERPRETATION_CORRECTION_V0_42_1_AR.md` | v0.42 interpretation correction |
| `results/input_gain_v0_41/development_diagnostic/report_ar.md` | v0.41 input-gain diagnostic |
| `results/ema_tau_v0_39_official/report_ar.md` | v0.39 official EMA-tau confirmation |
| `results/local_inhibition_v0_37/official_run/report_ar.md` | v0.37 official local-inhibition report |
| `docs/README_AR.md` | Archived Arabic documentation index |

The reports under `results/` are the authoritative source for their own numbers. This README summarizes selected results and states their limits; it does not replace the reports.

---

## 15. License and citation

### License

No license file is currently included in this repository. If you intend to reuse the code or data, please open an issue to discuss licensing before redistribution.

### Citation

There is no peer-reviewed paper for this project yet. If you reference the repository, cite it by URL and commit hash:

```text
Nucleus Pulse. GitHub repository, abdullahnuby/necleus.
https://github.com/abdullahnuby/necleus
Commit: <full SHA at the time of access>
```

For a specific result, cite the corresponding report path and the commit that contains it. Do not cite a smoke test or a stage-1 tooling run as a benchmark result.

---

## 16. Contact

- Repository: <https://github.com/abdullahnuby/necleus>
- Issues: <https://github.com/abdullahnuby/necleus/issues>
- Maintainer: [@abdullahnuby](https://github.com/abdullahnuby)

Questions about a result should reference the report path, the seed set, and the commit hash. Questions about the protocol should reference the relevant specification section.
