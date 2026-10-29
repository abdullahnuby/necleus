# Nucleus Pulse Documentation Index

This directory contains the specifications, protocols, architecture notes, archived reports, and the mandatory handoff system for Nucleus Pulse.

The comprehensive project overview is the repository root **[`README.md`](../README.md)**. It is written in English and summarizes the research question, the epistemic contract, the current status, the selected results, the v0.43 benchmark design, installation, testing, integrity rules, and limitations.

## Language policy

- The root `README.md` and this index are in English.
- Detailed specifications, protocols, and historical experiment reports remain in Arabic to preserve their original provenance. They are **not** retroactively translated.
- The previous Arabic documentation index is preserved unchanged as [`README_AR.md`](README_AR.md).
- When a result is cited, the authoritative source is the corresponding report under `results/`, not this index.

## Start here

| Document | Purpose |
|---|---|
| [`../README.md`](../README.md) | Full English overview of the project, evidence levels, results, architecture, benchmark design, and limitations |
| [`handoff/LATEST_REPORT.md`](handoff/LATEST_REPORT.md) | Latest task report, including commands, tests, hashes, and push status |
| [`PROJECT_SUMMARY_AR.md`](PROJECT_SUMMARY_AR.md) | Project decision and status after v0.42.2 (Arabic) |
| [`IMPLEMENTATION_STAGE1_V0_43_AR.md`](IMPLEMENTATION_STAGE1_V0_43_AR.md) | Implemented scope and hard limits of v0.43-stage1 (Arabic) |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history and documentation changes |

## v0.43 benchmark

| Document | Purpose |
|---|---|
| [`V0_43_BENCHMARK_SPEC_AR.md`](V0_43_BENCHMARK_SPEC_AR.md) | Full benchmark specification: grammar, splits, gates, baselines, abstention, feedback, retention |
| [`NEXT_STAGE_V0_43_PLAN_AR.md`](NEXT_STAGE_V0_43_PLAN_AR.md) | Binding build order and decision gates |
| [`V0_43_ERROR_CORRECTION_AND_RETENTION_PROTOCOL_AR.md`](V0_43_ERROR_CORRECTION_AND_RETENTION_PROTOCOL_AR.md) | Three-round correction protocol and forgetting measurements |
| [`V0_43_PREREGISTRATION_CHECKLIST_AR.md`](V0_43_PREREGISTRATION_CHECKLIST_AR.md) | Readiness checklist that must pass before the official test |
| [`V0_43_STAGE2_IDENTIFIABILITY_CENSUS_PROTOCOL_AR.md`](V0_43_STAGE2_IDENTIFIABILITY_CENSUS_PROTOCOL_AR.md) | Aborted stage-2 identifiability census protocol; no results were produced |
| [`V0_43_AUTONOMOUS_EXECUTION_ROADMAP_EN.md`](V0_43_AUTONOMOUS_EXECUTION_ROADMAP_EN.md) | Execution record for Phases 0–7, including failed scientific gates |
| [`BENCHMARK_INTEGRITY_V0_43.md`](BENCHMARK_INTEGRITY_V0_43.md) | Exposed smoke artifacts, local preservation, disqualification, and recurrence guards |
| [`V0_43_READINESS_CHECKLIST.md`](V0_43_READINESS_CHECKLIST.md) | PASS/FAIL/BLOCKED readiness review; official test remains closed |
| [`V0_43_OFFICIAL_TEST_COMMITMENT_PLAN.md`](V0_43_OFFICIAL_TEST_COMMITMENT_PLAN.md) | Commitment plan; not issued because readiness is blocked |

The benchmark code is in [`../skillbench_v043/`](../skillbench_v043/) and its tests are in [`../tests/v043/`](../tests/v043/).

## Architecture and simulator

| Document | Purpose |
|---|---|
| [`architecture_ar.md`](architecture_ar.md) | Per-synapse currents, LIF dynamics, inhibition, temporal AND cell, reference validation, performance limits |
| [`benchmark_protocol_v0_27_AR.md`](benchmark_protocol_v0_27_AR.md) | v0.27 temporal AND benchmark protocol |
| [`IMPLEMENTATION_AUDIT_NOTE_V0_35_AR.md`](IMPLEMENTATION_AUDIT_NOTE_V0_35_AR.md) | Why the first v0.35 run was discarded and rerun |

The simulator is in [`../pulse_core/`](../pulse_core/). The model uses the standard library; `sequential_stdp.py` requires NumPy.

## Exploration lab

| Document | Purpose |
|---|---|
| [`EXPLORATION_LAB.md`](EXPLORATION_LAB.md) | Local exploration loop, persistent spiking agent, action STDP, checkpoints, and learning-signal boundary |
| [`EXPLORATION_CITY.md`](EXPLORATION_CITY.md) | Shared 8-clone city, movement, collisions, interactions, and TASK-012 pilot |
| [`EXPLORATION_PREDICTION.md`](EXPLORATION_PREDICTION.md) | TASK-013 separate local next-observation predictor, pilot results, and non-claims |

The exploration code is in [`../exploration_lab/`](../exploration_lab/). It is separate from the v0.43 benchmark and makes no learning or skill claim.

## Experiment protocols

Each protocol fixes the question, seeds, data, baselines, and success criteria before the run.

| Protocol | Version |
|---|---|
| [`benchmark_protocol_v0_28_1_AR.md`](benchmark_protocol_v0_28_1_AR.md) | v0.28.1 sequential STDP |
| [`benchmark_protocol_v0_29_AR.md`](benchmark_protocol_v0_29_AR.md) | v0.29 sequential learning |
| [`LEARNING_CURVE_PROTOCOL_V0_30_AR.md`](LEARNING_CURVE_PROTOCOL_V0_30_AR.md) | v0.30 learning curve |
| [`INCREMENTAL_WINDOWS_PROTOCOL_V0_31_AR.md`](INCREMENTAL_WINDOWS_PROTOCOL_V0_31_AR.md) | v0.31 incremental windows |
| [`MULTI_EPOCH_PROTOCOL_V0_32_AR.md`](MULTI_EPOCH_PROTOCOL_V0_32_AR.md) | v0.32 repeated passes |
| [`DELAYED_RESERVOIR_PROTOCOL_V0_33_AR.md`](DELAYED_RESERVOIR_PROTOCOL_V0_33_AR.md) | v0.33 heterogeneous delays |
| [`INHIBITION_STRUCTURE_PROTOCOL_V0_34_AR.md`](INHIBITION_STRUCTURE_PROTOCOL_V0_34_AR.md) | v0.34 Dale-style inhibition |
| [`ACTIVITY_ADAPTIVE_INHIBITION_PROTOCOL_V0_35_AR.md`](ACTIVITY_ADAPTIVE_INHIBITION_PROTOCOL_V0_35_AR.md) | v0.35 activity-adaptive inhibition |
| [`benchmark_protocol_v0_36_AR.md`](benchmark_protocol_v0_36_AR.md) | v0.36 local inhibition tuning |
| [`benchmark_protocol_v0_37_AR.md`](benchmark_protocol_v0_37_AR.md) | v0.37 official local inhibition |
| [`benchmark_protocol_v0_38_STAGE1_AR.md`](benchmark_protocol_v0_38_STAGE1_AR.md) | v0.38 EMA tau stage 1 |
| [`benchmark_protocol_v0_38_STAGE2_AR.md`](benchmark_protocol_v0_38_STAGE2_AR.md) | v0.38 EMA tau stage 2 |
| [`benchmark_protocol_v0_39_AR.md`](benchmark_protocol_v0_39_AR.md) | v0.39 official EMA tau confirmation |
| [`benchmark_protocol_v0_40_STAGE1_AR.md`](benchmark_protocol_v0_40_STAGE1_AR.md) | v0.40 input-gain stage 1 |
| [`benchmark_protocol_v0_41_STAGE2_AR.md`](benchmark_protocol_v0_41_STAGE2_AR.md) | v0.41 input-gain stage 2 |
| [`STDP_VS_LINEAR_PROTOCOL_V0_42_AR.md`](STDP_VS_LINEAR_PROTOCOL_V0_42_AR.md) | v0.42 STDP versus linear reader |

Validation summaries and run logs are stored as `VALIDATION_SUMMARY_*.txt` and `RUN_LOG_*.txt` in this directory. SHA-256 manifests are stored as `MANIFEST_SHA256*.txt`.

## Results and interpretation

Reports live with their artifacts under `../results/`.

| Report | Content |
|---|---|
| [`../results/stdp_vs_linear_v0_42/report_ar.md`](../results/stdp_vs_linear_v0_42/report_ar.md) | v0.42 official report: 40 seeds, 50/100/200 examples |
| [`../results/stdp_vs_linear_v0_42/INTERPRETATION_CORRECTION_V0_42_1_AR.md`](../results/stdp_vs_linear_v0_42/INTERPRETATION_CORRECTION_V0_42_1_AR.md) | Correction: the result does not isolate the causal effect of STDP |
| [`../results/input_gain_v0_41/development_diagnostic/report_ar.md`](../results/input_gain_v0_41/development_diagnostic/report_ar.md) | v0.41 component diagnostic |
| [`../results/ema_tau_v0_39_official/report_ar.md`](../results/ema_tau_v0_39_official/report_ar.md) | v0.39 official EMA-tau confirmation |
| [`../results/local_inhibition_v0_37/official_run/report_ar.md`](../results/local_inhibition_v0_37/official_run/report_ar.md) | v0.37 official local-inhibition report |
| [`../results/sequential_learning_v0_29/report_ar.md`](../results/sequential_learning_v0_29/report_ar.md) | v0.29 sequential learning and eligibility limits |

Release notes:

- [`RELEASE_NOTE_V0_42_1_AR.md`](RELEASE_NOTE_V0_42_1_AR.md)
- [`RELEASE_NOTE_V0_42_2_AR.md`](RELEASE_NOTE_V0_42_2_AR.md)

## Workflow, handoff, and repository sync

| Document | Purpose |
|---|---|
| [`../AGENTS.md`](../AGENTS.md) | Entry-point instructions read automatically by OpenCode |
| [`../OPENCODE_RULES.md`](../OPENCODE_RULES.md) | Detailed mandatory workflow: task registration, reporting, verification, commit, push |
| [`handoff/TEMPLATE.md`](handoff/TEMPLATE.md) | Required eleven-section handoff report template |
| [`handoff/LATEST_REPORT.md`](handoff/LATEST_REPORT.md) | Current task report |
| [`handoff/history/`](handoff/history/) | Archived reports, one per task |
| [`handoff/logs/`](handoff/logs/) | Command logs, test outputs, environment records, hashes |
| [`../scripts/verify_handoff.py`](../scripts/verify_handoff.py) | Structural validator for the handoff system |

Every task with a real change must end with:

```bash
python -m pytest -q tests/v043
python scripts/verify_handoff.py
git add -A
git commit -m "<TASK-ID>: <short summary>"
git push origin main
```

The task is not complete if the push fails or the working tree is not clean. If a command was not run, it is recorded as `NOT RUN` with the reason.

## Requirements

| File | Scope |
|---|---|
| [`requirements.txt`](requirements.txt) | Core pulse network: standard library only |
| [`requirements-dev.txt`](requirements-dev.txt) | Tests and plots (`pytest`, `matplotlib`) |
| [`requirements-experiment.txt`](requirements-experiment.txt) | Experiments (`numpy`, `torch`) |
| [`requirements-benchmark.txt`](requirements-benchmark.txt) | Benchmark baselines (`numpy`, `scikit-learn`, `matplotlib`, `threadpoolctl`) |
| [`requirements-brian2-optional.txt`](requirements-brian2-optional.txt) | Optional Brian2 comparison; not executed in the authoring environment |

## Repository map

```text
pulse_core/          pulse simulator, tasks, reference recurrence, legacy learning
skillbench_v043/     finite DSL, task generator, auditor, CLI
experiments/         runnable experiment scripts
tests/               unit and integration tests; tests/v043
results/             frozen outputs and reports
docs/                specifications, protocols, handoff system
scripts/             handoff validator, optional Brian2 comparison, analysis helpers
```

## Reading a result correctly

1. Check the evidence level in the root README. A tooling test is not a learning result.
2. Open the experiment protocol to confirm the question and the preregistered gate.
3. Open the report for the numbers, seed counts, exclusions, and confidence intervals.
4. Read any appended interpretation correction before quoting a causal claim.
5. Do not cite a smoke test, a development diagnostic, or a stage-1 tooling run as benchmark evidence.
