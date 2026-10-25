"""Command-line entry points for the deterministic v0.43 development pipeline.

The official v0.43 test is deliberately not a command here. Every command
operates on training or development data and writes evaluator-private smoke
artifacts only under the Git-ignored ``.local_private/`` tree.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

from .audit import audit_target_splits
from .dsl import ast_depth, canonical_semantic_programs, enumerate_programs, family_code, family_id, semantic_mask
from .generator import count_family_splits, generate_task, split_audit, stable_family_split
from .integrity import disqualification_report, official_test_family_ids


def _write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _inspect(seed: int) -> dict[str, Any]:
    programs = enumerate_programs()
    canonical = canonical_semantic_programs(programs)
    family_splits = count_family_splits(canonical)
    families_by_split: dict[str, set[str]] = {k: set() for k in ("train", "dev", "test")}
    for expr in canonical:
        families_by_split[stable_family_split(family_id(expr))].add(family_id(expr))
    eligible_test_families = official_test_family_ids(families_by_split["test"])
    report = {
        "stage": "v0.43-development",
        "grammar": "v0.43-core: 12 atoms (LT/EVEN/DIV3/DIV5), atom negation, one binary node, or nested two-binary-node shape; max AST depth 3",
        "input_domain": {"x0": [0, 99], "x1": [0, 99], "size": 10000},
        "syntactic_programs": len(programs),
        "unique_semantic_functions": len(canonical),
        "family_count_by_split": family_splits,
        "canonical_targets_by_split": dict(Counter(stable_family_split(family_id(x)) for x in canonical)),
        "official_test_family_count_after_disqualification": len(eligible_test_families),
        "family_minimums_met": (
            family_splits["train"] >= 100
            and family_splits["dev"] >= 20
            and family_splits["test"] >= 30
        ),
        "official_test_ready": False,
        "official_test_opened": False,
        "disqualification": disqualification_report(),
        "blocked_until": [
            "DeepSets abstention gate fails on development identifiable tasks",
            "retention improvement/forgetting targets fail on development skills",
            "external readiness review and a private official-test commitment",
        ],
        "commands": [
            "inspect", "smoke", "corpus", "dev", "train", "abstention",
            "new-operation", "correction-retention", "dev-family",
            "validate-splits", "train-family", "diagnostics", "remediation",
            "retention-compare", "summary",
        ],
    }
    return report


def _smoke(args: argparse.Namespace) -> dict[str, Any]:
    programs = enumerate_programs()
    canonical = canonical_semantic_programs(programs)
    candidates: dict[str, list] = {k: [] for k in ("train", "dev", "test")}
    for expr in canonical:
        split = stable_family_split(family_id(expr))
        mask = semantic_mask(expr)
        positives = mask.bit_count()
        negatives = 10000 - positives
        if positives >= 54 and negatives >= 54:
            candidates[split].append(expr)
    records = []
    for i, split in enumerate(("train", "dev", "test")):
        if not candidates[split]:
            raise RuntimeError(f"no balanced smoke target available in {split}")
        last_error = None
        for j, expr in enumerate(candidates[split][:100]):
            try:
                record = generate_task(
                    expr, split=split, seed=args.seed + i * 1000 + j, task_index=i
                )
                records.append(record)
                break
            except (RuntimeError, ValueError) as exc:
                last_error = exc
        else:
            raise RuntimeError(f"could not generate smoke task for {split}: {last_error}")

    public_out = Path(args.output)
    private_out = Path(args.private_output)
    public_out.mkdir(parents=True, exist_ok=True)
    private_out.mkdir(parents=True, exist_ok=True)
    for record in records:
        split = record.split
        public_path = public_out / f"{split}_public.jsonl"
        private_path = private_out / f"{split}_evaluator_private.jsonl"
        public_path.write_text(
            json.dumps(
                record.public_payload(include_query_labels=split != "test"),
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        private_path.write_text(
            json.dumps(record.evaluator_payload(), ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
    report = {
        "stage": "v0.43-cli-smoke",
        "official_test_opened": False,
        "public_output": str(public_out),
        "private_output": str(private_out),
        "task_count": len(records),
        "classes": {r.split: r.classification_private for r in records},
        "identifiability": {r.split: r.audit_private for r in records},
        "split_audit": split_audit(records),
        "note": "plumbing smoke test only; evaluator-private payloads stay under .local_private/",
    }
    _write_json(public_out / "stage1_smoke_report.json", report)
    return report


def _corpus(args: argparse.Namespace) -> dict[str, Any]:
    from .corpus import build_corpus_manifest

    manifest = build_corpus_manifest(n_episodes=args.episodes, seed=args.seed)
    _write_json(args.output, manifest)
    return {
        "schema": manifest["schema"],
        "output": str(args.output),
        "episodes": manifest["actual_episodes"],
        "unique_families": manifest["unique_families"],
        "unique_targets": manifest["unique_targets"],
        "episode_sha256": manifest["episode_sha256"],
    }


def _dev(args: argparse.Namespace) -> dict[str, Any]:
    from .corpus import build_development_manifest

    manifest = build_development_manifest(
        seed=args.seed,
        identifiable_count=args.identifiable,
        ambiguous_count=args.ambiguous,
        contradictory_count=args.contradictory,
        ood_count=args.ood,
    )
    _write_json(args.output, manifest)
    return {
        "schema": manifest["schema"],
        "output": str(args.output),
        "task_count": manifest["task_count"],
        "family_count": manifest["family_count"],
        "class_counts": manifest["class_counts"],
        "split_audit": manifest["split_audit"],
    }


def _train(args: argparse.Namespace) -> dict[str, Any]:
    from .pipeline import run_development

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_development(
        seeds=seeds,
        epochs=args.epochs,
        n_episodes=args.episodes,
        hidden=args.hidden,
        output_dir=args.output,
        checkpoint_dir=args.checkpoints,
        corpus_seed=args.corpus_seed,
        dev_seed=args.seed,
        identifiable_count=args.identifiable,
        ambiguous_count=args.ambiguous,
        contradictory_count=args.contradictory,
        ood_count=args.ood,
    )
    return {
        "output": str(Path(args.output) / "development_results.json"),
        "corpus_episodes": result["corpus_manifest"]["actual_episodes"],
        "development_tasks": result["development_manifest"]["task_count"],
        "seeds": list(seeds),
        "thresholds": {row["seed"]: row["threshold"] for row in result["seed_results"]},
        "checkpoints": {row["seed"]: row["checkpoint_sha256"] for row in result["seed_results"]},
        "runtime_seconds": result["compute"]["runtime_seconds"],
        "official_test_opened": False,
    }


def _abstention(args: argparse.Namespace) -> dict[str, Any]:
    from .abstention import run_abstention_evaluation

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_abstention_evaluation(
        seed=args.seed,
        seeds=seeds,
        checkpoint_dir=args.checkpoints,
        output_dir=args.output,
    )
    return {
        "output": str(Path(args.output) / "abstention_results.json"),
        "symbolic_task_gate": result["symbolic_task_level"]["gate"],
        "symbolic_query_gate": result["symbolic_query_level"]["gate"],
        "deepsets": [
            {"seed": row["seed"], "status": row["status"], "gate": row.get("gate")}
            for row in result["deepsets"]
        ],
        "official_test_opened": False,
    }


def _new_operation(args: argparse.Namespace) -> dict[str, Any]:
    from .new_operation import run_new_operation_learning

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_new_operation_learning(seeds=seeds)
    _write_json(args.output, result)
    return {
        "output": str(args.output),
        "by_budget": result["by_budget"],
        "official_test_opened": False,
    }


def _correction_retention(args: argparse.Namespace) -> dict[str, Any]:
    from .continual import run_correction_retention

    result = run_correction_retention(
        n_skills=args.skills,
        seed=args.seed,
        knn_k=args.knn_k,
        memory_mode=args.memory_mode,
    )
    _write_json(args.output, result)
    return {
        "output": str(args.output),
        "skill_count": result["skill_count"],
        "improvement_round0_to_round3": result["improvement_round0_to_round3"],
        "forgetting": {
            "memory": result["forgetting"]["memory"]["average_forgetting"],
            "parameter": result["forgetting"]["parameter"]["average_forgetting"],
            "parameter_worst": result["forgetting"]["parameter"]["worst_task_forgetting"],
        },
        "official_test_opened": False,
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dev_family(args: argparse.Namespace) -> dict[str, Any]:
    from .corpus import build_family_aware_development_manifest, write_manifest

    manifest = build_family_aware_development_manifest(seed=args.seed)
    write_manifest(args.output, manifest)
    path = Path(args.output)
    return {
        "schema": manifest["schema"],
        "protocol_version": manifest["protocol_version"],
        "output": str(path),
        "sha256": _file_sha256(path),
        "task_count": manifest["task_count"],
        "family_count": manifest["family_count"],
        "subset_counts": manifest["subset_counts"],
        "subset_class_counts": manifest["subset_class_counts"],
        "ood_operations": [row["operation_id"] for row in manifest["ood_operations"]],
        "official_test_opened": False,
    }


def _validate_splits(args: argparse.Namespace) -> dict[str, Any]:
    from .corpus import build_family_aware_development_tasks
    from .pipeline import load_family_aware_subsets

    records, _, _ = build_family_aware_development_tasks(seed=args.seed)
    subsets, manifest, audit = load_family_aware_subsets(args.manifest, records)
    return {
        "manifest": str(Path(args.manifest)),
        "manifest_sha256": _file_sha256(Path(args.manifest)),
        "protocol_version": manifest["protocol_version"],
        "subset_counts": {name: len(rows) for name, rows in subsets.items()},
        "split_audit": audit,
        "official_test_opened": False,
    }


def _train_family(args: argparse.Namespace) -> dict[str, Any]:
    from .pipeline import run_family_aware_development

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_family_aware_development(
        seeds=seeds,
        epochs=args.epochs,
        n_episodes=args.episodes,
        hidden=args.hidden,
        output_dir=args.output,
        checkpoint_dir=args.checkpoints,
        corpus_seed=args.corpus_seed,
        dev_seed=args.seed,
        manifest_path=args.manifest,
        cluster_bootstrap=args.cluster_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    return {
        "output": str(Path(args.output) / "development_results.json"),
        "run_metadata": result["run_metadata"],
        "subset_task_counts": result["subset_task_counts"],
        "gate_summary": result["gate_summary"],
        "thresholds": {
            str(row["seed"]): {
                "feasible": row["threshold_feasible"],
                "threshold": row["threshold"],
                "diagnostic_fallback": row["diagnostic_fallback_threshold"],
            }
            for row in result["seed_results"]
        },
        "checkpoints": {
            str(row["seed"]): row["checkpoint_sha256"] for row in result["seed_results"]
        },
        "official_test_opened": False,
    }


def _diagnostics(args: argparse.Namespace) -> dict[str, Any]:
    from .diagnostics import run_deepsets_diagnostics

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_deepsets_diagnostics(
        checkpoint_dir=args.checkpoints,
        seeds=seeds,
        manifest_path=args.manifest,
        output_dir=args.output,
        checkpoint_pattern=args.pattern,
        output_name=args.output_name,
    )
    return {
        "output": str(Path(args.output) / args.output_name),
        "manifest_sha256": result["manifest_sha256"],
        "seed_count": len(result["seed_results"]),
        "not_run": result["not_run"],
        "indicators": {
            str(row["config"]["seed"]): row["diagnostic_indicators"]
            for row in result["seed_results"]
        },
        "official_test_opened": False,
    }


def _remediation(args: argparse.Namespace) -> dict[str, Any]:
    from .remediation import run_remediation

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_remediation(
        plan_path=args.plan,
        preregistered_plan_sha256=args.plan_sha256,
        seeds=seeds,
        output_dir=args.output,
        manifest_path=args.manifest,
        control_checkpoint_dir=args.control_checkpoints,
        remediation_checkpoint_dir=args.checkpoints,
        cluster_bootstrap=args.cluster_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
    )
    return {
        "output": str(Path(args.output) / "remediation_results.json"),
        "preregistered_plan_sha256": result["preregistered_plan_sha256"],
        "decision": result["decision"],
        "experiments": [
            {
                "experiment_id": experiment["experiment_id"],
                "aggregate": experiment["aggregate"],
            }
            for experiment in result["experiments"]
        ],
        "official_test_opened": False,
    }


def _retention_compare(args: argparse.Namespace) -> dict[str, Any]:
    from .continual import run_retention_comparison

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    result = run_retention_comparison(
        n_skills=args.skills,
        seeds=seeds,
        strategies=tuple(args.strategies.split(",")),
        memory_mode=args.memory_mode,
        output_dir=args.output,
    )
    return {
        "output": str(Path(args.output) / "retention_comparison.json"),
        "sequence_audit": result["sequence_audit"],
        "aggregate": result["aggregate"],
        "decision": result["decision"],
        "official_test_opened": False,
    }


def _summary(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output)
    artifacts: dict[str, Any] = {}
    definitions = {
        "development_manifest": "dev_manifest.json",
        "development_results": "development_results.json",
        "deepsets_diagnostics_control": "deepsets_diagnostics_v2_control.json",
        "remediation_results": "remediation_results.json",
        "retention_comparison": "retention_comparison.json",
        "new_operation_results": "new_operation_results.json",
    }
    for name, filename in definitions.items():
        path = root / filename
        if not path.exists():
            artifacts[name] = {"status": "NOT RUN", "path": str(path)}
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        item: dict[str, Any] = {
            "status": "AVAILABLE",
            "path": str(path),
            "sha256": _file_sha256(path),
            "source": "existing_artifact",
        }
        if "run_metadata" in payload:
            item["run_metadata"] = payload["run_metadata"]
        if "decision" in payload:
            item["decision"] = payload["decision"]
        if "gate_summary" in payload:
            item["gate_summary"] = payload["gate_summary"]
        if "diagnostic_indicators" in payload:
            item["seed_indicators"] = {
                str(row["config"]["seed"]): row["diagnostic_indicators"]
                for row in payload.get("seed_results", [])
            }
        if "aggregate" in payload:
            item["aggregate"] = payload["aggregate"]
        artifacts[name] = item
    return {
        "schema": "nucleus-pulse-v0.43-task010-summary-v1",
        "official_test_opened": False,
        "official_test_ready": False,
        "artifacts": artifacts,
        "note": "Summary reads existing artifacts and labels them by hash; it never presents stale output as a fresh run.",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="inspect the grammar and split counts")
    inspect_parser.add_argument("--seed", type=int, default=430043)

    smoke_parser = subparsers.add_parser("smoke", help="write a private-safe smoke test")
    smoke_parser.add_argument("--output", default="results/v043_cli_smoke")
    smoke_parser.add_argument("--private-output", default=".local_private/v043_cli_smoke")
    smoke_parser.add_argument("--seed", type=int, default=430043)

    corpus_parser = subparsers.add_parser("corpus", help="generate a deterministic training corpus manifest")
    corpus_parser.add_argument("--episodes", type=int, default=10_000)
    corpus_parser.add_argument("--seed", type=int, default=430043)
    corpus_parser.add_argument("--output", default="results/v043_development/corpus_manifest.json")

    dev_parser = subparsers.add_parser("dev", help="generate the development task manifest")
    dev_parser.add_argument("--seed", type=int, default=430043)
    dev_parser.add_argument("--identifiable", type=int, default=8)
    dev_parser.add_argument("--ambiguous", type=int, default=8)
    dev_parser.add_argument("--contradictory", type=int, default=8)
    dev_parser.add_argument("--ood", type=int, default=4)
    dev_parser.add_argument("--output", default="results/v043_development/dev_manifest.json")

    train_parser = subparsers.add_parser("train", help="train and evaluate the development baselines")
    train_parser.add_argument("--seeds", default="430043,430044,430045")
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--episodes", type=int, default=10_000)
    train_parser.add_argument("--hidden", type=int, default=128)
    train_parser.add_argument("--seed", type=int, default=430043)
    train_parser.add_argument("--corpus-seed", type=int, default=430043)
    train_parser.add_argument("--identifiable", type=int, default=8)
    train_parser.add_argument("--ambiguous", type=int, default=8)
    train_parser.add_argument("--contradictory", type=int, default=8)
    train_parser.add_argument("--ood", type=int, default=4)
    train_parser.add_argument("--output", default="results/v043_development")
    train_parser.add_argument("--checkpoints", default=".local_private/v043_development/checkpoints")

    abstention_parser = subparsers.add_parser("abstention", help="evaluate development abstention")
    abstention_parser.add_argument("--seeds", default="430043,430044,430045")
    abstention_parser.add_argument("--seed", type=int, default=430043)
    abstention_parser.add_argument("--output", default="results/v043_development")
    abstention_parser.add_argument("--checkpoints", default=".local_private/v043_development/checkpoints")

    newop_parser = subparsers.add_parser("new-operation", help="evaluate held-out operation learning")
    newop_parser.add_argument("--seeds", default="430043,430044,430045")
    newop_parser.add_argument("--output", default="results/v043_development/new_operation_results.json")

    retention_parser = subparsers.add_parser("correction-retention", help="run correction and retention evaluation")
    retention_parser.add_argument("--skills", type=int, default=20)
    retention_parser.add_argument("--seed", type=int, default=430043)
    retention_parser.add_argument("--knn-k", type=int, default=3)
    retention_parser.add_argument("--memory-mode", choices=("program", "knn"), default="program")
    retention_parser.add_argument("--output", default="results/v043_development/correction_retention_results.json")

    dev_family_parser = subparsers.add_parser(
        "dev-family", help="write the frozen family-aware development manifest"
    )
    dev_family_parser.add_argument("--seed", type=int, default=430043)
    dev_family_parser.add_argument("--output", default="results/v043_task010/dev_manifest.json")

    validate_parser = subparsers.add_parser(
        "validate-splits", help="audit family-disjoint development subsets against a frozen manifest"
    )
    validate_parser.add_argument("--seed", type=int, default=430043)
    validate_parser.add_argument("--manifest", default="results/v043_task010/dev_manifest.json")

    train_family_parser = subparsers.add_parser(
        "train-family", help="train and evaluate on family-aware development subsets"
    )
    train_family_parser.add_argument("--seeds", default="430043,430044,430045")
    train_family_parser.add_argument("--epochs", type=int, default=10)
    train_family_parser.add_argument("--episodes", type=int, default=10_000)
    train_family_parser.add_argument("--hidden", type=int, default=128)
    train_family_parser.add_argument("--seed", type=int, default=430043)
    train_family_parser.add_argument("--corpus-seed", type=int, default=430043)
    train_family_parser.add_argument("--cluster-bootstrap", type=int, default=500)
    train_family_parser.add_argument("--bootstrap-seed", type=int, default=430043)
    train_family_parser.add_argument("--manifest", default="results/v043_task010/dev_manifest.json")
    train_family_parser.add_argument("--output", default="results/v043_task010")
    train_family_parser.add_argument(
        "--checkpoints", default=".local_private/v043_task010/checkpoints"
    )

    diagnostics_parser = subparsers.add_parser(
        "diagnostics", help="diagnose DeepSets confidence/error structure"
    )
    diagnostics_parser.add_argument("--seeds", default="430043,430044,430045")
    diagnostics_parser.add_argument("--manifest", default="results/v043_task010/dev_manifest.json")
    diagnostics_parser.add_argument("--output", default="results/v043_task010")
    diagnostics_parser.add_argument(
        "--checkpoints", default=".local_private/v043_task010/checkpoints"
    )
    diagnostics_parser.add_argument("--pattern", default="deepsets_seed_{seed}.npz")
    diagnostics_parser.add_argument("--output-name", default="deepsets_diagnostics.json")

    remediation_parser = subparsers.add_parser(
        "remediation", help="run the preregistered bounded DeepSets remediation cycle"
    )
    remediation_parser.add_argument("--seeds", default="430043,430044,430045")
    remediation_parser.add_argument("--plan", default="docs/V0_43_DEEPSETS_REMEDIATION_PLAN.md")
    remediation_parser.add_argument("--plan-sha256", default=None)
    remediation_parser.add_argument("--manifest", default="results/v043_task010/dev_manifest.json")
    remediation_parser.add_argument("--output", default="results/v043_task010")
    remediation_parser.add_argument(
        "--control-checkpoints", default=".local_private/v043_task010/checkpoints"
    )
    remediation_parser.add_argument(
        "--checkpoints", default=".local_private/v043_task010/remediation/checkpoints"
    )
    remediation_parser.add_argument("--cluster-bootstrap", type=int, default=500)
    remediation_parser.add_argument("--bootstrap-seed", type=int, default=430043)

    retention_compare_parser = subparsers.add_parser(
        "retention-compare", help="compare retention strategies across independent sequences"
    )
    retention_compare_parser.add_argument("--skills", type=int, default=20)
    retention_compare_parser.add_argument("--seeds", default="430043,430044,430045")
    retention_compare_parser.add_argument(
        "--strategies", default="control,balanced_replay,balanced_support_replay"
    )
    retention_compare_parser.add_argument(
        "--memory-mode", choices=("program", "knn"), default="program"
    )
    retention_compare_parser.add_argument("--output", default="results/v043_task010")

    summary_parser = subparsers.add_parser(
        "summary", help="summarize existing TASK-010 artifacts with hashes"
    )
    summary_parser.add_argument("--output", default="results/v043_task010")

    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "inspect"
    if command == "inspect":
        report = _inspect(args.seed)
    elif command == "smoke":
        report = _smoke(args)
    elif command == "corpus":
        report = _corpus(args)
    elif command == "dev":
        report = _dev(args)
    elif command == "train":
        report = _train(args)
    elif command == "abstention":
        report = _abstention(args)
    elif command == "new-operation":
        report = _new_operation(args)
    elif command == "correction-retention":
        report = _correction_retention(args)
    elif command == "dev-family":
        report = _dev_family(args)
    elif command == "validate-splits":
        report = _validate_splits(args)
    elif command == "train-family":
        report = _train_family(args)
    elif command == "diagnostics":
        report = _diagnostics(args)
    elif command == "remediation":
        report = _remediation(args)
    elif command == "retention-compare":
        report = _retention_compare(args)
    elif command == "summary":
        report = _summary(args)
    else:  # pragma: no cover - argparse rejects unknown commands
        raise ValueError(f"unknown command: {command}")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return report


if __name__ == "__main__":
    main()
