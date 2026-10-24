"""Inspect and smoke-test the deterministic v0.43 task grammar."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from .dsl import enumerate_programs, canonical_semantic_programs, ast_depth, family_id, family_code, semantic_mask
from .generator import count_family_splits, stable_family_split, generate_task, split_audit
from .audit import audit_target_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "smoke"), nargs="?", default="inspect")
    parser.add_argument("--output", default="results/v043_stage1")
    parser.add_argument("--seed", type=int, default=430043)
    args = parser.parse_args()

    programs = enumerate_programs()
    canonical = canonical_semantic_programs(programs)
    family_splits = count_family_splits(canonical)
    families_by_split: dict[str, set[str]] = {k: set() for k in ("train", "dev", "test")}
    for expr in canonical:
        families_by_split[stable_family_split(family_id(expr))].add(family_id(expr))
    report = {
        "stage": "v0.43-stage1",
        "grammar": "v0.43-core: 12 atoms (LT/EVEN/DIV3/DIV5), atom negation, one binary node, or nested two-binary-node shape; max AST depth 3",
        "input_domain": {"x0": [0, 99], "x1": [0, 99], "size": 10000},
        "syntactic_programs": len(programs),
        "unique_semantic_functions": len(canonical),
        "family_count_by_split": family_splits,
        "canonical_targets_by_split": Counter(stable_family_split(family_id(x)) for x in canonical),
        "family_minimums_met": family_splits["train"] >= 100 and family_splits["dev"] >= 20 and family_splits["test"] >= 30,
        "official_test_ready": False,
        "blocked_until": ["10,000 meta-learning episodes", "100 train/20 dev/30 test families", "all baselines", "feedback loop", "retention tests", "preregistration and hashes"],
    }
    report["canonical_targets_by_split"] = dict(report["canonical_targets_by_split"])

    if args.command == "smoke":
        records = []
        # Pick one target from each available split and generate any support-defined
        # category; the smoke run validates data mechanics, not benchmark performance.
        candidates: dict[str, list] = {k: [] for k in ("train", "dev", "test")}
        for expr in canonical:
            split = stable_family_split(family_id(expr))
            # Must have at least 54 positive and negative inputs to support 50/50 query sampling.
            mask = semantic_mask(expr)
            positives = mask.bit_count()
            negatives = 10000 - positives
            if positives >= 54 and negatives >= 54:
                candidates[split].append(expr)
        for i, split in enumerate(("train", "dev", "test")):
            if not candidates[split]:
                raise RuntimeError(f"no balanced smoke target available in {split}")
            # Search several deterministic targets until an auditable task succeeds.
            last_error = None
            for j, expr in enumerate(candidates[split][:100]):
                try:
                    record = generate_task(expr, split=split, seed=args.seed + i * 1000 + j, task_index=i)
                    records.append(record)
                    break
                except RuntimeError as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"could not generate smoke task for {split}: {last_error}")
        smoke_audit = split_audit(records)
        report["smoke"] = {
            "task_count": len(records),
            "classes": {r.split: r.classification_private for r in records},
            "identifiability": {r.split: r.audit_private for r in records},
            "split_audit": smoke_audit,
            "note": "plumbing smoke test only; not an official benchmark result",
        }
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        for record in records:
            split = record.split
            with (out / f"{split}_public.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(record.public_payload(include_query_labels=split != "test"), ensure_ascii=False) + "\n")
            with (out / f"{split}_evaluator_private.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps(record.evaluator_payload(), ensure_ascii=False, default=str) + "\n")
        (out / "stage1_smoke_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
