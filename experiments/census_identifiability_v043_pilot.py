"""v0.43 Stage 2 pilot: dev-only identifiability census.

Question: can the Stage 1 generator supply both the identifiable and the
ambiguous stratum at usable rates, and which support mode is needed?

This is a generator-validity diagnostic. It uses only the `dev` split, never
touches `test`, trains no model, and selects no thresholds. See
docs/V0_43_STAGE2_IDENTIFIABILITY_CENSUS_PROTOCOL_AR.md.

Run:
    python -m experiments.census_identifiability_v043_pilot \
        --output results/v043_stage2_census_dev --families 20 \
        --targets-per-family 2 --random-draws 5 --contrastive-draws 1
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from skillbench_v043.dsl import canonical_semantic_programs, family_id, semantic_mask
from skillbench_v043.generator import generate_task, stable_family_split

SPLIT = "dev"
BASE_SEED = 430_200
MAX_ATTEMPTS = 25


def _balanced(expr) -> bool:
    positives = semantic_mask(expr).bit_count()
    return positives >= 54 and (10_000 - positives) >= 54


def _evenly_spaced(items: list[Any], count: int) -> list[Any]:
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[0]]
    positions = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[i] for i in positions]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"task_count": 0}
    classes = Counter(row["classification"] for row in rows)
    valid = [row for row in rows if "error" not in row]
    if not valid:
        return {
            "task_count": len(rows),
            "class_counts": dict(sorted(classes.items())),
            "identifiable_rate": 0.0,
            "ambiguous_rate": 0.0,
            "ambiguous_admissible_rate": None,
            "consistent_classes_median": None,
            "consistent_classes_range": None,
            "disagreement_median": None,
            "disagreement_range": None,
            "failures": len(rows),
            "wall_seconds": round(sum(row["wall_seconds"] for row in rows), 3),
        }
    classes_with_disagreement = [
        row["candidate_disagreement"] for row in valid
        if row["candidate_disagreement"] is not None
    ]
    ambiguous = [
        row for row in valid if row["classification"] == "IN_SCOPE_AMBIGUOUS"
    ]
    ident = [row for row in valid if row["classification"] == "IN_SCOPE_IDENTIFIABLE"]
    return {
        "task_count": len(rows),
        "class_counts": dict(sorted(classes.items())),
        "identifiable_rate": len(ident) / len(valid),
        "ambiguous_rate": len(ambiguous) / len(valid),
        "ambiguous_admissible_rate": (
            sum(1 for r in ambiguous if r["candidate_disagreement"] >= 0.20) / len(ambiguous)
            if ambiguous else None
        ),
        "consistent_classes_median": statistics.median(
            row["consistent_semantic_classes"] for row in rows
        ),
        "consistent_classes_range": [
            min(row["consistent_semantic_classes"] for row in rows),
            max(row["consistent_semantic_classes"] for row in rows),
        ],
        "disagreement_median": (
            statistics.median(classes_with_disagreement) if classes_with_disagreement else None
        ),
        "disagreement_range": (
            [min(classes_with_disagreement), max(classes_with_disagreement)]
            if classes_with_disagreement else None
        ),
        "failures": len(rows) - len(valid),
        "wall_seconds": round(sum(row["wall_seconds"] for row in rows), 3),
    }


def _run_target(target, family: str, seed_base: int, mode: str, draws: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for draw in range(draws):
        started = time.perf_counter()
        row: dict[str, Any] = {
            "family_id": family,
            "split": SPLIT,
            "mode": mode,
            "draw": draw,
        }
        try:
            record = generate_task(
                target,
                split=SPLIT,
                seed=seed_base + draw,
                support_mode=mode,
                max_attempts=MAX_ATTEMPTS,
            )
            row.update(
                classification=record.classification_private,
                consistent_program_count=record.audit_private["consistent_program_count"],
                consistent_semantic_classes=record.audit_private["consistent_semantic_classes"],
                candidate_disagreement=record.audit_private["candidate_disagreement"],
            )
        except RuntimeError as exc:
            row["error"] = str(exc)
            row.update(
                classification="GENERATION_FAILED",
                consistent_program_count=None,
                consistent_semantic_classes=None,
                candidate_disagreement=None,
            )
        row["wall_seconds"] = round(time.perf_counter() - started, 3)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/v043_stage2_census_dev")
    parser.add_argument("--families", type=int, default=20)
    parser.add_argument("--targets-per-family", type=int, default=2)
    parser.add_argument("--random-draws", type=int, default=5)
    parser.add_argument("--contrastive-draws", type=int, default=1)
    args = parser.parse_args()

    dev_targets = [
        expr for expr in canonical_semantic_programs()
        if stable_family_split(family_id(expr)) == SPLIT and _balanced(expr)
    ]
    by_family: dict[str, list] = {}
    for expr in dev_targets:
        by_family.setdefault(family_id(expr), []).append(expr)
    families = sorted(by_family)
    chosen_families = _evenly_spaced(families, args.families)

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for family_index, family in enumerate(chosen_families):
        targets = _evenly_spaced(sorted(by_family[family], key=repr), args.targets_per_family)
        for target_index, target in enumerate(targets):
            target_split = stable_family_split(family_id(target))
            if target_split != SPLIT:  # defensive: never let a non-dev target in
                raise AssertionError(f"target from {target_split} reached the dev census")
            seed_base = BASE_SEED + family_index * 10_000 + target_index * 100
            rows.extend(_run_target(target, family, seed_base, "random", args.random_draws))
            rows.extend(
                _run_target(target, family, seed_base + 50, "contrastive", args.contrastive_draws)
            )

    report = {
        "stage": "v0.43-stage2-pilot",
        "status": "generator-validity diagnostic on dev only; not a model result",
        "scope": {
            "split": SPLIT,
            "dev_targets_available": len(dev_targets),
            "dev_families_available": len(families),
            "families_sampled": len(chosen_families),
            "targets_per_family": args.targets_per_family,
            "random_draws_per_target": args.random_draws,
            "contrastive_draws_per_target": args.contrastive_draws,
            "base_seed": BASE_SEED,
            "max_attempts": MAX_ATTEMPTS,
        },
        "summary_by_mode": {
            mode: _summarize([row for row in rows if row["mode"] == mode])
            for mode in ("random", "contrastive")
        },
        "rows": rows,
        "wall_seconds_total": round(time.perf_counter() - started, 3),
        "note": (
            "Contrastive support selects examples using hypothesis enumeration and the "
            "target labels. If it is required for identifiable tasks, that design choice "
            "must be disclosed; it does not by itself show generalization."
        ),
    }

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "census.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# تعداد قابلية التحديد — v0.43 المرحلة 2 (تطوير فقط)",
        "",
        "**تشخيص للمولد. ليس نتيجة تعلّم ولا يفتح الاختبار النهائي.**",
        "",
        f"- عائلات dev المتاحة: {len(families)} | التي شملتها العينة: {len(chosen_families)}",
        f"- أهداف dev الصالحة: {len(dev_targets)} | أهداف لكل عائلة: {args.targets_per_family}",
        f"- سحبات عشوائية لكل هدف: {args.random_draws} | سحبات contrastive: {args.contrastive_draws}",
        "",
        "| النمط | المهام | قابلة للتحديد | ملتبسة | ملتبسة مقبولة (≥20%) | فئات متسقة (وسيط) | زمن (ث) |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in ("random", "contrastive"):
        summary = report["summary_by_mode"][mode]
        lines.append(
            "| {mode} | {tasks} | {ident:.0%} | {amb:.0%} | {adm} | {cls} | {secs} |".format(
                mode=mode,
                tasks=summary["task_count"],
                ident=summary["identifiable_rate"],
                amb=summary["ambiguous_rate"],
                adm=(
                    f"{summary['ambiguous_admissible_rate']:.0%}"
                    if summary["ambiguous_admissible_rate"] is not None else "—"
                ),
                cls=summary["consistent_classes_median"],
                secs=summary["wall_seconds"],
            )
        )
    lines += [
        "",
        "## ما يعنيه هذا",
        "",
        "- إذا كان الدعم العشوائي ينتج طبقة قابلة للتحديد بمعدل كافٍ، فهو الخيار المفضل لأنه لا يستخدم معرفة الهدف.",
        "- إذا احتاجت الطبقة القابلة للتحديد إلى contrastive، فهذا تصميم دعم موجّه بتعداد الفرضيات ويجب التصريح به.",
        "- الأرقام أعلاه لا تقيس ذكاء أي نموذج، ولا تُستخدم لاختيار عتبات الامتناع.",
        "",
        "انظر `census.json` للتفاصيل والبذور.",
    ]
    (out / "report_ar.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report["summary_by_mode"], ensure_ascii=False, indent=2))
    print("wall_seconds_total", report["wall_seconds_total"])


if __name__ == "__main__":
    main()
