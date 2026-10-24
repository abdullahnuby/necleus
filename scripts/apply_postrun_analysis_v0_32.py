"""Post-run statistical correction/reporting for v0.32 only.

Does not change model fits, datasets, preregistered parameters, or the checkpoint
selection rule. It replaces degenerate percentile-bootstrap intervals for binary
threshold-hit rates with Wilson score intervals and adds paired 10-vs-1 pass gains.
Accuracy intervals remain bootstrap across seeds as originally specified.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.run_multi_epoch_v0_32 import atomic_json, render_report, write_csv, _stable_seed, bootstrap_ci  # noqa: E402

OUT = ROOT / "results" / "multi_epoch_v0_32"
Z = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = Z) -> tuple[float, float, float]:
    if total <= 0:
        raise ValueError("total must be positive")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return 100.0 * p, 100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)


def summarize_paired_gains(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], dict[int, dict[int, dict]]] = {}
    for row in records:
        key = (row["task"], row["feature_set"])
        grouped.setdefault(key, {}).setdefault(int(row["epochs"]), {})[int(row["seed"])] = row
    out = []
    for (task, feature), epochs in sorted(grouped.items()):
        seeds_1 = epochs[1]
        seeds_10 = epochs[10]
        shared = sorted(set(seeds_1) & set(seeds_10))
        row = {"task": task, "feature_set": feature, "n_paired_seeds": len(shared)}
        for split in ("validation", "test"):
            diffs = np.asarray([seeds_10[s][f"{split}_accuracy"] - seeds_1[s][f"{split}_accuracy"] for s in shared])
            mean, low, high = bootstrap_ci(diffs, _stable_seed(task, feature, split, "paired_gain_10_minus_1"))
            row[f"{split}_accuracy_gain_10_minus_1_mean_percentage_points"] = mean
            row[f"{split}_accuracy_gain_ci95_low_pp"] = low
            row[f"{split}_accuracy_gain_ci95_high_pp"] = high
        out.append(row)
    return out


def main() -> None:
    path = OUT / "results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != "0.32" or len(data.get("records", [])) != 960 or len(data.get("integrity", [])) != 40:
        raise RuntimeError("Post-run correction refuses incomplete or unexpected result data")

    # Preserve the original bootstrap result so the post-run correction is auditable.
    if "threshold_summary_bootstrap_raw" not in data:
        data["threshold_summary_bootstrap_raw"] = data["threshold_summary"]
    corrected = []
    for row in data["threshold_summary_bootstrap_raw"]:
        r = dict(row)
        rate, low, high = wilson_interval(int(r["n_reached_85"]), int(r["n_seeds"]))
        r["reach_rate_percent"] = rate
        r["reach_rate_ci95_low_percent"] = low
        r["reach_rate_ci95_high_percent"] = high
        r["reach_rate_ci_method"] = "Wilson score interval"
        corrected.append(r)
    data["threshold_summary"] = corrected
    data["postrun_analysis_correction"] = {
        "applied": True,
        "reason": "Percentile bootstrap over seeds degenerates to [0,0] for 0/40 binary threshold hits; Wilson score intervals are used for hit rate only.",
        "model_or_data_changed": False,
        "accuracy_ci_method": "percentile bootstrap over seeds (10,000 resamples)",
        "threshold_hit_rate_ci_method": "Wilson score interval 95%",
        "paired_epoch_gain_added": "10 passes minus 1 pass, bootstrap over paired seeds",
    }

    write_csv(OUT / "threshold_summary.csv", corrected)
    paired = summarize_paired_gains(data["records"])
    data["paired_epoch_gain_summary"] = paired
    write_csv(OUT / "paired_epoch_gain_summary.csv", paired)

    # Re-render canonical report using the corrected threshold intervals.
    report = render_report(data["records"], data["summary"], corrected, data["integrity"],
                           data["runtime_seconds"], data["execution_workers"])
    marker = "## سلامة التجربة"
    note = (
        "**ملاحظة تصحيح إحصائي بعد التشغيل:** فاصل percentile-bootstrap لمعدل حدث ثنائي قد يصبح عديم العرض عند 0 نجاح (مثل 0/40)، "
        "لذلك يُستخدم فاصل Wilson 95% لنسبة البذور التي وصلت إلى 85%. فواصل دقة التحقق/الاختبار تظل bootstrap على مستوى البذرة. "
        "لم تتغير البيانات أو النماذج أو العتبة أو اختيار المرور. تفاصيل القيم الخام السابقة محفوظة في `threshold_summary_bootstrap_raw` داخل JSON.\n\n"
        "## فرق الدقة المزدوج بين 10 و1 مرور\n\n"
        "هذه فروق لكل بذرة (10 مرورّات ناقص مرور واحد) ثم متوسطها وفاصل bootstrap 95%، وليست مقارنة عينات مستقلة.\n\n"
        "| المهمة | التمثيل | مكسب التحقق بالنقاط المئوية [95% CI] | مكسب الاختبار بالنقاط المئوية [95% CI] |\n"
        "|---|---|---:|---:|\n"
    )
    for row in paired:
        note += (f"| {row['task']} | {row['feature_set']} | "
                 f"{row['validation_accuracy_gain_10_minus_1_mean_percentage_points']:.2f} "
                 f"[{row['validation_accuracy_gain_ci95_low_pp']:.2f}, {row['validation_accuracy_gain_ci95_high_pp']:.2f}] | "
                 f"{row['test_accuracy_gain_10_minus_1_mean_percentage_points']:.2f} "
                 f"[{row['test_accuracy_gain_ci95_low_pp']:.2f}, {row['test_accuracy_gain_ci95_high_pp']:.2f}] |\n")
    note += "\n**تنبيه:** دقة الاختبار عند أول نقطة اختيار مشروطة بأن البذرة اجتازت العتبة على التحقق؛ وخصوصًا عند أعداد قليلة من البذور، لا تمثل تلك الدقة تقديرًا غير مشروط لكل البذور.\n\n"
    if marker not in report:
        raise RuntimeError("Could not find report insertion marker")
    report = report.replace(marker, note + marker, 1)
    (OUT / "report_ar.md").write_text(report, encoding="utf-8")
    atomic_json(path, data)
    print("POST-RUN ANALYSIS CORRECTION SAVED")
    for row in corrected:
        print(f"{row['task']} {row['feature_set']}: hit {row['n_reached_85']}/{row['n_seeds']} "
              f"Wilson {row['reach_rate_ci95_low_percent']:.2f}–{row['reach_rate_ci95_high_percent']:.2f}%")
    print("Paired gains:")
    for row in paired:
        print(f"{row['task']} {row['feature_set']}: test +{row['test_accuracy_gain_10_minus_1_mean_percentage_points']:.2f}pp "
              f"[{row['test_accuracy_gain_ci95_low_pp']:.2f}, {row['test_accuracy_gain_ci95_high_pp']:.2f}]")

if __name__ == "__main__":
    main()
