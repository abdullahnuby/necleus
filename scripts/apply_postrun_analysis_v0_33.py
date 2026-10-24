"""Append descriptive firing-activity diagnostics to the v0.33 report only.

This script does not modify the model, labels, model predictions, or any
registered metric. It summarizes activity diagnostics already saved with each
seed and adds a transparent post-run note to the Arabic report.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "delayed_reservoir_v0_33"


def main() -> None:
    result_path = OUT / "results.json"
    report_path = OUT / "report_ar.md"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    lines = ["## تشخيص وصفي لنشاط الخزان بعد التشغيل", "",
             "هذه المقاييس وصفية مجمعة على أمثلة التدريب والتحقق والاختبار بعد توليد الخزان. لم تُستخدم لاختيار النموذج أو تعديل الإعدادات.",
             "", "| المهمة | الخزان | متوسط النبضات لكل مثال | الخلايا التي أطلقت نبضة واحدة على الأقل | أمثلة بخزان صامت |", "|---|---|---:|---:|---:|"]
    rows_out = []
    for task in ("sync", "order"):
        for condition in result["conditions"]:
            items = [seed_item["tasks"][task]["spike_diagnostics"][condition]
                     for seed_item in result["integrity"]]
            summary = {
                "task": task, "condition": condition, "n_seeds": len(items),
                "mean_spikes_per_trial": float(np.mean([x["mean_spikes_per_trial"] for x in items])),
                "fraction_hidden_cells_with_at_least_one_spike_percent": 100 * float(np.mean([x["fraction_hidden_cells_with_at_least_one_spike"] for x in items])),
                "fraction_trials_with_zero_hidden_spikes_percent": 100 * float(np.mean([x["fraction_trials_with_zero_hidden_spikes"] for x in items])),
                "mean_spike_rate_per_hidden_tick_percent": 100 * float(np.mean([x["mean_spike_rate_per_hidden_tick"] for x in items])),
            }
            rows_out.append(summary)
            lines.append(f"| {task} | {condition} | {summary['mean_spikes_per_trial']:.2f} | {summary['fraction_hidden_cells_with_at_least_one_spike_percent']:.2f}% | {summary['fraction_trials_with_zero_hidden_spikes_percent']:.2f}% |")
    lines += ["", "في المتوسط، التأخيرات المتنوعة خفّضت عدد النبضات في المثال قليلًا، مع زيادة طفيفة في نسبة الخلايا التي نشطت. لذلك لا ينبغي تفسير فرق الدقة وحده كأثر توقيت خالص من دون دراسة آلية إضافية؛ هذا الملخص لا يثبت الوساطة السببية.", ""]
    (OUT / "activity_diagnostics_summary.json").write_text(json.dumps(rows_out, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (ROOT / "docs" / "POSTRUN_ANALYSIS_NOTE_V0_33_AR.md").write_text("# ملاحظة تحليل بعد التشغيل — v0.33\n\n" + "\n".join(lines[2:]) + "\n", encoding="utf-8")
    report = report_path.read_text(encoding="utf-8")
    marker = "## الملفات\n"
    if "## تشخيص وصفي لنشاط الخزان بعد التشغيل" not in report:
        if marker not in report:
            raise RuntimeError("Could not find report insertion point")
        report = report.replace(marker, "\n".join(lines) + "\n" + marker)
    report_path.write_text(report, encoding="utf-8")
    print("Post-run activity note appended; no model or performance result changed.")


if __name__ == "__main__":
    main()
