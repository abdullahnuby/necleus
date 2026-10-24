"""Stage-1 component diagnostic for EMA response time; not an official benchmark.

Uses one reserved diagnostic seed and a small, class-balanced slice of each task.
Only EMA tau changes (2, 4, 8 ticks); recurrent matrix, trial list, local gain
slope, target rate and feature extraction stay fixed. No readout is trained.
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.sequential_stdp import DynamicPulseReservoir
from experiments import run_activity_adaptive_inhibition_v0_35 as v35
from experiments.tune_local_inhibition_v0_37 import simulate_local
from experiments.run_inhibition_structure_v0_34 import extract_features, spectral_radius
from experiments.run_learning_curve_v0_30 import N_HIDDEN

SEED = 13990  # reserved for component smoke; not a tuning or official evaluation seed
TASKS = ("sync", "order")
TAUS = (2.0, 4.0, 8.0)
LOCAL_SLOPE = 2.0  # fixed from v0.37 development results; not tuned here
N_PER_CLASS = 16
OUTDIR = ROOT / "results" / "ema_tau_v0_38" / "component_smoke"


def ema_decay_factor(tau: float) -> float:
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be finite and > 0")
    return float(np.exp(-1.0 / tau))


def balanced_slice(concatenated, labels, n_per_class=N_PER_CLASS):
    ys = labels["train"]
    idx0 = np.flatnonzero(ys == 0)[:n_per_class]
    idx1 = np.flatnonzero(ys == 1)[:n_per_class]
    if len(idx0) != n_per_class or len(idx1) != n_per_class:
        raise RuntimeError("not enough examples in both classes for balanced smoke slice")
    idx = np.sort(np.concatenate([idx0, idx1]))
    trials = [concatenated[int(i)] for i in idx]
    selected_labels = ys[idx]
    if int(np.sum(selected_labels == 0)) != n_per_class or int(np.sum(selected_labels == 1)) != n_per_class:
        raise RuntimeError("smoke subset class balance failed")
    return trials, selected_labels


def run() -> dict:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    reservoir = DynamicPulseReservoir(seed=SEED, n_hidden=N_HIDDEN)
    matrices, matrix_diag = v35.make_shared_dale_matrix(reservoir, SEED)
    weights = np.asarray(matrices["dale_fixed_25"], dtype=np.float32)
    radius = float(spectral_radius(weights))
    if not np.isclose(radius, v35.TARGET_SPECTRAL_RADIUS, atol=5e-4):
        raise RuntimeError(f"unexpected spectral radius {radius}")

    rows = []
    integrity = {"seed": SEED, "tasks": {}, "same_recurrent_matrix_for_all_taus": True,
                 "recurrent_spectral_radius": radius, "n_hidden": N_HIDDEN,
                 "tau_ticks": list(TAUS), "local_slope": LOCAL_SLOPE,
                 "no_readout_training": True, "official_test_used": False}

    for task in TASKS:
        concatenated, bounds, labels, split_integrity = v35.make_data(SEED, task)
        trials, selected_labels = balanced_slice(concatenated, labels)
        task_rows = []
        outputs = {}
        base_inputs = None
        for tau in TAUS:
            inputs, spikes, diag = simulate_local(
                reservoir, trials, weights, slope=LOCAL_SLOPE, tau=tau,
                target_rate=v35.TARGET_ACTIVITY_RATE,
                gain_min=v35.INHIBITORY_GAIN_MIN,
                gain_max=v35.INHIBITORY_GAIN_MAX,
                return_diagnostics=True,
            )
            if base_inputs is None:
                base_inputs = inputs.copy()
            elif not np.array_equal(inputs, base_inputs):
                raise RuntimeError("input raster changed when only EMA tau changed")
            features = extract_features(spikes)
            if not np.isfinite(spikes).all() or not np.isfinite(features).all():
                raise RuntimeError(f"non-finite output for task={task}, tau={tau}")
            if not np.isclose(diag["first_tick_gain_mean"], 1.0, atol=1e-7):
                raise RuntimeError(f"initial gain must equal 1.0, got {diag['first_tick_gain_mean']}")
            outputs[tau] = {"spikes": spikes, "features": features}
            row = {
                "seed": SEED, "task": task, "tau_ticks": tau,
                "ema_decay_factor_rho": ema_decay_factor(tau),
                "n_trials": len(trials), "class_0": int(np.sum(selected_labels == 0)),
                "class_1": int(np.sum(selected_labels == 1)),
                "recurrent_spectral_radius": radius,
                "first_tick_gain_mean": diag["first_tick_gain_mean"],
                "mean_gain": diag["mean_gain"],
                "min_gain_seen": diag["min_gain_seen"], "max_gain_seen": diag["max_gain_seen"],
                "fraction_at_min": diag["fraction_at_min"], "fraction_at_max": diag["fraction_at_max"],
                "mean_within_trial_gain_std": diag["mean_within_trial_gain_std"],
                "mean_spikes_per_trial": float(spikes.sum(axis=(1, 2)).mean()),
                "mean_spike_rate_per_hidden_tick": float(spikes.mean()),
                "feature_shape": list(features.shape),
                "feature_std_mean": float(features.std(axis=0).mean()),
            }
            task_rows.append(row)

        base = outputs[4.0]
        for row in task_rows:
            tau = float(row["tau_ticks"])
            row["spike_disagreement_vs_tau4"] = float(np.mean(outputs[tau]["spikes"] != base["spikes"]))
            row["feature_relative_l2_delta_vs_tau4"] = float(
                np.linalg.norm(outputs[tau]["features"] - base["features"])
                / max(np.linalg.norm(base["features"]), 1e-12)
            )
            rows.append(row)
        integrity["tasks"][task] = {
            "n_trials": len(trials), "class_balance": {"0": int(np.sum(selected_labels == 0)), "1": int(np.sum(selected_labels == 1))},
            "same_inputs_across_tau": True,
            "tau4_spike_count": int(outputs[4.0]["spikes"].sum()),
            "spike_counts_by_tau": {str(t): int(outputs[t]["spikes"].sum()) for t in TAUS},
            "spike_outputs_changed_vs_tau4": {str(t): bool(np.any(outputs[t]["spikes"] != outputs[4.0]["spikes"])) for t in TAUS},
        }

    summary = {
        "title": "Nucleus Pulse v0.38 EMA tau component smoke test",
        "stage": "component-only diagnostic; not a tuning result and not an official benchmark",
        "seed": SEED, "taus": list(TAUS), "slope_fixed_from_v0_37": LOCAL_SLOPE,
        "target_rate": v35.TARGET_ACTIVITY_RATE, "gain_bounds": [v35.INHIBITORY_GAIN_MIN, v35.INHIBITORY_GAIN_MAX],
        "rows": rows, "integrity": integrity,
        "interpretation_limit": "One diagnostic seed and 32 balanced examples per task; no classifier training or accuracy claims.",
    }
    (OUTDIR / "results.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    with (OUTDIR / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[k for k, v in rows[0].items() if not isinstance(v, list)])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: v for k, v in r.items() if not isinstance(v, list)})
    lines = [
        "# Nucleus Pulse v0.38 — فحص مكوّن EMA tau فقط", "",
        "## نطاق الاختبار", "",
        f"- بذرة تشخيص واحدة محجوزة: {SEED}.",
        f"- لكل مهمة: {N_PER_CLASS} مثالًا من كل فئة (32 إجمالًا).",
        f"- قيم tau المختبرة: {list(TAUS)} ticks؛ الميل المحلي ثابت عند {LOCAL_SLOPE}.",
        "- لا تدريب لقارئ، ولا حساب دقة، ولا استخدام لمجموعة الاختبار الرسمية.",
        "- مصفوفة الخزان والمدخلات والعتبات والميزات ثابتة؛ المتغير الوحيد tau.", "",
        "## نتائج المحاكاة", "",
        "| المهمة | tau | rho=exp(-1/tau) | نبضات/مثال | متوسط الكسب | نسبة التشبع الأدنى | نسبة التشبع الأعلى | اختلاف النبضات عن tau=4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['task']} | {r['tau_ticks']:g} | {r['ema_decay_factor_rho']:.4f} | {r['mean_spikes_per_trial']:.2f} | {r['mean_gain']:.4f} | {100*r['fraction_at_min']:.2f}% | {100*r['fraction_at_max']:.2f}% | {100*r['spike_disagreement_vs_tau4']:.2f}% |")
    lines += ["", "## حدود الاستنتاج", "", "هذا اختبار مكوّن صغير بذرّة واحدة و32 مثالًا متوازنًا لكل مهمة. يثبت فقط أن معامل tau يمر إلى المحاكي ويغير/يحافظ على السلوك كما تسجله المقاييس. لا يسمح باستنتاج تفوق في التعلم، ولا ينبغي استخدامه لاختيار إعداد نهائي. الخطوة التالية — بعد مراجعة هذا الفحص — هي مسح تطوير منفصل على عدة بذور وقيم tau، باستخدام التحقق فقط.", ""]
    (OUTDIR / "report_ar.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": "passed", "seed": SEED, "rows": len(result["rows"]), "tasks": list(result["integrity"]["tasks"])}, indent=2))
