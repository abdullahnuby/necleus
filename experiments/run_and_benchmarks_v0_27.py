"""v0.27: saturated pulse-AND, varied pulse trains and CPU baselines.

This is a mechanism/implementation experiment, not a claim that the AND circuit
learned its weights. Circuit parameters are fixed by design and reported openly.
"""
from __future__ import annotations
import json, math, platform, random, statistics, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "and_benchmarks_v0_27"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from pulse_core.model import LIFNeuron, Synapse, PulseNetwork
from pulse_core.tasks import make_dataset, make_varied_example, coincidence_rule, count_only_rule, binned_features
from pulse_core.reference import reference_lif_trace

DURATION = 32
TEST_SEEDS = [27001, 27011, 27023, 27031, 27041, 27053, 27061, 27071, 27083, 27091]
TRAIN_SEED = 270007
TRAIN_N = 5000
TRAIN_EPOCHS = 60


def build_saturated_and() -> PulseNetwork:
    neuron = LIFNeuron(
        "and", tau_mem=1.0, tau_syn=0.8, threshold=1.02,
        refractory_steps=2, clear_synaptic_current_on_spike=True,
    )
    synapses = [
        Synapse("A_to_and", "A", "and", 1.0, delay=1, tau_syn=0.8, current_cap=1.0),
        Synapse("B_to_and", "B", "and", 1.0, delay=1, tau_syn=0.8, current_cap=1.0),
    ]
    return PulseNetwork([neuron], synapses)


def pulse_predict(net, ex) -> int:
    res = net.run({"A": ex.a, "B": ex.b}, duration=DURATION)
    return int(bool(res.spike_times["and"]))


def make_mlp_dataset(examples):
    x = np.asarray([binned_features(e, DURATION) for e in examples], dtype=np.float32)
    y = np.asarray([e.target for e in examples], dtype=np.float32).reshape(-1, 1)
    return torch.from_numpy(x), torch.from_numpy(y)


def fit_temporal_mlp(seed: int):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(TRAIN_SEED)
    train = [make_varied_example(rng, DURATION) for _ in range(TRAIN_N)]
    x, y = make_mlp_dataset(train)
    model = nn.Sequential(nn.Linear(DURATION * 2, 32), nn.Tanh(), nn.Linear(32, 16), nn.Tanh(), nn.Linear(16, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)
    gen = torch.Generator().manual_seed(seed + 13)
    batch_size = 128
    for _ in range(TRAIN_EPOCHS):
        order = torch.randperm(len(x), generator=gen)
        for start in range(0, len(x), batch_size):
            idx = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[idx])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, y[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return model, sum(p.numel() for p in model.parameters())


def metrics(y_true, y_pred):
    tp = sum(int(a == 1 and b == 1) for a, b in zip(y_true, y_pred))
    tn = sum(int(a == 0 and b == 0) for a, b in zip(y_true, y_pred))
    fp = sum(int(a == 0 and b == 1) for a, b in zip(y_true, y_pred))
    fn = sum(int(a == 1 and b == 0) for a, b in zip(y_true, y_pred))
    n = max(1, len(y_true))
    return {"n": len(y_true), "accuracy": (tp+tn)/n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "positive_recall": tp/max(1, tp+fn), "negative_recall": tn/max(1, tn+fp)}


def analytical_reference_check():
    # High threshold prevents spikes; independently calculate the same discrete
    # exponential recurrence for a capped synaptic current lane.
    event_times = [2, 5, 10, 11]
    tau_m, tau_s, cap = 2.3, 1.7, 0.9
    net = PulseNetwork(
        [LIFNeuron("ref", tau_mem=tau_m, tau_syn=tau_s, threshold=100.0)],
        [Synapse("src_ref", "SRC", "ref", 1.0, delay=0, tau_syn=tau_s, current_cap=cap)],
    )
    # All input events use delay zero, so times align with reference events.
    got = net.run({"SRC": event_times}, duration=20)
    # Independent reference recurrence uses the same declared discrete LIF
    # equation but separate code. This checks implementation parity, not Brian2.
    ref_i, ref_v, ref_spikes = reference_lif_trace(
        duration=20, input_events=[(t, 1.0) for t in event_times],
        tau_mem=tau_m, tau_syn=tau_s, threshold=100.0, current_cap=cap,
    )
    got_i = got.current_trace["ref"]
    got_v = got.voltage_trace["ref"]
    return {"current_max_abs_error": max(abs(a-b) for a,b in zip(got_i, ref_i)),
            "voltage_max_abs_error": max(abs(a-b) for a,b in zip(got_v, ref_v)),
            "note": "The independent recurrence is used for numerical cross-check. Brian2 was unavailable in this offline runtime and was not executed."}


def main():
    # Fit three independently seeded MLP baselines on the same labeled task family.
    mlps = []
    mlp_param_counts = []
    for seed in [311, 317, 331]:
        model, count = fit_temporal_mlp(seed)
        mlps.append(model); mlp_param_counts.append(count)

    all_examples = []
    per_seed = []
    net = build_saturated_and()
    for seed in TEST_SEEDS:
        examples = make_dataset(seed, n=300, duration=DURATION)  # 30 examples per pattern kind
        all_examples.extend(examples)
        y = [e.target for e in examples]
        pulse = [pulse_predict(net, e) for e in examples]
        count = [count_only_rule(e.a, e.b) for e in examples]
        oracle = [coincidence_rule(e.a, e.b, window=1) for e in examples]
        x_test = torch.from_numpy(np.asarray([binned_features(e, DURATION) for e in examples], dtype=np.float32))
        mlp_preds = []
        with torch.no_grad():
            for model in mlps:
                logits = model(x_test).squeeze(-1)
                mlp_preds.append((logits >= 0).to(torch.int32).tolist())
        mlp_vote = [int(sum(pred[i] for pred in mlp_preds) >= 2) for i in range(len(examples))]
        per_seed.append({
            "seed": seed,
            "pulse_network": metrics(y, pulse),
            "count_only": metrics(y, count),
            "analytic_coincidence_rule": metrics(y, oracle),
            "temporal_mlp_majority_3_seeds": metrics(y, mlp_vote),
            "by_kind": {},
        })
        for kind in sorted({e.kind for e in examples}):
            idx = [i for i,e in enumerate(examples) if e.kind == kind]
            per_seed[-1]["by_kind"][kind] = {
                "pulse_network": metrics([y[i] for i in idx], [pulse[i] for i in idx]),
                "count_only": metrics([y[i] for i in idx], [count[i] for i in idx]),
                "temporal_mlp": metrics([y[i] for i in idx], [mlp_vote[i] for i in idx]),
            }

    # Inference time per sample is reported as measured, not assumed faster.
    bench_examples = all_examples[:500]
    bench_x = torch.from_numpy(np.asarray([binned_features(e, DURATION) for e in bench_examples], dtype=np.float32))
    start = time.perf_counter()
    for e in bench_examples: pulse_predict(net, e)
    pulse_elapsed = time.perf_counter()-start
    start = time.perf_counter()
    with torch.no_grad():
        for row in bench_x:
            _ = mlps[0](row.unsqueeze(0))
    mlp_single_elapsed = time.perf_counter()-start
    start = time.perf_counter()
    with torch.no_grad():
        _ = mlps[0](bench_x)
    mlp_batch_elapsed = time.perf_counter()-start
    start = time.perf_counter()
    for e in bench_examples: _ = coincidence_rule(e.a, e.b, 1)
    rule_elapsed = time.perf_counter()-start

    # Test inhibitory polarity explicitly: excitation plus inhibition must be
    # weaker than excitation alone, all else held constant.
    inh_neurons = [LIFNeuron("target", tau_mem=1.0, tau_syn=1.0, threshold=0.8)]
    exc_only = PulseNetwork(inh_neurons, [Synapse("exc", "A", "target", 1.5, delay=0, tau_syn=1.0)])
    with_inhib = PulseNetwork([LIFNeuron("target", tau_mem=1.0, tau_syn=1.0, threshold=0.8)], [
        Synapse("exc", "A", "target", 1.5, delay=0, tau_syn=1.0),
        Synapse("inh", "B", "target", 0.6, delay=0, polarity=-1.0, tau_syn=1.0),
    ])
    exc_res = exc_only.run({"A": [2]}, duration=10)
    inh_res = with_inhib.run({"A": [2], "B": [2]}, duration=10)

    # Verify capped-input burst behavior and AND truth table/lag window.
    fixed_cases = [
        ({"A": [5], "B": [5]}, 1, "simultaneous"),
        ({"A": [5], "B": [6]}, 1, "A_before_B_by_1"),
        ({"A": [6], "B": [5]}, 1, "B_before_A_by_1"),
        ({"A": [5], "B": [7]}, 0, "gap_2"),
        ({"A": [5], "B": [9]}, 0, "gap_4"),
        ({"A": list(range(2, 27)), "B": []}, 0, "A_high_rate_burst_only"),
        ({"A": [], "B": list(range(2, 27))}, 0, "B_high_rate_burst_only"),
        ({"A": [2,8,14,20], "B": [4,10,16,22]}, 0, "interleaved_gap_2_trains"),
        ({"A": [2,8,14,20], "B": [2,8,14,20]}, 1, "synchronous_trains"),
    ]
    fixed = []
    for inp, target, name in fixed_cases:
        pred = pulse_predict(net, type("Ex", (), {"a": tuple(inp.get("A", [])), "b": tuple(inp.get("B", []))})())
        fixed.append({"name": name, "target": target, "prediction": pred, "correct": pred == target})

    y_all = [e.target for e in all_examples]
    pulse_all = [pulse_predict(net, e) for e in all_examples]
    count_all = [count_only_rule(e.a, e.b) for e in all_examples]
    oracle_all = [coincidence_rule(e.a, e.b, 1) for e in all_examples]
    x_all = torch.from_numpy(np.asarray([binned_features(e, DURATION) for e in all_examples], dtype=np.float32))
    mlp_predictions = []
    with torch.no_grad():
        for model in mlps:
            mlp_predictions.append((model(x_all).squeeze(-1) >= 0).to(torch.int32).tolist())
    mlp_vote_all = [int(sum(pred[i] for pred in mlp_predictions) >= 2) for i in range(len(all_examples))]

    result = {
        "version": "v0.27",
        "purpose": "Saturation/inhibition AND validation with varied pulse trains and fair reference baselines",
        "parameters": {"duration": DURATION, "threshold": 1.02, "tau_mem": 1.0, "tau_syn": 0.8,
                        "weight_A": 1.0, "weight_B": 1.0, "per_synapse_current_cap": 1.0,
                        "coincidence_window_ticks": 1, "test_seeds": TEST_SEEDS,
                        "test_examples_per_seed": 300, "mlp_train_examples": TRAIN_N,
                        "mlp_epochs": TRAIN_EPOCHS, "mlp_hidden": [32,16], "mlp_seeds": [311,317,331]},
        "dataset": {"examples": len(all_examples), "pattern_kinds": sorted({e.kind for e in all_examples}),
                    "positive_rate": sum(y_all)/len(y_all)},
        "overall": {
            "pulse_saturated_and": metrics(y_all,pulse_all),
            "count_only_baseline": metrics(y_all,count_all),
            "single_layer_analytic_rule_ceiling": metrics(y_all,oracle_all),
            "nonspiking_temporal_mlp_3_model_majority": metrics(y_all,mlp_vote_all),
        },
        "per_seed": per_seed,
        "fixed_stress_cases": fixed,
        "inhibitory_polarity_check": {
            "exc_only_spikes": exc_res.spike_times["target"],
            "exc_plus_inhib_spikes": inh_res.spike_times["target"],
            "exc_only_max_voltage": max(exc_res.voltage_trace["target"]),
            "exc_plus_inhib_max_voltage": max(inh_res.voltage_trace["target"]),
        },
        "inference_timing_500_examples_seconds": {
            "pulse_event_python": pulse_elapsed,
            "temporal_mlp_one_model_single_sample_loop": mlp_single_elapsed,
            "temporal_mlp_one_model_batched_500": mlp_batch_elapsed,
            "analytic_rule": rule_elapsed,
            "warning": "Not a hardware speedup claim: Python event loops and Torch kernels differ in batching/vectorization and overhead.",
        },
        "reference_validation": analytical_reference_check(),
        "external_simulator": {"name": "Brian2", "available": False, "executed": False,
                               "reason": "Brian2 is not installed and package installation was blocked by unavailable network access in this runtime.",
                               "fallback": "Independent exact-discrete recurrence cross-check is provided; Brian2 parity remains an open validation item."},
        "limits": [
            "The pulse AND cell has manually specified parameters; this experiment validates the mechanism and dynamics, not learned weights.",
            "The temporal MLP is trained on 5000 examples; the pulse circuit is configured analytically, so this is not an equal-training-budget learning comparison.",
            "The analytic coincidence rule is an oracle for the task definition, not a learned model.",
            "A later experiment will train a network sequentially on synchrony then A-before-B and compare reward-modulated STDP with a same-order MLP baseline.",
        ],
    }
    out_path = OUT / "results.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    report = [
        "# Nucleus Pulse Software v0.27 — AND والتوقيت والمراجع",
        "",
        "## الهدف",
        "إصلاح AND زمني باستخدام تشبع تيار كل مشبك مستقلًا، مع اختبار قطارات نبضات متنوعة ومقارنة مع مرجع قاعدي أحادي وموديل زمني غير نبضي.",
        "",
        "## النتيجة الإجمالية",
        f"- أمثلة الاختبار: {len(all_examples)} عبر 10 بذور.",
        f"- شبكة النبضات المشبعة: {result['overall']['pulse_saturated_and']['accuracy']*100:.2f}%.",
        f"- baseline عدّ القنوات دون توقيت: {result['overall']['count_only_baseline']['accuracy']*100:.2f}%.",
        f"- MLP غير نبضي بمدخلات زمنية، تصويت 3 نماذج: {result['overall']['nonspiking_temporal_mlp_3_model_majority']['accuracy']*100:.2f}%.",
        f"- القاعدة التحليلية المحددة للمهمة: {result['overall']['single_layer_analytic_rule_ceiling']['accuracy']*100:.2f}% (oracle، ليست نموذجًا متعلمًا).",
        "",
        "## اختبارات قطارات النبض",
    ]
    for r in fixed:
        report.append(f"- {r['name']}: target={r['target']}, prediction={r['prediction']}, correct={r['correct']}.")
    report += ["", "## التثبيط والتشبع", f"- أقصى جهد مع الإثارة فقط: {result['inhibitory_polarity_check']['exc_only_max_voltage']:.6f}.",
               f"- أقصى جهد مع الإثارة والتثبيط المتزامن: {result['inhibitory_polarity_check']['exc_plus_inhib_max_voltage']:.6f}.",
               f"- spikes excitation only: {result['inhibitory_polarity_check']['exc_only_spikes']}.",
               f"- spikes excitation + inhibition: {result['inhibitory_polarity_check']['exc_plus_inhib_spikes']}.", "",
               "## سلامة الحساب", f"- فرق التيار الأقصى عن المرجع المستقل: {result['reference_validation']['current_max_abs_error']:.3e}.",
               f"- فرق الجهد الأقصى عن المرجع المستقل: {result['reference_validation']['voltage_max_abs_error']:.3e}.",
               "- لم تُنفّذ مقارنة Brian2 لأن المكتبة غير مثبتة وبيئة التنفيذ لم تسمح بتنزيلها؛ لا ندعي parity مع simulator خارجي بعد.", "",
               "## تكلفة التشغيل على CPU", f"- 500 مثال، شبكة أحداث Python: {pulse_elapsed:.4f} ثانية.",
               f"- نموذج MLP واحد، استدلال عينة بعينة: {mlp_single_elapsed:.4f} ثانية.",
               f"- نموذج MLP واحد، استدلال دفعي 500 مثال: {mlp_batch_elapsed:.4f} ثانية.",
               f"- القاعدة التحليلية: {rule_elapsed:.4f} ثانية.",
               "- هذه قياسات بيئة واحدة وليست حكمًا عامًا على الأداء. التنفيذ الحدثي في Python ليس مضمونًا أن يكون أسرع من العمليات المصفوفية؛ العتاد المتخصص قصة مختلفة.", "",
               "## حدود التجربة", *[f"- {x}" for x in result["limits"]], "",
               "## التفسير", "نجاح AND هنا ناتج عن تصميم خلية بتشبع لكل مشبك وثوابت زمن مناسبة؛ لم نتعلم هذه المعمارية من البيانات. التجربة تثبت صحة آلية التشبع والتوقيت على القطارات المختبرة، لا تعلمًا عامًا ولا تفوقًا شاملًا على MLP.", ""]
    (OUT / "report_ar.md").write_text("\n".join(report), encoding="utf-8")

    # CSV of per-pattern/per-seed metrics for auditable inspection.
    csv_lines = ["seed,pattern,pulse_accuracy,count_accuracy,mlp_accuracy,n"]
    for s in per_seed:
        for kind, vals in s["by_kind"].items():
            csv_lines.append(f"{s['seed']},{kind},{vals['pulse_network']['accuracy']:.6f},{vals['count_only']['accuracy']:.6f},{vals['temporal_mlp']['accuracy']:.6f},{vals['pulse_network']['n']}")
    (OUT / "per_pattern_results.csv").write_text("\n".join(csv_lines)+"\n", encoding="utf-8")

    # Representative traces: synchronous, separated, and single-channel burst.
    import csv as csv_module
    trace_cases = [
        ("synchronous", {"A": [8], "B": [8]}),
        ("lag_one", {"A": [8], "B": [9]}),
        ("lag_two", {"A": [8], "B": [10]}),
        ("A_burst_only", {"A": list(range(2, 27, 1)), "B": []}),
    ]
    trace_rows = []
    trace_pack = {}
    for label, inp in trace_cases:
        tr = net.run(inp, duration=DURATION)
        trace_pack[label] = tr
        for t in range(DURATION):
            trace_rows.append({
                "case": label, "t": t, "voltage": tr.voltage_trace["and"][t],
                "total_current": tr.current_trace["and"][t],
                "current_A": tr.synapse_current_trace["A_to_and"][t],
                "current_B": tr.synapse_current_trace["B_to_and"][t],
                "spike": int(t in tr.spike_times["and"]),
            })
    with (OUT / "representative_traces.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=list(trace_rows[0]))
        writer.writeheader(); writer.writerows(trace_rows)
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        colors = {"synchronous":"#16803c", "lag_one":"#2374ab", "lag_two":"#c77d00", "A_burst_only":"#8754a1"}
        for label, tr in trace_pack.items():
            axes[0].plot(range(DURATION), tr.voltage_trace["and"], label=label, color=colors[label])
        axes[0].axhline(1.02, color="black", linestyle="--", linewidth=1, label="threshold")
        axes[0].set_ylabel("Membrane voltage")
        axes[0].legend(ncol=3, fontsize=8); axes[0].grid(alpha=.2)
        tr = trace_pack["synchronous"]
        axes[1].plot(range(DURATION), tr.synapse_current_trace["A_to_and"], label="A synaptic current")
        axes[1].plot(range(DURATION), tr.synapse_current_trace["B_to_and"], label="B synaptic current")
        axes[1].plot(range(DURATION), tr.current_trace["and"], label="total current", linewidth=2)
        axes[1].set_xlabel("Discrete time step"); axes[1].set_ylabel("Synaptic current")
        axes[1].legend(fontsize=8); axes[1].grid(alpha=.2)
        fig.suptitle("Nucleus Pulse v0.27 — saturated AND traces (software model)")
        fig.tight_layout()
        fig.savefig(OUT / "pulse_and_traces.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass
    print((OUT / "report_ar.md").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
