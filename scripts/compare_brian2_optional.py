"""Optional parity check against Brian2 when that simulator is installed.

The network uses Brian2's scheduler with the same discrete exponential update
order as PulseNetwork. This script was syntax-checked in the release environment
but could not be executed there because Brian2 was unavailable and package
installation had no network access.
"""
from __future__ import annotations
import json, math, sys
from pathlib import Path

try:
    import brian2 as b2
except ImportError:
    print("SKIP: Brian2 is not installed. Install brian2, then rerun this script.")
    raise SystemExit(0)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pulse_core.model import LIFNeuron, Synapse, PulseNetwork


def build_pulse_model(a_times, b_times, duration=32):
    n = LIFNeuron("and", tau_mem=1.0, tau_syn=0.8, threshold=1.02,
                  refractory_steps=2, clear_synaptic_current_on_spike=True)
    ss = [Synapse("A", "A", "and", 1.0, delay=1, tau_syn=0.8, current_cap=1.0),
          Synapse("B", "B", "and", 1.0, delay=1, tau_syn=0.8, current_cap=1.0)]
    net = PulseNetwork([n], ss)
    return net.run({"A": a_times, "B": b_times}, duration=duration).spike_times["and"]


def build_brian2_model(a_times, b_times, duration=32):
    b2.start_scope()
    dt = 1*b2.ms
    b2.defaultclock.dt = dt
    # Input bins are shifted one tick for the PulseNetwork's source-to-target delay=1.
    aa = [0.0] * duration
    bb = [0.0] * duration
    for t in a_times:
        if t+1 < duration: aa[t+1] = 1.0
    for t in b_times:
        if t+1 < duration: bb[t+1] = 1.0
    pulse_a = b2.TimedArray(aa, dt=dt)
    pulse_b = b2.TimedArray(bb, dt=dt)
    dm, ds, cap, threshold = math.exp(-1/1.0), math.exp(-1/0.8), 1.0, 1.02
    g = b2.NeuronGroup(
        1,
        model="v : 1\nia : 1\nib : 1",
        threshold="v >= 1.02",
        reset="v = 0; ia = 0; ib = 0",
        refractory=2*dt,
        method="euler",
    )
    g.v = 0; g.ia = 0; g.ib = 0
    # Match: decay synaptic state, add event and saturate; then integrate membrane.
    g.run_regularly(
        f"ia = clip(ia*{ds:.17g} + pulse_a(t), -{cap}, {cap})\n"
        f"ib = clip(ib*{ds:.17g} + pulse_b(t), -{cap}, {cap})",
        dt=dt, when="start", order=0,
    )
    g.run_regularly(
        f"v = v*{dm:.17g} + (1-{dm:.17g})*(ia+ib)",
        dt=dt, when="start", order=1,
    )
    mon = b2.SpikeMonitor(g)
    b2.Network(g, mon).run(duration*dt)
    return [int(round(float(t/dt))) for t in mon.t]


def main():
    cases = [
        (([5], [5]), "simultaneous"),
        (([5], [6]), "lag_one"),
        (([5], [7]), "lag_two"),
        ((list(range(2, 27)), []), "A_burst_only"),
        (([2,8,14,20], [4,10,16,22]), "interleaved_gap_two"),
        (([2,8,14,20], [2,8,14,20]), "synchronous_trains"),
    ]
    results = []
    for (a, b), name in cases:
        ours = build_pulse_model(a, b)
        ref = build_brian2_model(a, b)
        results.append({"case": name, "pulse_core_spikes": ours, "brian2_spikes": ref,
                        "exact_match": ours == ref})
    out = ROOT / "results" / "brian2_optional_comparison.json"
    out.write_text(json.dumps({"simulator":"Brian2", "results":results,
                               "all_exact_match":all(r["exact_match"] for r in results)}, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    if not all(r["exact_match"] for r in results):
        raise SystemExit(1)

if __name__ == "__main__":
    main()
