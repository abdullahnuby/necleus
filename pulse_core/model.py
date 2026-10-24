"""Event-driven LIF pulse network with per-synapse currents, inhibition and saturation.

This is a discrete-time software model. Pulses are events; postsynaptic currents
are accumulated independently per synapse, optionally clipped to a saturation
limit, then summed at each neuron. This is not a physical circuit simulator.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import math


@dataclass
class Synapse:
    name: str
    pre: str
    post: str
    weight: float
    delay: int = 1
    plastic: bool = False
    polarity: float = 1.0  # +1 excitatory, -1 inhibitory
    tau_syn: Optional[float] = None
    current_cap: Optional[float] = None  # symmetric bound on this synapse's current state

    def transmit(self) -> float:
        return float(self.weight) * float(self.polarity)


@dataclass
class LIFNeuron:
    name: str
    tau_mem: float = 2.0
    tau_syn: float = 6.0
    threshold: float = 1.0
    reset_voltage: float = 0.0
    refractory_steps: int = 2
    clear_synaptic_current_on_spike: bool = False
    voltage: float = 0.0
    current: float = 0.0
    refractory_left: int = 0

    def reset_state(self) -> None:
        self.voltage = self.reset_voltage
        self.current = 0.0
        self.refractory_left = 0

    @property
    def membrane_decay(self) -> float:
        return math.exp(-1.0 / max(self.tau_mem, 1e-6))

    @property
    def synaptic_decay(self) -> float:
        return math.exp(-1.0 / max(self.tau_syn, 1e-6))


@dataclass
class RunResult:
    spike_times: Dict[str, List[int]]
    voltage_trace: Dict[str, List[float]]
    current_trace: Dict[str, List[float]]
    delivered_pulses: List[Tuple[int, str, str, float]]
    synapse_current_trace: Optional[Dict[str, List[float]]] = None


class PulseNetwork:
    """Sparse event queue for discrete-time pulse propagation.

    Each synapse owns a separate decaying current state. When `current_cap` is
    configured, repeated input events saturate that synapse's state instead of
    increasing it without bound. Excitatory and inhibitory synapses share the
    same mechanism but have opposite current polarity.

    Source events may arrive at delay 0. A spike emitted by a neuron is causal:
    an internal propagation step is at least one tick even if `delay=0`.
    """
    def __init__(self, neurons: Sequence[LIFNeuron], synapses: Sequence[Synapse]):
        self.neurons: Dict[str, LIFNeuron] = {n.name: n for n in neurons}
        if len(self.neurons) != len(neurons):
            raise ValueError("Neuron names must be unique")
        self.synapses: List[Synapse] = list(synapses)
        self.outgoing: Dict[str, List[int]] = defaultdict(list)
        self.incoming: Dict[str, List[int]] = defaultdict(list)
        for idx, s in enumerate(self.synapses):
            if s.post not in self.neurons:
                raise ValueError(f"Unknown postsynaptic neuron: {s.post}")
            if s.delay < 0:
                raise ValueError("Synaptic delay cannot be negative")
            if s.tau_syn is not None and s.tau_syn <= 0:
                raise ValueError("Synaptic tau_syn must be positive")
            if s.current_cap is not None and s.current_cap <= 0:
                raise ValueError("current_cap must be positive")
            self.outgoing[s.pre].append(idx)
            self.incoming[s.post].append(idx)

    def reset(self) -> None:
        for n in self.neurons.values():
            n.reset_state()

    def run(self, input_spikes: Mapping[str, Iterable[int]], duration: int = 28) -> RunResult:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self.reset()
        queue: Dict[int, List[Tuple[str, float, int]]] = defaultdict(list)
        for source, times in input_spikes.items():
            for t in times:
                if t < 0 or t >= duration:
                    raise ValueError(f"Input spike {t} out of duration [0,{duration})")
                for idx in self.outgoing.get(source, []):
                    syn = self.synapses[idx]
                    arrival = t + syn.delay
                    if arrival < duration:
                        queue[arrival].append((syn.post, syn.transmit(), idx))

        spikes: Dict[str, List[int]] = {name: [] for name in self.neurons}
        v_trace: Dict[str, List[float]] = {name: [] for name in self.neurons}
        i_trace: Dict[str, List[float]] = {name: [] for name in self.neurons}
        lane_trace: Dict[str, List[float]] = {s.name: [] for s in self.synapses}
        lane_current: Dict[int, float] = {idx: 0.0 for idx in range(len(self.synapses))}
        delivered: List[Tuple[int, str, str, float]] = []

        for t in range(duration):
            # Independent synaptic decay; this preserves distinct synaptic kinetics.
            for idx, syn in enumerate(self.synapses):
                post = self.neurons[syn.post]
                tau = syn.tau_syn if syn.tau_syn is not None else post.tau_syn
                lane_current[idx] *= math.exp(-1.0 / max(tau, 1e-6))

            for target, amplitude, idx in queue.get(t, []):
                lane_current[idx] += amplitude
                cap = self.synapses[idx].current_cap
                if cap is not None:
                    lane_current[idx] = max(-cap, min(cap, lane_current[idx]))
                syn = self.synapses[idx]
                delivered.append((t, syn.pre, target, amplitude))

            # Trace synaptic states as they are presented to the membrane integrator,
            # before any spike-triggered reset clears them for the next step.
            for idx, syn in enumerate(self.synapses):
                lane_trace[syn.name].append(lane_current[idx])

            for name, n in self.neurons.items():
                total_current = sum(lane_current[idx] for idx in self.incoming.get(name, []))
                n.current = total_current
                recorded_voltage = n.voltage
                if n.refractory_left > 0:
                    n.refractory_left -= 1
                else:
                    d = n.membrane_decay
                    n.voltage = n.reset_voltage + (n.voltage - n.reset_voltage) * d + (1.0 - d) * n.current
                    recorded_voltage = n.voltage  # preserve threshold-crossing peak in the trace
                    if n.voltage >= n.threshold:
                        spikes[name].append(t)
                        n.voltage = n.reset_voltage
                        n.refractory_left = n.refractory_steps
                        if n.clear_synaptic_current_on_spike:
                            for idx in self.incoming.get(name, []):
                                lane_current[idx] = 0.0
                        for idx in self.outgoing.get(name, []):
                            syn = self.synapses[idx]
                            # Prevent same-tick zero-delay cascades; causality is at least one tick.
                            arrival = t + max(1, syn.delay)
                            if arrival < duration:
                                queue[arrival].append((syn.post, syn.transmit(), idx))
                v_trace[name].append(recorded_voltage)
                i_trace[name].append(total_current)

        return RunResult(spikes, v_trace, i_trace, delivered, lane_trace)


def unit_impulse_eligibility(
    spike_times: Mapping[str, Iterable[int]], duration: int, tau_mem: float,
    tau_syn: float, synapse_delay: Mapping[str, int],
) -> Dict[str, float]:
    """Simple linear subthreshold trace for the legacy supervised learning demo."""
    dm = math.exp(-1.0 / max(tau_mem, 1e-6))
    ds = math.exp(-1.0 / max(tau_syn, 1e-6))
    parts: Dict[str, List[float]] = {}
    for source, times_iter in spike_times.items():
        times = set(times_iter)
        current = voltage = 0.0
        vals: List[float] = []
        delay = synapse_delay.get(source, 1)
        for t in range(duration):
            current *= ds
            if t - delay in times:
                current += 1.0
            voltage = dm * voltage + (1.0 - dm) * current
            vals.append(voltage)
        parts[source] = vals
    if not parts:
        return {}
    total = [sum(vals[t] for vals in parts.values()) for t in range(duration)]
    peak_t = max(range(duration), key=lambda t: total[t])
    return {source: vals[peak_t] for source, vals in parts.items()}
