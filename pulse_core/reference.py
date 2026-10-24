"""Independent small-step reference implementation for current-based LIF.

The update order is intentionally explicit and separate from PulseNetwork.run.
This is a code cross-check, not an external simulator such as Brian2.
"""
from __future__ import annotations
import math
from typing import Sequence, Tuple, List


def reference_lif_trace(
    duration: int,
    input_events: Sequence[Tuple[int, float]],
    tau_mem: float,
    tau_syn: float,
    threshold: float,
    reset_voltage: float = 0.0,
    current_cap: float | None = None,
) -> Tuple[List[float], List[float], List[int]]:
    """One input lane, exact exponential decay, same dt=1 discrete recurrence."""
    dm = math.exp(-1.0 / tau_mem)
    ds = math.exp(-1.0 / tau_syn)
    event_map = {}
    for t, amp in input_events:
        event_map.setdefault(t, []).append(amp)
    current = voltage = 0.0
    currents, voltages, spikes = [], [], []
    for t in range(duration):
        current *= ds
        for amp in event_map.get(t, []):
            current += amp
            if current_cap is not None:
                current = max(-current_cap, min(current_cap, current))
        voltage = reset_voltage + (voltage-reset_voltage)*dm + (1-dm)*current
        if voltage >= threshold:
            spikes.append(t)
            voltage = reset_voltage
            current = 0.0
        currents.append(current)
        voltages.append(voltage)
    return currents, voltages, spikes
