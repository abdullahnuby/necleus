"""Sequential reward-modulated STDP experiment for Nucleus Pulse v0.28.

The reservoir is a discrete-time LIF spiking network with fixed recurrent/input
connections. Only the output synapses are plastic. A local pre/post timing
eligibility is modulated by a scalar reward-prediction error (three-factor rule).
This is a software experiment, not a physical circuit or a biological model.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math
import random

import numpy as np

DURATION = 32
INPUT_CHANNELS = 4  # A, B, task_sync, task_order
N_OUTPUTS = 2       # shared negative/positive outputs; context pulse identifies the task


@dataclass(frozen=True)
class TemporalTrial:
    task: str
    a: Tuple[int, ...]
    b: Tuple[int, ...]
    label: int
    kind: str

    @property
    def target_class(self) -> int:
        # Both tasks share the same two readout neurons. The task identity is an
        # input context pulse, not a separate output head.
        return int(self.label)


def task_label(task: str, a: Sequence[int], b: Sequence[int]) -> int:
    if task == "sync":
        return int(any(abs(x - y) <= 1 for x in a for y in b))
    if task == "order":
        return int(any(2 <= y - x <= 4 for x in a for y in b))
    raise ValueError(f"Unknown task {task}")


def make_trial(rng: random.Random, task: str, label: int, phase: str, jitter: int = 0) -> TemporalTrial:
    """Generate temporal spike trains with optional unrelated distractor pulses.

    Training/validation and IID holdout use the same onset range but independent
    random pulse patterns. A shifted-phase split is available separately. Labels
    are derived from the actual presented train after jitter.
    """
    if phase in ("train", "test"):
        t_lo, t_hi = 4, 16
    elif phase == "shifted_test":
        t_lo, t_hi = 17, 24
    elif phase == "validation":
        t_lo, t_hi = 10, 17
    else:
        raise ValueError(phase)

    if task == "sync":
        if label:
            lag = rng.choice((-1, 0, 1))
            kind = "sync_positive"
        else:
            lag = rng.choice((-8, -6, -4, 4, 6, 8))
            kind = "sync_negative"
    elif task == "order":
        if label:
            lag = rng.choice((2, 3, 4))
            kind = "a_before_b_positive"
        else:
            lag = rng.choice((-8, -5, -4, -2, 0, 6, 8))
            kind = "order_negative"
    else:
        raise ValueError(f"Unknown task {task}")

    for _ in range(1000):
        ta = rng.randint(t_lo, t_hi)
        tb = ta + lag
        if not (2 <= tb < DURATION - 2):
            continue
        a_set, b_set = {ta}, {tb}
        # Add a variable number of distractor events to enlarge the hypothesis
        # space without hardcoding task labels from a single-pulse shortcut.
        for target_set in (a_set, b_set):
            n_extra = rng.choices((0, 1, 2), weights=(0.35, 0.45, 0.20), k=1)[0]
            for _j in range(n_extra):
                target_set.add(rng.randint(2, DURATION - 3))
        a, b = tuple(sorted(a_set)), tuple(sorted(b_set))
        actual = task_label(task, a, b)
        if jitter:
            a = tuple(sorted(set(max(2, min(DURATION - 3, x + rng.randint(-jitter, jitter))) for x in a)))
            b = tuple(sorted(set(max(2, min(DURATION - 3, x + rng.randint(-jitter, jitter))) for x in b)))
            actual = task_label(task, a, b)
            return TemporalTrial(task, a, b, actual, kind + "_jittered")
        if actual == int(label):
            return TemporalTrial(task, a, b, actual, kind)
    raise RuntimeError(f"Could not draw a valid {task} example with label={label}")


def make_dataset(seed: int, task: str, n: int, phase: str, jitter: int = 0) -> List[TemporalTrial]:
    rng = random.Random(seed)
    out: List[TemporalTrial] = []
    for i in range(n):
        out.append(make_trial(rng, task, i % 2, phase, jitter=jitter))
    rng.shuffle(out)
    return out




def make_disjoint_dataset(seed: int, task: str, n: int, phase: str,
                          forbidden: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]] = ()) -> List[TemporalTrial]:
    """Create a balanced dataset with unique event sequences and no forbidden overlap."""
    rng = random.Random(seed)
    forbidden_set = set(forbidden)
    seen = set(forbidden_set)
    out: List[TemporalTrial] = []
    attempts = 0
    limit = max(5000, n * 100)
    while len(out) < n and attempts < limit:
        label = len(out) % 2
        tr = make_trial(rng, task, label, phase)
        key = (tr.a, tr.b)
        attempts += 1
        if key in seen:
            continue
        seen.add(key)
        out.append(tr)
    if len(out) != n:
        raise RuntimeError(f"Could only make {len(out)}/{n} unique examples for {task}/{phase}")
    rng.shuffle(out)
    return out

def make_jittered_copy(seed: int, trials: Sequence[TemporalTrial], jitter: int = 1, forbidden: Sequence[Tuple[Tuple[int, ...], Tuple[int, ...]]] = ()) -> List[TemporalTrial]:
    """Perturb event times while preserving each trial's target label.

    This is a robustness test of timing noise rather than a relabelled new task:
    a candidate jitter is accepted only if the true analytic rule keeps the label.
    If no label-preserving jitter is found after repeated attempts, the original
    example is retained unchanged and counted in the report.
    """
    rng = random.Random(seed)
    forbidden_set = set(forbidden)
    out: List[TemporalTrial] = []
    for trial in trials:
        selected = trial  # original held-out trial is known not to overlap train
        for _ in range(200):
            a = tuple(sorted(set(max(2, min(DURATION - 3, t + rng.randint(-jitter, jitter))) for t in trial.a)))
            b = tuple(sorted(set(max(2, min(DURATION - 3, t + rng.randint(-jitter, jitter))) for t in trial.b)))
            y = task_label(trial.task, a, b)
            if y == trial.label and (a, b) not in forbidden_set:
                selected = TemporalTrial(trial.task, a, b, y, trial.kind + "_stable_jitter")
                break
        out.append(selected)
    return out

def rasterize(trial: TemporalTrial, duration: int = DURATION) -> np.ndarray:
    x = np.zeros((duration, INPUT_CHANNELS), dtype=np.float32)
    for t in trial.a:
        x[t, 0] = 1.0
    for t in trial.b:
        x[t, 1] = 1.0
    x[0, 2 if trial.task == "sync" else 3] = 1.0
    return x


class DynamicPulseReservoir:
    """Fixed sparse recurrent LIF reservoir; output synapses learn by reward STDP."""
    def __init__(self, seed: int, n_hidden: int = 24, duration: int = DURATION):
        self.seed = int(seed)
        self.n_hidden = int(n_hidden)
        self.duration = int(duration)
        rng = np.random.default_rng(seed)
        self.w_in = rng.normal(0.0, 0.92, (n_hidden, INPUT_CHANNELS)).astype(np.float32)
        # Give context channels adequate but not deterministic influence.
        self.w_in[:, 2:] *= 0.85
        mask = rng.random((n_hidden, n_hidden)) < 0.14
        np.fill_diagonal(mask, False)
        w_rec = rng.normal(0.0, 0.22, (n_hidden, n_hidden)) * mask
        eig = np.linalg.eigvals(w_rec)
        radius = float(np.max(np.abs(eig))) if eig.size else 1.0
        if radius > 1e-8:
            w_rec *= 0.72 / radius
        self.w_rec = w_rec.astype(np.float32)
        self.tau_mem = rng.uniform(2.0, 5.0, n_hidden).astype(np.float32)
        self.tau_syn = rng.uniform(2.0, 7.0, n_hidden).astype(np.float32)
        self.threshold = rng.uniform(0.27, 0.49, n_hidden).astype(np.float32)
        self.reset_leak = np.exp(-1.0 / self.tau_mem).astype(np.float32)
        self.syn_leak = np.exp(-1.0 / self.tau_syn).astype(np.float32)

    def simulate(self, trial: TemporalTrial) -> np.ndarray:
        x = rasterize(trial, self.duration)
        v = np.zeros(self.n_hidden, dtype=np.float32)
        i_syn = np.zeros(self.n_hidden, dtype=np.float32)
        refractory = np.zeros(self.n_hidden, dtype=np.int8)
        previous_spikes = np.zeros(self.n_hidden, dtype=np.float32)
        spikes = np.zeros((self.duration, self.n_hidden), dtype=np.float32)
        for t in range(self.duration):
            i_syn = self.syn_leak * i_syn + self.w_in @ x[t] + self.w_rec @ previous_spikes
            v = self.reset_leak * v + (1.0 - self.reset_leak) * i_syn
            active = (v >= self.threshold) & (refractory == 0)
            spikes[t] = active.astype(np.float32)
            v[active] = 0.0
            refractory = np.maximum(refractory - 1, 0)
            refractory[active] = 1
            previous_spikes = spikes[t]
        return spikes


def make_stdp_eligibility(spike_raster: np.ndarray, post_time: int | Sequence[int],
                          tau_plus: float = 4.0, tau_minus: float = 4.0) -> np.ndarray:
    """Pair-based eligibility across all selected-output spikes.

    Each pre/post pair contributes causal LTP (pre before post) or anti-causal
    LTD (post before pre). A virtual decision event is represented by a one-item
    sequence when the readout remains silent.
    """
    n = spike_raster.shape[1]
    elig = np.zeros(n, dtype=np.float32)
    post_times = [int(post_time)] if isinstance(post_time, (int, np.integer)) else [int(t) for t in post_time]
    if not post_times:
        return elig
    for t_pre in range(spike_raster.shape[0]):
        active = spike_raster[t_pre]
        coeff_sum = 0.0
        for t_post in post_times:
            dt = t_post - t_pre
            if dt >= 0:
                coeff_sum += math.exp(-dt / tau_plus)
            else:
                coeff_sum -= math.exp(dt / tau_minus)
        elig += active * coeff_sum
    # Normalize so learning rate does not grow unbounded with event count.
    norm = max(1.0, float(np.sum(np.abs(elig))))
    return elig / norm


class RewardSTDPReadout:
    """Event-based readout trained by a reward-modulated three-factor rule.

    The fixed reservoir emits real simulated spikes. Each hidden neuron projects
    through four phase-specific eligibility banks with local temporal kernels. A softmax action selector emits one output pulse
    at the decision time. Synaptic changes combine (1) pre/post timing eligibility,
    (2) the selected postsynaptic action, and (3) global reward-prediction error.

    This is a simplified software model of reward-modulated STDP; phase markers
    are a computational abstraction, not literal biological spikes.
    """
    # Four phase-specific eligibility banks. Each bank stores a local pre-spike
    # trace referenced to its phase marker; separate banks preserve temporal order.
    PHASE_BINS = ((0, 8), (8, 16), (16, 24), (24, 32))
    TRACE_TAU = 4.0

    def __init__(self, seed: int, n_hidden: int, lr: float = 0.08,
                 temperature: float = 0.70):
        rng = np.random.default_rng(seed)
        self.n_hidden = int(n_hidden)
        self.n_traces = len(self.PHASE_BINS)
        self.n_synapses = self.n_hidden * self.n_traces
        self.w = rng.normal(0.0, 0.05, (N_OUTPUTS, self.n_synapses)).astype(np.float32)
        self.lr = float(lr)
        self.temperature = float(temperature)
        self.reward_baseline = 0.0

    def temporal_eligibility(self, spikes: np.ndarray, decision_time: int | None = None) -> np.ndarray:
        """Phase-specific pre/post eligibility traces.

        Each bank has a phase marker at the end of its time window. Pre-spikes
        before the marker contribute causal potentiation, while later pre-spikes
        contribute anti-causal depression. These local traces remain available
        until the delayed scalar reward arrives.
        """
        if decision_time is None:
            decision_time = spikes.shape[0] - 1
        parts = []
        for start, stop in self.PHASE_BINS:
            marker = min(stop - 1, int(decision_time))
            trace = np.zeros(self.n_hidden, dtype=np.float32)
            for t in range(spikes.shape[0]):
                if t < start or t >= stop:
                    continue
                dt = marker - t
                kernel = math.exp(-dt / self.TRACE_TAU) if dt >= 0 else -math.exp(dt / self.TRACE_TAU)
                trace += spikes[t] * kernel
            parts.append(trace)
        feat = np.concatenate(parts)
        norm = max(float(np.linalg.norm(feat)), 1e-8)
        return feat / norm

    @staticmethod
    def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
        centered = x - float(np.mean(x))
        scale = max(float(np.std(x)), 0.05)
        z = centered / (scale * max(temperature, 1e-4))
        e = np.exp(np.clip(z, -20, 20))
        return e / np.sum(e)

    def score_trial(self, spikes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        feat = self.temporal_eligibility(spikes, decision_time=spikes.shape[0]-1)
        scores = self.w @ feat
        return scores.astype(np.float32), feat

    def predict(self, spikes: np.ndarray) -> Tuple[int, np.ndarray, int, bool]:
        scores, feat = self.score_trial(spikes)
        action = int(np.argmax(scores))
        # The selected action is emitted as a decision pulse at the end of the trial.
        return action, scores, int(spikes.shape[0]-1), False

    def train_trial(self, spikes: np.ndarray, target: int, rng: np.random.Generator) -> Dict:
        scores, eligibility = self.score_trial(spikes)
        p = self._softmax(scores, self.temperature)
        action = int(rng.choice(N_OUTPUTS, p=p))
        reward = 1.0 if action == int(target) else -1.0
        reward_error = reward - self.reward_baseline
        policy_factor = -p.copy()
        policy_factor[action] += 1.0
        # Three factors: synapse-local temporal eligibility, selected output pulse,
        # and global reward-prediction error. The policy factor allocates reward
        # across alternative output pulses rather than reinforcing every class.
        self.w += self.lr * reward_error * policy_factor[:, None] * eligibility[None, :]
        self.w[:] = np.clip(self.w, -1.5, 1.5)
        self.reward_baseline = 0.97 * self.reward_baseline + 0.03 * reward
        return {"action": action, "target": int(target), "correct": int(action == int(target)),
                "reward": reward, "reward_prediction_error": float(reward_error),
                "post_time": int(spikes.shape[0]-1), "post_spike_count": 1,
                "virtual_decision": 0, "eligibility_l1": float(np.sum(np.abs(eligibility)))}


def example_raster_features(trial: TemporalTrial) -> np.ndarray:
    """MLP baseline gets the same four channels in time-binned form."""
    return rasterize(trial).reshape(-1).astype(np.float32)
