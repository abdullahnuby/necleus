"""Legacy v0.26 supervised coincidence learner kept for regression tests.

Not used by the v0.27 saturated-AND benchmark. Its error-modulated update is
not STDP and is retained only to preserve historical behavior/tests.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import random
from .model import LIFNeuron, Synapse, PulseNetwork, unit_impulse_eligibility


@dataclass
class Example:
    a: Tuple[int, ...]
    b: Tuple[int, ...]
    target: int
    kind: str


def build_network(initial_weights=(0.5, 0.5), tau_mem=2.0, tau_syn=6.0) -> PulseNetwork:
    # Two adjustable afferent synapses feed a dynamic coincidence cell.
    # Its emitted spike then travels through a separate readout synapse.
    neurons = [
        LIFNeuron("coincidence", tau_mem=tau_mem, tau_syn=tau_syn, threshold=1.0, refractory_steps=2),
        LIFNeuron("readout", tau_mem=2.0, tau_syn=6.0, threshold=1.0, refractory_steps=2),
    ]
    synapses = [
        Synapse("A_to_coincidence", "A", "coincidence", float(initial_weights[0]), delay=1, plastic=True),
        Synapse("B_to_coincidence", "B", "coincidence", float(initial_weights[1]), delay=1, plastic=True),
        Synapse("coincidence_to_readout", "coincidence", "readout", 2.0, delay=1, plastic=False),
    ]
    return PulseNetwork(neurons, synapses)


def predict(network: PulseNetwork, a: Iterable[int], b: Iterable[int], duration: int = 28) -> Tuple[int, object]:
    result = network.run({"A": tuple(a), "B": tuple(b)}, duration=duration)
    return int(bool(result.spike_times["readout"])), result


def gen_example(rng: random.Random, train: bool = True) -> Example:
    # Input onset times and exact patterns are sampled per call.
    t = rng.randint(2, 8) if train else rng.randint(10, 15)
    u = rng.random()
    if u < 0.40:
        # Positive: near-coincident pulses; temporal window <= 1 tick.
        lag = rng.choice((-1, 0, 1))
        a, b = (t,), (t + lag,)
        if min(a + b) < 0:
            a, b = (t,), (t + 1,)
        return Example(a, b, 1, "coincident")
    if u < 0.76:
        # Negative: both inputs fire, but their separation is large.
        sep = rng.randint(4, 9)
        return Example((t,), (t + sep,), 0, "separated")
    if u < 0.88:
        return Example((t,), (), 0, "A_only")
    return Example((), (t,), 0, "B_only")


def make_train_set(seed: int, n: int = 120) -> List[Example]:
    rng = random.Random(seed)
    # Controlled balanced pool: positives, delayed pairs, and single-channel cases.
    examples: List[Example] = []
    for _ in range(n // 3):
        t = rng.randint(2, 8)
        lag = rng.choice((-1, 0, 1))
        a, b = (t,), (t + lag,)
        examples.append(Example(a, b, 1, "coincident"))
    for _ in range(n // 3):
        t = rng.randint(2, 8)
        sep = rng.randint(4, 9)
        examples.append(Example((t,), (t + sep,), 0, "separated"))
    for _ in range(n // 6):
        examples.append(Example((rng.randint(2, 12),), (), 0, "A_only"))
        examples.append(Example((), (rng.randint(2, 12),), 0, "B_only"))
    rng.shuffle(examples)
    return examples


def make_test_set(seed: int, n_each: int = 60) -> List[Example]:
    rng = random.Random(seed + 900_001)
    out: List[Example] = []
    # New onset phase range (10..15) is disjoint from training range (2..8).
    for _ in range(n_each):
        t = rng.randint(10, 15)
        lag = rng.choice((-1, 0, 1))
        out.append(Example((t,), (t + lag,), 1, "coincident_unseen_phase"))
    for _ in range(n_each):
        t = rng.randint(10, 15)
        sep = rng.randint(4, 9)
        out.append(Example((t,), (t + sep,), 0, "separated_unseen_phase"))
    for _ in range(n_each):
        out.append(Example((rng.randint(10, 15),), (), 0, "A_only_unseen_phase"))
        out.append(Example((), (rng.randint(10, 15),), 0, "B_only_unseen_phase"))
    rng.shuffle(out)
    return out


def train_one_seed(seed: int, epochs: int = 30, lr: float = 0.5) -> Dict:
    rng = random.Random(seed)
    initial = (rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9))
    net = build_network(initial)
    train = make_train_set(seed)
    epoch_errors: List[int] = []
    for _ in range(epochs):
        rng.shuffle(train)
        errors = 0
        for ex in train:
            pred, _ = predict(net, ex.a, ex.b)
            if pred == ex.target:
                continue
            errors += 1
            teaching_error = ex.target - pred
            elig = unit_impulse_eligibility(
                {"A": ex.a, "B": ex.b}, duration=28,
                tau_mem=net.neurons["coincidence"].tau_mem,
                tau_syn=net.neurons["coincidence"].tau_syn,
                synapse_delay={"A": 1, "B": 1},
            )
            for idx, source in ((0, "A"), (1, "B")):
                syn = net.synapses[idx]
                if not (ex.a if source == "A" else ex.b):
                    continue
                syn.weight = max(0.05, min(2.5, syn.weight + lr * teaching_error * elig.get(source, 0.0)))
        epoch_errors.append(errors)
    return {"seed": seed, "initial_weights": list(initial), "final_weights": [net.synapses[0].weight, net.synapses[1].weight],
            "epoch_errors": epoch_errors, "network": net}
