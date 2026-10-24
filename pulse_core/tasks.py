"""Input generators and baselines for temporal AND / coincidence tasks."""
from __future__ import annotations
from dataclasses import dataclass
import random
from typing import Tuple, List


@dataclass(frozen=True)
class PulseExample:
    a: Tuple[int, ...]
    b: Tuple[int, ...]
    target: int
    kind: str


def coincidence_rule(a, b, window: int = 1) -> int:
    """Analytic task definition: output iff a pair of A/B pulses is within `window`."""
    return int(any(abs(x-y) <= window for x in a for y in b))


def count_only_rule(a, b) -> int:
    """Timing-blind baseline: positive whenever both channels fired at least once."""
    return int(bool(a) and bool(b))


def _extras(rng: random.Random, lo: int, hi: int, n: int, exclude=()) -> Tuple[int, ...]:
    allowed = [t for t in range(lo, hi + 1) if t not in set(exclude)]
    rng.shuffle(allowed)
    return tuple(sorted(allowed[:max(0, min(n, len(allowed)))]))


def make_varied_example(rng: random.Random, duration: int = 32, kind: str | None = None) -> PulseExample:
    """Generate varied pulse trains; labels come from the frozen coincidence rule."""
    choices = [
        "coincident_single", "near_single", "separated_single", "a_only_burst",
        "b_only_burst", "interleaved_gap2", "synchronous_trains", "noisy_positive",
        "random_positive", "random_negative",
    ]
    kind = kind or rng.choice(choices)
    lo, hi = 2, duration - 4

    if kind == "coincident_single":
        t = rng.randint(lo, hi)
        a, b = (t,), (t,)
    elif kind == "near_single":
        t = rng.randint(lo, hi-1)
        lag = rng.choice((-1, 1))
        a, b = (t,), (t+lag,)
    elif kind == "separated_single":
        t = rng.randint(lo, hi-5)
        lag = rng.choice((-5, -4, -3, 2, 3, 4, 5))
        u = max(lo, min(hi, t + lag))
        if abs(u-t) <= 1:
            u = min(hi, t+3)
        a, b = (t,), (u,)
    elif kind == "a_only_burst":
        a = tuple(sorted(rng.sample(range(lo, hi+1), rng.randint(3, 8))))
        b = ()
    elif kind == "b_only_burst":
        a = ()
        b = tuple(sorted(rng.sample(range(lo, hi+1), rng.randint(3, 8))))
    elif kind == "interleaved_gap2":
        start = rng.randint(lo, lo+2)
        a = tuple(t for t in range(start, min(hi+1, start+20), 6))
        b = tuple(t+2 for t in a if t+2 <= hi)
    elif kind == "synchronous_trains":
        start = rng.randint(lo, lo+3)
        base = tuple(t for t in range(start, min(hi+1, start+20), rng.choice((4, 5, 6))))
        a, b = base, base
    elif kind == "noisy_positive":
        t = rng.randint(lo, hi-1)
        lag = rng.choice((-1, 0, 1))
        a_set = {t} | set(_extras(rng, lo, hi, rng.randint(1, 4), exclude={t}))
        b_set = {t+lag} | set(_extras(rng, lo, hi, rng.randint(1, 4), exclude={t+lag}))
        a, b = tuple(sorted(a_set)), tuple(sorted(b_set))
    elif kind == "random_positive":
        a_set = set(rng.sample(range(lo, hi+1), rng.randint(1, 5)))
        b_set = set(rng.sample(range(lo, hi+1), rng.randint(1, 5)))
        t = rng.randint(lo, hi-1)
        u = max(lo, min(hi, t+rng.choice((-1, 0, 1))))
        a_set.add(t); b_set.add(u)
        a, b = tuple(sorted(a_set)), tuple(sorted(b_set))
    elif kind == "random_negative":
        for _ in range(500):
            a0 = tuple(sorted(rng.sample(range(lo, hi+1), rng.randint(1, 5))))
            b0 = tuple(sorted(rng.sample(range(lo, hi+1), rng.randint(1, 5))))
            if coincidence_rule(a0, b0) == 0:
                a, b = a0, b0
                break
        else:
            a, b = (lo,), (hi,)
    else:
        raise ValueError(f"Unknown pattern kind: {kind}")

    a, b = tuple(sorted(set(a))), tuple(sorted(set(b)))
    return PulseExample(a, b, coincidence_rule(a, b), kind)


def make_dataset(seed: int, n: int = 1000, duration: int = 32) -> List[PulseExample]:
    rng = random.Random(seed)
    kinds = [
        "coincident_single", "near_single", "separated_single", "a_only_burst",
        "b_only_burst", "interleaved_gap2", "synchronous_trains", "noisy_positive",
        "random_positive", "random_negative",
    ]
    return [make_varied_example(rng, duration, kinds[i % len(kinds)]) for i in range(n)]


def binned_features(ex: PulseExample, duration: int = 32) -> List[float]:
    """Non-spiking baseline input: raw binary time bins for A then B."""
    x = [0.0] * (2 * duration)
    for t in ex.a:
        if 0 <= t < duration: x[t] = 1.0
    for t in ex.b:
        if 0 <= t < duration: x[duration+t] = 1.0
    return x
