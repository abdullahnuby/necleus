"""Observation encoding for the exploration lab.

Each game already returns a fixed-length numeric vector in ``[0, 1]`` (with a
small number of signed relative features).  The agent uses a deterministic binary
threshold encoder: a feature emits an input spike when its value is at or above
``threshold``.  No rate code, learned embedding, label, or game-specific policy is
involved.  The threshold is a fixed interface constant, not a tuned parameter.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np

from .games import FEATURE_DIM

DEFAULT_SPIKE_THRESHOLD = 0.5


def spikes_from_features(
    features: Iterable[float],
    threshold: float = DEFAULT_SPIKE_THRESHOLD,
) -> np.ndarray:
    vector = np.asarray(tuple(features), dtype=np.float64)
    if vector.shape != (FEATURE_DIM,):
        raise ValueError(f"expected {FEATURE_DIM} features, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("observation contains non-finite values")
    return (vector >= threshold).astype(np.float64)


def feature_digest(features: Iterable[float]) -> str:
    """Stable digest used in chronological logs instead of raw private payloads."""
    rounded = ",".join(f"{float(v):.4f}" for v in features)
    return hashlib.sha256(rounded.encode("ascii")).hexdigest()
