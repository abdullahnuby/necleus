"""Signed prediction-residual encoding for the exploration agent.

The local predictor produces a feature-level residual::

    residual = actual_next_observation - predicted_next_observation

The residual is not collapsed into a scalar surprise or reward.  Positive and
negative deviations are exposed as two non-negative channels so their direction is
preserved.  The transformation below is a fixed numerical interface, not a tuned
policy:

* validate the 128-feature shape and finite values;
* clip each feature to ``[-1, 1]`` (observations and predictions are both in
  ``[0, 1]``);
* split into ``positive = max(residual, 0)`` and
  ``negative = max(-residual, 0)``.

The agent projects these channels through separate fixed weight matrices.  No
reward, score, target action, correctness label, or city outcome is involved.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .games import FEATURE_DIM


@dataclass(frozen=True)
class ResidualConfig:
    """Fixed encoding parameters for the signed residual channels."""

    input_dim: int = FEATURE_DIM
    clip: float = 1.0

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("residual input_dim must be positive")
        if not np.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("residual clip must be a positive finite value")


def split_residual(
    residual: Iterable[float],
    config: ResidualConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the non-negative positive and negative residual channels.

    The returned arrays have shape ``(input_dim,)``.  Their difference is the
    clipped signed residual, so directional information is preserved exactly at
    the encoding boundary.
    """
    cfg = config or ResidualConfig()
    vector = np.asarray(tuple(residual), dtype=np.float64)
    if vector.shape != (cfg.input_dim,):
        raise ValueError(f"expected {cfg.input_dim} residual features, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("residual contains non-finite values")
    clipped = np.clip(vector, -cfg.clip, cfg.clip)
    positive = np.clip(clipped, 0.0, cfg.clip)
    negative = np.clip(-clipped, 0.0, cfg.clip)
    return positive, negative


def residual_summary(residual: Iterable[float] | None, config: ResidualConfig | None = None) -> dict[str, float]:
    """Return an interpretable, JSON-safe summary of a signed residual."""
    cfg = config or ResidualConfig()
    if residual is None:
        return {
            "abs_mean": 0.0,
            "signed_mean": 0.0,
            "positive_mean": 0.0,
            "negative_mean": 0.0,
            "max_abs": 0.0,
        }
    positive, negative = split_residual(residual, cfg)
    signed = positive - negative
    return {
        "abs_mean": float(np.abs(signed).mean()),
        "signed_mean": float(signed.mean()),
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
        "max_abs": float(np.abs(signed).max()),
    }
