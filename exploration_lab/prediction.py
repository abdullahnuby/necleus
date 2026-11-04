"""Local, self-supervised prediction head for the exploration city.

The head predicts the next 128-feature observation from three local signals:

* the clone's current hidden activity,
* the chosen action as a one-hot vector,
* the current observation.

It is trained with a local delta rule (LMS): each prediction neuron updates its
incoming weights using its own error and the presynaptic activity.  There is no
backpropagation, no reward, no score, no target action, and no city outcome.  The
prediction error is never used as a reward or to update the action STDP pathway.
TASK-014 optionally exposes the signed feature-level residual to the same clone's
separate residual-input pathway on the next perception-action cycle; the predictor
itself still learns only from its own next-observation error.

This is deliberately a small, inspectable model.  It is not a claim that the
clone learns useful behavior or a world model in the reinforcement-learning
sense.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any, Sequence

import numpy as np

from .games import FEATURE_DIM

PREDICTION_SCHEMA = "nucleus-exploration-prediction-v1"


@dataclass(frozen=True)
class PredictionConfig:
    hidden_dim: int = 64
    action_dim: int = 4
    input_dim: int = FEATURE_DIM
    learning_rate: float = 0.001
    weight_max: float = 2.0
    weight_decay: float = 1e-6
    init_scale: float = 0.01

    @property
    def predictor_dim(self) -> int:
        return self.hidden_dim + self.action_dim + self.input_dim


@dataclass(frozen=True)
class PredictionMetrics:
    mae: float
    mse: float
    prediction_updates: int


class LocalPredictor:
    """A per-clone linear forward model trained by local prediction error."""

    def __init__(self, config: PredictionConfig | None = None, seed: int = 0) -> None:
        self.config = config or PredictionConfig()
        self._validate_config()
        cfg = self.config
        rng = random.Random(seed)
        self.weights = np.asarray(
            [
                [rng.uniform(-cfg.init_scale, cfg.init_scale) for _ in range(cfg.input_dim)]
                for _ in range(cfg.predictor_dim)
            ],
            dtype=np.float64,
        )
        self.bias = np.zeros(cfg.input_dim, dtype=np.float64)
        self.steps = 0
        self.total_updates = 0
        self._last_input: np.ndarray | None = None
        self._last_prediction: np.ndarray | None = None
        self.last_metrics: PredictionMetrics | None = None
        # Signed feature-level residual from the most recent observe_next().
        # It is persisted for diagnostics/continuation and is never a reward.
        self.last_residual: np.ndarray | None = None

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.input_dim != FEATURE_DIM:
            raise ValueError(f"input_dim must equal the city feature size {FEATURE_DIM}")
        if cfg.hidden_dim <= 0 or cfg.action_dim <= 0:
            raise ValueError("hidden_dim and action_dim must be positive")
        if not np.isfinite(cfg.learning_rate) or cfg.learning_rate < 0.0:
            raise ValueError("learning_rate must be non-negative and finite")
        if not np.isfinite(cfg.weight_max) or cfg.weight_max <= 0.0:
            raise ValueError("weight_max must be positive and finite")
        if not np.isfinite(cfg.weight_decay) or cfg.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative and finite")
        if not np.isfinite(cfg.init_scale) or cfg.init_scale < 0.0:
            raise ValueError("init_scale must be non-negative and finite")

    def _build_input(
        self,
        features: Sequence[float],
        hidden_activity: Sequence[float],
        action: int,
    ) -> np.ndarray:
        cfg = self.config
        feature_vector = np.asarray(tuple(features), dtype=np.float64)
        hidden_vector = np.asarray(tuple(hidden_activity), dtype=np.float64)
        if feature_vector.shape != (cfg.input_dim,):
            raise ValueError(f"expected {cfg.input_dim} features, got {feature_vector.shape}")
        if hidden_vector.shape != (cfg.hidden_dim,):
            raise ValueError(f"expected {cfg.hidden_dim} hidden activities, got {hidden_vector.shape}")
        if not np.all(np.isfinite(feature_vector)) or not np.all(np.isfinite(hidden_vector)):
            raise ValueError("prediction inputs contain non-finite values")
        if action < 0 or action >= cfg.action_dim:
            raise ValueError("action outside prediction action_dim")
        action_vector = np.zeros(cfg.action_dim, dtype=np.float64)
        action_vector[action] = 1.0
        return np.concatenate((hidden_vector, action_vector, feature_vector))

    def predict(
        self,
        features: Sequence[float],
        hidden_activity: Sequence[float],
        action: int,
    ) -> np.ndarray:
        """Return a clipped prediction of the next observation."""
        pre = self._build_input(features, hidden_activity, action)
        raw = pre @ self.weights + self.bias
        prediction = np.clip(raw, 0.0, 1.0)
        self._last_input = pre
        self._last_prediction = prediction
        return prediction

    def observe_next(self, next_features: Sequence[float]) -> PredictionMetrics:
        """Compute local error, update only this predictor, and return metrics."""
        if self._last_input is None or self._last_prediction is None:
            raise RuntimeError("predict() must be called before observe_next()")
        target = np.asarray(tuple(next_features), dtype=np.float64)
        if target.shape != (self.config.input_dim,):
            raise ValueError(f"expected {self.config.input_dim} next features, got {target.shape}")
        if not np.all(np.isfinite(target)):
            raise ValueError("next observation contains non-finite values")
        error = target - self._last_prediction
        self.last_residual = error.copy()
        mae = float(np.abs(error).mean())
        mse = float(np.mean(error * error))
        if self.config.learning_rate > 0.0:
            self.weights += self.config.learning_rate * np.outer(self._last_input, error)
            self.bias += self.config.learning_rate * error
            self.weights *= (1.0 - self.config.weight_decay)
            np.clip(self.weights, -self.config.weight_max, self.config.weight_max, out=self.weights)
            np.clip(self.bias, -1.0, 1.0, out=self.bias)
            self.total_updates += 1
        self.steps += 1
        metrics = PredictionMetrics(
            mae=mae,
            mse=mse,
            prediction_updates=self.total_updates,
        )
        self.last_metrics = metrics
        return metrics

    def weight_stats(self) -> dict[str, float]:
        return {
            "mean": float(self.weights.mean()),
            "std": float(self.weights.std()),
            "min": float(self.weights.min()),
            "max": float(self.weights.max()),
            "bias_abs_mean": float(np.abs(self.bias).mean()),
            "total_updates": float(self.total_updates),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": PREDICTION_SCHEMA,
            "config": asdict(self.config),
            "weights": self.weights.tolist(),
            "bias": self.bias.tolist(),
            "steps": self.steps,
            "total_updates": self.total_updates,
            "last_residual": (
                self.last_residual.tolist() if self.last_residual is not None else None
            ),
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "LocalPredictor":
        if not isinstance(data, dict):
            raise ValueError("prediction checkpoint must be a JSON object")
        if data.get("schema") != PREDICTION_SCHEMA:
            raise ValueError("unsupported prediction checkpoint schema")
        try:
            config = PredictionConfig(**data["config"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction config: {exc}") from exc
        predictor = cls(config, seed=0)
        try:
            weights = np.asarray(data["weights"], dtype=np.float64)
            bias = np.asarray(data["bias"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction tensors: {exc}") from exc
        if weights.shape != (config.predictor_dim, config.input_dim):
            raise ValueError(
                f"prediction weights have shape {weights.shape}, expected "
                f"({config.predictor_dim}, {config.input_dim})"
            )
        if bias.shape != (config.input_dim,):
            raise ValueError(
                f"prediction bias has shape {bias.shape}, expected ({config.input_dim},)"
            )
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("prediction checkpoint contains non-finite values")
        tolerance = 1e-9
        if np.any(weights < -config.weight_max - tolerance) or np.any(
            weights > config.weight_max + tolerance
        ):
            raise ValueError(
                f"prediction weights are outside [{-config.weight_max}, {config.weight_max}]"
            )
        if np.any(bias < -1.0 - tolerance) or np.any(bias > 1.0 + tolerance):
            raise ValueError("prediction bias is outside [-1, 1]")

        try:
            steps = int(data["steps"])
            total_updates = int(data["total_updates"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid prediction counters: {exc}") from exc
        if steps < 0 or total_updates < 0:
            raise ValueError("prediction counters must be non-negative")
        if total_updates > steps:
            raise ValueError("prediction total_updates cannot exceed steps")

        last_residual = data.get("last_residual")
        if last_residual is not None:
            try:
                residual = np.asarray(last_residual, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"last_residual is not numeric: {exc}") from exc
            if residual.shape != (config.input_dim,):
                raise ValueError(
                    f"last_residual has shape {residual.shape}, expected ({config.input_dim},)"
                )
            if not np.all(np.isfinite(residual)):
                raise ValueError("last_residual contains non-finite values")
            if np.any(residual < -1.0 - tolerance) or np.any(residual > 1.0 + tolerance):
                raise ValueError("last_residual is outside [-1, 1]")
            predictor.last_residual = np.clip(residual, -1.0, 1.0)
        else:
            predictor.last_residual = None

        predictor.weights = weights
        predictor.bias = bias
        predictor.steps = steps
        predictor.total_updates = total_updates
        return predictor
