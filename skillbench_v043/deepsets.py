"""Small NumPy DeepSets meta-learner for the v0.43 development pipeline.

Architecture (frozen for the development stage):

* ``phi``: support-example encoder, 3 -> 128 -> 128, ReLU.
* mean pooling over the eight support examples.
* ``rho``: query-conditioned head, 130 -> 128 -> 128 -> 1, ReLU.

The model receives only support inputs, support labels, and query inputs. It
never receives target programs, query labels, task IDs, family IDs, or evaluator
metadata. PyTorch is not available in the local environment, so this module
implements the same architecture with explicit NumPy forward/backward passes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .corpus import TrainingEpisode


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def _bce_with_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(np.maximum(logits, 0.0) - logits * labels
                         + np.log1p(np.exp(-np.abs(logits)))))


def _normalize_x(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64) / 99.0


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key in params:
            g = grads[key]
            self.m[key] = self.beta1 * self.m[key] + (1.0 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1.0 - self.beta2) * (g * g)
            m_hat = self.m[key] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1.0 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class DeepSetsConfig:
    hidden: int = 128
    seed: int = 430043
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 10
    l2: float = 1e-4
    gradient_clip: float = 5.0


class DeepSetsNumpy:
    def __init__(self, config: DeepSetsConfig | None = None):
        self.config = config or DeepSetsConfig()
        rng = np.random.default_rng(self.config.seed)
        h = self.config.hidden
        scale1 = np.sqrt(2.0 / 3.0)
        scale2 = np.sqrt(2.0 / h)
        self.params: dict[str, np.ndarray] = {
            "W1": rng.normal(0.0, scale1, (3, h)),
            "b1": np.zeros(h),
            "W2": rng.normal(0.0, scale2, (h, h)),
            "b2": np.zeros(h),
            "W3": rng.normal(0.0, np.sqrt(2.0 / (h + 2)), (h + 2, h)),
            "b3": np.zeros(h),
            "W4": rng.normal(0.0, scale2, (h, h)),
            "b4": np.zeros(h),
            "w5": rng.normal(0.0, np.sqrt(2.0 / h), (h,)),
            "b5": np.zeros(1),
        }
        self.history: list[dict[str, float]] = []

    def _forward(self, support_x: np.ndarray, support_y: np.ndarray,
                 query_x: np.ndarray) -> np.ndarray:
        p = self.params
        sx = _normalize_x(support_x)
        qx = _normalize_x(query_x)
        phi_in = np.concatenate([sx, support_y[..., None]], axis=-1)
        phi1_pre = phi_in @ p["W1"] + p["b1"]
        phi1 = _relu(phi1_pre)
        phi2_pre = phi1 @ p["W2"] + p["b2"]
        phi2 = _relu(phi2_pre)
        context = phi2.mean(axis=1)
        expanded = np.broadcast_to(context[:, None, :], (qx.shape[0], qx.shape[1], p["W1"].shape[1]))
        rho_in = np.concatenate([expanded, qx], axis=-1)
        rho1_pre = rho_in @ p["W3"] + p["b3"]
        rho1 = _relu(rho1_pre)
        rho2_pre = rho1 @ p["W4"] + p["b4"]
        rho2 = _relu(rho2_pre)
        logits = rho2 @ p["w5"] + p["b5"][0]
        self._cache = {
            "phi_in": phi_in, "phi1_pre": phi1_pre, "phi1": phi1,
            "phi2_pre": phi2_pre, "phi2": phi2, "context": context,
            "rho_in": rho_in, "rho1_pre": rho1_pre, "rho1": rho1,
            "rho2_pre": rho2_pre, "rho2": rho2, "logits": logits,
        }
        return logits

    def _backward(self, labels: np.ndarray) -> dict[str, np.ndarray]:
        p = self.params
        c = self._cache
        logits = c["logits"]
        b, q = logits.shape
        dz = (_sigmoid(logits) - labels) / (b * q)
        grads: dict[str, np.ndarray] = {}
        grads["w5"] = c["rho2"].reshape(-1, self.config.hidden).T @ dz.reshape(-1, 1)
        grads["w5"] = grads["w5"][:, 0]
        grads["b5"] = np.array([dz.sum()])
        drho2 = dz[:, :, None] * p["w5"][None, None, :]
        drho2_pre = drho2 * (c["rho2_pre"] > 0.0)
        grads["W4"] = c["rho1"].reshape(-1, self.config.hidden).T @ drho2_pre.reshape(-1, self.config.hidden)
        grads["b4"] = drho2_pre.sum(axis=(0, 1))
        drho1 = drho2_pre @ p["W4"].T
        drho1_pre = drho1 * (c["rho1_pre"] > 0.0)
        grads["W3"] = c["rho_in"].reshape(-1, self.config.hidden + 2).T @ drho1_pre.reshape(-1, self.config.hidden)
        grads["b3"] = drho1_pre.sum(axis=(0, 1))
        drho_in = drho1_pre @ p["W3"].T
        dcontext = drho_in[:, :, :self.config.hidden].sum(axis=1)
        dphi2 = np.broadcast_to(dcontext[:, None, :] / 8.0,
                                (b, 8, self.config.hidden))
        dphi2_pre = dphi2 * (c["phi2_pre"] > 0.0)
        grads["W2"] = c["phi1"].reshape(-1, self.config.hidden).T @ dphi2_pre.reshape(-1, self.config.hidden)
        grads["b2"] = dphi2_pre.sum(axis=(0, 1))
        dphi1 = dphi2_pre @ p["W2"].T
        dphi1_pre = dphi1 * (c["phi1_pre"] > 0.0)
        grads["W1"] = c["phi_in"].reshape(-1, 3).T @ dphi1_pre.reshape(-1, self.config.hidden)
        grads["b1"] = dphi1_pre.sum(axis=(0, 1))
        if self.config.l2:
            for key in ("W1", "W2", "W3", "W4"):
                grads[key] = grads[key] + self.config.l2 * p[key]
        return grads

    def predict_proba(self, support_x: np.ndarray, support_y: np.ndarray,
                      query_x: np.ndarray) -> np.ndarray:
        return _sigmoid(self._forward(support_x, support_y, query_x))

    def predict(self, support_x: np.ndarray, support_y: np.ndarray,
                query_x: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(support_x, support_y, query_x) >= threshold).astype(np.int64)

    def _clip_grads(self, grads: dict[str, np.ndarray]) -> None:
        norm = np.sqrt(sum(float(np.sum(g * g)) for g in grads.values()))
        if norm > self.config.gradient_clip:
            scale = self.config.gradient_clip / (norm + 1e-12)
            for key in grads:
                grads[key] *= scale

    def fit(self, episodes: Sequence[TrainingEpisode], *,
            validation: Sequence[TrainingEpisode] | None = None) -> list[dict[str, float]]:
        if not episodes:
            raise ValueError("training episodes are empty")
        support_x, support_y, query_x, query_y = stack_episodes(episodes)
        rng = np.random.default_rng(self.config.seed + 1)
        optimizer = Adam(self.params, lr=self.config.learning_rate)
        n = len(episodes)
        self.history = []
        for epoch in range(self.config.epochs):
            order = rng.permutation(n)
            total_loss = 0.0
            batches = 0
            for start in range(0, n, self.config.batch_size):
                idx = order[start:start + self.config.batch_size]
                logits = self._forward(support_x[idx], support_y[idx], query_x[idx])
                loss = _bce_with_logits(logits, query_y[idx])
                grads = self._backward(query_y[idx])
                self._clip_grads(grads)
                optimizer.step(self.params, grads)
                total_loss += loss
                batches += 1
            row = {"epoch": float(epoch + 1), "train_loss": total_loss / max(batches, 1)}
            if validation:
                vs, vy, vq, vql = stack_episodes(validation)
                vlogits = self._forward(vs, vy, vq)
                row["validation_loss"] = _bce_with_logits(vlogits, vql)
                row["validation_accuracy"] = float(np.mean((_sigmoid(vlogits) >= 0.5) == (vql == 1)))
            self.history.append(row)
        return self.history

    def save(self, path: str | Path, metadata: dict | None = None) -> str:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **self.params)
        meta = {
            "config": self.config.__dict__,
            "history": self.history,
            "architecture": "DeepSetsNumpy phi(3->128->128) mean rho(130->128->128->1)",
        }
        if metadata:
            meta.update(metadata)
        output.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return hashlib.sha256(output.read_bytes()).hexdigest()

    @classmethod
    def load(cls, path: str | Path, config: DeepSetsConfig | None = None) -> "DeepSetsNumpy":
        source = Path(path)
        model = cls(config)
        with np.load(source) as data:
            for key in model.params:
                model.params[key] = np.asarray(data[key], dtype=np.float64)
        meta_path = source.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model.history = meta.get("history", [])
        return model


def stack_episodes(episodes: Sequence[TrainingEpisode]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not episodes:
        raise ValueError("episodes are empty")
    support_x = np.asarray([[[ex.x[0], ex.x[1]] for ex in ep.support] for ep in episodes], dtype=np.float64)
    support_y = np.asarray([[ex.y for ex in ep.support] for ep in episodes], dtype=np.float64)
    query_x = np.asarray([[[x[0], x[1]] for x in ep.query_x] for ep in episodes], dtype=np.float64)
    query_y = np.asarray([list(ep.query_y) for ep in episodes], dtype=np.float64)
    return support_x, support_y, query_x, query_y
