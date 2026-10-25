"""Persistent, step-based spiking agent with reward-free local STDP.

The agent keeps its own LIF state between environment steps.  It is intentionally
separate from ``pulse_core.PulseNetwork.run()``: that method resets all neuron
state at the start of every call and is therefore unsuitable for a continuous
perception-action loop.  This module reuses the ``pulse_core.model.LIFNeuron``
dataclass only for its documented membrane/synaptic time constants and decay
math; the stepping loop is implemented here.

Learning is pair-based spike-timing-dependent plasticity (STDP) on two groups of
synapses:

* observation input -> hidden neurons
* hidden neurons -> action readout neurons

The update uses only the agent's own pre/post spike times.  It never receives a
reward, score, done flag, target action, label, or game outcome.  Weights are
clipped to ``[0, weight_max]``, decayed slightly, and periodically column-scaled
to a bounded total.  Those are numerical stability constraints, not a policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any, Iterable, Sequence

import numpy as np

from pulse_core.model import LIFNeuron

from .encoding import DEFAULT_SPIKE_THRESHOLD, spikes_from_features
from .games import FEATURE_DIM

CHECKPOINT_SCHEMA = "nucleus-exploration-agent-v1"


@dataclass(frozen=True)
class AgentConfig:
    input_dim: int = FEATURE_DIM
    hidden_dim: int = 64
    action_dim: int = 4
    ticks_per_step: int = 4
    hidden_threshold: float = 1.8
    action_threshold: float = 0.5
    hidden_refractory: int = 1
    action_refractory: int = 1
    hidden_bias_current: float = 0.15
    action_bias_current: float = 0.02
    tau_mem: float = 2.0
    tau_syn: float = 6.0
    lr_ltp: float = 0.0020
    lr_ltd: float = 0.0021
    trace_tau: float = 8.0
    weight_max: float = 1.0
    weight_decay: float = 1e-5
    max_column_sum: float = 8.0
    homeostasis_every: int = 50
    temperature: float = 0.6
    min_activity_signal: float = 0.05
    spike_threshold: float = DEFAULT_SPIKE_THRESHOLD


@dataclass(frozen=True)
class ActionDecision:
    action: int
    scores: tuple[float, ...]
    probabilities: tuple[float, ...]
    random_fallback: bool
    selection_reason: str
    hidden_spikes: int
    action_spikes: tuple[int, ...]


def _rng_to_json(rng: random.Random) -> dict[str, Any]:
    version, state, gauss_next = rng.getstate()
    return {"version": version, "state": list(state), "gauss_next": gauss_next}


def _rng_from_json(data: dict[str, Any]) -> random.Random:
    rng = random.Random()
    rng.setstate((int(data["version"]), tuple(int(x) for x in data["state"]), data["gauss_next"]))
    return rng


class SpikingExplorer:
    """A small persistent spiking network driven by interaction history."""

    def __init__(self, config: AgentConfig | None = None, seed: int | None = None) -> None:
        self.config = config or AgentConfig()
        self._validate_config()
        self._rng = random.Random(seed)
        self.steps = 0
        self.ticks = 0
        self.total_weight_updates = 0

        cfg = self.config
        self.w_input_hidden = self._random_matrix(cfg.input_dim, cfg.hidden_dim, 0.0, 0.15)
        self.w_hidden_hidden = self._random_recurrent(cfg.hidden_dim)
        self.w_hidden_action = self._random_matrix(cfg.hidden_dim, cfg.action_dim, 0.0, 0.12)

        self.v_hidden = np.zeros(cfg.hidden_dim, dtype=np.float64)
        self.v_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self.i_hidden = np.zeros(cfg.hidden_dim, dtype=np.float64)
        self.i_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self.refractory_hidden = np.zeros(cfg.hidden_dim, dtype=np.int64)
        self.refractory_action = np.zeros(cfg.action_dim, dtype=np.int64)
        self.trace_input = np.zeros(cfg.input_dim, dtype=np.float64)
        self.trace_hidden = np.zeros(cfg.hidden_dim, dtype=np.float64)
        self.trace_action = np.zeros(cfg.action_dim, dtype=np.float64)
        self._previous_hidden_spikes = np.zeros(cfg.hidden_dim, dtype=np.float64)
        self.last_decision: ActionDecision | None = None

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.input_dim != FEATURE_DIM:
            raise ValueError(f"input_dim must equal the game feature size {FEATURE_DIM}")
        if cfg.hidden_dim <= 0 or cfg.action_dim <= 0:
            raise ValueError("dimensions must be positive")
        if cfg.action_dim < 4:
            raise ValueError("action_dim must be at least 4 for the unified action space")
        if cfg.ticks_per_step <= 0:
            raise ValueError("ticks_per_step must be positive")
        if cfg.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 < cfg.spike_threshold <= 1.0:
            raise ValueError("spike_threshold must be in (0, 1]")
        if cfg.weight_max <= 0 or cfg.max_column_sum <= 0:
            raise ValueError("weight bounds must be positive")

    def _random_matrix(self, rows: int, cols: int, low: float, high: float) -> np.ndarray:
        return np.asarray(
            [[self._rng.uniform(low, high) for _ in range(cols)] for _ in range(rows)],
            dtype=np.float64,
        )

    def _random_recurrent(self, size: int) -> np.ndarray:
        matrix = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                if i != j and self._rng.random() < 0.20:
                    matrix[i, j] = self._rng.uniform(-0.15, 0.15)
        return matrix

    # ------------------------------------------------------------------
    # Dynamics and plasticity
    # ------------------------------------------------------------------

    def act(self, features: Iterable[float], legal_actions: Sequence[int]) -> ActionDecision:
        """Run internal ticks, apply local STDP, and select one legal action.

        ``features`` is the current observation only.  No outcome information is
        accepted by this method.
        """
        legal = tuple(int(a) for a in legal_actions)
        if not legal:
            raise ValueError("legal_actions cannot be empty")
        if any(a < 0 or a >= self.config.action_dim for a in legal):
            raise ValueError("legal action outside action_dim")

        input_spikes = spikes_from_features(features, self.config.spike_threshold)
        spike_counts, max_voltages, hidden_count = self._run_ticks(input_spikes)
        scores = spike_counts + max_voltages

        masked = np.asarray([scores[a] for a in legal], dtype=np.float64)
        probabilities = np.zeros(self.config.action_dim, dtype=np.float64)
        if float(masked.max()) < self.config.min_activity_signal:
            probabilities[list(legal)] = 1.0 / len(legal)
            fallback = True
            reason = "no_activity_signal_uniform_random"
        else:
            shifted = (masked - masked.max()) / self.config.temperature
            weights = np.exp(shifted)
            weights /= weights.sum()
            probabilities[list(legal)] = weights
            fallback = False
            reason = "softmax_over_action_activity"

        draw = self._rng.random()
        cumulative = 0.0
        chosen = legal[-1]
        for action in legal:
            cumulative += float(probabilities[action])
            if draw <= cumulative:
                chosen = action
                break

        decision = ActionDecision(
            action=chosen,
            scores=tuple(float(x) for x in scores),
            probabilities=tuple(float(x) for x in probabilities),
            random_fallback=fallback,
            selection_reason=reason,
            hidden_spikes=int(hidden_count),
            action_spikes=tuple(int(x) for x in spike_counts),
        )
        self.last_decision = decision
        return decision

    def _run_ticks(self, input_spikes: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        cfg = self.config
        trace_decay = math.exp(-1.0 / cfg.trace_tau)

        # Reuse the documented LIF time constants from the core model.
        hidden_neuron = LIFNeuron("hidden", tau_mem=cfg.tau_mem, tau_syn=cfg.tau_syn,
                                  threshold=cfg.hidden_threshold,
                                  refractory_steps=cfg.hidden_refractory)
        action_neuron = LIFNeuron("action", tau_mem=cfg.tau_mem, tau_syn=cfg.tau_syn,
                                  threshold=cfg.action_threshold,
                                  refractory_steps=cfg.action_refractory)
        hidden_decay_m = hidden_neuron.membrane_decay
        hidden_decay_s = hidden_neuron.synaptic_decay
        action_decay_m = action_neuron.membrane_decay
        action_decay_s = action_neuron.synaptic_decay

        spike_counts = np.zeros(cfg.action_dim, dtype=np.float64)
        max_voltages = np.zeros(cfg.action_dim, dtype=np.float64)
        hidden_count = 0
        previous_hidden = self._previous_hidden_spikes

        for tick in range(cfg.ticks_per_step):
            tick_input = input_spikes if tick == 0 else np.zeros_like(input_spikes)
            self.i_hidden *= hidden_decay_s
            self.i_action *= action_decay_s
            # A small constant bias keeps the network excitable after local
            # plasticity has weakened its input weights.  It is not a reward,
            # target, or game-specific signal.
            self.i_hidden += cfg.hidden_bias_current
            self.i_action += cfg.action_bias_current
            if tick == 0:
                self.i_hidden += self.w_input_hidden.T @ tick_input
            self.i_hidden += self.w_hidden_hidden.T @ previous_hidden
            self.i_action += self.w_hidden_action.T @ previous_hidden

            self.trace_input *= trace_decay
            self.trace_hidden *= trace_decay
            self.trace_action *= trace_decay

            hidden_spikes, self.v_hidden, self.refractory_hidden, _ = self._lif_update(
                self.v_hidden, self.i_hidden, self.refractory_hidden,
                hidden_decay_m, cfg.hidden_threshold, cfg.hidden_refractory,
            )
            action_spikes, self.v_action, self.refractory_action, action_peak = self._lif_update(
                self.v_action, self.i_action, self.refractory_action,
                action_decay_m, cfg.action_threshold, cfg.action_refractory,
            )
            spike_counts += action_spikes
            max_voltages = np.maximum(max_voltages, action_peak)
            hidden_count += int(hidden_spikes.sum())

            self._apply_stdp(tick_input, hidden_spikes, action_spikes)
            self.trace_input += tick_input
            self.trace_hidden += hidden_spikes
            self.trace_action += action_spikes
            previous_hidden = hidden_spikes
            self.ticks += 1

        self._previous_hidden_spikes = previous_hidden
        self.steps += 1
        self._apply_stability()
        return spike_counts, max_voltages, hidden_count

    def _lif_update(
        self,
        voltage: np.ndarray,
        current: np.ndarray,
        refractory: np.ndarray,
        membrane_decay: float,
        threshold: float,
        refractory_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        skipped = refractory > 0
        refractory = np.maximum(refractory - 1, 0)
        updated = voltage * membrane_decay + (1.0 - membrane_decay) * current
        active = ~skipped
        candidate = np.where(active, updated, voltage)
        spikes = active & (candidate >= threshold)
        new_voltage = np.where(spikes, 0.0, candidate)
        refractory = np.where(spikes, refractory_steps, refractory)
        return spikes.astype(np.float64), new_voltage, refractory, candidate

    def _apply_stdp(
        self,
        input_spikes: np.ndarray,
        hidden_spikes: np.ndarray,
        action_spikes: np.ndarray,
    ) -> None:
        cfg = self.config
        if not (input_spikes.any() or hidden_spikes.any() or action_spikes.any()):
            return
        # Causal pair: a presynaptic spike followed by a postsynaptic spike
        # potentiates the synapse.  Acausal order depresses it.  Traces hold
        # recent activity, so both terms are local to the two neurons involved.
        self.w_input_hidden -= cfg.lr_ltd * np.outer(input_spikes, self.trace_hidden)
        self.w_input_hidden += cfg.lr_ltp * np.outer(self.trace_input, hidden_spikes)
        self.w_hidden_action -= cfg.lr_ltd * np.outer(hidden_spikes, self.trace_action)
        self.w_hidden_action += cfg.lr_ltp * np.outer(self.trace_hidden, action_spikes)
        np.clip(self.w_input_hidden, 0.0, cfg.weight_max, out=self.w_input_hidden)
        np.clip(self.w_hidden_action, 0.0, cfg.weight_max, out=self.w_hidden_action)
        self.total_weight_updates += 1

    def _apply_stability(self) -> None:
        cfg = self.config
        self.w_input_hidden *= (1.0 - cfg.weight_decay)
        self.w_hidden_action *= (1.0 - cfg.weight_decay)
        np.clip(self.w_input_hidden, 0.0, cfg.weight_max, out=self.w_input_hidden)
        np.clip(self.w_hidden_action, 0.0, cfg.weight_max, out=self.w_hidden_action)
        if cfg.homeostasis_every > 0 and self.steps % cfg.homeostasis_every == 0:
            self._scale_columns(self.w_input_hidden)
            self._scale_columns(self.w_hidden_action)

    def _scale_columns(self, matrix: np.ndarray) -> None:
        sums = matrix.sum(axis=0)
        scale = np.minimum(1.0, self.config.max_column_sum / np.maximum(sums, 1e-12))
        matrix *= scale[np.newaxis, :]

    def reset_transient(self) -> None:
        """Clear activity between episodes while preserving weights and counters."""
        cfg = self.config
        self.v_hidden.fill(0.0)
        self.v_action.fill(0.0)
        self.i_hidden.fill(0.0)
        self.i_action.fill(0.0)
        self.refractory_hidden.fill(0)
        self.refractory_action.fill(0)
        self.trace_input.fill(0.0)
        self.trace_hidden.fill(0.0)
        self.trace_action.fill(0.0)
        self._previous_hidden_spikes.fill(0.0)

    def weight_stats(self) -> dict[str, float]:
        def stats(matrix: np.ndarray) -> dict[str, float]:
            return {
                "mean": float(matrix.mean()),
                "std": float(matrix.std()),
                "min": float(matrix.min()),
                "max": float(matrix.max()),
            }
        return {
            "input_hidden": stats(self.w_input_hidden),
            "hidden_action": stats(self.w_hidden_action),
            "hidden_hidden_abs_mean": float(np.abs(self.w_hidden_hidden).mean()),
            "total_weight_updates": float(self.total_weight_updates),
        }

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": CHECKPOINT_SCHEMA,
            "config": asdict(self.config),
            "rng": _rng_to_json(self._rng),
            "steps": self.steps,
            "ticks": self.ticks,
            "total_weight_updates": self.total_weight_updates,
            "w_input_hidden": self.w_input_hidden.tolist(),
            "w_hidden_hidden": self.w_hidden_hidden.tolist(),
            "w_hidden_action": self.w_hidden_action.tolist(),
            "v_hidden": self.v_hidden.tolist(),
            "v_action": self.v_action.tolist(),
            "i_hidden": self.i_hidden.tolist(),
            "i_action": self.i_action.tolist(),
            "refractory_hidden": self.refractory_hidden.tolist(),
            "refractory_action": self.refractory_action.tolist(),
            "trace_input": self.trace_input.tolist(),
            "trace_hidden": self.trace_hidden.tolist(),
            "trace_action": self.trace_action.tolist(),
            "previous_hidden_spikes": self._previous_hidden_spikes.tolist(),
        }

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "SpikingExplorer":
        if data.get("schema") != CHECKPOINT_SCHEMA:
            raise ValueError("unsupported agent checkpoint schema")
        config = AgentConfig(**data["config"])
        agent = cls(config, seed=0)
        cfg = agent.config

        def matrix(key: str, shape: tuple[int, int], low: float | None, high: float | None) -> np.ndarray:
            value = np.asarray(data[key], dtype=np.float64)
            if value.shape != shape:
                raise ValueError(f"{key} has shape {value.shape}, expected {shape}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{key} contains non-finite values")
            if low is not None and high is not None and (value.min() < low - 1e-9 or value.max() > high + 1e-9):
                raise ValueError(f"{key} is outside [{low}, {high}]")
            return value

        agent.w_input_hidden = matrix("w_input_hidden", (cfg.input_dim, cfg.hidden_dim), 0.0, cfg.weight_max)
        agent.w_hidden_hidden = matrix("w_hidden_hidden", (cfg.hidden_dim, cfg.hidden_dim), None, None)
        agent.w_hidden_action = matrix("w_hidden_action", (cfg.hidden_dim, cfg.action_dim), 0.0, cfg.weight_max)

        def vector(key: str, size: int) -> np.ndarray:
            value = np.asarray(data[key], dtype=np.float64)
            if value.shape != (size,):
                raise ValueError(f"{key} has shape {value.shape}, expected ({size},)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{key} contains non-finite values")
            return value

        agent.v_hidden = vector("v_hidden", cfg.hidden_dim)
        agent.v_action = vector("v_action", cfg.action_dim)
        agent.i_hidden = vector("i_hidden", cfg.hidden_dim)
        agent.i_action = vector("i_action", cfg.action_dim)

        def refractory(key: str, size: int) -> np.ndarray:
            value = np.asarray(data[key], dtype=np.int64)
            if value.shape != (size,):
                raise ValueError(f"{key} has shape {value.shape}, expected ({size},)")
            if np.any(value < 0):
                raise ValueError(f"{key} contains negative refractory counters")
            return value

        agent.refractory_hidden = refractory("refractory_hidden", cfg.hidden_dim)
        agent.refractory_action = refractory("refractory_action", cfg.action_dim)
        agent.trace_input = vector("trace_input", cfg.input_dim)
        agent.trace_hidden = vector("trace_hidden", cfg.hidden_dim)
        agent.trace_action = vector("trace_action", cfg.action_dim)
        agent._previous_hidden_spikes = vector("previous_hidden_spikes", cfg.hidden_dim)
        agent._rng = _rng_from_json(data["rng"])
        agent.steps = int(data["steps"])
        agent.ticks = int(data["ticks"])
        agent.total_weight_updates = int(data["total_weight_updates"])
        return agent
