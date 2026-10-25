"""Nucleus autonomous exploration lab.

A local, reward-free perception-action loop over three small games.  This package
is intentionally separate from the benchmark and experiment code; it makes no
claim about learning, intelligence, or game-playing ability.
"""
from .agent import ActionDecision, AgentConfig, SpikingExplorer
from .games import GAME_REGISTRY, GridWorld, Observation, PongLite, Snake, StepResult, make_game

__all__ = [
    "ActionDecision",
    "AgentConfig",
    "GAME_REGISTRY",
    "GridWorld",
    "Observation",
    "PongLite",
    "Snake",
    "SpikingExplorer",
    "StepResult",
    "make_game",
]
