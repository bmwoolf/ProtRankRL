"""
Simplified Protein Environment for RL-based Protein Target Prioritization.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ProteinEnv(gym.Env):
    """
    Simplified Gym environment for protein target prioritization.
    """

    def __init__(
        self,
        feats: np.ndarray,
        targets: np.ndarray,
        normalize_features: bool = True
    ):
        # Basic validation
        if len(feats) != len(targets):
            raise ValueError("feats and targets must have same length")

        self.feats = feats.astype(np.float32)
        self.targets = targets.astype(np.int32)
        self.num_proteins = len(feats)
        self.feature_dim = feats.shape[1]

        # Normalize features if requested
        if normalize_features:
            self.feats = self._normalize_features(self.feats)

        # Initialize episode state
        self.current_idx = 0

        # Define spaces
        self.action_space = spaces.Discrete(self.num_proteins)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32
        )

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        """Normalize features to [-1, 1] range."""
        feats_min = np.min(feats, axis=0, keepdims=True)
        feats_max = np.max(feats, axis=0, keepdims=True)
        feats_range = feats_max - feats_min
        feats_range[feats_range == 0] = 1.0
        normalized = 2.0 * (feats - feats_min) / feats_range - 1.0
        return normalized.astype(np.float32)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.current_idx = 0
        return self.feats[self.current_idx], {"protein_idx": 0}

    def step(self, action: int) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:
        # Calculate reward
        reward = float(self.targets[action])

        # Move to next protein
        self.current_idx += 1
        done = self.current_idx >= self.num_proteins

        # Next observation
        next_obs = None if done else self.feats[self.current_idx]

        info = {
            "protein_idx": self.current_idx - 1,
            "was_hit": bool(self.targets[action]),
            "remaining": max(0, self.num_proteins - self.current_idx)
        }

        return next_obs, reward, done, False, info

    def render(self):
        pass

    def close(self):
        pass


def create_synthetic_env(
    num_proteins: int = 64,
    feature_dim: int = 128,
    hit_rate: float = 0.2,
    seed: int | None = None
) -> ProteinEnv:
    """Create environment with synthetic data."""
    if seed is not None:
        np.random.seed(seed)

    # Generate synthetic features
    feats = np.random.randn(num_proteins, feature_dim).astype(np.float32)

    # Generate synthetic targets
    num_hits = int(num_proteins * hit_rate)
    targets = np.zeros(num_proteins, dtype=np.int32)
    targets[:num_hits] = 1
    np.random.shuffle(targets)

    return ProteinEnv(feats, targets)
