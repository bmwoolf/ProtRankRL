"""
Protein Environment for Reinforcement Learning-based Protein Target Prioritization.

This module implements a Gym-compatible environment for training RL agents
to prioritize protein targets based on scientific embeddings and known outcomes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any


class ProteinEnv(gym.Env):
    """
    Gym-compatible environment for protein target prioritization.
    
    The environment models a triage decision over a fixed batch of proteins.
    Each episode processes a batch of proteins sequentially, with the agent
    making ranking decisions for each protein.
    
    Attributes:
        feats: Protein feature matrix of shape (N, D)
        targets: Binary labels of shape (N,) indicating known hits
        current_idx: Current protein index in the batch
        action_space: Discrete action space for protein ranking
        observation_space: Continuous observation space for protein features
    """
    
    def __init__(
        self,
        feats: np.ndarray,
        targets: np.ndarray,
        feature_dim: Optional[int] = None,
        normalize_features: bool = True
    ):
        """
        Initialize the Protein Environment.
        
        Args:
            feats: Protein feature matrix of shape (N, D)
            targets: Binary labels of shape (N,) where 1 = known hit, 0 = non-hit
            feature_dim: Expected feature dimension (inferred from feats if None)
            normalize_features: Whether to normalize features to [-1, 1] range
        """
        self._validate_inputs(feats, targets)
        
        self.feats = feats.astype(np.float32)
        self.targets = targets.astype(np.int32)
        self.num_proteins = len(feats)
        self.feature_dim = feature_dim or feats.shape[1]
        
        # Normalize features if requested
        if normalize_features:
            self.feats = self._normalize_features(self.feats)
        
        # Initialize episode state
        self.current_idx = 0
        self.episode_rewards = []
        self.episode_actions = []
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.num_proteins)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.feature_dim,),
            dtype=np.float32
        )
        
        # logger.info(f"ProteinEnv initialized with {self.num_proteins} proteins, "
        #             f"feature_dim={self.feature_dim}, normalize={self.normalize_features}")
    
    def _validate_inputs(self, feats: np.ndarray, targets: np.ndarray) -> None:
        """
        Validate input arrays for shape and type consistency.
        
        Args:
            feats: Protein feature matrix
            targets: Binary labels
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not isinstance(feats, np.ndarray) or not isinstance(targets, np.ndarray):
            raise ValueError("feats and targets must be numpy arrays")
        
        if feats.ndim != 2:
            raise ValueError(f"feats must be 2D array, got shape {feats.shape}")
        
        if targets.ndim != 1:
            raise ValueError(f"targets must be 1D array, got shape {targets.shape}")
        
        if len(feats) != len(targets):
            raise ValueError(f"feats and targets must have same length: "
                           f"{len(feats)} vs {len(targets)}")
        
        if len(feats) == 0:
            raise ValueError("feats cannot be empty")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(feats)) or np.any(np.isinf(feats)):
            raise ValueError("feats contains NaN or infinite values")
        
        # Validate targets are binary
        unique_targets = np.unique(targets)
        if not np.all(np.isin(unique_targets, [0, 1])):
            raise ValueError(f"targets must be binary (0 or 1), got {unique_targets}")
    
    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        """
        Normalize features to [-1, 1] range using min-max normalization.
        
        Args:
            feats: Input feature matrix
            
        Returns:
            Normalized feature matrix
        """
        feats_min = np.min(feats, axis=0, keepdims=True)
        feats_max = np.max(feats, axis=0, keepdims=True)
        
        # Avoid division by zero
        feats_range = feats_max - feats_min
        feats_range[feats_range == 0] = 1.0
        
        normalized = 2.0 * (feats - feats_min) / feats_range - 1.0
        return normalized.astype(np.float32)
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to start a new episode.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        super().reset(seed=seed)
        
        # Reset episode state
        self.current_idx = 0
        self.episode_rewards = []
        self.episode_actions = []
        
        # Return first protein's features
        initial_obs = self.feats[self.current_idx]
        info = {
            "protein_idx": self.current_idx,
            "total_proteins": self.num_proteins,
            "num_hits": np.sum(self.targets)
        }
        
        return initial_obs, info
    
    def step(
        self, action: int
    ) -> Tuple[Optional[np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Integer action in [0, N-1] representing protein ranking
            
        Returns:
            Tuple of (next_observation, reward, terminated, truncated, info)
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in [0, {self.num_proteins-1}]")
        
        # Calculate reward based on whether the selected protein is a hit
        reward = float(self.targets[action])
        
        # Store episode data
        self.episode_rewards.append(reward)
        self.episode_actions.append(action)
        
        # Move to next protein
        self.current_idx += 1
        
        # Check if episode is done
        done = self.current_idx >= self.num_proteins
        
        # Prepare next observation
        if done:
            next_obs = None
        else:
            next_obs = self.feats[self.current_idx]
        
        # Prepare info dict
        info = {
            "protein_idx": self.current_idx - 1,  # Current protein that was just processed
            "action_taken": action,
            "was_hit": bool(self.targets[action]),
            "episode_reward_sum": sum(self.episode_rewards),
            "episode_length": len(self.episode_rewards),
            "remaining_proteins": max(0, self.num_proteins - self.current_idx)
        }
        
        return next_obs, reward, done, False, info
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """
        Get statistics for the current episode.
        
        Returns:
            Dictionary containing episode statistics
        """
        if not self.episode_rewards:
            return {}
        
        return {
            "total_reward": sum(self.episode_rewards),
            "num_hits_found": sum(self.episode_rewards),
            "hit_rate": sum(self.episode_rewards) / len(self.episode_rewards),
            "episode_length": len(self.episode_rewards),
            "actions_taken": self.episode_actions.copy(),
            "rewards": self.episode_rewards.copy()
        }
    
    def render(self):
        """Render the environment (not implemented for this environment)."""
        pass
    
    def close(self):
        """Close the environment and clean up resources."""
        pass


class ProteinEnvFactory:
    """
    Factory class for creating ProteinEnv instances with different configurations.
    
    This class provides convenient methods for creating environments with
    synthetic data or loading from external sources.
    """
    
    @staticmethod
    def create_synthetic_env(
        num_proteins: int = 64,
        feature_dim: int = 128,
        hit_rate: float = 0.2,
        seed: Optional[int] = None
    ) -> ProteinEnv:
        """
        Create a ProteinEnv with synthetic data for testing.
        
        Args:
            num_proteins: Number of proteins in the batch
            feature_dim: Dimension of protein features
            hit_rate: Fraction of proteins that are hits (0.0 to 1.0)
            seed: Random seed for reproducibility
            
        Returns:
            Configured ProteinEnv instance
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Generate synthetic features (normal distribution)
        feats = np.random.randn(num_proteins, feature_dim).astype(np.float32)
        
        # Generate synthetic targets based on hit rate
        num_hits = int(num_proteins * hit_rate)
        targets = np.zeros(num_proteins, dtype=np.int32)
        targets[:num_hits] = 1
        np.random.shuffle(targets)
        
        return ProteinEnv(feats, targets, feature_dim)
    
    @staticmethod
    def create_from_data(
        feats: np.ndarray,
        targets: np.ndarray,
        feature_dim: Optional[int] = None,
        normalize_features: bool = True
    ) -> ProteinEnv:
        """
        Create a ProteinEnv from provided data.
        
        Args:
            feats: Protein feature matrix
            targets: Binary labels
            feature_dim: Expected feature dimension
            normalize_features: Whether to normalize features
            
        Returns:
            Configured ProteinEnv instance
        """
        return ProteinEnv(feats, targets, feature_dim, normalize_features) 