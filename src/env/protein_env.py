"""
Simplified Protein Environment for RL-based Protein Target Prioritization.
"""

from typing import Any, Optional, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data_loader import load_experimental_data


class ProteinEnv(gym.Env):
    """
    Simplified Gym environment for protein target prioritization.
    Supports both synthetic and real experimental data.
    """

    def __init__(
        self,
        feats: np.ndarray,
        targets: np.ndarray,
        normalize_features: bool = True,
        reward_type: str = "binary",
        experimental_data: Optional[Dict] = None
    ) -> None:
        # Basic validation
        if len(feats) != len(targets):
            raise ValueError("feats and targets must have same length")

        self.feats = feats.astype(np.float32)
        self.targets = targets.astype(np.int32)
        self.num_proteins = len(feats)
        self.feature_dim = feats.shape[1]
        self.reward_type = reward_type
        self.experimental_data = experimental_data

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
        return normalized.astype(np.float32)  # type: ignore

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on reward type and experimental data."""
        if self.reward_type == "binary":
            return float(self.targets[action])
        
        elif self.reward_type == "affinity_based" and self.experimental_data is not None:
            # Use binding affinity for reward calculation
            protein_idx = action
            if protein_idx < len(self.experimental_data):
                affinity = self.experimental_data[protein_idx].get('binding_affinity_kd')
                if affinity is not None and affinity > 0:
                    # Convert to nM and calculate reward
                    affinity_nm = affinity * 1e9
                    # Higher reward for lower affinity (stronger binding)
                    reward = max(0, 10 - np.log10(affinity_nm))
                    return reward
            return float(self.targets[action])
        
        elif self.reward_type == "multi_objective" and self.experimental_data is not None:
            # Multi-objective reward considering multiple experimental metrics
            protein_idx = action
            if protein_idx < len(self.experimental_data):
                data = self.experimental_data[protein_idx]
                
                # Base reward from hit status
                base_reward = float(self.targets[action])
                
                # Affinity reward
                affinity_reward = 0.0
                if data.get('binding_affinity_kd'):
                    affinity_nm = data['binding_affinity_kd'] * 1e9
                    affinity_reward = max(0, 5 - np.log10(affinity_nm))
                
                # Functional activity reward
                activity_reward = data.get('functional_activity', 0.0) * 5.0
                
                # Toxicity penalty (lower is better)
                toxicity_penalty = data.get('toxicity_score', 0.0) * -2.0
                
                # Expression reward
                expression_reward = data.get('expression_level', 1.0) * 2.0
                
                total_reward = base_reward + affinity_reward + activity_reward + toxicity_penalty + expression_reward
                return total_reward
            
            return float(self.targets[action])
        
        else:
            # Fallback to binary reward
            return float(self.targets[action])

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.current_idx = 0
        return self.feats[self.current_idx], {"protein_idx": 0}

    def step(self, action: int) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:
        # Calculate reward based on reward type
        reward = self._calculate_reward(action)

        # Move to next protein
        self.current_idx += 1
        done = self.current_idx >= self.num_proteins

        # Next observation
        next_obs = None if done else self.feats[self.current_idx]

        # Enhanced info with experimental data
        info = {
            "protein_idx": self.current_idx - 1,
            "was_hit": bool(self.targets[action]),
            "remaining": max(0, self.num_proteins - self.current_idx),
            "reward_type": self.reward_type
        }
        
        # Add experimental data to info if available
        if self.experimental_data and action < len(self.experimental_data):
            exp_data = self.experimental_data[action]
            info.update({
                "binding_affinity_kd": exp_data.get('binding_affinity_kd'),
                "functional_activity": exp_data.get('functional_activity'),
                "toxicity_score": exp_data.get('toxicity_score'),
                "expression_level": exp_data.get('expression_level'),
                "protein_id": exp_data.get('protein_id'),
                "pref_name": exp_data.get('pref_name')
            })

        return next_obs, reward, done, False, info

    def render(self) -> None:
        pass

    def close(self) -> None:
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


def create_experimental_env(
    data_source: str = "uniprot_bindingdb",
    data_path: Optional[str] = None,
    target_column: str = 'is_hit',
    feature_columns: Optional[List[str]] = None,
    reward_type: str = "multi_objective",
    normalize_features: bool = True,
    max_proteins: Optional[int] = None
) -> ProteinEnv:
    """
    Create environment with real experimental data (UniProt+BindingDB or legacy).
    Args:
        data_source: Source of data ('uniprot_bindingdb', 'legacy', or file path)
        data_path: Path to data file
        target_column: Column name for target labels
        feature_columns: List of feature columns to use
        reward_type: Type of reward function ('binary', 'affinity_based', 'multi_objective')
        normalize_features: Whether to normalize features
        max_proteins: Maximum number of proteins to use (for testing)
    Returns:
        ProteinEnv with real experimental data
    """
    features, targets, summary_stats = load_experimental_data(
        data_source=data_source,
        data_path=data_path,
        target_column=target_column,
        feature_columns=feature_columns,
        normalize_features=normalize_features
    )
    if max_proteins and len(features) > max_proteins:
        indices = np.random.choice(len(features), max_proteins, replace=False)
        features = features[indices]
        targets = targets[indices]
    experimental_data = None
    if reward_type in ["affinity_based", "multi_objective"]:
        try:
            from .data_loader import ExperimentalDataLoader
            loader = ExperimentalDataLoader()
            if data_source == "uniprot_bindingdb":
                data_df = loader.load_uniprot_bindingdb_data(data_path)
            elif data_source == "legacy":
                if data_path is None:
                    data_path = "protein_inputs/SHRT_experimental_labels.csv"
                data_df = loader.load_legacy_data(data_path)
            else:
                data_df = loader.load_legacy_data(data_source)
            if max_proteins:
                data_df = data_df.iloc[:max_proteins]
            experimental_data = data_df.to_dict('records')
        except Exception as e:
            print(f"Warning: Could not load experimental data for enhanced rewards: {e}")
            experimental_data = None
    print(f"Created experimental environment with {len(features)} proteins")
    print(f"Data summary: {summary_stats}")
    return ProteinEnv(
        feats=features,
        targets=targets,
        normalize_features=False,  # Already normalized by loader
        reward_type=reward_type,
        experimental_data=experimental_data
    )
