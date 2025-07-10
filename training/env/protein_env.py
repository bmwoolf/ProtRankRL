"""
Enhanced Protein Environment for RL-based Protein Target Prioritization.
Supports multi-objective rewards, diversity constraints, and advanced evaluation.
"""

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .data_loader import load_experimental_data
from ..utils import normalize_to_unit_range, capped_cosine_similarity

logger = logging.getLogger(__name__)


class ProteinEnv(gym.Env):
    """
    Enhanced Gym environment for protein target prioritization.
    Supports multi-objective rewards, diversity constraints, and batch selection.
    """

    def __init__(
        self,
        feats: np.ndarray,
        targets: np.ndarray,
        normalize_features: bool = True,
        reward_type: str = "enhanced_multi_objective",
        experimental_data: dict | None = None,
        diversity_weight: float = 0.3,
        novelty_weight: float = 0.2,
        cost_weight: float = 0.1,
        similarity_threshold: float = 0.8,
        batch_size: int = 1,
        max_episode_length: int | None = None,
        early_stopping_patience: int = 10,
        min_diversity_ratio: float = 0.5,
    ) -> None:
        # Basic validation
        if len(feats) != len(targets):
            raise ValueError("feats and targets must have same length")

        self.feats = feats.astype(np.float32)
        self.targets = targets.astype(np.int32)
        self.num_proteins = len(feats)
        self.feature_dim = feats.shape[1]
        self.reward_type = reward_type
        self.experimental_data = experimental_data or {}

        # Multi-objective parameters
        self.diversity_weight = diversity_weight
        self.novelty_weight = novelty_weight
        self.cost_weight = cost_weight
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.max_episode_length = max_episode_length or self.num_proteins
        self.early_stopping_patience = early_stopping_patience
        self.min_diversity_ratio = min_diversity_ratio

        # Normalize features if requested
        if normalize_features:
            self.feats = self._normalize_features(self.feats)

        # Initialize episode state
        self.current_idx = 0
        self.selected_proteins = []
        self.episode_rewards = []
        self.consecutive_no_improvement = 0
        self.best_episode_reward = float("-inf")

        # Precompute protein similarities for diversity calculation
        self._compute_protein_similarities()

        # Compute novelty scores based on activity count
        self._compute_novelty_scores()

        # Define spaces
        self.action_space = spaces.Discrete(self.num_proteins)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32
        )

    def _normalize_features(self, feats: np.ndarray) -> np.ndarray:
        """Normalize features to [-1, 1] range."""
        # Use utility function to normalize to [0, 1], then scale to [-1, 1]
        normalized_01 = normalize_to_unit_range(feats)
        normalized = 2.0 * normalized_01 - 1.0
        return normalized.astype(np.float32)

    def _compute_protein_similarities(self) -> None:
        """Precompute pairwise protein similarities for diversity calculation."""
        logger.info("Computing protein similarity matrix...")
        self.similarity_matrix = capped_cosine_similarity(self.feats)
        logger.info(f"Computed similarity matrix: {self.similarity_matrix.shape}")

    def _compute_novelty_scores(self) -> None:
        """Compute novelty scores based on activity count (less studied = more novel)."""
        if self.experimental_data:
            activity_counts = []
            for i in range(self.num_proteins):
                if i < len(self.experimental_data):
                    count = self.experimental_data[i].get("activity_count", 0)
                    activity_counts.append(count)
                else:
                    activity_counts.append(0)

            # Normalize to [0, 1] where 1 = most novel (least studied)
            max_count = max(activity_counts) if activity_counts else 1
            self.novelty_scores = [
                1.0 - (count / max_count) for count in activity_counts
            ]
        else:
            # Fallback: random novelty scores
            self.novelty_scores = np.random.random(self.num_proteins).tolist()

    def _calculate_diversity_reward(self, action: int) -> float:
        """Calculate diversity reward based on similarity to previously selected proteins."""
        if not self.selected_proteins:
            return 1.0  # First selection gets full diversity reward

        # Calculate similarity to all previously selected proteins
        similarities = []
        for selected_idx in self.selected_proteins:
            if selected_idx < len(self.similarity_matrix) and action < len(
                self.similarity_matrix
            ):
                sim = self.similarity_matrix[selected_idx, action]
                similarities.append(sim)

        if not similarities:
            return 1.0

        # Average similarity to selected proteins
        avg_similarity = np.mean(similarities)

        # Higher reward for lower similarity (more diverse)
        diversity_reward = 1.0 - avg_similarity
        return max(0.0, diversity_reward)

    def _calculate_novelty_reward(self, action: int) -> float:
        """Calculate novelty reward based on how well-studied the protein is."""
        if action < len(self.novelty_scores):
            return self.novelty_scores[action]
        return 0.5  # Default novelty score

    def _calculate_cost_penalty(self, action: int) -> float:
        """Calculate cost penalty based on experimental complexity."""
        if action >= len(self.experimental_data):
            return 0.0

        data = self.experimental_data[action]

        # Cost factors (higher values = more expensive)
        activity_count = data.get("activity_count", 0)
        pchembl_std = data.get("pchembl_std", 0)

        # Normalize cost factors
        if self.experimental_data:
            max_activities = max(
                [d.get("activity_count", 0) for d in self.experimental_data]
            )
            max_std = max([d.get("pchembl_std", 0) for d in self.experimental_data])
        else:
            max_activities = 1
            max_std = 1

        activity_cost = activity_count / max_activities if max_activities > 0 else 0
        std_cost = pchembl_std / max_std if max_std > 0 else 0

        # Combined cost penalty (higher = more expensive)
        total_cost = (activity_cost + std_cost) / 2.0
        return total_cost

    def _calculate_activity_strength_reward(self, action: int) -> float:
        """Calculate reward based on activity strength (pChemBL values)."""
        if action >= len(self.experimental_data):
            return float(self.targets[action])  # Fallback to binary

        data = self.experimental_data[action]

        # Use pChemBL mean as activity strength indicator
        pchembl_mean = data.get("pchembl_mean", 0)

        # Normalize pChemBL (typically 0-14, where higher is better)
        # Convert to [0, 1] scale
        normalized_strength = min(1.0, pchembl_mean / 10.0)  # 10+ pChemBL is excellent

        # Combine with binary hit status
        hit_bonus = float(self.targets[action]) * 0.5
        strength_reward = normalized_strength * 0.5

        return hit_bonus + strength_reward

    def _calculate_enhanced_multi_objective_reward(self, action: int) -> float:
        """Calculate enhanced multi-objective reward considering all factors."""
        # Base activity reward
        activity_reward = self._calculate_activity_strength_reward(action)

        # Diversity reward
        diversity_reward = self._calculate_diversity_reward(action)

        # Novelty reward
        novelty_reward = self._calculate_novelty_reward(action)

        # Cost penalty (negative reward)
        cost_penalty = self._calculate_cost_penalty(action)

        # Combine rewards with weights
        total_reward = (
            activity_reward
            + self.diversity_weight * diversity_reward
            + self.novelty_weight * novelty_reward
            - self.cost_weight * cost_penalty
        )

        # Ensure reward is non-negative
        return max(0.0, total_reward)

    def _calculate_reward(self, action: int) -> float:
        """Calculate reward based on reward type and experimental data."""
        if self.reward_type == "binary":
            return float(self.targets[action])

        elif self.reward_type == "affinity_based" and self.experimental_data:
            # Use binding affinity for reward calculation
            protein_idx = action
            if protein_idx < len(self.experimental_data):
                affinity = self.experimental_data[protein_idx].get(
                    "binding_affinity_kd"
                )
                if affinity is not None and affinity > 0:
                    # Convert to nM and calculate reward
                    affinity_nm = affinity * 1e9
                    # Higher reward for lower affinity (stronger binding)
                    reward = max(0, 10 - np.log10(affinity_nm))
                    return reward
            return float(self.targets[action])

        elif self.reward_type == "multi_objective" and self.experimental_data:
            # Legacy multi-objective reward
            protein_idx = action
            if protein_idx < len(self.experimental_data):
                data = self.experimental_data[protein_idx]

                # Base reward from hit status
                base_reward = float(self.targets[action])

                # Affinity reward
                affinity_reward = 0.0
                if data.get("binding_affinity_kd"):
                    affinity_nm = data["binding_affinity_kd"] * 1e9
                    affinity_reward = max(0, 5 - np.log10(affinity_nm))

                # Functional activity reward
                activity_reward = data.get("functional_activity", 0.0) * 5.0

                # Toxicity penalty (lower is better)
                toxicity_penalty = data.get("toxicity_score", 0.0) * -2.0

                # Expression reward
                expression_reward = data.get("expression_level", 1.0) * 2.0

                total_reward = (
                    base_reward
                    + affinity_reward
                    + activity_reward
                    + toxicity_penalty
                    + expression_reward
                )
                return total_reward

            return float(self.targets[action])

        elif self.reward_type == "enhanced_multi_objective":
            # Enhanced multi-objective reward with diversity and novelty
            return self._calculate_enhanced_multi_objective_reward(action)

        else:
            # Fallback to binary reward
            return float(self.targets[action])

    def _check_early_stopping(self) -> bool:
        """Check if training should stop early based on reward convergence."""
        if len(self.episode_rewards) < self.early_stopping_patience:
            return False

        # Check if recent rewards are improving
        recent_rewards = self.episode_rewards[-self.early_stopping_patience :]
        current_avg = np.mean(recent_rewards)

        if current_avg > self.best_episode_reward:
            self.best_episode_reward = current_avg
            self.consecutive_no_improvement = 0
        else:
            self.consecutive_no_improvement += 1

        # Stop if no improvement for patience steps
        return self.consecutive_no_improvement >= self.early_stopping_patience

    def _check_diversity_constraint(self, action: int) -> bool:
        """Check if action satisfies diversity constraint."""
        if not self.selected_proteins:
            return True

        # Calculate similarity to selected proteins
        similarities = []
        for selected_idx in self.selected_proteins:
            if selected_idx < len(self.similarity_matrix) and action < len(
                self.similarity_matrix
            ):
                sim = self.similarity_matrix[selected_idx, action]
                similarities.append(sim)

        if not similarities:
            return True

        # Check if average similarity is below threshold
        avg_similarity = np.mean(similarities)
        return avg_similarity < self.similarity_threshold

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

        self.current_idx = 0
        self.selected_proteins = []
        self.episode_rewards = []
        self.consecutive_no_improvement = 0
        self.best_episode_reward = float("-inf")

        obs = self.feats[self.current_idx]
        return obs, {"protein_idx": 0}

    def step(
        self, action: int
    ) -> tuple[np.ndarray | None, float, bool, bool, dict[str, Any]]:
        # Validate action
        if action < 0 or action >= self.num_proteins:
            reward = -1.0  # Penalty for invalid action
            done = True
            info = {"error": "Invalid action"}
            return None, reward, done, False, info

        # Check diversity constraint
        if not self._check_diversity_constraint(action):
            reward = -0.5  # Penalty for violating diversity constraint
        else:
            # Calculate reward based on reward type
            reward = self._calculate_reward(action)

        # Track selected protein
        self.selected_proteins.append(action)
        self.episode_rewards.append(reward)

        # Move to next protein
        self.current_idx += 1
        done = (
            self.current_idx >= self.max_episode_length or self._check_early_stopping()
        )

        # Next observation
        next_obs = None if done else self.feats[self.current_idx]

        # Enhanced info with experimental data and metrics
        info = {
            "protein_idx": self.current_idx - 1,
            "was_hit": bool(self.targets[action]),
            "remaining": max(0, self.num_proteins - self.current_idx),
            "reward_type": self.reward_type,
            "selected_proteins": self.selected_proteins.copy(),
            "episode_reward": sum(self.episode_rewards),
            "diversity_score": self._calculate_diversity_reward(action),
            "novelty_score": self._calculate_novelty_reward(action),
            "cost_penalty": self._calculate_cost_penalty(action),
        }

        # Add experimental data to info if available
        if self.experimental_data and action < len(self.experimental_data):
            exp_data = self.experimental_data[action]
            info.update(
                {
                    "binding_affinity_kd": exp_data.get("binding_affinity_kd"),
                    "functional_activity": exp_data.get("functional_activity"),
                    "toxicity_score": exp_data.get("toxicity_score"),
                    "expression_level": exp_data.get("expression_level"),
                    "protein_id": exp_data.get("protein_id"),
                    "pref_name": exp_data.get("pref_name"),
                    "activity_count": exp_data.get("activity_count"),
                    "pchembl_mean": exp_data.get("pchembl_mean"),
                    "pchembl_std": exp_data.get("pchembl_std"),
                }
            )

        return next_obs, reward, done, False, info

    def render(self) -> None:
        pass

    def close(self) -> None:
        pass

    def get_evaluation_metrics(self) -> dict[str, float]:
        """Get comprehensive evaluation metrics for the episode."""
        if not self.selected_proteins:
            return {}

        # Basic metrics
        total_reward = sum(self.episode_rewards)
        hit_count = sum(1 for idx in self.selected_proteins if self.targets[idx])
        hit_rate = (
            hit_count / len(self.selected_proteins) if self.selected_proteins else 0
        )

        # Diversity metrics
        diversity_scores = []
        for i, protein_idx in enumerate(self.selected_proteins):
            if i == 0:
                diversity_scores.append(1.0)  # First selection
            else:
                # Calculate similarity to all previously selected
                similarities = []
                for prev_idx in self.selected_proteins[:i]:
                    if prev_idx < len(self.similarity_matrix) and protein_idx < len(
                        self.similarity_matrix
                    ):
                        sim = self.similarity_matrix[prev_idx, protein_idx]
                        similarities.append(sim)
                if similarities:
                    avg_sim = np.mean(similarities)
                    diversity_scores.append(1.0 - avg_sim)
                else:
                    diversity_scores.append(1.0)

        avg_diversity = np.mean(diversity_scores) if diversity_scores else 0

        # Novelty metrics
        novelty_scores = []
        for protein_idx in self.selected_proteins:
            if protein_idx < len(self.novelty_scores):
                novelty_scores.append(self.novelty_scores[protein_idx])

        avg_novelty = np.mean(novelty_scores) if novelty_scores else 0

        # Activity strength metrics
        activity_scores = []
        for protein_idx in self.selected_proteins:
            score = self._calculate_activity_strength_reward(protein_idx)
            activity_scores.append(score)

        avg_activity_strength = np.mean(activity_scores) if activity_scores else 0

        return {
            "total_reward": total_reward,
            "hit_count": hit_count,
            "hit_rate": hit_rate,
            "avg_diversity": avg_diversity,
            "avg_novelty": avg_novelty,
            "avg_activity_strength": avg_activity_strength,
            "episode_length": len(self.selected_proteins),
            "reward_per_step": (
                total_reward / len(self.selected_proteins)
                if self.selected_proteins
                else 0
            ),
        }


def create_synthetic_env(
    num_proteins: int = 64,
    feature_dim: int = 128,
    hit_rate: float = 0.2,
    seed: int | None = None,
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
    data_path: str | None = None,
    target_column: str = "has_activity",
    feature_columns: list[str] | None = None,
    reward_type: str = "enhanced_multi_objective",
    normalize_features: bool = True,
    max_proteins: int | None = None,
    diversity_weight: float = 0.3,
    novelty_weight: float = 0.2,
    cost_weight: float = 0.1,
    similarity_threshold: float = 0.8,
    batch_size: int = 1,
    max_episode_length: int | None = None,
    early_stopping_patience: int = 10,
    min_diversity_ratio: float = 0.5,
) -> ProteinEnv:
    """
    Create environment with real experimental data (UniProt+BindingDB or legacy).
    Args:
        data_source: Source of data ('uniprot_bindingdb', 'legacy', or file path)
        data_path: Path to data file
        target_column: Column name for target labels
        feature_columns: List of feature columns to use
        reward_type: Type of reward function ('binary', 'affinity_based', 'multi_objective', 'enhanced_multi_objective')
        normalize_features: Whether to normalize features
        max_proteins: Maximum number of proteins to use (for testing)
        diversity_weight: Weight for diversity reward component
        novelty_weight: Weight for novelty reward component
        cost_weight: Weight for cost penalty component
        similarity_threshold: Threshold for diversity constraint
        batch_size: Number of proteins to select per step
        max_episode_length: Maximum episode length
        early_stopping_patience: Number of steps without improvement before early stopping
        min_diversity_ratio: Minimum ratio of diverse selections required
    Returns:
        ProteinEnv with real experimental data
    """
    features, targets, summary_stats = load_experimental_data(
        data_source=data_source,
        data_path=data_path,
        target_column=target_column,
        feature_columns=feature_columns,
        normalize_features=normalize_features,
    )
    if max_proteins and len(features) > max_proteins:
        indices = np.random.choice(len(features), max_proteins, replace=False)
        features = features[indices]
        targets = targets[indices]

    experimental_data = None
    if reward_type in ["affinity_based", "multi_objective", "enhanced_multi_objective"]:
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
            experimental_data = data_df.to_dict("records")
        except Exception as e:
            print(
                f"Warning: Could not load experimental data for enhanced rewards: {e}"
            )
            experimental_data = None

    print(f"Created experimental environment with {len(features)} proteins")
    print(f"Data summary: {summary_stats}")
    print(f"Reward type: {reward_type}")
    print(f"Diversity weight: {diversity_weight}")
    print(f"Novelty weight: {novelty_weight}")
    print(f"Cost weight: {cost_weight}")

    return ProteinEnv(
        feats=features,
        targets=targets,
        normalize_features=False,  # Already normalized by loader
        reward_type=reward_type,
        experimental_data=experimental_data,
        diversity_weight=diversity_weight,
        novelty_weight=novelty_weight,
        cost_weight=cost_weight,
        similarity_threshold=similarity_threshold,
        batch_size=batch_size,
        max_episode_length=max_episode_length,
        early_stopping_patience=early_stopping_patience,
        min_diversity_ratio=min_diversity_ratio,
    )
