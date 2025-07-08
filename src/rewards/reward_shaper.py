"""
Configurable reward shaping module for protein target prioritization.

This module provides a modular, extensible reward system that supports
multiple objectives including binding affinity, diversity, novelty, and cost.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class RewardConfig(BaseModel):
    """Configuration for reward functions."""
    
    # Reward type and weights
    reward_type: str = Field("enhanced_multi_objective", description="Type of reward function to use")
    diversity_weight: float = Field(0.3, description="Weight for diversity reward component")
    novelty_weight: float = Field(0.2, description="Weight for novelty reward component") 
    cost_weight: float = Field(0.1, description="Weight for cost penalty component")
    affinity_weight: float = Field(1.0, description="Weight for binding affinity reward")
    
    # Diversity parameters
    similarity_threshold: float = Field(0.8, description="Threshold for diversity similarity")
    min_diversity_ratio: float = Field(0.5, description="Minimum diversity ratio required")
    
    # Affinity parameters
    affinity_normalization_factor: float = Field(10.0, description="Factor for normalizing affinity values")
    max_affinity_reward: float = Field(5.0, description="Maximum reward for binding affinity")
    
    # Cost parameters
    activity_cost_factor: float = Field(1.0, description="Weight for activity count in cost calculation")
    std_cost_factor: float = Field(1.0, description="Weight for standard deviation in cost calculation")
    
    # Novelty parameters
    novelty_baseline: float = Field(0.5, description="Baseline novelty score for proteins without data")


class BaseRewardFunction(ABC):
    """Abstract base class for reward functions."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
    
    @abstractmethod
    def calculate_reward(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculate reward for a given action."""
        pass
    
    def get_reward_components(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get individual reward components for analysis."""
        return {"total": self.calculate_reward(
            action, targets, experimental_data, selected_proteins, 
            similarity_matrix, novelty_scores
        )}


class BinaryRewardFunction(BaseRewardFunction):
    """Simple binary reward based on hit status."""
    
    def calculate_reward(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculate binary reward (0 or 1) based on hit status."""
        if action < 0 or action >= len(targets):
            return 0.0
        return float(targets[action])
    
    def get_reward_components(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get binary reward components."""
        reward = self.calculate_reward(action, targets, experimental_data, 
                                     selected_proteins, similarity_matrix, novelty_scores)
        return {
            "total": reward,
            "binary_hit": reward,
        }


class AffinityBasedRewardFunction(BaseRewardFunction):
    """Reward based on binding affinity values."""
    
    def calculate_reward(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculate reward based on binding affinity."""
        if action < 0 or action >= len(targets):
            return 0.0
            
        # Fallback to binary reward if no experimental data
        if not experimental_data or action >= len(experimental_data):
            return float(targets[action])
        
        data = experimental_data[action]
        affinity = data.get("binding_affinity_kd")
        
        if affinity is not None and affinity > 0:
            # Convert to nM and calculate reward
            affinity_nm = affinity * 1e9
            # Higher reward for lower affinity (stronger binding)
            reward = max(0, self.config.max_affinity_reward - np.log10(affinity_nm))
            return reward
        
        # Fallback to binary reward
        return float(targets[action])
    
    def get_reward_components(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get affinity-based reward components."""
        reward = self.calculate_reward(action, targets, experimental_data, 
                                     selected_proteins, similarity_matrix, novelty_scores)
        
        affinity_component = 0.0
        if experimental_data and action < len(experimental_data):
            data = experimental_data[action]
            affinity = data.get("binding_affinity_kd")
            if affinity is not None and affinity > 0:
                affinity_nm = affinity * 1e9
                affinity_component = max(0, self.config.max_affinity_reward - np.log10(affinity_nm))
        
        return {
            "total": reward,
            "affinity": affinity_component,
            "binary_fallback": float(targets[action]) if action < len(targets) else 0.0,
        }


class MultiObjectiveRewardFunction(BaseRewardFunction):
    """Legacy multi-objective reward function."""
    
    def calculate_reward(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculate multi-objective reward."""
        if action < 0 or action >= len(targets):
            return 0.0
            
        # Fallback to binary reward if no experimental data
        if not experimental_data or action >= len(experimental_data):
            return float(targets[action])
        
        data = experimental_data[action]
        
        # Base reward from hit status
        base_reward = float(targets[action])
        
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
        
        return max(0.0, total_reward)
    
    def get_reward_components(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get multi-objective reward components."""
        if action < 0 or action >= len(targets):
            return {"total": 0.0}
            
        if not experimental_data or action >= len(experimental_data):
            return {"total": float(targets[action]), "binary": float(targets[action])}
        
        data = experimental_data[action]
        
        base_reward = float(targets[action])
        
        affinity_reward = 0.0
        if data.get("binding_affinity_kd"):
            affinity_nm = data["binding_affinity_kd"] * 1e9
            affinity_reward = max(0, 5 - np.log10(affinity_nm))
        
        activity_reward = data.get("functional_activity", 0.0) * 5.0
        toxicity_penalty = data.get("toxicity_score", 0.0) * -2.0
        expression_reward = data.get("expression_level", 1.0) * 2.0
        
        total_reward = (
            base_reward
            + affinity_reward
            + activity_reward
            + toxicity_penalty
            + expression_reward
        )
        
        return {
            "total": max(0.0, total_reward),
            "base": base_reward,
            "affinity": affinity_reward,
            "activity": activity_reward,
            "toxicity": toxicity_penalty,
            "expression": expression_reward,
        }


class EnhancedMultiObjectiveRewardFunction(BaseRewardFunction):
    """Enhanced multi-objective reward with diversity, novelty, and cost considerations."""
    
    def calculate_reward(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculate enhanced multi-objective reward."""
        if action < 0 or action >= len(targets):
            return 0.0
        
        # Base activity reward
        activity_reward = self._calculate_activity_strength_reward(action, targets, experimental_data)
        
        # Diversity reward
        diversity_reward = self._calculate_diversity_reward(action, selected_proteins, similarity_matrix)
        
        # Novelty reward
        novelty_reward = self._calculate_novelty_reward(action, novelty_scores)
        
        # Cost penalty (negative reward)
        cost_penalty = self._calculate_cost_penalty(action, experimental_data)
        
        # Combine rewards with weights
        total_reward = (
            activity_reward
            + self.config.diversity_weight * diversity_reward
            + self.config.novelty_weight * novelty_reward
            - self.config.cost_weight * cost_penalty
        )
        
        # Ensure reward is non-negative
        return max(0.0, total_reward)
    
    def _calculate_activity_strength_reward(
        self, 
        action: int, 
        targets: np.ndarray, 
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> float:
        """Calculate reward based on activity strength (pChemBL values)."""
        if not experimental_data or action >= len(experimental_data):
            return float(targets[action])  # Fallback to binary
        
        data = experimental_data[action]
        
        # Use pChemBL mean as activity strength indicator
        pchembl_mean = data.get("pchembl_mean", 0)
        
        # Normalize pChemBL (typically 0-14, where higher is better)
        normalized_strength = min(1.0, pchembl_mean / self.config.affinity_normalization_factor)
        
        # Combine with binary hit status
        hit_bonus = float(targets[action]) * 0.5
        strength_reward = normalized_strength * 0.5
        
        return hit_bonus + strength_reward
    
    def _calculate_diversity_reward(
        self, 
        action: int, 
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None
    ) -> float:
        """Calculate diversity reward based on similarity to previously selected proteins."""
        if not selected_proteins or similarity_matrix is None:
            return 1.0  # First selection gets full diversity reward
        
        if action >= similarity_matrix.shape[0]:
            return 1.0
        
        # Calculate similarity to all previously selected proteins
        similarities = []
        for selected_idx in selected_proteins:
            if selected_idx < similarity_matrix.shape[0]:
                sim = similarity_matrix[selected_idx, action]
                similarities.append(sim)
        
        if not similarities:
            return 1.0
        
        # Average similarity to selected proteins
        avg_similarity = np.mean(similarities)
        
        # Higher reward for lower similarity (more diverse)
        diversity_reward = 1.0 - avg_similarity
        return max(0.0, diversity_reward)
    
    def _calculate_novelty_reward(
        self, 
        action: int, 
        novelty_scores: Optional[List[float]] = None
    ) -> float:
        """Calculate novelty reward based on how well-studied the protein is."""
        if novelty_scores and action < len(novelty_scores):
            return novelty_scores[action]
        return self.config.novelty_baseline
    
    def _calculate_cost_penalty(
        self, 
        action: int, 
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None
    ) -> float:
        """Calculate cost penalty based on experimental complexity."""
        if not experimental_data or action >= len(experimental_data):
            return 0.0
        
        data = experimental_data[action]
        
        # Cost factors (higher values = more expensive)
        activity_count = data.get("activity_count", 0)
        pchembl_std = data.get("pchembl_std", 0)
        
        # Normalize cost factors
        if experimental_data:
            max_activities = max([d.get("activity_count", 0) for d in experimental_data.values()])
            max_std = max([d.get("pchembl_std", 0) for d in experimental_data.values()])
        else:
            max_activities = 1
            max_std = 1
        
        activity_cost = activity_count / max_activities if max_activities > 0 else 0
        std_cost = pchembl_std / max_std if max_std > 0 else 0
        
        # Combined cost penalty (higher = more expensive)
        total_cost = (
            self.config.activity_cost_factor * activity_cost + 
            self.config.std_cost_factor * std_cost
        ) / 2.0
        
        return total_cost
    
    def get_reward_components(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get enhanced multi-objective reward components."""
        if action < 0 or action >= len(targets):
            return {"total": 0.0}
        
        activity_reward = self._calculate_activity_strength_reward(action, targets, experimental_data)
        diversity_reward = self._calculate_diversity_reward(action, selected_proteins, similarity_matrix)
        novelty_reward = self._calculate_novelty_reward(action, novelty_scores)
        cost_penalty = self._calculate_cost_penalty(action, experimental_data)
        
        total_reward = (
            activity_reward
            + self.config.diversity_weight * diversity_reward
            + self.config.novelty_weight * novelty_reward
            - self.config.cost_weight * cost_penalty
        )
        
        return {
            "total": max(0.0, total_reward),
            "activity": activity_reward,
            "diversity": diversity_reward,
            "novelty": novelty_reward,
            "cost_penalty": cost_penalty,
            "weighted_diversity": self.config.diversity_weight * diversity_reward,
            "weighted_novelty": self.config.novelty_weight * novelty_reward,
            "weighted_cost": self.config.cost_weight * cost_penalty,
        }


class RewardShaper:
    """Main reward shaper that manages different reward functions."""
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.reward_function = self._create_reward_function()
    
    def _create_reward_function(self) -> BaseRewardFunction:
        """Create reward function based on configuration."""
        reward_type = self.config.reward_type.lower()
        
        if reward_type == "binary":
            return BinaryRewardFunction(self.config)
        elif reward_type == "affinity_based":
            return AffinityBasedRewardFunction(self.config)
        elif reward_type == "multi_objective":
            return MultiObjectiveRewardFunction(self.config)
        elif reward_type == "enhanced_multi_objective":
            return EnhancedMultiObjectiveRewardFunction(self.config)
        else:
            logger.warning(f"Unknown reward type: {reward_type}. Using enhanced_multi_objective.")
            return EnhancedMultiObjectiveRewardFunction(self.config)
    
    def calculate_reward(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> float:
        """Calculate reward using the configured reward function."""
        return self.reward_function.calculate_reward(
            action, targets, experimental_data, selected_proteins, 
            similarity_matrix, novelty_scores
        )
    
    def get_reward_components(
        self,
        action: int,
        targets: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        selected_proteins: Optional[List[int]] = None,
        similarity_matrix: Optional[np.ndarray] = None,
        novelty_scores: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """Get detailed reward components for analysis."""
        return self.reward_function.get_reward_components(
            action, targets, experimental_data, selected_proteins, 
            similarity_matrix, novelty_scores
        )
    
    def update_config(self, new_config: RewardConfig) -> None:
        """Update configuration and recreate reward function."""
        self.config = new_config
        self.reward_function = self._create_reward_function()
        logger.info(f"Updated reward configuration: {new_config.reward_type}")
    
    def get_config(self) -> RewardConfig:
        """Get current configuration."""
        return self.config


def create_reward_shaper(
    reward_type: str = "enhanced_multi_objective",
    diversity_weight: float = 0.3,
    novelty_weight: float = 0.2,
    cost_weight: float = 0.1,
    **kwargs
) -> RewardShaper:
    """Factory function to create a reward shaper with common configurations."""
    config = RewardConfig(
        reward_type=reward_type,
        diversity_weight=diversity_weight,
        novelty_weight=novelty_weight,
        cost_weight=cost_weight,
        **kwargs
    )
    return RewardShaper(config) 