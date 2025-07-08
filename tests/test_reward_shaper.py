"""
Comprehensive tests for the reward shaping module.

Tests all reward functions, edge cases, and configuration options.
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.rewards.reward_shaper import (
    RewardConfig,
    BaseRewardFunction,
    BinaryRewardFunction,
    AffinityBasedRewardFunction,
    MultiObjectiveRewardFunction,
    EnhancedMultiObjectiveRewardFunction,
    RewardShaper,
    create_reward_shaper,
)


class TestRewardConfig:
    """Test reward configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = RewardConfig()
        assert config.reward_type == "enhanced_multi_objective"
        assert config.diversity_weight == 0.3
        assert config.novelty_weight == 0.2
        assert config.cost_weight == 0.1
        assert config.similarity_threshold == 0.8
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = RewardConfig(
            reward_type="binary",
            diversity_weight=0.5,
            novelty_weight=0.3,
            cost_weight=0.2,
            similarity_threshold=0.9
        )
        assert config.reward_type == "binary"
        assert config.diversity_weight == 0.5
        assert config.novelty_weight == 0.3
        assert config.cost_weight == 0.2
        assert config.similarity_threshold == 0.9
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Should not raise error for valid config
        config = RewardConfig(
            reward_type="enhanced_multi_objective",
            diversity_weight=0.5,
            novelty_weight=0.3,
            cost_weight=0.2
        )
        assert config is not None


class TestBinaryRewardFunction:
    """Test binary reward function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RewardConfig(reward_type="binary")
        self.reward_func = BinaryRewardFunction(self.config)
        self.targets = np.array([0, 1, 0, 1, 0])
    
    def test_valid_action(self):
        """Test reward calculation for valid actions."""
        # Test hit (target = 1)
        reward = self.reward_func.calculate_reward(1, self.targets)
        assert reward == 1.0
        
        # Test miss (target = 0)
        reward = self.reward_func.calculate_reward(0, self.targets)
        assert reward == 0.0
    
    def test_invalid_action(self):
        """Test reward calculation for invalid actions."""
        # Test negative action
        reward = self.reward_func.calculate_reward(-1, self.targets)
        assert reward == 0.0
        
        # Test action out of bounds
        reward = self.reward_func.calculate_reward(10, self.targets)
        assert reward == 0.0
    
    def test_reward_components(self):
        """Test reward component breakdown."""
        components = self.reward_func.get_reward_components(1, self.targets)
        assert components["total"] == 1.0
        assert components["binary_hit"] == 1.0
        
        components = self.reward_func.get_reward_components(0, self.targets)
        assert components["total"] == 0.0
        assert components["binary_hit"] == 0.0


class TestAffinityBasedRewardFunction:
    """Test affinity-based reward function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RewardConfig(reward_type="affinity_based")
        self.reward_func = AffinityBasedRewardFunction(self.config)
        self.targets = np.array([0, 1, 0, 1, 0])
        self.experimental_data = {
            0: {"binding_affinity_kd": 1e-6},  # 1 nM
            1: {"binding_affinity_kd": 1e-9},  # 1 pM
            2: {"binding_affinity_kd": 1e-3},  # 1 μM
            3: {"binding_affinity_kd": None},
            4: {},
        }
    
    def test_affinity_reward_calculation(self):
        """Test affinity-based reward calculation."""
        # Strong binding (1 pM) should give high reward
        reward = self.reward_func.calculate_reward(1, self.targets, self.experimental_data)
        assert reward > 4.0  # Should be close to max_affinity_reward (5.0)
        
        # Weak binding (1 μM) should give low reward
        reward = self.reward_func.calculate_reward(2, self.targets, self.experimental_data)
        assert reward < 2.0
        
        # Medium binding (1 nM) should give medium reward
        reward = self.reward_func.calculate_reward(0, self.targets, self.experimental_data)
        assert 2.0 <= reward <= 4.0
    
    def test_fallback_to_binary(self):
        """Test fallback to binary reward when no affinity data."""
        # No experimental data
        reward = self.reward_func.calculate_reward(1, self.targets, None)
        assert reward == 1.0
        
        # Missing affinity data
        reward = self.reward_func.calculate_reward(3, self.targets, self.experimental_data)
        assert reward == 1.0  # target[3] = 1
        
        # Empty experimental data
        reward = self.reward_func.calculate_reward(4, self.targets, self.experimental_data)
        assert reward == 0.0  # target[4] = 0
    
    def test_reward_components(self):
        """Test affinity reward components."""
        components = self.reward_func.get_reward_components(1, self.targets, self.experimental_data)
        assert "total" in components
        assert "affinity" in components
        assert "binary_fallback" in components
        assert components["affinity"] > 0


class TestMultiObjectiveRewardFunction:
    """Test multi-objective reward function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RewardConfig(reward_type="multi_objective")
        self.reward_func = MultiObjectiveRewardFunction(self.config)
        self.targets = np.array([0, 1, 0, 1, 0])
        self.experimental_data = {
            0: {
                "binding_affinity_kd": 1e-6,
                "functional_activity": 0.8,
                "toxicity_score": 0.1,
                "expression_level": 0.9
            },
            1: {
                "binding_affinity_kd": 1e-9,
                "functional_activity": 0.9,
                "toxicity_score": 0.05,
                "expression_level": 1.0
            },
            2: {
                "binding_affinity_kd": 1e-3,
                "functional_activity": 0.2,
                "toxicity_score": 0.8,
                "expression_level": 0.3
            }
        }
    
    def test_multi_objective_calculation(self):
        """Test multi-objective reward calculation."""
        # Good protein (high affinity, activity, low toxicity, high expression)
        reward = self.reward_func.calculate_reward(1, self.targets, self.experimental_data)
        assert reward > 5.0  # Should be high due to multiple positive factors
        
        # Poor protein (low affinity, activity, high toxicity, low expression)
        reward = self.reward_func.calculate_reward(2, self.targets, self.experimental_data)
        assert reward < 3.0  # Should be lower due to negative factors
    
    def test_fallback_to_binary(self):
        """Test fallback to binary reward when no experimental data."""
        reward = self.reward_func.calculate_reward(1, self.targets, None)
        assert reward == 1.0
    
    def test_reward_components(self):
        """Test multi-objective reward components."""
        components = self.reward_func.get_reward_components(1, self.targets, self.experimental_data)
        assert "total" in components
        assert "base" in components
        assert "affinity" in components
        assert "activity" in components
        assert "toxicity" in components
        assert "expression" in components


class TestEnhancedMultiObjectiveRewardFunction:
    """Test enhanced multi-objective reward function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RewardConfig(
            reward_type="enhanced_multi_objective",
            diversity_weight=0.3,
            novelty_weight=0.2,
            cost_weight=0.1
        )
        self.reward_func = EnhancedMultiObjectiveRewardFunction(self.config)
        self.targets = np.array([0, 1, 0, 1, 0])
        self.experimental_data = {
            0: {
                "pchembl_mean": 8.5,
                "activity_count": 10,
                "pchembl_std": 0.5
            },
            1: {
                "pchembl_mean": 9.2,
                "activity_count": 5,
                "pchembl_std": 0.3
            },
            2: {
                "pchembl_mean": 6.0,
                "activity_count": 20,
                "pchembl_std": 1.0
            }
        }
        self.similarity_matrix = np.array([
            [1.0, 0.8, 0.3, 0.2, 0.1],
            [0.8, 1.0, 0.4, 0.3, 0.2],
            [0.3, 0.4, 1.0, 0.7, 0.6],
            [0.2, 0.3, 0.7, 1.0, 0.8],
            [0.1, 0.2, 0.6, 0.8, 1.0]
        ])
        self.novelty_scores = [0.8, 0.9, 0.3, 0.7, 0.6]
    
    def test_activity_strength_reward(self):
        """Test activity strength reward calculation."""
        # High pChemBL should give higher reward
        reward = self.reward_func._calculate_activity_strength_reward(1, self.targets, self.experimental_data)
        assert reward > 0.5
        
        # Low pChemBL should give lower reward
        reward = self.reward_func._calculate_activity_strength_reward(2, self.targets, self.experimental_data)
        assert reward < 0.5
    
    def test_diversity_reward(self):
        """Test diversity reward calculation."""
        # First selection should get full diversity reward
        reward = self.reward_func._calculate_diversity_reward(0, [], self.similarity_matrix)
        assert reward == 1.0
        
        # Selection similar to previous should get low diversity reward
        selected = [0]  # Protein 0 is similar to protein 1 (similarity = 0.8)
        reward = self.reward_func._calculate_diversity_reward(1, selected, self.similarity_matrix)
        assert reward < 0.3  # Should be low due to high similarity
        
        # Selection dissimilar to previous should get high diversity reward
        reward = self.reward_func._calculate_diversity_reward(4, selected, self.similarity_matrix)
        assert reward > 0.8  # Should be high due to low similarity
    
    def test_novelty_reward(self):
        """Test novelty reward calculation."""
        # High novelty score should give high reward
        reward = self.reward_func._calculate_novelty_reward(1, self.novelty_scores)
        assert reward == 0.9
        
        # Low novelty score should give low reward
        reward = self.reward_func._calculate_novelty_reward(2, self.novelty_scores)
        assert reward == 0.3
        
        # No novelty scores should use baseline
        reward = self.reward_func._calculate_novelty_reward(0, None)
        assert reward == self.config.novelty_baseline
    
    def test_cost_penalty(self):
        """Test cost penalty calculation."""
        # High activity count and std should give high penalty
        penalty = self.reward_func._calculate_cost_penalty(2, self.experimental_data)
        assert penalty > 0.5
        
        # Low activity count and std should give low penalty
        penalty = self.reward_func._calculate_cost_penalty(1, self.experimental_data)
        assert penalty < 0.3
    
    def test_enhanced_multi_objective_calculation(self):
        """Test complete enhanced multi-objective reward calculation."""
        # Test with diversity and novelty
        reward = self.reward_func.calculate_reward(
            1, self.targets, self.experimental_data, 
            selected_proteins=[0], similarity_matrix=self.similarity_matrix, 
            novelty_scores=self.novelty_scores
        )
        assert reward > 0.0
        
        # Test without diversity (first selection)
        reward = self.reward_func.calculate_reward(
            0, self.targets, self.experimental_data, 
            selected_proteins=[], similarity_matrix=self.similarity_matrix, 
            novelty_scores=self.novelty_scores
        )
        assert reward > 0.0
    
    def test_reward_components(self):
        """Test enhanced multi-objective reward components."""
        components = self.reward_func.get_reward_components(
            1, self.targets, self.experimental_data, 
            selected_proteins=[0], similarity_matrix=self.similarity_matrix, 
            novelty_scores=self.novelty_scores
        )
        assert "total" in components
        assert "activity" in components
        assert "diversity" in components
        assert "novelty" in components
        assert "cost_penalty" in components
        assert "weighted_diversity" in components
        assert "weighted_novelty" in components
        assert "weighted_cost" in components


class TestRewardShaper:
    """Test main reward shaper class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RewardConfig(reward_type="enhanced_multi_objective")
        self.shaper = RewardShaper(self.config)
        self.targets = np.array([0, 1, 0, 1, 0])
    
    def test_create_reward_function(self):
        """Test reward function creation."""
        # Test enhanced multi-objective
        shaper = RewardShaper(RewardConfig(reward_type="enhanced_multi_objective"))
        assert isinstance(shaper.reward_function, EnhancedMultiObjectiveRewardFunction)
        
        # Test binary
        shaper = RewardShaper(RewardConfig(reward_type="binary"))
        assert isinstance(shaper.reward_function, BinaryRewardFunction)
        
        # Test affinity based
        shaper = RewardShaper(RewardConfig(reward_type="affinity_based"))
        assert isinstance(shaper.reward_function, AffinityBasedRewardFunction)
        
        # Test multi objective
        shaper = RewardShaper(RewardConfig(reward_type="multi_objective"))
        assert isinstance(shaper.reward_function, MultiObjectiveRewardFunction)
        
        # Test unknown type (should default to enhanced_multi_objective)
        shaper = RewardShaper(RewardConfig(reward_type="unknown"))
        assert isinstance(shaper.reward_function, EnhancedMultiObjectiveRewardFunction)
    
    def test_calculate_reward(self):
        """Test reward calculation through shaper."""
        reward = self.shaper.calculate_reward(1, self.targets)
        assert reward > 0.0
    
    def test_get_reward_components(self):
        """Test reward component retrieval through shaper."""
        components = self.shaper.get_reward_components(1, self.targets)
        assert "total" in components
    
    def test_update_config(self):
        """Test configuration update."""
        new_config = RewardConfig(reward_type="binary")
        self.shaper.update_config(new_config)
        assert isinstance(self.shaper.reward_function, BinaryRewardFunction)
        assert self.shaper.config.reward_type == "binary"
    
    def test_get_config(self):
        """Test configuration retrieval."""
        config = self.shaper.get_config()
        assert config.reward_type == "enhanced_multi_objective"


class TestCreateRewardShaper:
    """Test factory function."""
    
    def test_default_creation(self):
        """Test default reward shaper creation."""
        shaper = create_reward_shaper()
        assert isinstance(shaper.reward_function, EnhancedMultiObjectiveRewardFunction)
        assert shaper.config.diversity_weight == 0.3
        assert shaper.config.novelty_weight == 0.2
        assert shaper.config.cost_weight == 0.1
    
    def test_custom_creation(self):
        """Test custom reward shaper creation."""
        shaper = create_reward_shaper(
            reward_type="binary",
            diversity_weight=0.5,
            novelty_weight=0.4,
            cost_weight=0.3
        )
        assert isinstance(shaper.reward_function, BinaryRewardFunction)
        assert shaper.config.diversity_weight == 0.5
        assert shaper.config.novelty_weight == 0.4
        assert shaper.config.cost_weight == 0.3


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_targets(self):
        """Test behavior with empty targets array."""
        config = RewardConfig(reward_type="binary")
        reward_func = BinaryRewardFunction(config)
        targets = np.array([])
        
        reward = reward_func.calculate_reward(0, targets)
        assert reward == 0.0
    
    def test_none_experimental_data(self):
        """Test behavior with None experimental data."""
        config = RewardConfig(reward_type="enhanced_multi_objective")
        reward_func = EnhancedMultiObjectiveRewardFunction(config)
        targets = np.array([0, 1, 0])
        
        reward = reward_func.calculate_reward(1, targets, None)
        assert reward > 0.0  # Should fallback to binary reward
    
    def test_empty_experimental_data(self):
        """Test behavior with empty experimental data."""
        config = RewardConfig(reward_type="enhanced_multi_objective")
        reward_func = EnhancedMultiObjectiveRewardFunction(config)
        targets = np.array([0, 1, 0])
        experimental_data = {}
        
        reward = reward_func.calculate_reward(1, targets, experimental_data)
        assert reward > 0.0  # Should fallback to binary reward
    
    def test_missing_similarity_matrix(self):
        """Test behavior with missing similarity matrix."""
        config = RewardConfig(reward_type="enhanced_multi_objective")
        reward_func = EnhancedMultiObjectiveRewardFunction(config)
        targets = np.array([0, 1, 0])
        
        reward = reward_func.calculate_reward(1, targets, None, [0], None)
        assert reward > 0.0  # Should handle missing similarity matrix gracefully
    
    def test_missing_novelty_scores(self):
        """Test behavior with missing novelty scores."""
        config = RewardConfig(reward_type="enhanced_multi_objective")
        reward_func = EnhancedMultiObjectiveRewardFunction(config)
        targets = np.array([0, 1, 0])
        
        reward = reward_func.calculate_reward(1, targets, None, None, None, None)
        assert reward > 0.0  # Should use baseline novelty score


if __name__ == "__main__":
    pytest.main([__file__]) 