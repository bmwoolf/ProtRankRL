"""
Unit tests for ProteinEnv class.
"""

import pytest
import numpy as np
from src.env import ProteinEnv, ProteinEnvFactory


class TestProteinEnv:
    def test_init(self):
        feats = np.random.randn(10, 5).astype(np.float32)
        targets = np.random.randint(0, 2, 10, dtype=np.int32)
        env = ProteinEnv(feats, targets)
        
        assert env.num_proteins == 10
        assert env.feature_dim == 5
        assert env.action_space.n == 10
        assert env.observation_space.shape == (5,)
    
    def test_init_validation(self):
        feats = np.random.randn(10, 5)
        targets = np.random.randint(0, 2, 9)  # Wrong length
        
        with pytest.raises(ValueError, match="same length"):
            ProteinEnv(feats, targets)
    
    def test_reset(self):
        feats = np.random.randn(8, 4).astype(np.float32)
        targets = np.random.randint(0, 2, 8, dtype=np.int32)
        env = ProteinEnv(feats, targets)
        
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,)
        assert info["protein_idx"] == 0
        assert info["total_proteins"] == 8
    
    def test_step_loop(self):
        feats = np.random.randn(5, 3).astype(np.float32)
        targets = np.array([1, 0, 1, 0, 0], dtype=np.int32)
        env = ProteinEnv(feats, targets)
        
        obs, info = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            
            if done:
                break
        
        assert step_count == 5
        assert total_reward >= 0
        assert total_reward <= 5
    
    def test_seed_reproducibility(self):
        feats = np.random.randn(10, 4).astype(np.float32)
        targets = np.random.randint(0, 2, 10, dtype=np.int32)
        env = ProteinEnv(feats, targets)
        
        # First run
        env.reset(seed=123)
        actions1 = []
        while True:
            action = env.action_space.sample()
            actions1.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        # Second run with same seed
        env.reset(seed=123)
        actions2 = []
        while True:
            action = env.action_space.sample()
            actions2.append(action)
            obs, reward, done, truncated, info = env.step(action)
            if done:
                break
        
        # Convert to regular integers for comparison
        actions1 = [int(a) for a in actions1]
        actions2 = [int(a) for a in actions2]
        assert actions1 == actions2
    
    def test_episode_stats(self):
        feats = np.random.randn(4, 2).astype(np.float32)
        targets = np.array([1, 0, 1, 0], dtype=np.int32)
        env = ProteinEnv(feats, targets)
        
        env.reset()
        for i in range(4):
            env.step(i)
        
        stats = env.get_episode_stats()
        assert stats["total_reward"] == 2.0
        assert stats["num_hits_found"] == 2
        assert stats["hit_rate"] == 0.5
        assert stats["episode_length"] == 4


class TestProteinEnvFactory:
    def test_create_synthetic_env(self):
        env = ProteinEnvFactory.create_synthetic_env(
            num_proteins=20,
            feature_dim=10,
            hit_rate=0.3,
            seed=42
        )
        
        assert env.num_proteins == 20
        assert env.feature_dim == 10
        assert np.sum(env.targets) == 6  # 20 * 0.3 = 6
    
    def test_create_from_data(self):
        feats = np.random.randn(15, 6).astype(np.float32)
        targets = np.random.randint(0, 2, 15, dtype=np.int32)
        
        env = ProteinEnvFactory.create_from_data(feats, targets)
        assert env.num_proteins == 15
        assert env.feature_dim == 6
        np.testing.assert_array_equal(env.targets, targets) 