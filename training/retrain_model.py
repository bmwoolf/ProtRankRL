#!/usr/bin/env python3
"""
Retrain PPO Model with Enhanced State Builder

This script retrains the PPO agent using the enhanced state builder
that creates 1283-dim feature vectors (1280 ESM + 3 experimental features).
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import time
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces
import torch.nn as nn

from src.env.state_builder import ProteinStateBuilder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProteinEnv(gym.Env):
    """Simple protein environment for retraining with enhanced state builder."""
    
    def __init__(self, state_builder: ProteinStateBuilder, max_proteins: int = 50):
        self.state_builder = state_builder
        self.max_proteins = max_proteins
        
        # Get all available proteins
        self.all_proteins = list(state_builder.get_all_feature_vectors().keys())
        if len(self.all_proteins) > max_proteins:
            self.all_proteins = self.all_proteins[:max_proteins]
        
        self.num_proteins = len(self.all_proteins)
        self.feature_dim = state_builder.get_feature_dim()
        
        # Define spaces
        self.action_space = spaces.Discrete(self.num_proteins)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.feature_dim,), dtype=np.float32
        )
        
        # Episode state
        self.current_step = 0
        self.selected_proteins = []
        self.episode_rewards = []
        
        logger.info(f"Created environment with {self.num_proteins} proteins and {self.feature_dim} features")
    
    def reset(self, seed=None, options=None):
        """Reset the environment for a new episode."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.selected_proteins = []
        self.episode_rewards = []
        
        # Return initial observation (first protein's features)
        initial_protein = self.all_proteins[0]
        initial_features = self.state_builder.get_feature_vector(initial_protein)
        
        return initial_features, {}
    
    def step(self, action):
        """Take a step in the environment."""
        if action >= self.num_proteins:
            action = 0  # Default to first protein if invalid action
        
        selected_protein = self.all_proteins[action]
        self.selected_proteins.append(selected_protein)
        
        # Calculate reward based on experimental data
        reward = self._calculate_reward(selected_protein)
        self.episode_rewards.append(reward)
        
        # Check if episode is done
        done = self.current_step >= self.num_proteins - 1
        truncated = False
        
        # Get next observation
        if not done:
            next_protein = self.all_proteins[self.current_step + 1]
            next_features = self.state_builder.get_feature_vector(next_protein)
        else:
            next_features = np.zeros(self.feature_dim, dtype=np.float32)
        
        self.current_step += 1
        
        return next_features, reward, done, truncated, {}
    
    def _calculate_reward(self, protein_id: str) -> float:
        """Calculate reward based on experimental data."""
        # Get experimental data from the state builder's dataframe
        df = self.state_builder.df
        row = df[df["uniprot_id"] == protein_id]
        
        if row.empty:
            return 0.0
        
        row = row.iloc[0]
        
        # Reward based on activity strength (pChemBL values)
        pchembl_mean = row.get("pchembl_mean", 0.0)
        activity_count = row.get("activity_count", 0)
        has_activity = row.get("has_activity", 0)
        
        # Normalize pChemBL (0-14 scale, higher is better)
        pchembl_reward = min(1.0, pchembl_mean / 10.0) if pchembl_mean > 0 else 0.0
        
        # Activity bonus
        activity_bonus = float(has_activity) * 0.5
        
        # Combined reward
        total_reward = pchembl_reward * 0.5 + activity_bonus
        
        return total_reward


def create_env(data_path: str = "data/processed/unified_protein_dataset.csv", max_proteins: int = 50):
    """Create environment using enhanced state builder."""
    def _make_env():
        state_builder = ProteinStateBuilder(data_path)
        env = ProteinEnv(state_builder, max_proteins)
        return Monitor(env)
    return _make_env


def train_model(
    timesteps: int = 100000,
    max_proteins: int = 50,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    model_name: str = "ppo_enhanced_1283d"
):
    """Train the PPO model with enhanced state builder."""
    
    print(f"\n{'='*60}")
    print("Training PPO Model with Enhanced State Builder")
    print(f"{'='*60}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Max proteins: {max_proteins}")
    print(f"Feature dimension: 1283 (1280 ESM + 3 experimental)")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    # Create environments
    train_env = DummyVecEnv([create_env(max_proteins=max_proteins) for _ in range(4)])
    eval_env = DummyVecEnv([create_env(max_proteins=max_proteins) for _ in range(2)])
    
    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}_best_",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1,
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"./models/{model_name}_checkpoints/",
        name_prefix=model_name,
        verbose=1,
    )
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=f"./logs/{model_name}_tensorboard/",
        policy_kwargs={
            "net_arch": {"pi": [256, 256], "vf": [256, 256]},
            "activation_fn": nn.ReLU,
        },
    )
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )
    
    training_time = time.time() - start_time
    
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Model saved to: ./models/{model_name}_best_")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain PPO model with enhanced state builder")
    parser.add_argument('--timesteps', type=int, default=50000, help='Number of training timesteps')
    parser.add_argument('--max_proteins', type=int, default=50, help='Number of proteins to use')
    parser.add_argument('--model_name', type=str, default='ppo_enhanced_1283d', help='Model name for saving')
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--n_steps', type=int, default=2048, help='Number of steps per update')
    parser.add_argument('--n_epochs', type=int, default=10, help='Number of epochs per update')
    args = parser.parse_args()

    print("\nParsed arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    model = train_model(
        timesteps=args.timesteps,
        max_proteins=args.max_proteins,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        model_name=args.model_name
    )
    
    print("\nTraining completed successfully!")
    print("The model is now ready to use with 1283-dim features.") 