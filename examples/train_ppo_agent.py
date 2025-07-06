#!/usr/bin/env python3
"""PPO agent training for ProtRankRL."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.env import create_synthetic_env


def make_env():
    """Create environment for vectorized training."""
    def _make_env():
        return create_synthetic_env(
            num_proteins=32,
            feature_dim=64,
            hit_rate=0.2,
            seed=None
        )
    return _make_env


def main():
    print("ProtRankRL PPO Training")
    print("=" * 40)
    
    # Create vectorized environments
    train_env = DummyVecEnv([make_env() for _ in range(4)])
    eval_env = DummyVecEnv([make_env() for _ in range(2)])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="./logs/",
        eval_freq=1000,
        deterministic=True,
        render=False
    )
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=None,  # Disable tensorboard logging
    )
    
    print(f"Training for 100k timesteps...")
    model.learn(
        total_timesteps=100_000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Test trained model
    print("\nTesting trained model...")
    test_env = create_synthetic_env(
        num_proteins=32,  # Match training environment size
        feature_dim=64,
        hit_rate=0.2,
        seed=123
    )
    
    obs, info = test_env.reset()
    total_reward = 0
    step_count = 0
    hits_found = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        hits_found += int(reward)
        step_count += 1
        
        if done:
            break
    
    hit_rate = hits_found / step_count if step_count > 0 else 0
    print(f"Test episode: reward={total_reward:.1f}, "
          f"hits={hits_found}, rate={hit_rate:.2f}")
    
    # Save final model
    model.save("models/ppo_protrank_final")
    print("Training complete! Model saved to models/ppo_protrank_final")


if __name__ == "__main__":
    main() 