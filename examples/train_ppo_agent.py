#!/usr/bin/env python3
"""PPO agent training for ProtRankRL - Clean version with multiple timestep tests."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
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


def train_and_evaluate(timesteps, model_name):
    """Train model for specified timesteps and evaluate."""
    print(f"\n{'='*50}")
    print(f"Training for {timesteps:,} timesteps")
    print(f"{'='*50}")
    
    # Create environments
    train_env = DummyVecEnv([make_env() for _ in range(4)])
    eval_env = DummyVecEnv([make_env() for _ in range(2)])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}_",
        log_path="./logs/",
        eval_freq=max(1000, timesteps // 20),  # Evaluate ~20 times during training
        deterministic=True,
        render=False,
        verbose=0  # Reduce callback verbosity
    )
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,  # Reduce PPO verbosity
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=None,
    )
    
    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=eval_callback,
        progress_bar=False  # Disable progress bar
    )
    
    # Test trained model
    test_env = create_synthetic_env(
        num_proteins=32,
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
    
    # Print results
    print(f"Results for {timesteps:,} timesteps:")
    print(f"  Test Episode Reward: {total_reward:.1f}")
    print(f"  Hits Found: {hits_found}")
    print(f"  Hit Rate: {hit_rate:.3f} ({hits_found}/{step_count})")
    
    # Save model
    model.save(f"models/{model_name}")
    print(f"  Model saved: models/{model_name}")
    
    return {
        'timesteps': timesteps,
        'reward': total_reward,
        'hits': hits_found,
        'hit_rate': hit_rate
    }


def main():
    print("ProtRankRL PPO Training - Multiple Timestep Comparison")
    print("=" * 60)
    
    # Configure logging to reduce output
    configure(folder="./logs", format_strings=["stdout"])
    
    # Test different timestep configurations
    timestep_configs = [
        (100_000, "ppo_100k"),
        (200_000, "ppo_200k"), 
        (500_000, "ppo_500k")
    ]
    
    results = []
    
    for timesteps, model_name in timestep_configs:
        result = train_and_evaluate(timesteps, model_name)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF ALL TRAINING RUNS")
    print(f"{'='*60}")
    print(f"{'Timesteps':<12} {'Reward':<8} {'Hits':<6} {'Hit Rate':<10}")
    print("-" * 40)
    
    for result in results:
        print(f"{result['timesteps']:<12,} {result['reward']:<8.1f} "
              f"{result['hits']:<6} {result['hit_rate']:<10.3f}")
    
    # Find best performing model
    best_result = max(results, key=lambda x: x['hit_rate'])
    print(f"\nBest performing model: {best_result['timesteps']:,} timesteps")
    print(f"Hit rate: {best_result['hit_rate']:.3f} ({best_result['hits']} hits)")


if __name__ == "__main__":
    main() 