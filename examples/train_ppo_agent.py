#!/usr/bin/env python3
"""PPO agent training for ProtRankRL."""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from src.env import ProteinEnvFactory


def make_env():
    """Create environment for vectorized training."""
    def _make_env():
        return ProteinEnvFactory.create_synthetic_env(
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
    
    # Initialize PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./logs/"
    )
    
    print(f"Training for 100k timesteps...")
    model.learn(
        total_timesteps=100_000,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Test trained model
    print("\nTesting trained model...")
    test_env = ProteinEnvFactory.create_synthetic_env(
        num_proteins=16,
        feature_dim=64,
        hit_rate=0.2,
        seed=123
    )
    
    obs, info = test_env.reset()
    total_reward = 0
    step_count = 0
    
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = test_env.step(action)
        total_reward += reward
        step_count += 1
        
        if done:
            break
    
    stats = test_env.get_episode_stats()
    print(f"Test episode: reward={stats['total_reward']:.1f}, "
          f"hits={stats['num_hits_found']}, rate={stats['hit_rate']:.2f}")
    
    # Save final model
    model.save("models/ppo_protrank_final")
    print("Training complete! Model saved to models/ppo_protrank_final")


if __name__ == "__main__":
    main() 