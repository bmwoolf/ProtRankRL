#!/usr/bin/env python3
"""
Test/benchmark PPO, DQN, and A2C (Actor-Critic) from Stable Baselines3 on the same environment.
Compare their performance using the multi-objective reward function.
"""

import sys
import os
sys.path.append(os.path.abspath("."))

from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.env import create_experimental_env

ALGOS = [
    ("PPO", PPO),
    ("DQN", DQN),
    ("A2C", A2C),
]

def evaluate_agent(model, env, n_episodes=5):
    rewards = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if done:
                break
        rewards.append(episode_reward)
    return np.mean(rewards), rewards

def main():
    print("Benchmarking PPO, DQN, and A2C on multi-objective reward environment\n")
    env_fn = lambda: create_experimental_env(
        data_source="protein_inputs/processed/unified_protein_dataset.csv",
        reward_type="enhanced_multi_objective",
        max_proteins=30,
        diversity_weight=0.3,
        novelty_weight=0.2,
        cost_weight=0.1,
        max_episode_length=10,
    )
    results = {}
    for name, Algo in ALGOS:
        print(f"Training {name}...")
        env = DummyVecEnv([env_fn])
        model = Algo("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=10_000)
        print(f"Evaluating {name}...")
        eval_env = env_fn()
        mean_reward, rewards = evaluate_agent(model, eval_env, n_episodes=5)
        results[name] = {
            "mean_reward": mean_reward,
            "rewards": rewards,
        }
        print(f"{name} mean reward: {mean_reward:.2f}\n")
    print("\nSummary:")
    for name, res in results.items():
        print(f"{name}: Mean Reward = {res['mean_reward']:.2f}, Rewards = {res['rewards']}")

    # --- Visualization ---
    agent_names = list(results.keys())
    mean_rewards = [results[name]["mean_reward"] for name in agent_names]
    all_rewards = [results[name]["rewards"] for name in agent_names]

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(10, 5))
    plt.title("Mean Reward per Agent (Multi-Objective RL)")
    sns.barplot(x=agent_names, y=mean_rewards, palette="viridis")
    plt.ylabel("Mean Reward")
    plt.xlabel("Agent")
    plt.tight_layout()
    plt.savefig("outputs/agent_mean_rewards.png", dpi=150)
    plt.show()

    plt.figure(figsize=(12, 6))
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, marker="o", label=agent_names[i])
    plt.title("Episode Rewards per Agent")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/agent_episode_rewards.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    main() 