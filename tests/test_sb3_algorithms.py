#!/usr/bin/env python3
"""
Test/benchmark PPO, DQN, and A2C (Actor-Critic) from Stable Baselines3 on the same environment.
Compare their performance using the multi-objective reward function.
"""

import os
import sys

sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    ndcg_score,
    roc_auc_score,
)
from sbx import A2C, DQN, PPO
from sbx.vec_env import DummyVecEnv

from src.env import create_experimental_env

ALGOS = [
    ("PPO", PPO),
    ("DQN", DQN),
    ("A2C", A2C),
]


def env_fn():
    return create_experimental_env(
        data_source="protein_inputs/processed/unified_protein_dataset.csv",
        reward_type="enhanced_multi_objective",
        max_proteins=30,
        diversity_weight=0.3,
        novelty_weight=0.2,
        cost_weight=0.1,
        max_episode_length=10,
    )


def compute_metrics(selected_hits, all_hits, rewards, k=5):
    # selected_hits: list of 0/1 for each selected protein (was_hit)
    # all_hits: list of 0/1 for all proteins in the environment
    # rewards: list of rewards per episode
    # k: top-k for precision/recall
    n_selected = len(selected_hits)
    n_hits = sum(selected_hits)
    n_total_hits = sum(all_hits)
    hit_rate = n_hits / n_selected if n_selected else 0.0
    avg_reward = np.mean(rewards)
    cum_reward = np.sum(rewards)
    # Precision@k, Recall@k, F1@k
    top_k = selected_hits[:k]
    precision_at_k = sum(top_k) / k if k else 0.0
    recall_at_k = sum(top_k) / n_total_hits if n_total_hits else 0.0
    f1_at_k = (
        (2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k))
        if (precision_at_k + recall_at_k)
        else 0.0
    )
    # AUC-ROC (if possible)
    try:
        auc_roc = roc_auc_score(
            all_hits, [1.0] * len(all_hits)
        )  # Dummy: all selected, for demo
    except Exception:
        auc_roc = float("nan")
    # NDCG (ranking quality)
    try:
        ndcg = ndcg_score([all_hits], [selected_hits])
    except Exception:
        ndcg = float("nan")
    return {
        "hit_rate": hit_rate,
        "avg_reward": avg_reward,
        "cum_reward": cum_reward,
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "f1@k": f1_at_k,
        "auc_roc": auc_roc,
        "ndcg": ndcg,
    }


def evaluate_agent(model, env, n_episodes=5, k=5):
    rewards = []
    all_selected_hits = []
    all_env_hits = []
    for _ in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        selected_hits = []
        env_hits = []
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            selected_hits.append(int(info.get("was_hit", 0)))
            env_hits.append(int(info.get("was_hit", 0)))
            if done:
                break
        rewards.append(episode_reward)
        all_selected_hits.extend(selected_hits)
        all_env_hits.extend(env_hits)
    metrics = compute_metrics(all_selected_hits, all_env_hits, rewards, k=k)
    return np.mean(rewards), rewards, metrics


def main():
    print("Benchmarking PPO, DQN, and A2C on multi-objective reward environment\n")
    results = {}
    for name, algo in ALGOS:
        print(f"Training {name}...")
        env = DummyVecEnv([env_fn])
        model = algo("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=10_000)
        print(f"Evaluating {name}...")
        eval_env = env_fn()
        mean_reward, rewards, metrics = evaluate_agent(
            model, eval_env, n_episodes=5, k=5
        )
        results[name] = {
            "mean_reward": mean_reward,
            "rewards": rewards,
            "metrics": metrics,
        }
        print(f"{name} mean reward: {mean_reward:.2f}\n")
    print("\nSummary:")
    for name, res in results.items():
        print(
            f"{name}: Mean Reward = {res['mean_reward']:.2f}, Rewards = {res['rewards']}"
        )
        print(f"  Metrics: {res['metrics']}")

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
    plt.savefig("tests/charts/agent_mean_rewards.png", dpi=150)
    plt.show()

    plt.figure(figsize=(12, 6))
    for i, rewards in enumerate(all_rewards):
        plt.plot(rewards, marker="o", label=agent_names[i])
    plt.title("Episode Rewards per Agent")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig("tests/charts/agent_episode_rewards.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
