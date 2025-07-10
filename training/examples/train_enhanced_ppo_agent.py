#!/usr/bin/env python3
"""
Enhanced PPO Agent Training for ProtRankRL with Multi-Objective Rewards

This script trains a PPO agent using the enhanced multi-objective environment
with diversity constraints, novelty rewards, and cost considerations.

Features:
- Multi-objective reward optimization
- Diversity-aware protein selection
- Novelty-based exploration
- Cost-constrained optimization
- Comprehensive evaluation metrics
- Hyperparameter optimization
- Early stopping and convergence monitoring
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sbx import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env import create_experimental_env


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("enhanced_training.log"),
        ],
    )


def create_enhanced_env(
    data_path: str = "protein_inputs/processed/unified_protein_dataset.csv",
    max_proteins: int = 100,
    diversity_weight: float = 0.3,
    novelty_weight: float = 0.2,
    cost_weight: float = 0.1,
    similarity_threshold: float = 0.8,
    max_episode_length: int = 50,
    early_stopping_patience: int = 10,
) -> Any:
    """Create enhanced environment with multi-objective rewards."""

    def _make_env():
        env = create_experimental_env(
            data_source=data_path,
            reward_type="enhanced_multi_objective",
            max_proteins=max_proteins,
            diversity_weight=diversity_weight,
            novelty_weight=novelty_weight,
            cost_weight=cost_weight,
            similarity_threshold=similarity_threshold,
            max_episode_length=max_episode_length,
            early_stopping_patience=early_stopping_patience,
        )
        return Monitor(env)

    return _make_env


def evaluate_model(
    model: PPO, env: Any, n_eval_episodes: int = 10, deterministic: bool = True
) -> dict[str, float]:
    """Evaluate model and return comprehensive metrics."""
    all_metrics = []

    for _episode in range(n_eval_episodes):
        obs, info = env.reset()
        episode_rewards = []
        episode_actions = []

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)
            episode_actions.append(action)

            if done:
                break

        # Get episode metrics
        episode_metrics = env.get_evaluation_metrics()
        all_metrics.append(episode_metrics)

    # Aggregate metrics across episodes
    aggregated_metrics = {}
    for key in all_metrics[0].keys():
        values = [metrics[key] for metrics in all_metrics if key in metrics]
        aggregated_metrics[f"mean_{key}"] = np.mean(values)
        aggregated_metrics[f"std_{key}"] = np.std(values)
        aggregated_metrics[f"min_{key}"] = np.min(values)
        aggregated_metrics[f"max_{key}"] = np.max(values)

    return aggregated_metrics


def train_with_hyperparameters(
    timesteps: int,
    model_name: str,
    data_path: str,
    max_proteins: int = 100,
    diversity_weight: float = 0.3,
    novelty_weight: float = 0.2,
    cost_weight: float = 0.1,
    similarity_threshold: float = 0.8,
    max_episode_length: int = 50,
    early_stopping_patience: int = 10,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    n_envs: int = 4,
    eval_freq: int = 10000,
    verbose: int = 1,
) -> dict[str, Any]:
    """Train model with specified hyperparameters and return results."""

    print(f"\n{'='*60}")
    print("Training Enhanced PPO Agent")
    print(f"{'='*60}")
    print(f"Timesteps: {timesteps:,}")
    print(f"Max proteins: {max_proteins}")
    print(f"Diversity weight: {diversity_weight}")
    print(f"Novelty weight: {novelty_weight}")
    print(f"Cost weight: {cost_weight}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")

    start_time = time.time()

    # Create environments
    train_env = DummyVecEnv(
        [
            create_enhanced_env(
                data_path=data_path,
                max_proteins=max_proteins,
                diversity_weight=diversity_weight,
                novelty_weight=novelty_weight,
                cost_weight=cost_weight,
                similarity_threshold=similarity_threshold,
                max_episode_length=max_episode_length,
                early_stopping_patience=early_stopping_patience,
            )
            for _ in range(n_envs)
        ]
    )

    eval_env = DummyVecEnv(
        [
            create_enhanced_env(
                data_path=data_path,
                max_proteins=max_proteins,
                diversity_weight=diversity_weight,
                novelty_weight=novelty_weight,
                cost_weight=cost_weight,
                similarity_threshold=similarity_threshold,
                max_episode_length=max_episode_length,
                early_stopping_patience=early_stopping_patience,
            )
            for _ in range(2)
        ]
    )

    # Create callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./models/{model_name}_best_",
        log_path="./logs/",
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=verbose,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq * 2,
        save_path=f"./models/{model_name}_checkpoints/",
        name_prefix=model_name,
        verbose=verbose,
    )

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=verbose,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        tensorboard_log=f"./logs/{model_name}_tensorboard/",
        policy_kwargs={
            "net_arch": [{"pi": [256, 256], "vf": [256, 256]}],
            "activation_fn": "relu",
        },
    )

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True,
    )

    training_time = time.time() - start_time

    # Evaluate final model
    print("\nEvaluating final model...")
    final_metrics = evaluate_model(model, eval_env.envs[0], n_eval_episodes=5)

    # Save model
    model.save(f"models/{model_name}_final")

    # Save training results
    results = {
        "model_name": model_name,
        "timesteps": timesteps,
        "training_time": training_time,
        "hyperparameters": {
            "diversity_weight": diversity_weight,
            "novelty_weight": novelty_weight,
            "cost_weight": cost_weight,
            "similarity_threshold": similarity_threshold,
            "max_episode_length": max_episode_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        },
        "final_metrics": final_metrics,
    }

    with open(f"logs/{model_name}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nTraining completed in {training_time:.1f} seconds")
    print(f"Final metrics: {final_metrics}")
    print(f"Model saved: models/{model_name}_final")

    return results


def hyperparameter_optimization(
    data_path: str,
    max_proteins: int = 100,
    timesteps: int = 500000,
    n_trials: int = 5,
) -> dict[str, Any]:
    """Run hyperparameter optimization for the enhanced environment."""

    print(f"\n{'='*60}")
    print("Hyperparameter Optimization")
    print(f"{'='*60}")
    print(f"Trials: {n_trials}")
    print(f"Timesteps per trial: {timesteps:,}")
    print(f"{'='*60}")

    # Define hyperparameter search space
    hyperparameter_configs = [
        # Conservative weights
        {
            "diversity_weight": 0.2,
            "novelty_weight": 0.1,
            "cost_weight": 0.05,
            "similarity_threshold": 0.7,
            "learning_rate": 1e-4,
            "batch_size": 32,
        },
        # Balanced weights
        {
            "diversity_weight": 0.3,
            "novelty_weight": 0.2,
            "cost_weight": 0.1,
            "similarity_threshold": 0.8,
            "learning_rate": 3e-4,
            "batch_size": 64,
        },
        # High diversity focus
        {
            "diversity_weight": 0.5,
            "novelty_weight": 0.3,
            "cost_weight": 0.1,
            "similarity_threshold": 0.6,
            "learning_rate": 5e-4,
            "batch_size": 128,
        },
        # High novelty focus
        {
            "diversity_weight": 0.2,
            "novelty_weight": 0.5,
            "cost_weight": 0.05,
            "similarity_threshold": 0.8,
            "learning_rate": 2e-4,
            "batch_size": 64,
        },
        # Cost-aware
        {
            "diversity_weight": 0.3,
            "novelty_weight": 0.2,
            "cost_weight": 0.3,
            "similarity_threshold": 0.8,
            "learning_rate": 3e-4,
            "batch_size": 64,
        },
    ]

    best_result = None
    best_score = float("-inf")
    all_results = []

    for i, config in enumerate(hyperparameter_configs):
        print(f"\nTrial {i+1}/{len(hyperparameter_configs)}")
        print(f"Config: {config}")

        model_name = f"enhanced_ppo_trial_{i+1}"

        try:
            result = train_with_hyperparameters(
                timesteps=timesteps,
                model_name=model_name,
                data_path=data_path,
                max_proteins=max_proteins,
                **config,
                max_episode_length=50,
                early_stopping_patience=10,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                n_envs=4,
                eval_freq=10000,
                verbose=1,
            )

            # Calculate composite score
            metrics = result["final_metrics"]
            score = (
                metrics.get("mean_total_reward", 0) * 0.4
                + metrics.get("mean_hit_rate", 0) * 0.3
                + metrics.get("mean_avg_diversity", 0) * 0.2
                + metrics.get("mean_avg_novelty", 0) * 0.1
            )

            result["composite_score"] = score
            all_results.append(result)

            if score > best_score:
                best_score = score
                best_result = result

            print(f"Trial {i+1} score: {score:.4f}")

        except Exception as e:
            print(f"Trial {i+1} failed: {e}")
            continue

    # Save optimization results
    optimization_results = {
        "best_result": best_result,
        "all_results": all_results,
        "best_score": best_score,
    }

    with open("logs/hyperparameter_optimization_results.json", "w") as f:
        json.dump(optimization_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Optimization Complete")
    print(f"{'='*60}")
    print(f"Best score: {best_score:.4f}")
    print(f"Best config: {best_result['hyperparameters'] if best_result else 'None'}")

    return optimization_results


def plot_training_results(model_names: list[str], log_dir: str = "./logs/") -> None:
    """Plot training results for comparison."""
    try:
        # Plot training curves
        # plot_results(
        #     [f"{log_dir}/{name}" for name in model_names],
        #     timesteps_to_plot=1e6,
        #     title="Enhanced PPO Training Results",
        #     xlabel="Timesteps",
        #     ylabel="Episode Reward",
        # )
        # plt.savefig("logs/enhanced_training_curves.png", dpi=300, bbox_inches="tight")
        # plt.close()

        print("Training curves plotting is not directly supported by SBX.")

    except Exception as e:
        print(f"Could not plot training results: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train enhanced PPO agent with multi-objective rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Train with default settings
            python examples/train_enhanced_ppo_agent.py
            
            # Train with custom parameters
            python examples/train_enhanced_ppo_agent.py --max-proteins 50 --timesteps 1000000
            
            # Run hyperparameter optimization
            python examples/train_enhanced_ppo_agent.py --optimize --trials 10
            
            # Train with specific weights
            python examples/train_enhanced_ppo_agent.py --diversity-weight 0.4 --novelty-weight 0.3
        """,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default="protein_inputs/processed/unified_protein_dataset.csv",
        help="Path to unified protein dataset",
    )

    parser.add_argument(
        "--max-proteins",
        type=int,
        default=100,
        help="Maximum number of proteins to use",
    )

    parser.add_argument(
        "--timesteps", type=int, default=500000, help="Number of training timesteps"
    )

    parser.add_argument(
        "--diversity-weight",
        type=float,
        default=0.3,
        help="Weight for diversity reward component",
    )

    parser.add_argument(
        "--novelty-weight",
        type=float,
        default=0.2,
        help="Weight for novelty reward component",
    )

    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.1,
        help="Weight for cost penalty component",
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for diversity constraint",
    )

    parser.add_argument(
        "--max-episode-length", type=int, default=50, help="Maximum episode length"
    )

    parser.add_argument(
        "--optimize", action="store_true", help="Run hyperparameter optimization"
    )

    parser.add_argument(
        "--trials", type=int, default=5, help="Number of optimization trials"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    if args.optimize:
        # Run hyperparameter optimization
        optimization_results = hyperparameter_optimization(
            data_path=args.data_path,
            max_proteins=args.max_proteins,
            timesteps=args.timesteps,
            n_trials=args.trials,
        )

        # Train final model with best parameters
        if optimization_results["best_result"]:
            best_config = optimization_results["best_result"]["hyperparameters"]
            print("\nTraining final model with best parameters...")

            train_with_hyperparameters(
                timesteps=args.timesteps * 2,  # Longer training for final model
                model_name="enhanced_ppo_final",
                data_path=args.data_path,
                max_proteins=args.max_proteins,
                **best_config,
                max_episode_length=args.max_episode_length,
                early_stopping_patience=10,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                n_envs=4,
                eval_freq=10000,
                verbose=1,
            )
    else:
        # Single training run
        train_with_hyperparameters(
            timesteps=args.timesteps,
            model_name="enhanced_ppo_single",
            data_path=args.data_path,
            max_proteins=args.max_proteins,
            diversity_weight=args.diversity_weight,
            novelty_weight=args.novelty_weight,
            cost_weight=args.cost_weight,
            similarity_threshold=args.similarity_threshold,
            max_episode_length=args.max_episode_length,
            early_stopping_patience=10,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            n_envs=4,
            eval_freq=10000,
            verbose=1,
        )

    print(f"\n{'='*60}")
    print("Enhanced PPO Training Complete!")
    print(f"{'='*60}")
    print("Check logs/ directory for detailed results")
    print("Check models/ directory for saved models")


if __name__ == "__main__":
    main()
