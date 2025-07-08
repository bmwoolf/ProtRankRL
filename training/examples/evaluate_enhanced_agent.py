#!/usr/bin/env python3
"""
Enhanced Agent Evaluation for ProtRankRL

This script evaluates the enhanced multi-objective RL agent with comprehensive metrics
including diversity, novelty, cost efficiency, and activity strength.

Features:
- Multi-metric evaluation
- Comparison with baseline methods
- Detailed protein selection analysis
- Cost-benefit analysis
- Diversity and novelty assessment
"""

import argparse
import json
import logging
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

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
            logging.FileHandler("enhanced_evaluation.log"),
        ],
    )


def load_model(model_path: str) -> PPO:
    """Load a trained PPO model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    return PPO.load(model_path)


def create_evaluation_env(
    data_path: str,
    max_proteins: int = 100,
    diversity_weight: float = 0.3,
    novelty_weight: float = 0.2,
    cost_weight: float = 0.1,
    similarity_threshold: float = 0.8,
    max_episode_length: int = 50,
    early_stopping_patience: int = 10,
) -> Any:
    """Create environment for evaluation."""
    return create_experimental_env(
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


def run_random_baseline(
    env: Any, n_episodes: int = 10, seed: int = 42
) -> list[dict[str, float]]:
    """Run random baseline for comparison."""
    np.random.seed(seed)
    results = []

    for _episode in range(n_episodes):
        obs, info = env.reset()
        episode_rewards = []

        while True:
            # Random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)

            if done:
                break

        # Get episode metrics
        episode_metrics = env.get_evaluation_metrics()
        results.append(episode_metrics)

    return results


def run_greedy_baseline(
    env: Any, n_episodes: int = 10, seed: int = 42
) -> list[dict[str, float]]:
    """Run greedy baseline (always select highest activity protein)."""
    np.random.seed(seed)
    results = []

    for _episode in range(n_episodes):
        obs, info = env.reset()
        episode_rewards = []

        while True:
            # Greedy action: select protein with highest activity
            if hasattr(env, "targets") and hasattr(env, "experimental_data"):
                best_action = 0
                best_score = -1

                for i in range(env.num_proteins):
                    if i < len(env.targets):
                        # Score based on hit status and pChemBL
                        score = env.targets[i]
                        if i < len(env.experimental_data):
                            pchembl = env.experimental_data[i].get("pchembl_mean", 0)
                            score += pchembl / 10.0  # Normalize pChemBL

                        if score > best_score:
                            best_score = score
                            best_action = i

                action = best_action
            else:
                action = env.action_space.sample()

            obs, reward, done, truncated, info = env.step(action)
            episode_rewards.append(reward)

            if done:
                break

        # Get episode metrics
        episode_metrics = env.get_evaluation_metrics()
        results.append(episode_metrics)

    return results


def evaluate_model_comprehensive(
    model: PPO,
    env: Any,
    n_episodes: int = 10,
    deterministic: bool = True,
    save_trajectories: bool = True,
) -> dict[str, Any]:
    """Comprehensive model evaluation with detailed metrics."""

    all_metrics = []
    all_trajectories = []

    for _episode in range(n_episodes):
        obs, info = env.reset()
        episode_rewards = []
        episode_actions = []
        episode_info = []

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)

            episode_rewards.append(reward)
            episode_actions.append(action)
            episode_info.append(info.copy())

            if done:
                break

        # Get episode metrics
        episode_metrics = env.get_evaluation_metrics()
        all_metrics.append(episode_metrics)

        # Save trajectory if requested
        if save_trajectories:
            trajectory = {
                "episode": _episode,
                "actions": episode_actions,
                "rewards": episode_rewards,
                "info": episode_info,
                "metrics": episode_metrics,
            }
            all_trajectories.append(trajectory)

    # Aggregate metrics
    aggregated_metrics = {}
    for key in all_metrics[0].keys():
        values = [metrics[key] for metrics in all_metrics if key in metrics]
        aggregated_metrics[f"mean_{key}"] = np.mean(values)
        aggregated_metrics[f"std_{key}"] = np.std(values)
        aggregated_metrics[f"min_{key}"] = np.min(values)
        aggregated_metrics[f"max_{key}"] = np.max(values)
        aggregated_metrics[f"median_{key}"] = np.median(values)

    results = {
        "metrics": aggregated_metrics,
        "trajectories": all_trajectories if save_trajectories else None,
        "n_episodes": n_episodes,
    }

    return results


def compare_methods(
    model: PPO,
    env: Any,
    n_episodes: int = 10,
    save_results: bool = True,
) -> dict[str, Any]:
    """Compare RL agent with baseline methods."""

    print("Evaluating RL Agent...")
    rl_results = evaluate_model_comprehensive(model, env, n_episodes)

    print("Evaluating Random Baseline...")
    random_results = run_random_baseline(env, n_episodes)

    print("Evaluating Greedy Baseline...")
    greedy_results = run_greedy_baseline(env, n_episodes)

    # Aggregate baseline results
    def aggregate_baseline_results(results):
        aggregated = {}
        for key in results[0].keys():
            values = [r[key] for r in results if key in r]
            aggregated[f"mean_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)
        return aggregated

    random_metrics = aggregate_baseline_results(random_results)
    greedy_metrics = aggregate_baseline_results(greedy_results)

    comparison = {
        "rl_agent": rl_results["metrics"],
        "random_baseline": random_metrics,
        "greedy_baseline": greedy_metrics,
        "improvement_over_random": {},
        "improvement_over_greedy": {},
    }

    # Calculate improvements
    for key in rl_results["metrics"].keys():
        if key.startswith("mean_"):
            metric_name = key[5:]  # Remove "mean_" prefix

            if f"mean_{metric_name}" in random_metrics:
                random_val = random_metrics[f"mean_{metric_name}"]
                rl_val = rl_results["metrics"][f"mean_{metric_name}"]
                if random_val != 0:
                    improvement = (rl_val - random_val) / abs(random_val) * 100
                    comparison["improvement_over_random"][metric_name] = improvement

            if f"mean_{metric_name}" in greedy_metrics:
                greedy_val = greedy_metrics[f"mean_{metric_name}"]
                rl_val = rl_results["metrics"][f"mean_{metric_name}"]
                if greedy_val != 0:
                    improvement = (rl_val - greedy_val) / abs(greedy_val) * 100
                    comparison["improvement_over_greedy"][metric_name] = improvement

    if save_results:
        with open("logs/method_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)

    return comparison


def plot_evaluation_results(
    comparison_results: dict[str, Any],
    save_plots: bool = True,
) -> None:
    """Plot evaluation results and comparisons."""

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Enhanced RL Agent Evaluation Results", fontsize=16)

    # Extract metrics for plotting
    metrics_to_plot = ["total_reward", "hit_rate", "avg_diversity", "avg_novelty"]

    methods = ["RL Agent", "Random", "Greedy"]
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i // 2, i % 2]

        values = []
        errors = []

        # RL Agent
        rl_mean = comparison_results["rl_agent"].get(f"mean_{metric}", 0)
        rl_std = comparison_results["rl_agent"].get(f"std_{metric}", 0)
        values.append(rl_mean)
        errors.append(rl_std)

        # Random Baseline
        random_mean = comparison_results["random_baseline"].get(f"mean_{metric}", 0)
        random_std = comparison_results["random_baseline"].get(f"std_{metric}", 0)
        values.append(random_mean)
        errors.append(random_std)

        # Greedy Baseline
        greedy_mean = comparison_results["greedy_baseline"].get(f"mean_{metric}", 0)
        greedy_std = comparison_results["greedy_baseline"].get(f"std_{metric}", 0)
        values.append(greedy_mean)
        errors.append(greedy_std)

        # Create bar plot
        bars = ax.bar(methods, values, yerr=errors, capsize=5, color=colors, alpha=0.7)
        ax.set_title(f"{metric.replace('_', ' ').title()}")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values, strict=False):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()

    if save_plots:
        plt.savefig("logs/evaluation_results.png", dpi=300, bbox_inches="tight")
        print("Evaluation plots saved to logs/evaluation_results.png")

    plt.show()


def analyze_protein_selections(
    model: PPO,
    env: Any,
    n_episodes: int = 5,
    save_analysis: bool = True,
) -> dict[str, Any]:
    """Analyze protein selection patterns and characteristics."""

    print("Analyzing protein selection patterns...")

    all_selections = []
    selection_frequencies = {}

    for _episode in range(n_episodes):
        obs, info = env.reset()
        episode_selections = []

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

            episode_selections.append(
                {
                    "protein_idx": action,
                    "reward": reward,
                    "was_hit": info.get("was_hit", False),
                    "diversity_score": info.get("diversity_score", 0),
                    "novelty_score": info.get("novelty_score", 0),
                    "cost_penalty": info.get("cost_penalty", 0),
                }
            )

            # Track selection frequency
            selection_frequencies[action] = selection_frequencies.get(action, 0) + 1

            if done:
                break

        all_selections.append(episode_selections)

    # Analyze selection patterns
    analysis = {
        "total_selections": (
            len(all_selections) * len(all_selections[0]) if all_selections else 0
        ),
        "unique_proteins_selected": len(selection_frequencies),
        "selection_frequencies": selection_frequencies,
        "most_frequently_selected": sorted(
            selection_frequencies.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "average_reward_per_selection": np.mean(
            [s["reward"] for episode in all_selections for s in episode]
        ),
        "hit_rate": np.mean(
            [s["was_hit"] for episode in all_selections for s in episode]
        ),
        "average_diversity_score": np.mean(
            [s["diversity_score"] for episode in all_selections for s in episode]
        ),
        "average_novelty_score": np.mean(
            [s["novelty_score"] for episode in all_selections for s in episode]
        ),
        "average_cost_penalty": np.mean(
            [s["cost_penalty"] for episode in all_selections for s in episode]
        ),
    }

    if save_analysis:
        with open("logs/protein_selection_analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)

    return analysis


def print_evaluation_summary(comparison_results: dict[str, Any]) -> None:
    """Print a comprehensive evaluation summary."""

    print(f"\n{'='*80}")
    print("ENHANCED RL AGENT EVALUATION SUMMARY")
    print(f"{'='*80}")

    # RL Agent Performance
    print("\n🤖 RL AGENT PERFORMANCE:")
    rl_metrics = comparison_results["rl_agent"]
    print(
        f"  Total Reward: {rl_metrics.get('mean_total_reward', 0):.3f} ± {rl_metrics.get('std_total_reward', 0):.3f}"
    )
    print(
        f"  Hit Rate: {rl_metrics.get('mean_hit_rate', 0):.3f} ± {rl_metrics.get('std_hit_rate', 0):.3f}"
    )
    print(
        f"  Average Diversity: {rl_metrics.get('mean_avg_diversity', 0):.3f} ± {rl_metrics.get('std_avg_diversity', 0):.3f}"
    )
    print(
        f"  Average Novelty: {rl_metrics.get('mean_avg_novelty', 0):.3f} ± {rl_metrics.get('std_avg_novelty', 0):.3f}"
    )
    print(
        f"  Activity Strength: {rl_metrics.get('mean_avg_activity_strength', 0):.3f} ± {rl_metrics.get('std_avg_activity_strength', 0):.3f}"
    )

    # Improvements over baselines
    print("\n📈 IMPROVEMENTS OVER BASELINES:")
    improvements_random = comparison_results["improvement_over_random"]
    improvements_greedy = comparison_results["improvement_over_greedy"]

    for metric, improvement in improvements_random.items():
        print(f"  vs Random - {metric.replace('_', ' ').title()}: {improvement:+.1f}%")

    for metric, improvement in improvements_greedy.items():
        print(f"  vs Greedy - {metric.replace('_', ' ').title()}: {improvement:+.1f}%")

    print(f"\n{'='*80}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate enhanced multi-objective RL agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default settings
  python examples/evaluate_enhanced_agent.py --model-path models/enhanced_ppo_final
  
  # Evaluate with custom parameters
  python examples/evaluate_enhanced_agent.py --model-path models/enhanced_ppo_final --max-proteins 50 --episodes 20
  
  # Run comprehensive analysis
  python examples/evaluate_enhanced_agent.py --model-path models/enhanced_ppo_final --comprehensive
        """,
    )

    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to trained model"
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
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
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
        "--comprehensive",
        action="store_true",
        help="Run comprehensive analysis including protein selection patterns",
    )

    parser.add_argument(
        "--save-plots", action="store_true", help="Save evaluation plots"
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
    os.makedirs("logs", exist_ok=True)

    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path)

    print("Creating evaluation environment...")
    env = create_evaluation_env(
        data_path=args.data_path,
        max_proteins=args.max_proteins,
        diversity_weight=args.diversity_weight,
        novelty_weight=args.novelty_weight,
        cost_weight=args.cost_weight,
        similarity_threshold=args.similarity_threshold,
        max_episode_length=50,
        early_stopping_patience=10,
    )

    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    comparison_results = compare_methods(
        model=model,
        env=env,
        n_episodes=args.episodes,
        save_results=True,
    )

    # Print summary
    print_evaluation_summary(comparison_results)

    # Run additional analysis if requested
    if args.comprehensive:
        print("\nRunning comprehensive analysis...")
        selection_analysis = analyze_protein_selections(
            model=model,
            env=env,
            n_episodes=5,
            save_analysis=True,
        )

        print("Protein Selection Analysis:")
        print(f"  Total selections: {selection_analysis['total_selections']}")
        print(f"  Unique proteins: {selection_analysis['unique_proteins_selected']}")
        print(f"  Hit rate: {selection_analysis['hit_rate']:.3f}")
        print(
            f"  Average diversity: {selection_analysis['average_diversity_score']:.3f}"
        )
        print(f"  Average novelty: {selection_analysis['average_novelty_score']:.3f}")

    # Generate plots
    if args.save_plots:
        print("\nGenerating evaluation plots...")
        plot_evaluation_results(comparison_results, save_plots=True)

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print("Check logs/ directory for detailed results")
    print("Results saved to:")
    print("  - logs/method_comparison.json")
    if args.comprehensive:
        print("  - logs/protein_selection_analysis.json")
    if args.save_plots:
        print("  - logs/evaluation_results.png")


if __name__ == "__main__":
    main()
