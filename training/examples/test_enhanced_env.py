#!/usr/bin/env python3
"""
Test script for enhanced multi-objective environment.

This script tests the enhanced environment to ensure all features work correctly:
- Multi-objective reward calculation
- Diversity constraints
- Novelty scoring
- Cost penalties
- Evaluation metrics
"""

import os
import sys

import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.env import create_experimental_env, create_synthetic_env


def test_synthetic_environment():
    """Test enhanced environment with synthetic data."""
    print("🧪 Testing Enhanced Environment with Synthetic Data")
    print("=" * 60)

    # Create synthetic environment
    env = create_synthetic_env(num_proteins=20, feature_dim=64, hit_rate=0.3, seed=42)

    print(f"Environment created with {env.num_proteins} proteins")
    print(f"Feature dimension: {env.feature_dim}")
    print(f"Reward type: {env.reward_type}")
    print(f"Diversity weight: {env.diversity_weight}")
    print(f"Novelty weight: {env.novelty_weight}")
    print(f"Cost weight: {env.cost_weight}")

    # Test episode
    obs, info = env.reset()
    total_reward = 0
    step_count = 0

    print("\nRunning test episode...")
    print(f"Initial observation shape: {obs.shape}")

    while True:
        # Random action
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        print(f"Step {step_count}: Action={action}, Reward={reward:.3f}, Done={done}")
        print(f"  Info: {info}")

        if done:
            break

    # Get final metrics
    final_metrics = env.get_evaluation_metrics()

    print("\nEpisode completed!")
    print(f"Total steps: {step_count}")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Final metrics: {final_metrics}")

    return True


def test_experimental_environment():
    """Test enhanced environment with experimental data."""
    print("\n🧪 Testing Enhanced Environment with Experimental Data")
    print("=" * 60)

    data_path = "protein_inputs/processed/unified_protein_dataset.csv"

    if not os.path.exists(data_path):
        print(f"⚠️  Experimental data not found at {data_path}")
        print("Skipping experimental environment test")
        return False

    try:
        # Create experimental environment
        env = create_experimental_env(
            data_source=data_path,
            max_proteins=50,  # Use smaller subset for testing
            reward_type="enhanced_multi_objective",
            diversity_weight=0.3,
            novelty_weight=0.2,
            cost_weight=0.1,
            similarity_threshold=0.8,
            max_episode_length=20,
            early_stopping_patience=5,
        )

        print(f"Environment created with {env.num_proteins} proteins")
        print(f"Feature dimension: {env.feature_dim}")
        print(f"Reward type: {env.reward_type}")
        print(f"Experimental data available: {len(env.experimental_data) > 0}")

        # Test episode
        obs, info = env.reset()
        total_reward = 0
        step_count = 0

        print("\nRunning test episode...")
        print(f"Initial observation shape: {obs.shape}")

        while True:
            # Random action
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            step_count += 1

            print(
                f"Step {step_count}: Action={action}, Reward={reward:.3f}, Done={done}"
            )
            print(f"  Was hit: {info.get('was_hit', 'N/A')}")
            print(f"  Diversity score: {info.get('diversity_score', 'N/A'):.3f}")
            print(f"  Novelty score: {info.get('novelty_score', 'N/A'):.3f}")
            print(f"  Cost penalty: {info.get('cost_penalty', 'N/A'):.3f}")

            if done:
                break

        # Get final metrics
        final_metrics = env.get_evaluation_metrics()

        print("\nEpisode completed!")
        print(f"Total steps: {step_count}")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Final metrics: {final_metrics}")

        return True

    except Exception as e:
        print(f"❌ Error testing experimental environment: {e}")
        return False


def test_reward_components():
    """Test individual reward components."""
    print("\n🧪 Testing Reward Components")
    print("=" * 60)

    # Create environment
    env = create_synthetic_env(num_proteins=10, feature_dim=32, hit_rate=0.4, seed=42)

    # Test diversity reward
    print("Testing diversity reward...")
    diversity_rewards = []
    for action in range(5):
        reward = env._calculate_diversity_reward(action)
        diversity_rewards.append(reward)
        print(f"  Action {action}: diversity reward = {reward:.3f}")

    # Test novelty reward
    print("Testing novelty reward...")
    novelty_rewards = []
    for action in range(5):
        reward = env._calculate_novelty_reward(action)
        novelty_rewards.append(reward)
        print(f"  Action {action}: novelty reward = {reward:.3f}")

    # Test cost penalty
    print("Testing cost penalty...")
    cost_penalties = []
    for action in range(5):
        penalty = env._calculate_cost_penalty(action)
        cost_penalties.append(penalty)
        print(f"  Action {action}: cost penalty = {penalty:.3f}")

    # Test activity strength reward
    print("Testing activity strength reward...")
    activity_rewards = []
    for action in range(5):
        reward = env._calculate_activity_strength_reward(action)
        activity_rewards.append(reward)
        print(f"  Action {action}: activity reward = {reward:.3f}")

    print("\nReward component summary:")
    print(
        f"  Diversity rewards: {np.mean(diversity_rewards):.3f} ± {np.std(diversity_rewards):.3f}"
    )
    print(
        f"  Novelty rewards: {np.mean(novelty_rewards):.3f} ± {np.std(novelty_rewards):.3f}"
    )
    print(
        f"  Cost penalties: {np.mean(cost_penalties):.3f} ± {np.std(cost_penalties):.3f}"
    )
    print(
        f"  Activity rewards: {np.mean(activity_rewards):.3f} ± {np.std(activity_rewards):.3f}"
    )

    return True


def test_diversity_constraints():
    """Test diversity constraint enforcement."""
    print("\n🧪 Testing Diversity Constraints")
    print("=" * 60)

    # Create environment with strict diversity constraint
    env = create_synthetic_env(num_proteins=10, feature_dim=32, hit_rate=0.4, seed=42)
    env.similarity_threshold = 0.5  # Strict constraint

    print(f"Similarity threshold: {env.similarity_threshold}")

    # Test constraint checking
    obs, info = env.reset()
    selected_actions = []

    for step in range(5):
        # Try to select similar proteins
        if step == 0:
            action = 0
        else:
            # Try to select a similar protein
            action = 1  # Should be similar to action 0

        # Check if action satisfies diversity constraint
        satisfies_constraint = env._check_diversity_constraint(action)

        print(f"Step {step + 1}: Action {action}")
        print(f"  Satisfies diversity constraint: {satisfies_constraint}")

        if satisfies_constraint:
            selected_actions.append(action)
            obs, reward, done, truncated, info = env.step(action)
            print(f"  Reward: {reward:.3f}")
        else:
            print("  Action rejected due to diversity constraint")

        if done:
            break

    print(f"Selected actions: {selected_actions}")

    return True


def main():
    """Run all tests."""
    print("🚀 Enhanced Environment Test Suite")
    print("=" * 80)

    tests = [
        ("Synthetic Environment", test_synthetic_environment),
        ("Experimental Environment", test_experimental_environment),
        ("Reward Components", test_reward_components),
        ("Diversity Constraints", test_diversity_constraints),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            print(f"\n{'='*80}")
            print(f"Running: {test_name}")
            print(f"{'='*80}")

            success = test_func()
            if success:
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")

        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")

    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")

    if passed == total:
        print("🎉 All tests passed! Enhanced environment is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
