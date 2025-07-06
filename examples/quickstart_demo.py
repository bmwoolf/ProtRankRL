#!/usr/bin/env python3
"""Quickstart demo for ProtRankRL - single episode run."""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from src.env import create_synthetic_env


def main():
    print("ProtRankRL Quickstart Demo")
    print("=" * 40)
    
    # Create synthetic environment
    env = create_synthetic_env(
        num_proteins=16,
        feature_dim=32,
        hit_rate=0.25,
        seed=42
    )
    
    print(f"Environment: {env.num_proteins} proteins, {env.feature_dim} features")
    print(f"Target hits: {np.sum(env.targets)}/{env.num_proteins}")
    print()
    
    # Run single episode
    obs, info = env.reset()
    total_reward = 0
    step_count = 0
    hits_found = 0
    
    print("Episode progress:")
    while True:
        # Random action policy
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        hits_found += int(reward)
        step_count += 1
        
        print(f"  Step {step_count}: action={action}, reward={reward}, "
              f"hit={info['was_hit']}, remaining={info['remaining']}")
        
        if done:
            break
    
    # Episode statistics
    print()
    print("Episode complete!")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Hits found: {hits_found}")
    print(f"Hit rate: {hits_found/step_count:.2f}")
    print(f"Episode length: {step_count}")


if __name__ == "__main__":
    main() 