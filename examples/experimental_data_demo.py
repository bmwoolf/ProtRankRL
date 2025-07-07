#!/usr/bin/env python3
"""Demo of ProtRankRL with real experimental data from ChEMBL."""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.env import create_experimental_env, create_synthetic_env

console = Console()


def run_synthetic_comparison():
    """Compare synthetic vs experimental data environments."""
    console.print(Panel.fit("Synthetic vs Experimental Data Comparison", style="bold blue"))
    
    # Create synthetic environment
    console.print("\n[bold]Synthetic Environment:[/bold]")
    synthetic_env = create_synthetic_env(
        num_proteins=20,
        feature_dim=32,
        hit_rate=0.25,
        seed=42
    )
    
    console.print(f"  Proteins: {synthetic_env.num_proteins}")
    console.print(f"  Features: {synthetic_env.feature_dim}")
    console.print(f"  Hits: {np.sum(synthetic_env.targets)}/{synthetic_env.num_proteins}")
    
    # Run synthetic episode
    obs, info = synthetic_env.reset()
    total_reward = 0
    hits_found = 0
    
    while True:
        action = synthetic_env.action_space.sample()
        obs, reward, done, truncated, info = synthetic_env.step(action)
        
        total_reward += reward
        hits_found += int(reward)
        
        if done:
            break
    
    console.print(f"  Total reward: {total_reward:.1f}")
    console.print(f"  Hits found: {hits_found}")
    console.print(f"  Hit rate: {hits_found/synthetic_env.num_proteins:.2f}")


def run_experimental_demo():
    """Demo with real experimental data."""
    console.print("\n[bold]Experimental Data Demo:[/bold]")
    
    try:
        # Try to create experimental environment
        experimental_env = create_experimental_env(
            data_source="chembl",
            reward_type="multi_objective",
            max_proteins=20  # Limit for demo
        )
        
        console.print(f"  Proteins: {experimental_env.num_proteins}")
        console.print(f"  Features: {experimental_env.feature_dim}")
        console.print(f"  Hits: {np.sum(experimental_env.targets)}/{experimental_env.num_proteins}")
        console.print(f"  Reward type: {experimental_env.reward_type}")
        
        # Run experimental episode
        obs, info = experimental_env.reset()
        total_reward = 0
        hits_found = 0
        protein_details = []
        
        while True:
            action = experimental_env.action_space.sample()
            obs, reward, done, truncated, info = experimental_env.step(action)
            
            total_reward += reward
            hits_found += int(reward)
            
            # Collect protein details
            if 'protein_id' in info:
                protein_details.append({
                    'protein_id': info.get('protein_id', 'Unknown'),
                    'pref_name': info.get('pref_name', 'Unknown'),
                    'was_hit': info['was_hit'],
                    'reward': reward,
                    'affinity': info.get('binding_affinity_kd'),
                    'activity': info.get('functional_activity'),
                    'toxicity': info.get('toxicity_score')
                })
            
            if done:
                break
        
        console.print(f"  Total reward: {total_reward:.1f}")
        console.print(f"  Hits found: {hits_found}")
        console.print(f"  Hit rate: {hits_found/experimental_env.num_proteins:.2f}")
        
        # Show protein details table
        if protein_details:
            table = Table(title="Selected Proteins")
            table.add_column("Protein ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Hit", style="red")
            table.add_column("Reward", style="yellow")
            table.add_column("Affinity (M)", style="blue")
            table.add_column("Activity", style="magenta")
            
            for protein in protein_details[:10]:  # Show first 10
                table.add_row(
                    protein['protein_id'][:10] + "...",
                    protein['pref_name'][:15] + "..." if len(protein['pref_name']) > 15 else protein['pref_name'],
                    "✓" if protein['was_hit'] else "✗",
                    f"{protein['reward']:.2f}",
                    f"{protein['affinity']:.2e}" if protein['affinity'] else "N/A",
                    f"{protein['activity']:.2f}" if protein['activity'] else "N/A"
                )
            
            console.print(table)
        
    except FileNotFoundError:
        console.print("[red]No ChEMBL data found. Run scripts/download_chembl_data.py first.[/red]")
        
        # Fallback to legacy data
        console.print("\n[bold]Trying legacy data...[/bold]")
        try:
            legacy_env = create_experimental_env(
                data_source="legacy",
                data_path="protein_inputs/SHRT_experimental_labels.csv",
                reward_type="binary",
                max_proteins=5
            )
            
            console.print(f"  Proteins: {legacy_env.num_proteins}")
            console.print(f"  Features: {legacy_env.feature_dim}")
            console.print(f"  Hits: {np.sum(legacy_env.targets)}/{legacy_env.num_proteins}")
            
        except Exception as e:
            console.print(f"[red]Error loading legacy data: {e}[/red]")


def run_reward_comparison():
    """Compare different reward types."""
    console.print(Panel.fit("Reward Type Comparison", style="bold green"))
    
    try:
        # Create environments with different reward types
        reward_types = ["binary", "affinity_based", "multi_objective"]
        results = {}
        
        for reward_type in reward_types:
            console.print(f"\n[bold]{reward_type.upper()} rewards:[/bold]")
            
            try:
                env = create_experimental_env(
                    data_source="chembl",
                    reward_type=reward_type,
                    max_proteins=10
                )
                
                # Run multiple episodes
                episode_rewards = []
                for episode in range(5):
                    obs, info = env.reset()
                    total_reward = 0
                    
                    while True:
                        action = env.action_space.sample()
                        obs, reward, done, truncated, info = env.step(action)
                        total_reward += reward
                        
                        if done:
                            break
                    
                    episode_rewards.append(total_reward)
                
                avg_reward = np.mean(episode_rewards)
                std_reward = np.std(episode_rewards)
                
                console.print(f"  Average reward: {avg_reward:.2f} ± {std_reward:.2f}")
                results[reward_type] = avg_reward
                
            except Exception as e:
                console.print(f"  [red]Error: {e}[/red]")
                results[reward_type] = None
        
        # Summary table
        if any(v is not None for v in results.values()):
            table = Table(title="Reward Type Performance")
            table.add_column("Reward Type", style="cyan")
            table.add_column("Average Reward", style="green")
            
            for reward_type, avg_reward in results.items():
                if avg_reward is not None:
                    table.add_row(reward_type, f"{avg_reward:.2f}")
                else:
                    table.add_row(reward_type, "Error")
            
            console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error in reward comparison: {e}[/red]")


def main():
    """Main demo function."""
    console.print("ProtRankRL Experimental Data Demo", style="bold blue")
    console.print("=" * 50)
    
    # Run synthetic comparison
    run_synthetic_comparison()
    
    # Run experimental demo
    run_experimental_demo()
    
    # Run reward comparison
    run_reward_comparison()
    
    console.print("\n[bold green]Demo complete![/bold green]")
    console.print("\nTo get real experimental data, run:")
    console.print("  python scripts/download_chembl_data.py")


if __name__ == "__main__":
    main() 