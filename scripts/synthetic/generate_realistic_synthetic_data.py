#!/usr/bin/env python3
"""
Generate realistic synthetic protein data for ProtRankRL testing.
Creates diverse proteins with realistic hit rates and properties.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def generate_protein_sequence(length: int) -> str:
    """Generate a realistic protein sequence."""
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return ''.join(np.random.choice(list(amino_acids), length))


def calculate_hydrophobicity(sequence: str) -> float:
    """Calculate average hydrophobicity using Kyte-Doolittle scale."""
    hydrophobicity = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
        'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
        'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    if not sequence:
        return 0.0
    total = sum(hydrophobicity.get(aa, 0.0) for aa in sequence.upper())
    return total / len(sequence)


def calculate_charge(sequence: str) -> float:
    """Calculate net charge per residue."""
    positive = sum(1 for aa in sequence.upper() if aa in 'RHK')
    negative = sum(1 for aa in sequence.upper() if aa in 'DE')
    if not sequence:
        return 0.0
    return (positive - negative) / len(sequence)


def calculate_polarity(sequence: str) -> float:
    """Calculate average polarity."""
    polarity = {
        'A': 0.0, 'R': 1.0, 'N': 0.8, 'D': 1.0, 'C': 0.2, 'Q': 0.8, 'E': 1.0, 'G': 0.0,
        'H': 0.6, 'I': 0.0, 'L': 0.0, 'K': 1.0, 'M': 0.2, 'F': 0.0, 'P': 0.0, 'S': 0.6,
        'T': 0.6, 'W': 0.2, 'Y': 0.4, 'V': 0.0
    }
    if not sequence:
        return 0.0
    total = sum(polarity.get(aa, 0.0) for aa in sequence.upper())
    return total / len(sequence)


def generate_realistic_protein_data(
    num_proteins: int = 100,
    hit_rate: float = 0.25,
    min_length: int = 50,
    max_length: int = 500
) -> pd.DataFrame:
    """
    Generate realistic synthetic protein data.
    
    Args:
        num_proteins: Number of proteins to generate
        hit_rate: Fraction of proteins that should be hits (0.0-1.0)
        min_length: Minimum protein length
        max_length: Maximum protein length
        
    Returns:
        DataFrame with synthetic protein data
    """
    print(f"Generating {num_proteins} proteins with {hit_rate:.1%} hit rate...")
    
    # Determine which proteins will be hits
    num_hits = int(num_proteins * hit_rate)
    hit_indices = np.random.choice(num_proteins, num_hits, replace=False)
    
    protein_data = []
    
    for i in range(num_proteins):
        # Generate random protein length
        length = np.random.randint(min_length, max_length + 1)
        
        # Generate sequence
        sequence = generate_protein_sequence(length)
        
        # Calculate basic properties
        hydrophobicity = calculate_hydrophobicity(sequence)
        charge = calculate_charge(sequence)
        polarity = calculate_polarity(sequence)
        molecular_weight = length * 110  # Average amino acid weight
        
        # Determine if this protein is a hit
        is_hit = i in hit_indices
        
        # Generate realistic binding affinity based on hit status and properties
        if is_hit:
            # Hits have lower affinity (stronger binding)
            base_affinity = np.random.uniform(1, 100)  # 1-100 nM
            # Hydrophobicity affects binding - more hydrophobic = stronger binding
            hydrophobicity_bonus = max(0, hydrophobicity) * 0.5
            affinity_nm = base_affinity / (1 + hydrophobicity_bonus)
        else:
            # Non-hits have higher affinity (weaker binding)
            base_affinity = np.random.uniform(100, 10000)  # 100-10000 nM
            affinity_nm = base_affinity
        
        # Generate other properties
        hit_count = 1 if is_hit else 0
        total_assays = np.random.randint(1, 5)  # 1-4 assays per protein
        hit_rate_protein = hit_count / total_assays
        
        # Functional activity correlates with hit status
        if is_hit:
            functional_activity = np.random.uniform(0.3, 1.0)
        else:
            functional_activity = np.random.uniform(0.0, 0.3)
        
        # Toxicity (lower is better)
        toxicity_score = np.random.uniform(0.0, 0.5)
        
        # Expression level (higher is better)
        expression_level = np.random.uniform(0.5, 1.0)
        
        protein_data.append({
            'protein_id': f'SYNTH_{i+1:04d}',
            'uniprot_id': f'SYNTH{i+1:04d}',
            'pref_name': f'Synthetic Protein {i+1}',
            'organism': 'Synthetic',
            'sequence': sequence,
            'length': length,
            'total_assays': total_assays,
            'hit_count': hit_count,
            'hit_rate': hit_rate_protein,
            'avg_affinity_nm': affinity_nm,
            'min_affinity_nm': affinity_nm,
            'max_affinity_nm': affinity_nm,
            'std_affinity_nm': 0.0,
            'is_hit': is_hit,
            'binding_affinity_kd': affinity_nm / 1e9,  # Convert to M
            'functional_activity': functional_activity,
            'toxicity_score': toxicity_score,
            'expression_level': expression_level,
            'molecular_weight': molecular_weight,
            'avg_hydrophobicity': hydrophobicity,
            'avg_charge': charge,
            'avg_polarity': polarity
        })
    
    df = pd.DataFrame(protein_data)
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Generated {len(df)} proteins:")
    print(f"  Hits: {df['is_hit'].sum()}")
    print(f"  Non-hits: {(~df['is_hit']).sum()}")
    print(f"  Actual hit rate: {df['is_hit'].mean():.1%}")
    print(f"  Average affinity: {df['avg_affinity_nm'].mean():.1f} nM")
    print(f"  Average length: {df['length'].mean():.1f} aa")
    
    return df


def save_data(data_df: pd.DataFrame, output_dir: str = "protein_inputs/processed") -> None:
    """Save the synthetic data to files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save as CSV
    csv_path = output_path / "realistic_synthetic_data.csv"
    data_df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    
    # Save as JSON
    json_path = output_path / "realistic_synthetic_data.json"
    json_data = data_df.to_dict('records')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved to {json_path}")
    
    # Save sequences as FASTA
    fasta_path = Path("protein_inputs/raw/realistic_synthetic_proteins.fasta")
    with open(fasta_path, 'w') as f:
        for _, row in data_df.iterrows():
            f.write(f">{row['protein_id']}|{row['pref_name']}|{row['organism']}\n")
            f.write(f"{row['sequence']}\n")
    print(f"Saved sequences to {fasta_path}")


def main():
    """Main function to generate realistic synthetic data."""
    print("Realistic Synthetic Protein Data Generator")
    print("=" * 50)
    
    # Generate data with different hit rates for testing
    hit_rates = [0.1, 0.2, 0.3]  # 10%, 20%, 30% hit rates
    
    for hit_rate in hit_rates:
        print(f"\nGenerating dataset with {hit_rate:.1%} hit rate...")
        
        # Generate data
        data_df = generate_realistic_protein_data(
            num_proteins=50,  # Smaller dataset for testing
            hit_rate=hit_rate,
            min_length=50,
            max_length=300
        )
        
        # Save with hit rate in filename
        output_dir = f"protein_inputs/processed/hit_rate_{int(hit_rate*100)}pct"
        save_data(data_df, output_dir)
    
    # Also generate a larger dataset with 20% hit rate
    print(f"\nGenerating large dataset with 20% hit rate...")
    large_df = generate_realistic_protein_data(
        num_proteins=200,
        hit_rate=0.2,
        min_length=50,
        max_length=500
    )
    save_data(large_df, "protein_inputs/processed")
    
    print("\nData generation complete!")
    print("\nAvailable datasets:")
    print("  - protein_inputs/processed/realistic_synthetic_data.csv (200 proteins, 20% hits)")
    print("  - protein_inputs/processed/hit_rate_10pct/ (50 proteins, 10% hits)")
    print("  - protein_inputs/processed/hit_rate_20pct/ (50 proteins, 20% hits)")
    print("  - protein_inputs/processed/hit_rate_30pct/ (50 proteins, 30% hits)")


if __name__ == "__main__":
    main() 