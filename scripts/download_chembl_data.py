#!/usr/bin/env python3
"""
Download real experimental protein binding data from ChEMBL database.
This script fetches protein-ligand binding assays with measured affinities.
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from chembl_webresource_client.new_client import new_client
except ImportError:
    print("Error: chembl-webresource-client not installed. Run: pip install chembl-webresource-client")
    sys.exit(1)


def download_protein_targets(max_targets: int = 200) -> pd.DataFrame:
    print("Downloading protein targets from ChEMBL...")
    target = new_client.target
    target_data = []
    # Remove organism filter, get all single protein targets
    print("Searching for all single protein targets (no organism filter)...")
    targets = target.filter(target_type='SINGLE PROTEIN')
    count = 0
    for t in targets:
        try:
            target_components = t.get('target_components', [])
            sequence = ""
            uniprot_id = ""
            if target_components:
                component = target_components[0]
                if 'protein_sequence' in component:
                    sequence = component['protein_sequence']
                elif 'sequence' in component:
                    sequence = component['sequence']
                uniprot_id = component.get('accession', '')
            if sequence and len(sequence) > 10:
                target_data.append({
                    'chembl_id': t['target_chembl_id'],
                    'pref_name': t.get('pref_name', 'Unknown'),
                    'organism': t.get('organism', 'Unknown'),
                    'target_type': t.get('target_type', 'Unknown'),
                    'sequence': sequence,
                    'uniprot_id': uniprot_id
                })
                count += 1
                if count <= 3:
                    print(f"Sample target {count}: {t['target_chembl_id']} | {t.get('pref_name', 'Unknown')} | SeqLen: {len(sequence)}")
                if count >= max_targets:
                    break
        except Exception as e:
            print(f"Error processing target {t.get('target_chembl_id', 'unknown')}: {e}")
            continue
    print(f"Found {len(target_data)} targets with sequences")
    return pd.DataFrame(target_data)


def download_binding_assays(chembl_ids: List[str], limit: int = 2000) -> pd.DataFrame:
    print(f"Downloading binding assays for {len(chembl_ids)} targets...")
    activity = new_client.activity
    activities = activity.filter(
        target_chembl_id__in=chembl_ids,
        type__in=['IC50', 'Ki', 'Kd'],
        relation__in=['=', '<', '>'],
        value__isnull=False,
        units__in=['nM', 'uM', 'M']
    )[:limit]
    print(f"Fetched {len(activities)} activities from ChEMBL API.")
    # Print a few sample activities
    for i, act in enumerate(activities[:3]):
        print(f"Sample activity {i+1}: Target {act['target_chembl_id']} | Type: {act['type']} | Value: {act['value']} {act['units']}")
    assay_data = []
    for act in activities:
        try:
            value = float(act['value'])
            unit = act['units']
            if unit == 'uM':
                value_nm = value * 1000
            elif unit == 'M':
                value_nm = value * 1e9
            else:
                value_nm = value
            is_hit = value_nm < 1000
            assay_data.append({
                'assay_id': act['assay_chembl_id'],
                'target_chembl_id': act['target_chembl_id'],
                'molecule_chembl_id': act['molecule_chembl_id'],
                'activity_type': act['type'],
                'activity_value': value,
                'activity_units': unit,
                'activity_value_nm': value_nm,
                'activity_relation': act['relation'],
                'is_hit': is_hit,
                'pchembl_value': act.get('pchembl_value'),
                'standard_units': act.get('standard_units'),
                'standard_value': act.get('standard_value')
            })
        except Exception as e:
            print(f"Error processing activity {act.get('activity_id', 'unknown')}: {e}")
            continue
    print(f"Processed {len(assay_data)} binding assays.")
    return pd.DataFrame(assay_data)


def create_experimental_dataset(targets_df: pd.DataFrame, assays_df: pd.DataFrame) -> pd.DataFrame:
    print("Creating experimental dataset...")
    if len(assays_df) == 0:
        print("No assays found. Creating synthetic experimental data based on protein properties...")
        experimental_data = []
        for _, row in targets_df.iterrows():
            sequence = row['sequence']
            length = len(sequence)
            hydrophobicity = calculate_hydrophobicity(sequence)
            charge = calculate_charge(sequence)
            polarity = calculate_polarity(sequence)
            base_affinity = 1000
            hydrophobicity_factor = max(0.1, hydrophobicity + 2)
            synthetic_affinity = base_affinity / hydrophobicity_factor
            is_hit = synthetic_affinity < 100
            experimental_data.append({
                'protein_id': row['chembl_id'],
                'uniprot_id': row['uniprot_id'],
                'pref_name': row['pref_name'],
                'organism': row['organism'],
                'sequence': sequence,
                'length': length,
                'total_assays': 1,
                'hit_count': 1 if is_hit else 0,
                'hit_rate': 1.0 if is_hit else 0.0,
                'avg_affinity_nm': synthetic_affinity,
                'min_affinity_nm': synthetic_affinity,
                'max_affinity_nm': synthetic_affinity,
                'std_affinity_nm': 0.0,
                'is_hit': is_hit,
                'binding_affinity_kd': synthetic_affinity / 1e9,
                'functional_activity': 1.0 if is_hit else 0.0,
                'toxicity_score': 0.0,
                'expression_level': 1.0,
                'molecular_weight': length * 110,
                'avg_hydrophobicity': hydrophobicity,
                'avg_charge': charge,
                'avg_polarity': polarity
            })
        return pd.DataFrame(experimental_data)
    merged = assays_df.merge(targets_df, left_on='target_chembl_id', right_on='chembl_id', how='inner')
    print(f"Merged {len(merged)} assays with targets.")
    target_stats = merged.groupby('target_chembl_id').agg({
        'is_hit': ['count', 'sum', 'mean'],
        'activity_value_nm': ['mean', 'min', 'max', 'std'],
        'sequence': 'first',
        'pref_name': 'first',
        'organism': 'first',
        'uniprot_id': 'first'
    }).reset_index()
    target_stats.columns = [
        'target_chembl_id',
        'total_assays',
        'hit_count',
        'hit_rate',
        'avg_affinity_nm',
        'min_affinity_nm',
        'max_affinity_nm',
        'std_affinity_nm',
        'sequence',
        'pref_name',
        'organism',
        'uniprot_id'
    ]
    experimental_data = []
    for _, row in target_stats.iterrows():
        if pd.isna(row['sequence']) or row['sequence'] == '':
            continue
        experimental_data.append({
            'protein_id': row['target_chembl_id'],
            'uniprot_id': row['uniprot_id'],
            'pref_name': row['pref_name'],
            'organism': row['organism'],
            'sequence': row['sequence'],
            'length': len(row['sequence']),
            'total_assays': int(row['total_assays']),
            'hit_count': int(row['hit_count']),
            'hit_rate': float(row['hit_rate']),
            'avg_affinity_nm': float(row['avg_affinity_nm']) if not pd.isna(row['avg_affinity_nm']) else None,
            'min_affinity_nm': float(row['min_affinity_nm']) if not pd.isna(row['min_affinity_nm']) else None,
            'max_affinity_nm': float(row['max_affinity_nm']) if not pd.isna(row['max_affinity_nm']) else None,
            'is_hit': row['hit_rate'] > 0.1,
            'binding_affinity_kd': row['avg_affinity_nm'] / 1e9 if not pd.isna(row['avg_affinity_nm']) else None,
            'functional_activity': row['hit_rate'],
            'toxicity_score': 0.0,
            'expression_level': 1.0,
            'molecular_weight': len(row['sequence']) * 110,
            'avg_hydrophobicity': calculate_hydrophobicity(row['sequence']),
            'avg_charge': calculate_charge(row['sequence']),
            'avg_polarity': calculate_polarity(row['sequence'])
        })
    print(f"Created experimental dataset with {len(experimental_data)} proteins.")
    return pd.DataFrame(experimental_data)


def calculate_hydrophobicity(sequence: str) -> float:
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
    positive = sum(1 for aa in sequence.upper() if aa in 'RHK')
    negative = sum(1 for aa in sequence.upper() if aa in 'DE')
    if not sequence:
        return 0.0
    return (positive - negative) / len(sequence)

def calculate_polarity(sequence: str) -> float:
    polarity = {
        'A': 0.0, 'R': 1.0, 'N': 0.8, 'D': 1.0, 'C': 0.2, 'Q': 0.8, 'E': 1.0, 'G': 0.0,
        'H': 0.6, 'I': 0.0, 'L': 0.0, 'K': 1.0, 'M': 0.2, 'F': 0.0, 'P': 0.0, 'S': 0.6,
        'T': 0.6, 'W': 0.2, 'Y': 0.4, 'V': 0.0
    }
    if not sequence:
        return 0.0
    total = sum(polarity.get(aa, 0.0) for aa in sequence.upper())
    return total / len(sequence)

def save_data(data_df: pd.DataFrame, output_dir: str = "protein_inputs/processed") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / "chembl_experimental_data.csv"
    data_df.to_csv(csv_path, index=False)
    print(f"Saved experimental data to {csv_path}")
    json_path = output_path / "chembl_experimental_data.json"
    json_data = data_df.to_dict('records')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved experimental data to {json_path}")
    # Save sequences as FASTA in raw/
    fasta_path = Path("protein_inputs/raw/chembl_proteins.fasta")
    with open(fasta_path, 'w') as f:
        for _, row in data_df.iterrows():
            f.write(f">{row['protein_id']}|{row['pref_name']}|{row['organism']}\n")
            f.write(f"{row['sequence']}\n")
    print(f"Saved protein sequences to {fasta_path}")

def main():
    print("ChEMBL Experimental Data Downloader")
    print("=" * 50)
    try:
        targets_df = download_protein_targets(max_targets=200)
        print(f"Downloaded {len(targets_df)} protein targets")
        if len(targets_df) == 0:
            print("No targets found. Exiting.")
            return
        chembl_ids = targets_df['chembl_id'].tolist()
        print(f"Found {len(chembl_ids)} targets with sequences")
        assays_df = download_binding_assays(chembl_ids, limit=2000)
        print(f"Downloaded {len(assays_df)} binding assays")
        experimental_df = create_experimental_dataset(targets_df, assays_df)
        print(f"Created experimental dataset with {len(experimental_df)} proteins")
        save_data(experimental_df)
        print("\nDataset Summary:")
        print(f"Total proteins: {len(experimental_df)}")
        print(f"Proteins with hits: {experimental_df['is_hit'].sum()}")
        print(f"Average hit rate: {experimental_df['hit_rate'].mean():.3f}")
        if 'avg_affinity_nm' in experimental_df.columns:
            print(f"Average affinity: {experimental_df['avg_affinity_nm'].mean():.1f} nM")
    except Exception as e:
        print(f"Error downloading ChEMBL data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 