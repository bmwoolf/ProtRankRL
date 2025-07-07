#!/usr/bin/env python3
"""
Download protein-ligand binding data from UniProt + BindingDB.
This script fetches protein sequences from UniProt and binding affinities from BindingDB,
then merges them to create a comprehensive dataset for ProtRankRL.

BindingDB API docs: https://www.bindingdb.org/rwd/bind/BindingDBRESTfulAPI.jsp
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import requests
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BINDINGDB_BATCH_SIZE = 5  # Reduced from 10 to avoid timeouts
BINDINGDB_AFFINITY_CUTOFF = 10000  # nM
BINDINGDB_API = "https://bindingdb.org/rest/getLigandsByUniprots"


def download_uniprot_proteins(protein_ids: List[str], max_proteins: int = 100) -> pd.DataFrame:
    print(f"Downloading up to {max_proteins} proteins from UniProt...")
    base_url = "https://rest.uniprot.org/uniprotkb"
    protein_data = []
    protein_ids = protein_ids[:max_proteins]
    for i, protein_id in enumerate(protein_ids):
        try:
            print(f"Downloading {protein_id} ({i+1}/{len(protein_ids)})...")
            response = requests.get(f"{base_url}/{protein_id}", timeout=30)
            if response.status_code == 200:
                data = response.json()
                sequence = data.get('sequence', {}).get('value', '')
                if sequence and len(sequence) > 20:
                    protein_info = {
                        'uniprot_id': protein_id,
                        'protein_id': protein_id,
                        'pref_name': data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'Unknown'),
                        'organism': data.get('organism', {}).get('scientificName', 'Unknown'),
                        'sequence': sequence,
                        'length': len(sequence),
                        'entry_name': data.get('entryName', ''),
                        'gene_names': extract_gene_names(data),
                        'protein_families': extract_protein_families(data)
                    }
                    protein_data.append(protein_info)
                    time.sleep(0.1)
            else:
                print(f"Failed to download {protein_id}: {response.status_code}")
        except Exception as e:
            print(f"Error downloading {protein_id}: {e}")
            continue
    print(f"Successfully downloaded {len(protein_data)} proteins from UniProt")
    return pd.DataFrame(protein_data)

def extract_gene_names(data: Dict) -> str:
    gene_names = []
    if 'genes' in data:
        for gene in data['genes']:
            if 'geneName' in gene:
                gene_names.append(gene['geneName']['value'])
    return '; '.join(gene_names) if gene_names else 'Unknown'

def extract_protein_families(data: Dict) -> str:
    families = []
    if 'dbReferences' in data:
        for ref in data['dbReferences']:
            if ref.get('type') == 'Pfam':
                families.append(ref.get('id', ''))
    return '; '.join(families) if families else 'Unknown'

def download_bindingdb_data_by_uniprot(uniprot_ids: List[str], batch_size: int = BINDINGDB_BATCH_SIZE, affinity_cutoff: int = BINDINGDB_AFFINITY_CUTOFF) -> pd.DataFrame:
    print(f"Querying BindingDB for {len(uniprot_ids)} UniProt IDs in batches of {batch_size}...")
    all_results = []
    for i in range(0, len(uniprot_ids), batch_size):
        batch = uniprot_ids[i:i+batch_size]
        ids_str = ','.join(batch)
        # Remove cutoff and response parameters
        url = f"{BINDINGDB_API}?uniprot={ids_str}"
        print(f"  Querying batch {i//batch_size+1}: {url}")
        try:
            response = requests.get(url, timeout=120, headers={"User-Agent": "ProtRankRL/1.0"})
            print(f"  RAW RESPONSE: {response.text[:1000]}\n---END RAW---")
            with open(f"bindingdb_raw_response_{ids_str}.json", "w") as f:
                f.write(response.text)
            if response.status_code == 200 and response.text.strip():
                try:
                    data = response.json()
                    print(f"  DEBUG: Response type: {type(data)}")
                    print(f"  DEBUG: Response keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
                    
                    # Handle nested BindingDB response structure
                    if isinstance(data, dict):
                        # Check for the main response key
                        if 'getLindsByUniprotsResponse' in data:
                            binding_data = data['getLindsByUniprotsResponse']
                            print(f"  DEBUG: getLindsByUniprotsResponse type: {type(binding_data)}")
                            print(f"  DEBUG: getLindsByUniprotsResponse keys: {list(binding_data.keys()) if isinstance(binding_data, dict) else 'Not a dict'}")
                            
                            if isinstance(binding_data, list):
                                print(f"  DEBUG: List length: {len(binding_data)}")
                                if len(binding_data) > 0:
                                    print(f"  DEBUG: First item keys: {list(binding_data[0].keys())}")
                                all_results.extend(binding_data)
                            elif isinstance(binding_data, dict):
                                print(f"  DEBUG: Dict keys: {list(binding_data.keys())}")
                                all_results.append(binding_data)
                        else:
                            # Direct response without wrapper
                            print(f"  DEBUG: Direct response keys: {list(data.keys())}")
                            all_results.append(data)
                    elif isinstance(data, list):
                        print(f"  DEBUG: Direct list response, length: {len(data)}")
                        if len(data) > 0:
                            print(f"  DEBUG: First item keys: {list(data[0].keys())}")
                        all_results.extend(data)
                    else:
                        print(f"  Unexpected data type: {type(data)}")
                        continue
                    
                    # Count results for this batch
                    batch_results = 0
                    if isinstance(data, dict) and 'getLindsByUniprotsResponse' in data:
                        binding_data = data['getLindsByUniprotsResponse']
                        batch_results = len(binding_data) if isinstance(binding_data, list) else 1
                    elif isinstance(data, list):
                        batch_results = len(data)
                    else:
                        batch_results = 1
                    print(f"  Batch {i//batch_size+1}: Got {batch_results} results")
                    
                except Exception as e:
                    print(f"  Error parsing JSON: {e}")
                    print(f"  Response text: {response.text[:500]}...")
            else:
                print(f"  No data or error for batch {i//batch_size+1}: {response.status_code}")
                if response.status_code != 200:
                    print(f"  Response text: {response.text[:500]}...")
        except Exception as e:
            print(f"  Error querying BindingDB: {e}")
        time.sleep(1.0)  # Increased delay between requests
    
    print(f"Downloaded {len(all_results)} binding measurements from BindingDB.")
    if len(all_results) > 0:
        print(f"DEBUG: Final results columns: {list(all_results[0].keys())}")
        print(f"DEBUG: Sample result: {json.dumps(all_results[0], indent=2)[:500]}...")
    
    return pd.DataFrame(all_results)

def merge_uniprot_bindingdb_data(uniprot_df: pd.DataFrame, bindingdb_df: pd.DataFrame) -> pd.DataFrame:
    print("Merging UniProt and BindingDB data...")
    
    # Check if BindingDB data is empty
    if len(bindingdb_df) == 0:
        print("No BindingDB data available. Creating synthetic binding data...")
        # Create synthetic binding data for testing
        synthetic_data = []
        for _, row in uniprot_df.iterrows():
            # Generate synthetic binding data
            is_hit = np.random.choice([True, False], p=[0.3, 0.7])  # 30% hit rate
            if is_hit:
                affinity_nm = np.random.exponential(100)  # Exponential distribution
            else:
                affinity_nm = np.random.uniform(1000, 10000)
            
            synthetic_data.append({
                'uniprot_id': row['uniprot_id'],
                'protein_id': row['protein_id'],
                'pref_name': row['pref_name'],
                'organism': row['organism'],
                'sequence': row['sequence'],
                'length': row['length'],
                'gene_names': row['gene_names'],
                'protein_families': row['protein_families'],
                'is_hit': is_hit,
                'binding_affinity_kd': affinity_nm / 1e9,  # Convert to M
                'functional_activity': float(is_hit),
                'toxicity_score': np.random.uniform(0, 0.5),
                'expression_level': np.random.uniform(0.5, 2.0),
                'total_assays': 1,
                'hit_count': int(is_hit),
                'hit_rate': float(is_hit),
                'avg_affinity_nm': affinity_nm,
                'min_affinity_nm': affinity_nm,
                'max_affinity_nm': affinity_nm,
                'std_affinity_nm': 0.0,
                'molecular_weight': row['length'] * 110,
                'avg_hydrophobicity': calculate_hydrophobicity(row['sequence']),
                'avg_charge': calculate_charge(row['sequence']),
                'avg_polarity': calculate_polarity(row['sequence'])
            })
        
        merged_df = pd.DataFrame(synthetic_data)
        print(f"Created synthetic data for {len(merged_df)} proteins.")
        return merged_df
    
    # Extract real binding data from BindingDB response
    print("Extracting real binding data from BindingDB response...")
    all_affinities = []
    
    for _, bindingdb_row in bindingdb_df.iterrows():
        if 'affinities' in bindingdb_row and isinstance(bindingdb_row['affinities'], list):
            for affinity_data in bindingdb_row['affinities']:
                all_affinities.append(affinity_data)
    
    print(f"Extracted {len(all_affinities)} individual binding measurements")
    
    if len(all_affinities) == 0:
        print("No affinity data found. Using synthetic data.")
        return merge_uniprot_bindingdb_data(uniprot_df, pd.DataFrame())
    
    # Convert to DataFrame for easier processing
    affinities_df = pd.DataFrame(all_affinities)
    print(f"Affinities DataFrame columns: {affinities_df.columns.tolist()}")
    print(f"Sample affinity data:\n{affinities_df.head()}")
    
    # Match proteins by name (query field in BindingDB vs pref_name in UniProt)
    merged_data = []
    
    for _, uniprot_row in uniprot_df.iterrows():
        protein_name = uniprot_row['pref_name'].lower()
        
        # Find matching binding data
        matching_affinities = []
        for _, affinity_row in affinities_df.iterrows():
            query_name = affinity_row['query'].lower()
            
            # Simple name matching (can be improved with fuzzy matching)
            if (protein_name in query_name or query_name in protein_name or 
                any(word in query_name for word in protein_name.split()) or
                any(word in protein_name for word in query_name.split())):
                matching_affinities.append(affinity_row)
        
        if matching_affinities:
            print(f"Found {len(matching_affinities)} binding measurements for {uniprot_row['pref_name']}")
            
            # Calculate aggregate statistics
            affinities = []
            for aff in matching_affinities:
                try:
                    affinity_val = float(aff['affinity'])
                    affinities.append(affinity_val)
                except (ValueError, TypeError):
                    continue
            
            if affinities:
                avg_affinity = np.mean(affinities)
                min_affinity = np.min(affinities)
                max_affinity = np.max(affinities)
                std_affinity = np.std(affinities)
                
                # Determine hit status (affinity < 1000 nM)
                is_hit = min_affinity < 1000
                hit_rate = sum(1 for a in affinities if a < 1000) / len(affinities)
                
                # Determine binding affinity type and convert to Kd if needed
                binding_affinity_kd = None
                for aff in matching_affinities:
                    if aff['affinity_type'] == 'Kd':
                        try:
                            binding_affinity_kd = float(aff['affinity']) / 1e9  # Convert nM to M
                            break
                        except (ValueError, TypeError):
                            continue
                
                # If no Kd, use the minimum affinity as approximation
                if binding_affinity_kd is None:
                    binding_affinity_kd = min_affinity / 1e9
                
                merged_data.append({
                    'uniprot_id': uniprot_row['uniprot_id'],
                    'protein_id': uniprot_row['protein_id'],
                    'pref_name': uniprot_row['pref_name'],
                    'organism': uniprot_row['organism'],
                    'sequence': uniprot_row['sequence'],
                    'length': uniprot_row['length'],
                    'gene_names': uniprot_row['gene_names'],
                    'protein_families': uniprot_row['protein_families'],
                    'is_hit': is_hit,
                    'binding_affinity_kd': binding_affinity_kd,
                    'functional_activity': hit_rate,
                    'toxicity_score': 0.0,  # Not available in BindingDB
                    'expression_level': 1.0,  # Not available in BindingDB
                    'total_assays': len(matching_affinities),
                    'hit_count': sum(1 for a in affinities if a < 1000),
                    'hit_rate': hit_rate,
                    'avg_affinity_nm': avg_affinity,
                    'min_affinity_nm': min_affinity,
                    'max_affinity_nm': max_affinity,
                    'std_affinity_nm': std_affinity,
                    'molecular_weight': uniprot_row['length'] * 110,
                    'avg_hydrophobicity': calculate_hydrophobicity(uniprot_row['sequence']),
                    'avg_charge': calculate_charge(uniprot_row['sequence']),
                    'avg_polarity': calculate_polarity(uniprot_row['sequence'])
                })
            else:
                print(f"No valid affinity values for {uniprot_row['pref_name']}")
        else:
            print(f"No binding data found for {uniprot_row['pref_name']}")
    
    if len(merged_data) == 0:
        print("No successful matches. Using synthetic data.")
        return merge_uniprot_bindingdb_data(uniprot_df, pd.DataFrame())
    
    merged_df = pd.DataFrame(merged_data)
    print(f"Merged data contains {len(merged_df)} proteins with real binding data.")
    return merged_df

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

def get_sample_uniprot_ids() -> List[str]:
    # Focused test: Only use Prothrombin (P00734)
    return ['P00734']

def save_data(data_df: pd.DataFrame, output_dir: str = "protein_inputs/processed") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    csv_path = output_path / "uniprot_bindingdb_data.csv"
    data_df.to_csv(csv_path, index=False)
    print(f"Saved to {csv_path}")
    json_path = output_path / "uniprot_bindingdb_data.json"
    json_data = data_df.to_dict('records')
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved to {json_path}")
    fasta_path = Path("protein_inputs/raw/uniprot_bindingdb_proteins.fasta")
    with open(fasta_path, 'w') as f:
        for _, row in data_df.iterrows():
            f.write(f">{row['protein_id']}|{row['pref_name']}|{row['organism']}\n")
            f.write(f"{row['sequence']}\n")
    print(f"Saved sequences to {fasta_path}")

def main():
    print("UniProt + BindingDB Data Downloader")
    print("=" * 50)
    try:
        uniprot_ids = get_sample_uniprot_ids()
        print(f"Using {len(uniprot_ids)} sample UniProt IDs")
        uniprot_df = download_uniprot_proteins(uniprot_ids, max_proteins=50)
        if len(uniprot_df) == 0:
            print("No proteins downloaded from UniProt. Exiting.")
            return
        bindingdb_df = download_bindingdb_data_by_uniprot(uniprot_df['uniprot_id'].tolist())
        if len(bindingdb_df) == 0:
            print("No binding data found in BindingDB for these UniProt IDs.")
            return
        merged_df = merge_uniprot_bindingdb_data(uniprot_df, bindingdb_df)
        save_data(merged_df)
        print("\nDataset Summary:")
        print(f"Total proteins/assays: {len(merged_df)}")
        print(f"Proteins with hits: {merged_df['is_hit'].sum()}")
        print(f"Average hit rate: {merged_df['hit_rate'].mean():.3f}")
        print(f"Average affinity: {merged_df['avg_affinity_nm'].mean():.1f} nM")
        print(f"Average length: {merged_df['length'].mean():.1f} aa")
        print("\nSample proteins:")
        for _, row in merged_df.head(5).iterrows():
            print(f"  {row['protein_id']}: {row['pref_name']} ({row['organism']})")
            print(f"    Length: {row['length']} aa, Hit: {row['is_hit']}, Hit rate: {row['hit_rate']:.2f}")
    except Exception as e:
        print(f"Error in main process: {e}")
        print("Consider using synthetic data for testing.")

if __name__ == "__main__":
    main() 