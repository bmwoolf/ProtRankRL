import pandas as pd
import numpy as np
import json
from pathlib import Path

# Configuration
EMBEDDINGS_CSV = 'protein_inputs/embeddings/validated_proteins_esm_embeddings.csv'
CHEMBL_CSV = 'protein_inputs/processed/chembl_experimental_data.csv'
OUTPUT_DIR = Path('protein_inputs/processed')
OUTPUT_CSV = OUTPUT_DIR / 'unified_protein_dataset.csv'
OUTPUT_JSON = OUTPUT_DIR / 'unified_protein_dataset.json'

def load_embeddings():
    """Load protein embeddings from CSV."""
    print("Loading protein embeddings...")
    embeddings_df = pd.read_csv(EMBEDDINGS_CSV)
    print(f"Loaded embeddings for {len(embeddings_df)} proteins")
    return embeddings_df

def load_chembl_data():
    """Load ChEMBL experimental data."""
    print("Loading ChEMBL experimental data...")
    chembl_df = pd.read_csv(CHEMBL_CSV)
    print(f"Loaded {len(chembl_df)} experimental activities")
    return chembl_df

def aggregate_activities(chembl_df):
    """Aggregate experimental activities per protein."""
    print("Aggregating experimental activities per protein...")
    
    # Group by protein and calculate activity statistics
    activity_stats = chembl_df.groupby('uniprot_id').agg({
        'pchembl_value': ['count', 'mean', 'std', 'min', 'max'],
        'standard_value': ['mean', 'std', 'min', 'max'],
        'standard_type': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(3)
    
    # Flatten column names
    activity_stats.columns = [
        'activity_count', 'pchembl_mean', 'pchembl_std', 'pchembl_min', 'pchembl_max',
        'standard_value_mean', 'standard_value_std', 'standard_value_min', 'standard_value_max',
        'most_common_activity_type'
    ]
    
    activity_stats = activity_stats.reset_index()
    print(f"Aggregated activities for {len(activity_stats)} proteins")
    return activity_stats

def create_unified_dataset(embeddings_df, activity_stats):
    """Create unified dataset combining embeddings and activities."""
    print("Creating unified dataset...")
    
    # Merge embeddings with activity statistics
    unified_df = embeddings_df.merge(activity_stats, left_on='id', right_on='uniprot_id', how='left')
    
    # Fill missing activity data with zeros
    activity_columns = ['activity_count', 'pchembl_mean', 'pchembl_std', 'pchembl_min', 'pchembl_max',
                       'standard_value_mean', 'standard_value_std', 'standard_value_min', 'standard_value_max']
    unified_df[activity_columns] = unified_df[activity_columns].fillna(0)
    unified_df['most_common_activity_type'] = unified_df['most_common_activity_type'].fillna('No_data')
    
    # Create a binary "has_activity" column
    unified_df['has_activity'] = (unified_df['activity_count'] > 0).astype(int)
    
    # Reorder columns for clarity
    embedding_cols = [col for col in unified_df.columns if col.startswith('f')]
    activity_cols = ['activity_count', 'has_activity', 'pchembl_mean', 'pchembl_std', 'pchembl_min', 'pchembl_max',
                    'standard_value_mean', 'standard_value_std', 'standard_value_min', 'standard_value_max',
                    'most_common_activity_type']
    
    final_columns = ['id', 'uniprot_id'] + activity_cols + embedding_cols
    unified_df = unified_df[final_columns]
    
    print(f"Created unified dataset with {len(unified_df)} proteins")
    print(f"Features: {len(embedding_cols)} embedding dimensions + {len(activity_cols)} activity features")
    
    return unified_df

def save_dataset(unified_df):
    """Save the unified dataset in multiple formats."""
    print("Saving unified dataset...")
    
    # Save as CSV
    unified_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved CSV: {OUTPUT_CSV}")
    
    # Save as JSON for programmatic access
    dataset_dict = {
        'metadata': {
            'n_proteins': len(unified_df),
            'n_embedding_features': len([col for col in unified_df.columns if col.startswith('f')]),
            'n_activity_features': len([col for col in unified_df.columns if not col.startswith('f') and col not in ['id', 'uniprot_id']]),
            'proteins_with_activities': int(unified_df['has_activity'].sum()),
            'total_activities': int(unified_df['activity_count'].sum())
        },
        'proteins': unified_df.to_dict('records')
    }
    
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset_dict, f, indent=2)
    print(f"Saved JSON: {OUTPUT_JSON}")

def print_summary(unified_df):
    """Print summary statistics of the unified dataset."""
    print("\n" + "="*50)
    print("UNIFIED DATASET SUMMARY")
    print("="*50)
    
    print(f"Total proteins: {len(unified_df)}")
    print(f"Proteins with activities: {unified_df['has_activity'].sum()}")
    print(f"Proteins without activities: {(unified_df['has_activity'] == 0).sum()}")
    print(f"Total experimental activities: {unified_df['activity_count'].sum():,}")
    
    print(f"\nActivity distribution:")
    print(f"  Mean activities per protein: {unified_df['activity_count'].mean():.1f}")
    print(f"  Median activities per protein: {unified_df['activity_count'].median():.1f}")
    print(f"  Max activities per protein: {unified_df['activity_count'].max()}")
    
    print(f"\nEmbedding features: {len([col for col in unified_df.columns if col.startswith('f')])}")
    print(f"Activity features: {len([col for col in unified_df.columns if not col.startswith('f') and col not in ['id', 'uniprot_id']])}")
    
    print(f"\nMost common activity types:")
    activity_types = unified_df['most_common_activity_type'].value_counts()
    for activity_type, count in activity_types.head(5).items():
        print(f"  {activity_type}: {count}")

def main():
    """Main function to create unified dataset."""
    print("Creating unified protein dataset for RL environment...")
    
    # Load data
    embeddings_df = load_embeddings()
    chembl_df = load_chembl_data()
    
    # Process data
    activity_stats = aggregate_activities(chembl_df)
    unified_df = create_unified_dataset(embeddings_df, activity_stats)
    
    # Save dataset
    save_dataset(unified_df)
    
    # Print summary
    print_summary(unified_df)
    
    print(f"\n✅ Unified dataset created successfully!")
    print(f"📁 Files saved to: {OUTPUT_DIR}")
    print(f"🚀 Ready for RL environment integration!")

if __name__ == "__main__":
    main() 