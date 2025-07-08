"""
Protein database loader for the API.

Loads pre-processed protein data and provides fast lookup capabilities.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ProteinDatabase:
    """Loads and manages protein data for fast API access."""
    
    def __init__(self, data_path: str = "protein_inputs/processed/unified_protein_dataset.csv"):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None
        self.protein_index: Dict[str, int] = {}
        self.features: Optional[np.ndarray] = None
        self.targets: Optional[np.ndarray] = None
        self.experimental_data: Dict[str, Dict] = {}
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load protein data from CSV file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Protein database not found: {self.data_path}")
        
        logger.info(f"Loading protein database from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        # Create index for fast lookup
        for idx, uniprot_id in enumerate(self.df['uniprot_id']):
            self.protein_index[uniprot_id] = idx
        
        # Extract features (ESM embeddings)
        feature_cols = [col for col in self.df.columns if col.startswith('f')]
        self.features = self.df[feature_cols].values.astype(np.float32)
        
        # Extract targets (activity labels)
        self.targets = self.df['has_activity'].values.astype(np.int32)
        
        # Load experimental data
        self._load_experimental_data()
        
        logger.info(f"Loaded {len(self.df)} proteins with {self.features.shape[1]} features")
    
    def _load_experimental_data(self) -> None:
        """Load experimental data for proteins."""
        exp_data_path = Path("protein_inputs/processed/chembl_experimental_data.csv")
        if exp_data_path.exists():
            logger.info("Loading experimental data")
            exp_df = pd.read_csv(exp_data_path)
            
            for _, row in exp_df.iterrows():
                uniprot_id = row['uniprot_id']
                if uniprot_id in self.protein_index:
                    self.experimental_data[uniprot_id] = {
                        'activity_count': row.get('activity_count', 0),
                        'pchembl_mean': row.get('pchembl_mean', 0.0),
                        'pchembl_std': row.get('pchembl_std', 0.0),
                        'binding_affinity_kd': row.get('binding_affinity_kd', None),
                    }
    
    def get_protein_features(self, uniprot_id: str) -> Optional[np.ndarray]:
        """Get protein features by uniprot_id."""
        if uniprot_id not in self.protein_index:
            return None
        
        idx = self.protein_index[uniprot_id]
        return self.features[idx]
    
    def get_protein_target(self, uniprot_id: str) -> Optional[int]:
        """Get protein activity target by uniprot_id."""
        if uniprot_id not in self.protein_index:
            return None
        
        idx = self.protein_index[uniprot_id]
        return self.targets[idx]
    
    def get_experimental_data(self, uniprot_id: str) -> Optional[Dict]:
        """Get experimental data for protein."""
        return self.experimental_data.get(uniprot_id)
    
    def get_proteins_batch(self, uniprot_ids: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Get features and targets for a batch of proteins."""
        valid_ids = []
        features_list = []
        targets_list = []
        
        for uniprot_id in uniprot_ids:
            features = self.get_protein_features(uniprot_id)
            target = self.get_protein_target(uniprot_id)
            
            if features is not None and target is not None:
                valid_ids.append(uniprot_id)
                features_list.append(features)
                targets_list.append(target)
        
        if not features_list:
            return np.array([]), np.array([]), []
        
        return np.array(features_list), np.array(targets_list), valid_ids
    
    def get_all_proteins(self) -> List[str]:
        """Get list of all available protein IDs."""
        return list(self.protein_index.keys())
    
    def protein_exists(self, uniprot_id: str) -> bool:
        """Check if protein exists in database."""
        return uniprot_id in self.protein_index
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        return {
            'total_proteins': len(self.df) if self.df is not None else 0,
            'feature_dim': self.features.shape[1] if self.features is not None else 0,
            'active_proteins': int(np.sum(self.targets)) if self.targets is not None else 0,
            'experimental_data_count': len(self.experimental_data),
        }


# Global instance for API use
protein_db = ProteinDatabase() 