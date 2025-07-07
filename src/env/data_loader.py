"""
Data loader for real experimental protein data.
Handles loading and processing of experimental data from UniProt+BindingDB and other sources.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

class ExperimentalDataLoader:
    """Loader for real experimental protein data."""
    
    def __init__(self, data_dir: str = "protein_inputs"):
        self.data_dir = Path(data_dir)
        
    def load_uniprot_bindingdb_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load experimental data from UniProt+BindingDB.
        Args:
            file_path: Path to data file. If None, looks for default files.
        Returns:
            DataFrame with experimental data
        """
        if file_path is None:
            possible_files = [
                self.data_dir / "processed" / "uniprot_bindingdb_data.csv",
                self.data_dir / "processed" / "uniprot_bindingdb_data.json"
            ]
            for file_path in possible_files:
                if file_path.exists():
                    break
            else:
                raise FileNotFoundError(
                    "No UniProt+BindingDB data files found. Run scripts/download_uniprot_bindingdb_data.py first."
                )
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = pd.DataFrame(json.load(f))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        logger.info(f"Loaded {len(data)} proteins from {file_path}")
        return data
    
    def load_legacy_data(self, file_path: str) -> pd.DataFrame:
        file_path = Path(file_path)
        if file_path.suffix == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix == '.json':
            with open(file_path, 'r') as f:
                data = pd.DataFrame(json.load(f))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        logger.info(f"Loaded {len(data)} proteins from legacy file {file_path}")
        return data
    
    def prepare_environment_data(
        self, 
        data_df: pd.DataFrame,
        target_column: str = 'is_hit',
        feature_columns: Optional[List[str]] = None,
        normalize_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        if target_column not in data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")
        targets = data_df[target_column].astype(int).values
        if feature_columns is None:
            feature_columns = [
                'length', 'molecular_weight', 'avg_hydrophobicity', 
                'avg_charge', 'avg_polarity', 'hit_rate', 'avg_affinity_nm'
            ]
        available_features = [col for col in feature_columns if col in data_df.columns]
        missing_features = [col for col in feature_columns if col not in data_df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
        if not available_features:
            raise ValueError("No valid feature columns found")
        features = data_df[available_features].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)
        if normalize_features:
            features = self._normalize_features(features)
        logger.info(f"Prepared {len(features)} samples with {features.shape[1]} features")
        logger.info(f"Target distribution: {np.bincount(targets)}")
        return features, targets
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        features_min = np.min(features, axis=0, keepdims=True)
        features_max = np.max(features, axis=0, keepdims=True)
        features_range = features_max - features_min
        features_range[features_range == 0] = 1.0
        normalized = 2.0 * (features - features_min) / features_range - 1.0
        return normalized.astype(np.float32)
    
    def get_data_summary(self, data_df: pd.DataFrame) -> Dict[str, Union[int, float]]:
        summary = {
            'total_proteins': len(data_df),
            'proteins_with_hits': data_df['is_hit'].sum() if 'is_hit' in data_df.columns else 0,
            'hit_rate': data_df['is_hit'].mean() if 'is_hit' in data_df.columns else 0.0,
        }
        if 'avg_affinity_nm' in data_df.columns:
            summary['avg_affinity_nm'] = data_df['avg_affinity_nm'].mean()
            summary['min_affinity_nm'] = data_df['avg_affinity_nm'].min()
            summary['max_affinity_nm'] = data_df['avg_affinity_nm'].max()
        if 'hit_rate' in data_df.columns:
            summary['avg_hit_rate'] = data_df['hit_rate'].mean()
        if 'length' in data_df.columns:
            summary['avg_length'] = data_df['length'].mean()
            summary['min_length'] = data_df['length'].min()
            summary['max_length'] = data_df['length'].max()
        return summary

def load_experimental_data(
    data_source: str = "uniprot_bindingdb",
    data_path: Optional[str] = None,
    target_column: str = 'is_hit',
    feature_columns: Optional[List[str]] = None,
    normalize_features: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Union[int, float]]]:
    loader = ExperimentalDataLoader()
    if data_source == "uniprot_bindingdb":
        data_df = loader.load_uniprot_bindingdb_data(data_path)
    elif data_source == "legacy":
        if data_path is None:
            data_path = "protein_inputs/SHRT_experimental_labels.csv"
        data_df = loader.load_legacy_data(data_path)
    else:
        data_df = loader.load_legacy_data(data_source)
    features, targets = loader.prepare_environment_data(
        data_df, target_column, feature_columns, normalize_features
    )
    summary_stats = loader.get_data_summary(data_df)
    return features, targets, summary_stats 