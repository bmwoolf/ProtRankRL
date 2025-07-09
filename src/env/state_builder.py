"""
State Builder for RL Environment

The state builder is the foundation: it defines what the RL agent "sees." 
It builds feature vectors for each protein by concatenating ESM-2 embeddings and 
experimental/metadata features.

Components:

1. State Builder (this module): Converts protein data into 1283-dim feature vectors
2. Environment: Manages episodes, state transitions, and action spaces
3. Policy Network: PPO neural network that maps states to action probabilities
4. Reward Function: Multi-objective rewards based on activity strength, diversity, novelty
5. Loss Function: PPO loss with policy/value/entropy components
Training Loop: Orchestrates all components with hyperparameter optimization

The current system was trained on 64-dim features, but this state builder creates 1283-dim vectors.
This requires retraining the RL agent to work with the new feature dimensionality.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict


class ProteinStateBuilder:
    def __init__(
        self,
        unified_path: str = "data/processed/unified_protein_dataset.csv",
        esm_dim: int = 1280,
        extra_features: Optional[List[str]] = None,
    ):
        self.unified_path = Path(unified_path)
        self.esm_dim = esm_dim
        self.extra_features = extra_features or [
            "pchembl_mean",
            "pchembl_std",
            "activity_count",
        ]
        self.df = None
        self.feature_cols = None
        self._load()

    def _load(self):
        if not self.unified_path.exists():
            raise FileNotFoundError(f"Unified dataset not found: {self.unified_path}")
        self.df = pd.read_csv(self.unified_path)
        # ESM embedding columns: f0, f1, ..., f1279
        self.feature_cols = [f"f{i}" for i in range(self.esm_dim)]
        # Ensure all extra features exist
        for feat in self.extra_features:
            if feat not in self.df.columns:
                print(f"[WARN] Feature '{feat}' not found in dataset. Filling with zeros.")
                self.df[feat] = 0.0

    def get_feature_vector(self, uniprot_id: str) -> Optional[np.ndarray]:
        row = self.df[self.df["uniprot_id"] == uniprot_id]
        if row.empty:
            return None
        row = row.iloc[0]
        esm = row[self.feature_cols].values.astype(np.float32)
        extras = row[self.extra_features].values.astype(np.float32)
        return np.concatenate([esm, extras])

    def get_all_feature_vectors(self) -> Dict[str, np.ndarray]:
        """Return a dict of uniprot_id -> feature vector."""
        features = {}
        for _, row in self.df.iterrows():
            uniprot_id = row["uniprot_id"]
            esm = row[self.feature_cols].values.astype(np.float32)
            extras = row[self.extra_features].values.astype(np.float32)
            features[uniprot_id] = np.concatenate([esm, extras])
        return features

    def get_feature_dim(self) -> int:
        return self.esm_dim + len(self.extra_features)


# Example usage (for testing):
if __name__ == "__main__":
    builder = ProteinStateBuilder()
    vec = builder.get_feature_vector("O43451")
    print(f"Feature vector shape: {vec.shape}")
    print(f"First 10 values: {vec[:10]}")
    print(f"Feature dim: {builder.get_feature_dim()}") 