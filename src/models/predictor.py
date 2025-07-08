"""
Pre-trained model predictor for protein ranking.

Loads the trained RL model and provides scoring functionality.
"""

import logging
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class ProteinPredictor:
    """Loads pre-trained model and provides protein scoring."""
    
    def __init__(self, model_path: str = "models/best_model.zip"):
        self.model_path = Path(model_path)
        self.model: Optional[PPO] = None
        self.feature_dim: int = 1280  # ESM embedding dimension
        
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the pre-trained PPO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading pre-trained model from {self.model_path}")
        self.model = PPO.load(self.model_path)
        logger.info("Model loaded successfully")
    
    def score_proteins(self, features: np.ndarray) -> np.ndarray:
        """
        Score proteins using the pre-trained model.
        
        Args:
            features: Protein features array (n_proteins, feature_dim)
            
        Returns:
            Scores array (n_proteins,)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        if features.shape[1] != self.feature_dim:
            raise ValueError(f"Expected {self.feature_dim} features, got {features.shape[1]}")
        
        # Use model to predict action probabilities
        # For ranking, we'll use the model's action probabilities as scores
        scores = []
        
        for feature in features:
            # Reshape for model input
            obs = feature.reshape(1, -1)
            
            # Get action probabilities
            action_probs = self.model.policy.get_distribution(obs).distribution.probs
            score = float(action_probs.max())  # Use max probability as score
            scores.append(score)
        
        return np.array(scores)
    
    def rank_proteins(self, features: np.ndarray, uniprot_ids: List[str]) -> List[Tuple[str, float, int]]:
        """
        Rank proteins by their predicted scores.
        
        Args:
            features: Protein features array
            uniprot_ids: List of protein IDs
            
        Returns:
            List of (uniprot_id, score, rank) tuples, sorted by score descending
        """
        if len(features) != len(uniprot_ids):
            raise ValueError("Features and uniprot_ids must have same length")
        
        # Score proteins
        scores = self.score_proteins(features)
        
        # Create ranking
        protein_scores = list(zip(uniprot_ids, scores))
        protein_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Add ranks
        ranked_proteins = []
        for rank, (uniprot_id, score) in enumerate(protein_scores, 1):
            ranked_proteins.append((uniprot_id, score, rank))
        
        return ranked_proteins
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_path": str(self.model_path),
            "model_type": "PPO",
            "feature_dim": self.feature_dim,
        }


# Global instance for API use
predictor = ProteinPredictor() 