"""
Main protein ranker for the API.

Orchestrates the ranking process using the protein database and predictor.
"""

import logging
import time
from typing import Dict, List, Optional
import numpy as np

from ..data.protein_db import protein_db
from ..models.predictor import predictor

logger = logging.getLogger(__name__)


class ProteinRanker:
    """Main ranker that orchestrates protein ranking."""
    
    def __init__(self):
        self.db = protein_db
        self.predictor = predictor
        logger.info("ProteinRanker initialized")
    
    def rank(self, uniprot_ids: List[str]) -> Dict:
        """
        Rank proteins by their predicted activity scores.
        
        Args:
            uniprot_ids: List of UniProt IDs to rank
            
        Returns:
            Dictionary with rankings and metadata
        """
        start_time = time.time()
        
        # Validate input
        if not uniprot_ids:
            return {
                "rankings": [],
                "metadata": {
                    "total_proteins": 0,
                    "valid_proteins": 0,
                    "processing_time": 0.0,
                    "error": "No protein IDs provided"
                }
            }
        
        # Get protein features and targets
        features, targets, valid_ids = self.db.get_proteins_batch(uniprot_ids)
        
        if len(valid_ids) == 0:
            return {
                "rankings": [],
                "metadata": {
                    "total_proteins": len(uniprot_ids),
                    "valid_proteins": 0,
                    "processing_time": time.time() - start_time,
                    "error": "No valid proteins found in database"
                }
            }
        
        # Rank proteins
        ranked_proteins = self.predictor.rank_proteins(features, valid_ids)
        
        # Format results
        rankings = []
        for uniprot_id, score, rank in ranked_proteins:
            # Get additional data
            target = self.db.get_protein_target(uniprot_id)
            exp_data = self.db.get_experimental_data(uniprot_id)
            
            ranking_entry = {
                "uniprot_id": uniprot_id,
                "rank": rank,
                "score": float(score),
                "confidence": float(score),  # Use score as confidence for now
                "has_activity": bool(target) if target is not None else None,
                "experimental_data": exp_data
            }
            rankings.append(ranking_entry)
        
        processing_time = time.time() - start_time
        
        return {
            "rankings": rankings,
            "metadata": {
                "total_proteins": len(uniprot_ids),
                "valid_proteins": len(valid_ids),
                "processing_time": processing_time,
                "model_version": "v1.0",
                "database_stats": self.db.get_database_stats()
            }
        }
    
    def get_health_status(self) -> Dict:
        """Get health status of the ranker."""
        try:
            db_stats = self.db.get_database_stats()
            model_info = self.predictor.get_model_info()
            
            return {
                "status": "healthy",
                "database": db_stats,
                "model": model_info,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }


# Global instance for API use
ranker = ProteinRanker() 