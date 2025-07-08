"""
Tests for the ProtRankRL API.
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np

from src.api.main import app

client = TestClient(app)


class TestAPIEndpoints:
    """Test API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "ProtRankRL" in data["message"]
    
    def test_health_endpoint(self):
        """Test health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "model" in data
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "database" in data
        assert "model" in data
        assert "available_proteins" in data
    
    def test_rank_endpoint_valid_request(self):
        """Test ranking with valid protein IDs."""
        # Get some valid protein IDs from the database
        from src.data.protein_db import protein_db
        
        all_proteins = protein_db.get_all_proteins()
        if len(all_proteins) >= 3:
            test_proteins = all_proteins[:3]
            
            response = client.post("/rank", json={"uniprot_ids": test_proteins})
            assert response.status_code == 200
            
            data = response.json()
            assert "rankings" in data
            assert "metadata" in data
            assert len(data["rankings"]) == 3
            
            # Check ranking structure
            for ranking in data["rankings"]:
                assert "uniprot_id" in ranking
                assert "rank" in ranking
                assert "score" in ranking
                assert "confidence" in ranking
    
    def test_rank_endpoint_empty_request(self):
        """Test ranking with empty request."""
        response = client.post("/rank", json={"uniprot_ids": []})
        assert response.status_code == 422  # Validation error
    
    def test_rank_endpoint_invalid_proteins(self):
        """Test ranking with invalid protein IDs."""
        response = client.post("/rank", json={"uniprot_ids": ["INVALID1", "INVALID2"]})
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["rankings"]) == 0
        assert data["metadata"]["valid_proteins"] == 0


class TestRankerLogic:
    """Test the ranker logic directly."""
    
    def test_ranker_initialization(self):
        """Test ranker initialization."""
        from src.api.ranker import ranker
        
        assert ranker is not None
        assert hasattr(ranker, 'rank')
        assert hasattr(ranker, 'get_health_status')
    
    def test_ranker_health_status(self):
        """Test ranker health status."""
        from src.api.ranker import ranker
        
        health = ranker.get_health_status()
        assert "status" in health
        assert "database" in health
        assert "model" in health
    
    def test_ranker_with_valid_proteins(self):
        """Test ranker with valid protein IDs."""
        from src.api.ranker import ranker
        from src.data.protein_db import protein_db
        
        all_proteins = protein_db.get_all_proteins()
        if len(all_proteins) >= 2:
            test_proteins = all_proteins[:2]
            
            result = ranker.rank(test_proteins)
            
            assert "rankings" in result
            assert "metadata" in result
            assert len(result["rankings"]) == 2
            assert result["metadata"]["valid_proteins"] == 2


class TestDatabase:
    """Test the protein database."""
    
    def test_database_loading(self):
        """Test database loading."""
        from src.data.protein_db import protein_db
        
        stats = protein_db.get_database_stats()
        assert stats["total_proteins"] > 0
        assert stats["feature_dim"] > 0
    
    def test_protein_lookup(self):
        """Test protein lookup functionality."""
        from src.data.protein_db import protein_db
        
        all_proteins = protein_db.get_all_proteins()
        if all_proteins:
            test_protein = all_proteins[0]
            
            features = protein_db.get_protein_features(test_protein)
            target = protein_db.get_protein_target(test_protein)
            
            assert features is not None
            assert target is not None
            assert features.shape[0] > 0
    
    def test_batch_lookup(self):
        """Test batch protein lookup."""
        from src.data.protein_db import protein_db
        
        all_proteins = protein_db.get_all_proteins()
        if len(all_proteins) >= 3:
            test_proteins = all_proteins[:3]
            
            features, targets, valid_ids = protein_db.get_proteins_batch(test_proteins)
            
            assert len(features) == 3
            assert len(targets) == 3
            assert len(valid_ids) == 3
            assert features.shape[1] > 0


class TestPredictor:
    """Test the model predictor."""
    
    def test_predictor_loading(self):
        """Test predictor loading."""
        from src.models.predictor import predictor
        
        info = predictor.get_model_info()
        assert info["status"] == "loaded"
        assert "model_path" in info
    
    def test_protein_scoring(self):
        """Test protein scoring."""
        from src.models.predictor import predictor
        from src.data.protein_db import protein_db
        
        all_proteins = protein_db.get_all_proteins()
        if all_proteins:
            test_protein = all_proteins[0]
            features = protein_db.get_protein_features(test_protein)
            
            if features is not None:
                scores = predictor.score_proteins(features.reshape(1, -1))
                assert len(scores) == 1
                assert scores[0] >= 0
    
    def test_protein_ranking(self):
        """Test protein ranking."""
        from src.models.predictor import predictor
        from src.data.protein_db import protein_db
        
        all_proteins = protein_db.get_all_proteins()
        if len(all_proteins) >= 3:
            test_proteins = all_proteins[:3]
            features_list = []
            
            for protein in test_proteins:
                features = protein_db.get_protein_features(protein)
                if features is not None:
                    features_list.append(features)
            
            if features_list:
                features = np.array(features_list)
                rankings = predictor.rank_proteins(features, test_proteins[:len(features_list)])
                
                assert len(rankings) == len(features_list)
                for uniprot_id, score, rank in rankings:
                    assert rank >= 1
                    assert score >= 0 