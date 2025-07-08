"""
Comprehensive tests for the evaluation harness module.

Tests all evaluation metrics, edge cases, and visualization capabilities.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
from unittest.mock import patch, MagicMock

from src.evaluation.evaluator import (
    EvaluationConfig,
    RankingMetrics,
    DiversityMetrics,
    CostMetrics,
    NoveltyMetrics,
    EvaluationMetrics,
    ProteinEvaluator,
    create_evaluator,
)


class TestEvaluationConfig:
    """Test evaluation configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = EvaluationConfig()
        assert config.top_k_values == [5, 10, 20, 50]
        assert config.include_ndcg is True
        assert config.include_map is True
        assert config.similarity_threshold == 0.8
        assert config.save_plots is True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = EvaluationConfig(
            top_k_values=[3, 7, 15],
            include_ndcg=False,
            include_map=False,
            similarity_threshold=0.9,
            save_plots=False
        )
        assert config.top_k_values == [3, 7, 15]
        assert config.include_ndcg is False
        assert config.include_map is False
        assert config.similarity_threshold == 0.9
        assert config.save_plots is False


class TestRankingMetrics:
    """Test ranking metrics."""
    
    def test_default_metrics(self):
        """Test default ranking metrics."""
        metrics = RankingMetrics()
        assert metrics.precision_at_k == {}
        assert metrics.recall_at_k == {}
        assert metrics.f1_at_k == {}
        assert metrics.ndcg_at_k == {}
        assert metrics.map_score == 0.0
        assert metrics.auc_roc == 0.0
        assert metrics.auc_pr == 0.0
        assert metrics.total_hits == 0
        assert metrics.hits_found == 0
        assert metrics.hit_rate == 0.0
    
    def test_custom_metrics(self):
        """Test custom ranking metrics."""
        metrics = RankingMetrics(
            precision_at_k={5: 0.8, 10: 0.6},
            recall_at_k={5: 0.4, 10: 0.8},
            map_score=0.75,
            total_hits=10,
            hits_found=8,
            hit_rate=0.8
        )
        assert metrics.precision_at_k[5] == 0.8
        assert metrics.recall_at_k[10] == 0.8
        assert metrics.map_score == 0.75
        assert metrics.total_hits == 10
        assert metrics.hits_found == 8
        assert metrics.hit_rate == 0.8


class TestDiversityMetrics:
    """Test diversity metrics."""
    
    def test_default_metrics(self):
        """Test default diversity metrics."""
        metrics = DiversityMetrics()
        assert metrics.mean_similarity == 0.0
        assert metrics.diversity_score == 0.0
        assert metrics.unique_families == 0
        assert metrics.family_distribution == {}
        assert metrics.similarity_matrix is None
        assert metrics.diversity_trajectory == []
    
    def test_custom_metrics(self):
        """Test custom diversity metrics."""
        similarity_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        metrics = DiversityMetrics(
            mean_similarity=0.5,
            diversity_score=0.5,
            unique_families=2,
            similarity_matrix=similarity_matrix,
            diversity_trajectory=[1.0, 0.8, 0.6]
        )
        assert metrics.mean_similarity == 0.5
        assert metrics.diversity_score == 0.5
        assert metrics.unique_families == 2
        assert metrics.similarity_matrix is not None
        assert len(metrics.diversity_trajectory) == 3


class TestCostMetrics:
    """Test cost metrics."""
    
    def test_default_metrics(self):
        """Test default cost metrics."""
        metrics = CostMetrics()
        assert metrics.total_cost == 0.0
        assert metrics.mean_cost == 0.0
        assert metrics.cost_efficiency == 0.0
        assert metrics.cost_per_hit == 0.0
        assert metrics.cost_distribution == {}
        assert metrics.cost_vs_reward == []
    
    def test_custom_metrics(self):
        """Test custom cost metrics."""
        metrics = CostMetrics(
            total_cost=100.0,
            mean_cost=20.0,
            cost_efficiency=0.5,
            cost_per_hit=25.0,
            cost_vs_reward=[(10.0, 1.0), (20.0, 0.5)]
        )
        assert metrics.total_cost == 100.0
        assert metrics.mean_cost == 20.0
        assert metrics.cost_efficiency == 0.5
        assert metrics.cost_per_hit == 25.0
        assert len(metrics.cost_vs_reward) == 2


class TestNoveltyMetrics:
    """Test novelty metrics."""
    
    def test_default_metrics(self):
        """Test default novelty metrics."""
        metrics = NoveltyMetrics()
        assert metrics.mean_novelty == 0.0
        assert metrics.novelty_distribution == {}
        assert metrics.high_novelty_count == 0
        assert metrics.novelty_efficiency == 0.0
        assert metrics.novelty_vs_activity == []
        assert metrics.novelty_trajectory == []
    
    def test_custom_metrics(self):
        """Test custom novelty metrics."""
        metrics = NoveltyMetrics(
            mean_novelty=0.7,
            high_novelty_count=3,
            novelty_efficiency=0.35,
            novelty_vs_activity=[(0.8, 1.0), (0.6, 0.5)],
            novelty_trajectory=[0.8, 0.7, 0.6]
        )
        assert metrics.mean_novelty == 0.7
        assert metrics.high_novelty_count == 3
        assert metrics.novelty_efficiency == 0.35
        assert len(metrics.novelty_vs_activity) == 2
        assert len(metrics.novelty_trajectory) == 3


class TestEvaluationMetrics:
    """Test comprehensive evaluation metrics."""
    
    def test_default_metrics(self):
        """Test default evaluation metrics."""
        metrics = EvaluationMetrics()
        assert isinstance(metrics.ranking, RankingMetrics)
        assert isinstance(metrics.diversity, DiversityMetrics)
        assert isinstance(metrics.cost, CostMetrics)
        assert isinstance(metrics.novelty, NoveltyMetrics)
        assert metrics.overall_score == 0.0
        assert metrics.balanced_score == 0.0
        assert metrics.num_selections == 0
        assert metrics.evaluation_time == 0.0
    
    def test_custom_metrics(self):
        """Test custom evaluation metrics."""
        ranking = RankingMetrics(map_score=0.8)
        diversity = DiversityMetrics(diversity_score=0.7)
        cost = CostMetrics(cost_efficiency=0.6)
        novelty = NoveltyMetrics(mean_novelty=0.5)
        
        metrics = EvaluationMetrics(
            ranking=ranking,
            diversity=diversity,
            cost=cost,
            novelty=novelty,
            overall_score=0.7,
            balanced_score=0.65,
            num_selections=10,
            evaluation_time=1.5
        )
        
        assert metrics.ranking.map_score == 0.8
        assert metrics.diversity.diversity_score == 0.7
        assert metrics.cost.cost_efficiency == 0.6
        assert metrics.novelty.mean_novelty == 0.5
        assert metrics.overall_score == 0.7
        assert metrics.balanced_score == 0.65
        assert metrics.num_selections == 10
        assert metrics.evaluation_time == 1.5


class TestProteinEvaluator:
    """Test main evaluation engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EvaluationConfig(
            top_k_values=[3, 5],
            include_ndcg=True,
            include_map=True,
            save_plots=False
        )
        self.evaluator = ProteinEvaluator(self.config)
        
        # Test data
        self.targets = np.array([1, 0, 1, 0, 1, 0, 0, 1])  # 4 hits out of 8
        self.features = np.random.randn(8, 10)  # 8 proteins, 10 features
        self.scores = np.array([0.9, 0.3, 0.8, 0.2, 0.7, 0.1, 0.4, 0.6])
        self.selected_indices = [0, 2, 4, 7]  # All hits selected
        self.experimental_data = {
            0: {"activity_count": 5, "pchembl_std": 0.3, "protein_family": "A"},
            2: {"activity_count": 3, "pchembl_std": 0.2, "protein_family": "B"},
            4: {"activity_count": 8, "pchembl_std": 0.5, "protein_family": "A"},
            7: {"activity_count": 2, "pchembl_std": 0.1, "protein_family": "C"},
        }
    
    def test_evaluate_ranking(self):
        """Test ranking evaluation."""
        metrics = self.evaluator.evaluate_ranking(
            self.selected_indices, self.targets, self.scores, self.experimental_data
        )
        
        # Basic hit analysis
        assert metrics.total_hits == 4
        assert metrics.hits_found == 4
        assert metrics.hit_rate == 1.0
        
        # Top-K metrics
        assert metrics.precision_at_k[3] == 1.0  # All 3 selected are hits
        assert metrics.recall_at_k[3] == 0.75    # 3 out of 4 hits found
        assert metrics.precision_at_k[5] == 1.0  # All 4 selected are hits (k=5 but only 4 selections)
        assert metrics.recall_at_k[5] == 1.0     # All hits found
    
    def test_evaluate_ranking_no_hits(self):
        """Test ranking evaluation with no hits."""
        no_hit_targets = np.array([0, 0, 0, 0])
        no_hit_selections = [0, 1, 2, 3]
        
        metrics = self.evaluator.evaluate_ranking(
            no_hit_selections, no_hit_targets, None, None
        )
        
        assert metrics.total_hits == 0
        assert metrics.hits_found == 0
        assert metrics.hit_rate == 0.0
        assert metrics.precision_at_k[3] == 0.0
    
    def test_evaluate_diversity(self):
        """Test diversity evaluation."""
        metrics = self.evaluator.evaluate_diversity(
            self.selected_indices, self.features, self.experimental_data
        )
        
        assert metrics.diversity_score >= 0.0
        assert metrics.diversity_score <= 1.0
        assert metrics.mean_similarity >= -1.0
        assert metrics.mean_similarity <= 1.0
        assert metrics.similarity_matrix is not None
        assert len(metrics.diversity_trajectory) == 4
    
    def test_evaluate_diversity_single_selection(self):
        """Test diversity evaluation with single selection."""
        metrics = self.evaluator.evaluate_diversity([0], self.features, None)
        assert metrics.diversity_score == 1.0
        assert len(metrics.diversity_trajectory) == 1
    
    def test_evaluate_cost(self):
        """Test cost evaluation."""
        metrics = self.evaluator.evaluate_cost(
            self.selected_indices, self.experimental_data, self.targets
        )
        
        assert metrics.total_cost > 0.0
        assert metrics.mean_cost > 0.0
        assert metrics.cost_efficiency > 0.0
        assert metrics.cost_per_hit > 0.0
        assert len(metrics.cost_vs_reward) == 4
    
    def test_evaluate_cost_no_data(self):
        """Test cost evaluation with no experimental data."""
        metrics = self.evaluator.evaluate_cost([0, 1], None, None)
        assert metrics.total_cost == 0.0
        assert metrics.mean_cost == 0.0
    
    def test_evaluate_novelty(self):
        """Test novelty evaluation."""
        metrics = self.evaluator.evaluate_novelty(
            self.selected_indices, self.experimental_data, self.targets
        )
        
        assert metrics.mean_novelty >= 0.0
        assert metrics.mean_novelty <= 1.0
        assert metrics.high_novelty_count >= 0
        assert len(metrics.novelty_vs_activity) == 4
        assert len(metrics.novelty_trajectory) == 4
    
    def test_evaluate_novelty_no_data(self):
        """Test novelty evaluation with no experimental data."""
        metrics = self.evaluator.evaluate_novelty([0, 1], None, None)
        assert metrics.mean_novelty == self.config.novelty_baseline
    
    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation."""
        metrics = self.evaluator.evaluate(
            self.selected_indices,
            self.targets,
            self.features,
            self.scores,
            self.experimental_data
        )
        
        # Check all components are present
        assert isinstance(metrics.ranking, RankingMetrics)
        assert isinstance(metrics.diversity, DiversityMetrics)
        assert isinstance(metrics.cost, CostMetrics)
        assert isinstance(metrics.novelty, NoveltyMetrics)
        
        # Check combined scores
        assert metrics.overall_score > 0.0
        assert metrics.balanced_score > 0.0
        assert metrics.num_selections == 4
        assert metrics.evaluation_time > 0.0
    
    def test_calculate_overall_score(self):
        """Test overall score calculation."""
        ranking = RankingMetrics(map_score=0.8)
        diversity = DiversityMetrics(diversity_score=0.7)
        cost = CostMetrics(mean_cost=10.0)  # Will be converted to cost_score
        novelty = NoveltyMetrics(mean_novelty=0.5)
        
        score = self.evaluator._calculate_overall_score(ranking, diversity, cost, novelty)
        assert score > 0.0
        assert score <= 1.0
    
    def test_calculate_balanced_score(self):
        """Test balanced score calculation."""
        ranking = RankingMetrics(map_score=0.8)
        diversity = DiversityMetrics(diversity_score=0.7)
        cost = CostMetrics(mean_cost=10.0)
        novelty = NoveltyMetrics(mean_novelty=0.5)
        
        score = self.evaluator._calculate_balanced_score(ranking, diversity, cost, novelty)
        assert score > 0.0
        assert score <= 1.0
    
    def test_generate_report(self):
        """Test report generation."""
        metrics = self.evaluator.evaluate(
            self.selected_indices,
            self.targets,
            self.features,
            self.scores,
            self.experimental_data
        )
        
        report = self.evaluator.generate_report(metrics)
        assert "PROTEIN TARGET PRIORITIZATION EVALUATION REPORT" in report
        assert "SUMMARY" in report
        assert "RANKING PERFORMANCE" in report
        assert "DIVERSITY ANALYSIS" in report
        assert "COST ANALYSIS" in report
        assert "NOVELTY ANALYSIS" in report
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_create_visualizations(self, mock_figure, mock_savefig):
        """Test visualization creation."""
        metrics = self.evaluator.evaluate(
            self.selected_indices,
            self.targets,
            self.features,
            self.scores,
            self.experimental_data
        )
        
        # Mock figure creation
        mock_fig = MagicMock()
        mock_figure.return_value = mock_fig
        
        figures = self.evaluator.create_visualizations(metrics)
        
        # Should create multiple figures
        assert len(figures) > 0
        assert 'ranking_performance' in figures
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty selections
        metrics = self.evaluator.evaluate([], self.targets, self.features)
        assert metrics.num_selections == 0
        assert metrics.overall_score >= 0.0  # Should be non-negative
        
        # Invalid indices
        invalid_indices = [100, 200]  # Out of bounds
        metrics = self.evaluator.evaluate(invalid_indices, self.targets, self.features)
        assert metrics.num_selections == 2
        
        # All zeros targets
        zero_targets = np.zeros(8)
        metrics = self.evaluator.evaluate(self.selected_indices, zero_targets, self.features)
        assert metrics.ranking.total_hits == 0
        assert metrics.ranking.hit_rate == 0.0


class TestCreateEvaluator:
    """Test factory function."""
    
    def test_default_creation(self):
        """Test default evaluator creation."""
        evaluator = create_evaluator()
        assert isinstance(evaluator, ProteinEvaluator)
        assert evaluator.config.top_k_values == [5, 10, 20, 50]
        assert evaluator.config.include_ndcg is True
        assert evaluator.config.include_map is True
        assert evaluator.config.save_plots is True
    
    def test_custom_creation(self):
        """Test custom evaluator creation."""
        evaluator = create_evaluator(
            top_k_values=[3, 7, 15],
            include_ndcg=False,
            include_map=False,
            save_plots=False,
            similarity_threshold=0.9
        )
        assert isinstance(evaluator, ProteinEvaluator)
        assert evaluator.config.top_k_values == [3, 7, 15]
        assert evaluator.config.include_ndcg is False
        assert evaluator.config.include_map is False
        assert evaluator.config.save_plots is False
        assert evaluator.config.similarity_threshold == 0.9


class TestIntegration:
    """Integration tests for evaluation workflow."""
    
    def test_full_evaluation_workflow(self):
        """Test complete evaluation workflow."""
        # Create evaluator
        evaluator = create_evaluator(top_k_values=[2, 4], save_plots=False)
        
        # Create test data
        targets = np.array([1, 0, 1, 0, 1, 0, 0, 1])
        features = np.random.randn(8, 5)
        scores = np.array([0.9, 0.2, 0.8, 0.1, 0.7, 0.3, 0.4, 0.6])
        selected_indices = [0, 2, 4, 7]
        experimental_data = {
            0: {"activity_count": 3, "pchembl_std": 0.2},
            2: {"activity_count": 5, "pchembl_std": 0.4},
            4: {"activity_count": 2, "pchembl_std": 0.1},
            7: {"activity_count": 7, "pchembl_std": 0.6},
        }
        
        # Run evaluation
        metrics = evaluator.evaluate(
            selected_indices, targets, features, scores, experimental_data
        )
        
        # Generate report
        report = evaluator.generate_report(metrics)
        
        # Create visualizations
        figures = evaluator.create_visualizations(metrics)
        
        # Verify results
        assert metrics.num_selections == 4
        assert metrics.ranking.hit_rate == 1.0
        assert metrics.overall_score > 0.0
        assert len(report) > 0
        assert len(figures) > 0


if __name__ == "__main__":
    pytest.main([__file__]) 