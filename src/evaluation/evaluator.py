"""
Comprehensive evaluation harness for protein target prioritization.

This module provides detailed metrics and analysis tools to evaluate
the performance of protein ranking and selection algorithms.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, ndcg_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from ..utils import filter_valid_indices, normalize_to_unit_range, capped_cosine_similarity

logger = logging.getLogger(__name__)


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Ranking metrics
    top_k_values: List[int] = Field([5, 10, 20, 50], description="K values for top-K metrics")
    include_ndcg: bool = Field(True, description="Include NDCG (Normalized Discounted Cumulative Gain)")
    include_map: bool = Field(True, description="Include MAP (Mean Average Precision)")
    
    # Diversity metrics
    similarity_threshold: float = Field(0.8, description="Threshold for diversity similarity")
    diversity_weight: float = Field(1.0, description="Weight for diversity in combined metrics")
    
    # Cost metrics
    cost_normalization: str = Field("minmax", description="Cost normalization method: minmax, zscore, or none")
    include_cost_efficiency: bool = Field(True, description="Include cost efficiency metrics")
    
    # Novelty metrics
    novelty_baseline: float = Field(0.5, description="Baseline novelty score")
    include_novelty_analysis: bool = Field(True, description="Include novelty analysis")
    
    # Output settings
    save_plots: bool = Field(True, description="Save evaluation plots")
    plot_dir: str = Field("logs/evaluation_plots", description="Directory for evaluation plots")
    detailed_report: bool = Field(True, description="Generate detailed evaluation report")


class RankingMetrics(BaseModel):
    """Ranking performance metrics."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    precision_at_k: Dict[int, float] = Field(default_factory=dict, description="Precision@K for different K values")
    recall_at_k: Dict[int, float] = Field(default_factory=dict, description="Recall@K for different K values")
    f1_at_k: Dict[int, float] = Field(default_factory=dict, description="F1@K for different K values")
    ndcg_at_k: Dict[int, float] = Field(default_factory=dict, description="NDCG@K for different K values")
    map_score: float = Field(0.0, description="Mean Average Precision")
    auc_roc: float = Field(0.0, description="Area Under ROC Curve")
    auc_pr: float = Field(0.0, description="Area Under Precision-Recall Curve")
    
    # Hit analysis
    total_hits: int = Field(0, description="Total number of hits in dataset")
    hits_found: int = Field(0, description="Number of hits found by algorithm")
    hit_rate: float = Field(0.0, description="Proportion of hits found")
    
    # Ranking quality
    mean_reciprocal_rank: float = Field(0.0, description="Mean Reciprocal Rank of hits")
    mean_rank: float = Field(0.0, description="Mean rank of hits")


class DiversityMetrics(BaseModel):
    """Diversity analysis metrics."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    mean_similarity: float = Field(0.0, description="Mean similarity between selected proteins")
    diversity_score: float = Field(0.0, description="Overall diversity score (1 - mean_similarity)")
    unique_families: int = Field(0, description="Number of unique protein families selected")
    family_distribution: Dict[str, int] = Field(default_factory=dict, description="Distribution across protein families")
    
    # Pairwise analysis
    similarity_matrix: Optional[np.ndarray] = Field(None, description="Similarity matrix of selected proteins")
    min_similarity: float = Field(0.0, description="Minimum similarity between any two selected proteins")
    max_similarity: float = Field(0.0, description="Maximum similarity between any two selected proteins")
    
    # Diversity over selection order
    diversity_trajectory: List[float] = Field(default_factory=list, description="Diversity score as proteins are selected")


class CostMetrics(BaseModel):
    """Cost efficiency metrics."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    total_cost: float = Field(0.0, description="Total cost of selected proteins")
    mean_cost: float = Field(0.0, description="Mean cost per selected protein")
    cost_efficiency: float = Field(0.0, description="Hits per unit cost")
    cost_per_hit: float = Field(0.0, description="Cost per hit found")
    
    # Cost distribution
    cost_distribution: Dict[str, float] = Field(default_factory=dict, description="Cost distribution statistics")
    cost_vs_reward: List[Tuple[float, float]] = Field(default_factory=list, description="Cost vs reward pairs")
    
    # Budget analysis
    budget_utilization: float = Field(0.0, description="Percentage of budget utilized")
    cost_effectiveness: float = Field(0.0, description="Overall cost effectiveness score")


class NoveltyMetrics(BaseModel):
    """Novelty analysis metrics."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    mean_novelty: float = Field(0.0, description="Mean novelty score of selected proteins")
    novelty_distribution: Dict[str, float] = Field(default_factory=dict, description="Novelty distribution statistics")
    high_novelty_count: int = Field(0, description="Number of high novelty proteins selected")
    novelty_efficiency: float = Field(0.0, description="Novelty per selection")
    
    # Novelty vs performance
    novelty_vs_activity: List[Tuple[float, float]] = Field(default_factory=list, description="Novelty vs activity pairs")
    novelty_trajectory: List[float] = Field(default_factory=list, description="Novelty score as proteins are selected")


class EvaluationMetrics(BaseModel):
    """Comprehensive evaluation metrics."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    ranking: RankingMetrics = Field(default_factory=RankingMetrics, description="Ranking performance metrics")
    diversity: DiversityMetrics = Field(default_factory=DiversityMetrics, description="Diversity analysis metrics")
    cost: CostMetrics = Field(default_factory=CostMetrics, description="Cost efficiency metrics")
    novelty: NoveltyMetrics = Field(default_factory=NoveltyMetrics, description="Novelty analysis metrics")
    
    # Combined scores
    overall_score: float = Field(0.0, description="Overall performance score")
    balanced_score: float = Field(0.0, description="Balanced score considering all factors")
    
    # Metadata
    num_selections: int = Field(0, description="Number of proteins selected")
    evaluation_time: float = Field(0.0, description="Time taken for evaluation")
    config: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")


class ProteinEvaluator:
    """Main evaluation engine for protein target prioritization."""
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self._setup_plotting()
    
    def _setup_plotting(self) -> None:
        """Setup plotting configuration."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        if self.config.save_plots:
            plot_dir = Path(self.config.plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_ranking(
        self,
        selected_indices: List[int],
        targets: np.ndarray,
        scores: Optional[np.ndarray] = None,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> RankingMetrics:
        """Evaluate ranking performance."""
        metrics = RankingMetrics()
        
        # Filter out invalid indices
        valid_indices = filter_valid_indices(selected_indices, targets)
        if not valid_indices:
            return metrics
        
        # Basic hit analysis
        hits = targets[valid_indices]
        metrics.total_hits = int(np.sum(targets))
        metrics.hits_found = int(np.sum(hits))
        metrics.hit_rate = metrics.hits_found / metrics.total_hits if metrics.total_hits > 0 else 0.0
        
        # Top-K metrics
        for k in self.config.top_k_values:
            if len(valid_indices) >= k:
                top_k_hits = hits[:k]
                metrics.precision_at_k[k] = float(np.mean(top_k_hits))
                metrics.recall_at_k[k] = float(np.sum(top_k_hits) / metrics.total_hits) if metrics.total_hits > 0 else 0.0
                
                # F1@K
                if metrics.precision_at_k[k] + metrics.recall_at_k[k] > 0:
                    metrics.f1_at_k[k] = 2 * (metrics.precision_at_k[k] * metrics.recall_at_k[k]) / (metrics.precision_at_k[k] + metrics.recall_at_k[k])
                else:
                    metrics.f1_at_k[k] = 0.0
            else:
                # For k values larger than selection size, use all available selections
                metrics.precision_at_k[k] = float(np.mean(hits))
                metrics.recall_at_k[k] = float(np.sum(hits) / metrics.total_hits) if metrics.total_hits > 0 else 0.0
                metrics.f1_at_k[k] = 2 * (metrics.precision_at_k[k] * metrics.recall_at_k[k]) / (metrics.precision_at_k[k] + metrics.recall_at_k[k]) if (metrics.precision_at_k[k] + metrics.recall_at_k[k]) > 0 else 0.0
        
        # NDCG@K
        if self.config.include_ndcg and scores is not None:
            for k in self.config.top_k_values:
                if len(valid_indices) >= k:
                    try:
                        # Create relevance scores (1 for hits, 0 for misses)
                        relevance = targets[valid_indices[:k]]
                        # Use provided scores or fallback to relevance
                        score_values = scores[valid_indices[:k]] if scores is not None else relevance
                        
                        # Calculate NDCG
                        ndcg = ndcg_score([relevance], [score_values], k=k)
                        metrics.ndcg_at_k[k] = float(ndcg)
                    except Exception as e:
                        logger.warning(f"Could not calculate NDCG@K for k={k}: {e}")
                        metrics.ndcg_at_k[k] = 0.0
        
        # MAP and AUC scores
        if self.config.include_map:
            try:
                # Calculate MAP
                ap_scores = []
                for i, idx in enumerate(selected_indices):
                    if targets[idx] == 1:  # Hit
                        # Precision at this position
                        precision = np.sum(targets[selected_indices[:i+1]]) / (i + 1)
                        ap_scores.append(precision)
                
                metrics.map_score = float(np.mean(ap_scores)) if ap_scores else 0.0
            except Exception as e:
                logger.warning(f"Could not calculate MAP: {e}")
                metrics.map_score = 0.0
        
        # AUC scores
        if scores is not None:
            try:
                metrics.auc_roc = float(roc_auc_score(targets, scores))
                metrics.auc_pr = float(average_precision_score(targets, scores))
            except Exception as e:
                logger.warning(f"Could not calculate AUC scores: {e}")
                metrics.auc_roc = 0.0
                metrics.auc_pr = 0.0
        
        # Ranking quality metrics
        hit_ranks = [i + 1 for i, idx in enumerate(valid_indices) if targets[idx] == 1]
        if hit_ranks:
            metrics.mean_rank = float(np.mean(hit_ranks))
            metrics.mean_reciprocal_rank = float(np.mean([1.0 / rank for rank in hit_ranks]))
        
        return metrics
    
    def evaluate_diversity(
        self,
        selected_indices: List[int],
        features: np.ndarray,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> DiversityMetrics:
        """Evaluate diversity of selected proteins."""
        metrics = DiversityMetrics()
        
        # Filter out invalid indices
        valid_indices = filter_valid_indices(selected_indices, features)
        if len(valid_indices) < 2:
            metrics.diversity_score = 1.0
            metrics.diversity_trajectory = [1.0] * len(valid_indices)
            return metrics
        
        # Calculate similarity matrix
        selected_features = features[valid_indices]
        similarity_matrix = capped_cosine_similarity(selected_features)
        metrics.similarity_matrix = similarity_matrix
        
        # Basic diversity metrics
        # Exclude diagonal (self-similarity)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        metrics.mean_similarity = float(np.mean(upper_triangle))
        metrics.diversity_score = max(0.0, min(1.0, 1.0 - metrics.mean_similarity))  # Cap at [0, 1]
        metrics.min_similarity = float(np.min(upper_triangle))
        metrics.max_similarity = float(np.max(upper_triangle))
        
        # Diversity trajectory
        metrics.diversity_trajectory = []
        for i in range(1, len(valid_indices) + 1):
            if i == 1:
                metrics.diversity_trajectory.append(1.0)
            else:
                subset_features = features[valid_indices[:i]]
                subset_sim = capped_cosine_similarity(subset_features)
                subset_upper = subset_sim[np.triu_indices_from(subset_sim, k=1)]
                diversity = 1.0 - float(np.mean(subset_upper))
                metrics.diversity_trajectory.append(diversity)
        
        # Family analysis (if experimental data available)
        if experimental_data:
            families = {}
            for idx in valid_indices:
                if idx in experimental_data:
                    family = experimental_data[idx].get("protein_family", "unknown")
                    families[family] = families.get(family, 0) + 1
            
            metrics.unique_families = len(families)
            metrics.family_distribution = families
        
        return metrics
    
    def evaluate_cost(
        self,
        selected_indices: List[int],
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        targets: Optional[np.ndarray] = None,
    ) -> CostMetrics:
        """Evaluate cost efficiency of selected proteins."""
        metrics = CostMetrics()
        
        if not experimental_data or not selected_indices:
            return metrics
        
        # Calculate costs
        costs = []
        for idx in selected_indices:
            if idx in experimental_data:
                data = experimental_data[idx]
                # Cost based on activity count and variability
                activity_cost = data.get("activity_count", 0)
                std_cost = data.get("pchembl_std", 0)
                total_cost = (activity_cost + std_cost) / 2.0
                costs.append(total_cost)
            else:
                costs.append(0.0)
        
        if not costs:
            return metrics
        
        # Basic cost metrics
        metrics.total_cost = float(np.sum(costs))
        metrics.mean_cost = float(np.mean(costs))
        
        # Cost efficiency
        if metrics.total_cost > 0 and targets is not None:
            hits_found = int(np.sum(targets[selected_indices]))
            metrics.cost_efficiency = hits_found / metrics.total_cost
            metrics.cost_per_hit = metrics.total_cost / hits_found if hits_found > 0 else float('inf')
        
        # Cost distribution
        metrics.cost_distribution = {
            "min": float(np.min(costs)),
            "max": float(np.max(costs)),
            "std": float(np.std(costs)),
            "median": float(np.median(costs)),
        }
        
        # Cost vs reward analysis
        if targets is not None:
            for i, idx in enumerate(selected_indices):
                cost = costs[i] if i < len(costs) else 0.0
                reward = float(targets[idx])
                metrics.cost_vs_reward.append((cost, reward))
        
        return metrics
    
    def evaluate_novelty(
        self,
        selected_indices: List[int],
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
        targets: Optional[np.ndarray] = None,
    ) -> NoveltyMetrics:
        """Evaluate novelty of selected proteins."""
        metrics = NoveltyMetrics()
        
        if not selected_indices:
            return metrics
        
        if not experimental_data:
            # Use baseline novelty for all selections
            metrics.mean_novelty = self.config.novelty_baseline
            metrics.novelty_efficiency = self.config.novelty_baseline / len(selected_indices) if selected_indices else 0.0
            metrics.novelty_trajectory = [self.config.novelty_baseline] * len(selected_indices)
            return metrics
        
        # Calculate novelty scores
        novelty_scores = []
        for idx in selected_indices:
            if idx in experimental_data:
                data = experimental_data[idx]
                activity_count = data.get("activity_count", 0)
                # Novelty based on activity count (less studied = more novel)
                if activity_count == 0:
                    novelty = 1.0
                else:
                    # Normalize to [0, 1] where 1 = most novel
                    novelty = 1.0 / (1.0 + activity_count / 10.0)  # Scale factor
                novelty_scores.append(novelty)
            else:
                novelty_scores.append(self.config.novelty_baseline)
        
        if not novelty_scores:
            return metrics
        
        # Basic novelty metrics
        metrics.mean_novelty = float(np.mean(novelty_scores))
        metrics.high_novelty_count = sum(1 for score in novelty_scores if score > 0.7)
        metrics.novelty_efficiency = metrics.mean_novelty / len(selected_indices)
        
        # Novelty distribution
        metrics.novelty_distribution = {
            "min": float(np.min(novelty_scores)),
            "max": float(np.max(novelty_scores)),
            "std": float(np.std(novelty_scores)),
            "median": float(np.median(novelty_scores)),
        }
        
        # Novelty vs activity analysis
        if targets is not None:
            for i, idx in enumerate(selected_indices):
                novelty = novelty_scores[i] if i < len(novelty_scores) else self.config.novelty_baseline
                activity = float(targets[idx])
                metrics.novelty_vs_activity.append((novelty, activity))
        
        # Novelty trajectory
        metrics.novelty_trajectory = []
        for i in range(1, len(selected_indices) + 1):
            subset_novelty = novelty_scores[:i]
            avg_novelty = float(np.mean(subset_novelty))
            metrics.novelty_trajectory.append(avg_novelty)
        
        return metrics
    
    def evaluate(
        self,
        selected_indices: List[int],
        targets: np.ndarray,
        features: np.ndarray,
        scores: Optional[np.ndarray] = None,
        experimental_data: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> EvaluationMetrics:
        """Comprehensive evaluation of protein selection performance."""
        import time
        start_time = time.time()
        
        # Evaluate all aspects
        ranking_metrics = self.evaluate_ranking(selected_indices, targets, scores, experimental_data)
        diversity_metrics = self.evaluate_diversity(selected_indices, features, experimental_data)
        cost_metrics = self.evaluate_cost(selected_indices, experimental_data, targets)
        novelty_metrics = self.evaluate_novelty(selected_indices, experimental_data, targets)
        
        # Calculate combined scores
        overall_score = self._calculate_overall_score(ranking_metrics, diversity_metrics, cost_metrics, novelty_metrics)
        balanced_score = self._calculate_balanced_score(ranking_metrics, diversity_metrics, cost_metrics, novelty_metrics)
        
        # Create comprehensive metrics
        metrics = EvaluationMetrics(
            ranking=ranking_metrics,
            diversity=diversity_metrics,
            cost=cost_metrics,
            novelty=novelty_metrics,
            overall_score=overall_score,
            balanced_score=balanced_score,
            num_selections=len(selected_indices),
            evaluation_time=time.time() - start_time,
            config=self.config,
        )
        
        return metrics
    
    def _calculate_overall_score(
        self,
        ranking: RankingMetrics,
        diversity: DiversityMetrics,
        cost: CostMetrics,
        novelty: NoveltyMetrics,
    ) -> float:
        """Calculate overall performance score."""
        # Weighted combination of key metrics
        ranking_score = ranking.map_score if ranking.map_score > 0 else ranking.hit_rate
        diversity_score = diversity.diversity_score
        cost_score = 1.0 / (1.0 + cost.mean_cost) if cost.mean_cost > 0 else 1.0
        novelty_score = novelty.mean_novelty if novelty.mean_novelty > 0 else self.config.novelty_baseline
        
        # Combine with weights
        overall = (
            0.4 * ranking_score +
            0.3 * diversity_score +
            0.2 * cost_score +
            0.1 * novelty_score
        )
        
        return float(overall)
    
    def _calculate_balanced_score(
        self,
        ranking: RankingMetrics,
        diversity: DiversityMetrics,
        cost: CostMetrics,
        novelty: NoveltyMetrics,
    ) -> float:
        """Calculate balanced score considering all factors equally."""
        # Normalize all scores to [0, 1] range
        ranking_score = ranking.map_score if ranking.map_score > 0 else ranking.hit_rate
        diversity_score = diversity.diversity_score
        cost_score = 1.0 / (1.0 + cost.mean_cost) if cost.mean_cost > 0 else 1.0
        novelty_score = novelty.mean_novelty
        
        # Geometric mean for balanced score
        scores = [ranking_score, diversity_score, cost_score, novelty_score]
        balanced = np.power(np.prod(scores), 1.0 / len(scores))
        
        return float(balanced)
    
    def generate_report(
        self,
        metrics: EvaluationMetrics,
        save_path: Optional[str] = None,
    ) -> str:
        """Generate detailed evaluation report."""
        report = []
        report.append("=" * 60)
        report.append("PROTEIN TARGET PRIORITIZATION EVALUATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 20)
        report.append(f"Overall Score: {metrics.overall_score:.3f}")
        report.append(f"Balanced Score: {metrics.balanced_score:.3f}")
        report.append(f"Proteins Selected: {metrics.num_selections}")
        report.append(f"Evaluation Time: {metrics.evaluation_time:.2f}s")
        report.append("")
        
        # Ranking Performance
        report.append("RANKING PERFORMANCE")
        report.append("-" * 20)
        report.append(f"Hit Rate: {metrics.ranking.hit_rate:.3f} ({metrics.ranking.hits_found}/{metrics.ranking.total_hits})")
        report.append(f"MAP Score: {metrics.ranking.map_score:.3f}")
        report.append(f"AUC-ROC: {metrics.ranking.auc_roc:.3f}")
        report.append(f"AUC-PR: {metrics.ranking.auc_pr:.3f}")
        report.append(f"Mean Rank: {metrics.ranking.mean_rank:.1f}")
        report.append("")
        
        # Top-K Performance
        report.append("TOP-K PERFORMANCE")
        report.append("-" * 20)
        for k in self.config.top_k_values:
            if k in metrics.ranking.precision_at_k:
                report.append(f"Precision@{k}: {metrics.ranking.precision_at_k[k]:.3f}")
                report.append(f"Recall@{k}: {metrics.ranking.recall_at_k[k]:.3f}")
                report.append(f"F1@{k}: {metrics.ranking.f1_at_k[k]:.3f}")
                if k in metrics.ranking.ndcg_at_k:
                    report.append(f"NDCG@{k}: {metrics.ranking.ndcg_at_k[k]:.3f}")
                report.append("")
        
        # Diversity Analysis
        report.append("DIVERSITY ANALYSIS")
        report.append("-" * 20)
        report.append(f"Diversity Score: {metrics.diversity.diversity_score:.3f}")
        report.append(f"Mean Similarity: {metrics.diversity.mean_similarity:.3f}")
        report.append(f"Unique Families: {metrics.diversity.unique_families}")
        report.append("")
        
        # Cost Analysis
        report.append("COST ANALYSIS")
        report.append("-" * 20)
        report.append(f"Total Cost: {metrics.cost.total_cost:.2f}")
        report.append(f"Mean Cost: {metrics.cost.mean_cost:.2f}")
        report.append(f"Cost Efficiency: {metrics.cost.cost_efficiency:.3f}")
        report.append(f"Cost per Hit: {metrics.cost.cost_per_hit:.2f}")
        report.append("")
        
        # Novelty Analysis
        report.append("NOVELTY ANALYSIS")
        report.append("-" * 20)
        report.append(f"Mean Novelty: {metrics.novelty.mean_novelty:.3f}")
        report.append(f"High Novelty Count: {metrics.novelty.high_novelty_count}")
        report.append(f"Novelty Efficiency: {metrics.novelty.novelty_efficiency:.3f}")
        report.append("")
        
        report_text = "\n".join(report)
        
        # Save report if path provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text
    
    def create_visualizations(
        self,
        metrics: EvaluationMetrics,
        save_dir: Optional[str] = None,
    ) -> Dict[str, plt.Figure]:
        """Create evaluation visualizations."""
        if save_dir is None:
            save_dir = self.config.plot_dir
        
        figures = {}
        
        # 1. Ranking Performance Plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Ranking Performance Analysis", fontsize=16)
        
        # Precision-Recall curve
        k_values = list(metrics.ranking.precision_at_k.keys())
        precisions = [metrics.ranking.precision_at_k[k] for k in k_values]
        recalls = [metrics.ranking.recall_at_k[k] for k in k_values]
        
        axes[0, 0].plot(k_values, precisions, 'o-', label='Precision@K')
        axes[0, 0].set_xlabel('K')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_title('Precision@K')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(k_values, recalls, 's-', label='Recall@K')
        axes[0, 1].set_xlabel('K')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Recall@K')
        axes[0, 1].grid(True)
        
        # F1 and NDCG
        f1_scores = [metrics.ranking.f1_at_k.get(k, 0) for k in k_values]
        axes[1, 0].plot(k_values, f1_scores, '^-', label='F1@K')
        axes[1, 0].set_xlabel('K')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('F1@K')
        axes[1, 0].grid(True)
        
        ndcg_scores = [metrics.ranking.ndcg_at_k.get(k, 0) for k in k_values]
        axes[1, 1].plot(k_values, ndcg_scores, 'd-', label='NDCG@K')
        axes[1, 1].set_xlabel('K')
        axes[1, 1].set_ylabel('NDCG')
        axes[1, 1].set_title('NDCG@K')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        figures['ranking_performance'] = fig
        
        if self.config.save_plots:
            fig.savefig(f"{save_dir}/ranking_performance.png", dpi=300, bbox_inches='tight')
        
        # 2. Diversity Analysis
        if metrics.diversity.similarity_matrix is not None:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle("Diversity Analysis", fontsize=16)
            
            # Similarity matrix heatmap
            im = axes[0].imshow(metrics.diversity.similarity_matrix, cmap='viridis')
            axes[0].set_title('Similarity Matrix')
            axes[0].set_xlabel('Protein Index')
            axes[0].set_ylabel('Protein Index')
            plt.colorbar(im, ax=axes[0])
            
            # Diversity trajectory
            axes[1].plot(range(1, len(metrics.diversity.diversity_trajectory) + 1), 
                        metrics.diversity.diversity_trajectory, 'o-')
            axes[1].set_xlabel('Number of Selections')
            axes[1].set_ylabel('Diversity Score')
            axes[1].set_title('Diversity Trajectory')
            axes[1].grid(True)
            
            plt.tight_layout()
            figures['diversity_analysis'] = fig
            
            if self.config.save_plots:
                fig.savefig(f"{save_dir}/diversity_analysis.png", dpi=300, bbox_inches='tight')
        
        # 3. Cost vs Reward Analysis
        if metrics.cost.cost_vs_reward:
            fig, ax = plt.subplots(figsize=(8, 6))
            costs, rewards = zip(*metrics.cost.cost_vs_reward)
            ax.scatter(costs, rewards, alpha=0.6)
            ax.set_xlabel('Cost')
            ax.set_ylabel('Reward (Activity)')
            ax.set_title('Cost vs Reward Analysis')
            ax.grid(True)
            
            plt.tight_layout()
            figures['cost_vs_reward'] = fig
            
            if self.config.save_plots:
                fig.savefig(f"{save_dir}/cost_vs_reward.png", dpi=300, bbox_inches='tight')
        
        # 4. Novelty Analysis
        if metrics.novelty.novelty_vs_activity:
            fig, ax = plt.subplots(figsize=(8, 6))
            novelty, activity = zip(*metrics.novelty.novelty_vs_activity)
            ax.scatter(novelty, activity, alpha=0.6)
            ax.set_xlabel('Novelty Score')
            ax.set_ylabel('Activity')
            ax.set_title('Novelty vs Activity Analysis')
            ax.grid(True)
            
            plt.tight_layout()
            figures['novelty_analysis'] = fig
            
            if self.config.save_plots:
                fig.savefig(f"{save_dir}/novelty_analysis.png", dpi=300, bbox_inches='tight')
        
        return figures


def create_evaluator(
    top_k_values: Optional[List[int]] = None,
    include_ndcg: bool = True,
    include_map: bool = True,
    save_plots: bool = True,
    **kwargs
) -> ProteinEvaluator:
    """Factory function to create an evaluator with common configurations."""
    if top_k_values is None:
        top_k_values = [5, 10, 20, 50]
    
    config = EvaluationConfig(
        top_k_values=top_k_values,
        include_ndcg=include_ndcg,
        include_map=include_map,
        save_plots=save_plots,
        **kwargs
    )
    return ProteinEvaluator(config) 