from pydantic import BaseModel, Field
from typing import Optional

class RLConfig(BaseModel):
    # Environment hyperparameters
    gamma: float = Field(0.99, description="Discount factor")
    lam: float = Field(0.95, description="GAE lambda")
    entropy_coef: float = Field(0.01, description="Entropy regularization coefficient")
    reward_type: str = Field("enhanced_multi_objective", description="Reward function type")
    diversity_weight: float = Field(0.3, description="Weight for diversity reward")
    novelty_weight: float = Field(0.2, description="Weight for novelty reward")
    cost_weight: float = Field(0.1, description="Weight for cost penalty")
    similarity_threshold: float = Field(0.8, description="Diversity similarity threshold")
    batch_size: int = Field(1, description="Batch size for selection")
    max_episode_length: Optional[int] = Field(None, description="Max episode length")
    early_stopping_patience: int = Field(10, description="Early stopping patience")
    min_diversity_ratio: float = Field(0.5, description="Minimum diversity ratio")
    
    # Training hyperparameters
    total_timesteps: int = Field(100_000, description="Total training timesteps")
    learning_rate: float = Field(3e-4, description="Learning rate")
    n_steps: int = Field(2048, description="Number of steps per update")
    batch_size_train: int = Field(64, description="Batch size for training")
    n_epochs: int = Field(10, description="Number of epochs per update")
    
    # Data/config paths
    data_path: str = Field("protein_inputs/processed/unified_protein_dataset.csv", description="Path to protein data")
    model_save_path: str = Field("models/best_model.zip", description="Path to save trained model")
    log_dir: str = Field("logs/", description="Directory for logs") 