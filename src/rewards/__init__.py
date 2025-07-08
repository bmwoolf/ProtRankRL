"""
Reward shaping module for ProtRankRL.

This module provides configurable reward functions for protein target prioritization,
supporting multi-objective optimization with diversity, novelty, and cost considerations.
"""

from .reward_shaper import (
    BaseRewardFunction,
    BinaryRewardFunction,
    AffinityBasedRewardFunction,
    MultiObjectiveRewardFunction,
    EnhancedMultiObjectiveRewardFunction,
    RewardShaper,
    RewardConfig,
)

__all__ = [
    "BaseRewardFunction",
    "BinaryRewardFunction", 
    "AffinityBasedRewardFunction",
    "MultiObjectiveRewardFunction",
    "EnhancedMultiObjectiveRewardFunction",
    "RewardShaper",
    "RewardConfig",
] 