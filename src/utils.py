import numpy as np
from typing import List, Any
from sklearn.metrics.pairwise import cosine_similarity

def filter_valid_indices(indices: List[int], array: np.ndarray) -> List[int]:
    """Return only indices that are valid for the given array."""
    return [idx for idx in indices if 0 <= idx < len(array)]

def normalize_to_unit_range(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    if arr_max - arr_min == 0:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def capped_cosine_similarity(features: np.ndarray) -> np.ndarray:
    """Compute cosine similarity and cap values to [-1, 1]."""
    sim = cosine_similarity(features)
    return np.clip(sim, -1.0, 1.0) 