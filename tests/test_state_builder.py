import sys
import os
sys.path.append(os.path.abspath("."))
import numpy as np
from src.state_builder import build_protein_state

def test_build_protein_state_deterministic():
    protein_id = "P12345"  # Use a real ID from your dataset if possible
    vec1 = build_protein_state(protein_id)
    vec2 = build_protein_state(protein_id)
    assert isinstance(vec1, np.ndarray)
    assert vec1.shape == vec2.shape
    assert np.allclose(vec1, vec2), "State builder should be deterministic and cacheable"
    print(f"Vector shape for {protein_id}: {vec1.shape}")
    print(f"First 5 values: {vec1[:5]}")

if __name__ == "__main__":
    test_build_protein_state_deterministic() 