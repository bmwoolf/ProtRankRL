import numpy as np
import pandas as pd
import os
from typing import Optional
import hashlib

from .utils import normalize_to_unit_range

CACHE_PATH = "protein_inputs/processed/state_cache.parquet"

# Placeholder: load harmonized dataframes (UniProt, AlphaFold, etc.)
UNIPROT_PATH = "protein_inputs/processed/unified_protein_dataset.csv"
ALPHAFOLD_PATH = None  # Add path if available
KNOWLEDGE_GRAPH_PATH = None  # Add path if available

# Load UniProt/ESM features (as example)
if os.path.exists(UNIPROT_PATH):
    UNIPROT_DF = pd.read_csv(UNIPROT_PATH)
else:
    UNIPROT_DF = pd.DataFrame()


def build_protein_state(protein_id: str, data_version: str = "v1") -> np.ndarray:
    """
    Given a protein ID, return a deterministic, fixed-size feature vector
    using harmonized UniProt, AlphaFold, and Knowledge Graph data.
    Uses local cache for efficiency.
    """
    # Check cache
    cache_key = f"{protein_id}_{data_version}"
    cache_hash = hashlib.sha1(cache_key.encode()).hexdigest()
    if os.path.exists(CACHE_PATH):
        cache_df = pd.read_parquet(CACHE_PATH)
        cached = cache_df[cache_df["cache_hash"] == cache_hash]
        if not cached.empty:
            return np.array(cached.iloc[0]["vector"])
    else:
        cache_df = pd.DataFrame(columns=["cache_hash", "protein_id", "data_version", "vector"])

    # --- Harmonize features ---
    # Example: use ESM embedding from UniProt dataset
    if not UNIPROT_DF.empty and protein_id in UNIPROT_DF["uniprot_id"].values:
        row = UNIPROT_DF[UNIPROT_DF["uniprot_id"] == protein_id].iloc[0]
        embedding_cols = [col for col in UNIPROT_DF.columns if col.startswith("f")]
        vector = row[embedding_cols].values.astype(np.float32)
    else:
        # Fallback: zero vector
        vector = np.zeros(1280, dtype=np.float32)

    # TODO: Add AlphaFold, Knowledge Graph, etc. features here

    # Save to cache
    cache_row = {
        "cache_hash": cache_hash,
        "protein_id": protein_id,
        "data_version": data_version,
        "vector": vector,
    }
    cache_df = pd.concat([cache_df, pd.DataFrame([cache_row])], ignore_index=True)
    cache_df.to_parquet(CACHE_PATH, index=False)

    return vector 