"""
Data loader for real experimental protein data.
Handles loading and processing of unified protein dataset with ESM embeddings and ChEMBL activities.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UnifiedProteinDataLoader:
    """Loader for unified protein dataset with ESM embeddings and experimental activities."""

    def __init__(self, data_dir: str = "protein_inputs"):
        self.data_dir = Path(data_dir)

    def load_unified_dataset(self, file_path: str | None = None) -> pd.DataFrame:
        """
        Load unified protein dataset with ESM embeddings and ChEMBL activities.
        Args:
            file_path: Path to unified dataset file. If None, uses default location.
        Returns:
            DataFrame with protein embeddings and experimental activities
        """
        if file_path is None:
            file_path = self.data_dir / "processed" / "unified_protein_dataset.csv"

        if not Path(file_path).exists():
            raise FileNotFoundError(
                f"Unified dataset not found at {file_path}. "
                "Run scripts/data_collection/create_unified_dataset.py first."
            )

        data = pd.read_csv(file_path)
        logger.info(
            f"Loaded unified dataset with {len(data)} proteins from {file_path}"
        )
        return data

    def prepare_environment_data(
        self,
        data_df: pd.DataFrame,
        target_column: str = "has_activity",
        use_embeddings: bool = True,
        use_activity_features: bool = True,
        normalize_features: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for RL environment.
        Args:
            data_df: Unified protein dataset
            target_column: Column to use as target (default: 'has_activity')
            use_embeddings: Whether to include ESM embeddings
            use_activity_features: Whether to include activity features
            normalize_features: Whether to normalize features
        Returns:
            Tuple of (features, targets)
        """
        if target_column not in data_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data")

        targets = data_df[target_column].astype(int).values

        # Collect feature columns
        feature_columns = []

        if use_embeddings:
            # Add ESM embedding features (f0, f1, ..., f1279)
            embedding_cols = [col for col in data_df.columns if col.startswith("f")]
            feature_columns.extend(embedding_cols)
            logger.info(f"Using {len(embedding_cols)} ESM embedding features")

        if use_activity_features:
            # Add activity features (excluding target and ID columns)
            activity_cols = [
                "activity_count",
                "pchembl_mean",
                "pchembl_std",
                "pchembl_min",
                "pchembl_max",
                "standard_value_mean",
                "standard_value_std",
                "standard_value_min",
                "standard_value_max",
            ]
            available_activity_cols = [
                col for col in activity_cols if col in data_df.columns
            ]
            feature_columns.extend(available_activity_cols)
            logger.info(f"Using {len(available_activity_cols)} activity features")

        if not feature_columns:
            raise ValueError("No feature columns selected")

        # Extract features
        features = data_df[feature_columns].values.astype(np.float32)
        features = np.nan_to_num(features, nan=0.0)

        if normalize_features:
            features = self._normalize_features(features)

        logger.info(
            f"Prepared {len(features)} proteins with {features.shape[1]} features"
        )
        logger.info(f"Target distribution: {np.bincount(targets)}")

        return features, targets

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [-1, 1] range."""
        features_min = np.min(features, axis=0, keepdims=True)
        features_max = np.max(features, axis=0, keepdims=True)
        features_range = features_max - features_min
        features_range[features_range == 0] = 1.0
        normalized = 2.0 * (features - features_min) / features_range - 1.0
        return normalized.astype(np.float32)

    def get_data_summary(self, data_df: pd.DataFrame) -> dict[str, int | float]:
        """Get summary statistics of the dataset."""
        summary = {
            "total_proteins": len(data_df),
            "proteins_with_activities": int(data_df["has_activity"].sum()),
            "proteins_without_activities": int((data_df["has_activity"] == 0).sum()),
            "activity_rate": float(data_df["has_activity"].mean()),
            "total_activities": int(data_df["activity_count"].sum()),
        }

        if "activity_count" in data_df.columns:
            summary["mean_activities_per_protein"] = float(
                data_df["activity_count"].mean()
            )
            summary["median_activities_per_protein"] = float(
                data_df["activity_count"].median()
            )
            summary["max_activities_per_protein"] = int(data_df["activity_count"].max())

        if "pchembl_mean" in data_df.columns:
            active_proteins = data_df[data_df["activity_count"] > 0]
            if len(active_proteins) > 0:
                summary["mean_pchembl"] = float(active_proteins["pchembl_mean"].mean())
                summary["min_pchembl"] = float(active_proteins["pchembl_mean"].min())
                summary["max_pchembl"] = float(active_proteins["pchembl_mean"].max())

        # Count embedding features
        embedding_cols = [col for col in data_df.columns if col.startswith("f")]
        summary["n_embedding_features"] = len(embedding_cols)

        return summary


# Legacy support - keep old class for backward compatibility
class ExperimentalDataLoader(UnifiedProteinDataLoader):
    """Legacy loader for backward compatibility."""

    def load_uniprot_bindingdb_data(self, file_path: str | None = None) -> pd.DataFrame:
        """Legacy method - now loads unified dataset."""
        return self.load_unified_dataset(file_path)

    def load_legacy_data(self, file_path: str) -> pd.DataFrame:
        """Legacy method - now loads unified dataset."""
        return self.load_unified_dataset(file_path)


def load_unified_protein_data(
    data_path: str | None = None,
    target_column: str = "has_activity",
    use_embeddings: bool = True,
    use_activity_features: bool = True,
    normalize_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """
    Load unified protein dataset for RL environment.
    Args:
        data_path: Path to unified dataset file
        target_column: Column to use as target
        use_embeddings: Whether to include ESM embeddings
        use_activity_features: Whether to include activity features
        normalize_features: Whether to normalize features
    Returns:
        Tuple of (features, targets, summary_stats)
    """
    loader = UnifiedProteinDataLoader()
    data_df = loader.load_unified_dataset(data_path)
    features, targets = loader.prepare_environment_data(
        data_df,
        target_column,
        use_embeddings,
        use_activity_features,
        normalize_features,
    )
    summary_stats = loader.get_data_summary(data_df)
    return features, targets, summary_stats


# Legacy function for backward compatibility
def load_experimental_data(
    data_source: str = "unified",
    data_path: str | None = None,
    target_column: str = "has_activity",
    feature_columns: list[str] | None = None,
    normalize_features: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Legacy function - now uses unified dataset."""
    return load_unified_protein_data(
        data_path,
        target_column,
        use_embeddings=True,
        use_activity_features=True,
        normalize_features=normalize_features,
    )
