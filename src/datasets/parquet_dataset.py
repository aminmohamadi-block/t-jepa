"""
Parquet Dataset Integration for T-JEPA

This module provides a wrapper around LocalFilesDataset to integrate large-scale
parquet-based tabular datasets into T-JEPA's self-supervised learning framework.
"""

import logging
from typing import Optional, List
import polars as pl

from src.datasets.local_dataset import LocalFilesDataset

logger = logging.getLogger(__name__)


def load_feature_names_from_csv(feature_names_file: str) -> List[str]:
    """
    Load feature names from a CSV file (e.g., mi.csv from mutual information ranking).

    Expected CSV format:
    - Column 'feature': feature names
    - Column 'importance': importance scores (descending order)
    - Column 'method': method used for importance computation

    Args:
        feature_names_file: Path to CSV file with feature importance rankings

    Returns:
        List of feature names ordered by importance (descending)
    """
    logger.info(f"Loading feature names from: {feature_names_file}")
    df = pl.read_csv(feature_names_file)

    if "feature" not in df.columns:
        raise ValueError(f"CSV file {feature_names_file} must contain 'feature' column")

    # Sort by importance if available (descending)
    if "importance" in df.columns:
        df = df.sort("importance", descending=True)

    feature_names = df["feature"].to_list()
    logger.info(f"Loaded {len(feature_names)} features from {feature_names_file}")

    return feature_names


class TJEPAParquetDataset:
    """
    Wrapper for LocalFilesDataset adapted for T-JEPA unsupervised learning.

    This class provides a bridge between risk-tabular-slurm's LocalFilesDataset
    and T-JEPA's training pipeline. Key adaptations:

    1. Simplified initialization from argparse arguments
    2. Iterator that yields only features (targets kept for linear probe)
    3. Property accessors for T-JEPA compatibility

    Args:
        args: Argparse namespace with parquet-related configuration
    """

    def __init__(self, args):
        self.args = args

        # Validate required arguments
        self._validate_args()

        # Load feature names if provided
        external_feature_names = None
        if args.parquet_feature_names_file is not None:
            external_feature_names = load_feature_names_from_csv(
                args.parquet_feature_names_file
            )

        # Parse data_files if provided as comma-separated string
        data_files = None
        if args.parquet_data_files is not None:
            data_files = [f.strip() for f in args.parquet_data_files.split(",")]

        # Create feature_scaling configuration dict
        feature_scaling = {
            "method": args.parquet_scaling_method,
            "infill_value": args.parquet_infill_value,
            "transform": args.parquet_transform,
            "categorize_nan": args.parquet_categorize_nan,
            "clip_min": args.parquet_clip_min,
            "clip_max": args.parquet_clip_max,
        }

        # Initialize LocalFilesDataset
        logger.info("Initializing LocalFilesDataset with parquet configuration...")
        self.dataset = LocalFilesDataset(
            data_dir=args.parquet_data_dir,
            batch_size=args.batch_size,
            scaling_stats_file=args.parquet_scaling_stats_file,
            target_col=args.parquet_target_col,
            data_files=data_files,
            id_col=args.parquet_id_col,
            shuffle=args.parquet_shuffle,
            external_feature_names=external_feature_names,
            num_features_to_keep=args.parquet_num_features,
            feature_scaling=feature_scaling,
            sample_rates=None,  # No sampling for unsupervised learning
            preload_all_data=args.parquet_preload_data,
        )

        # Store metadata for T-JEPA compatibility
        self.N = len(self.dataset.all_features_tensor) if args.parquet_preload_data else None

        # Get actual feature count from LocalFilesDataset
        # LocalFilesDataset keeps target and ID separate from features
        self.D = len(self.dataset.feature_names_to_keep)

        # Account for categorize_nan doubling features
        if args.parquet_categorize_nan:
            self.D *= 2

        # All features are numerical after preprocessing in LocalFilesDataset
        self.num_features = list(range(self.D))
        self.cat_features = []
        self.cardinalities = []

        # Dataset name for logging/tracking
        self.dataset_name = "parquet_dataset"

        logger.info(f"TJEPAParquetDataset initialized: N={self.N}, D={self.D}")
        logger.info(f"Feature scaling: {args.parquet_scaling_method}, "
                   f"infill: {args.parquet_infill_value}, "
                   f"transform: {args.parquet_transform}")

    def _validate_args(self):
        """Validate that required arguments are provided."""
        required = [
            ("parquet_data_dir", "Directory containing parquet files"),
            ("parquet_scaling_stats_file", "Scaling statistics file"),
        ]

        missing = []
        for arg_name, description in required:
            if getattr(self.args, arg_name, None) is None:
                missing.append(f"--{arg_name} ({description})")

        if missing:
            raise ValueError(
                f"Missing required arguments for parquet dataset:\n" +
                "\n".join(f"  - {m}" for m in missing)
            )

    def __iter__(self):
        """
        Iterate over the dataset, yielding only features (not targets).

        For T-JEPA unsupervised learning, we don't need targets during training.
        The LocalFilesDataset yields (features, targets, ids), but we only return features.

        Yields:
            features: torch.Tensor of shape [batch_size, num_features]
        """
        for features, targets, ids in self.dataset:
            # For unsupervised T-JEPA, yield only features
            # Targets are kept in dataset for potential linear probe evaluation
            yield features

    def __len__(self):
        """Return number of batches if known."""
        return len(self.dataset) if self.dataset.batched_length is not None else 0

    @property
    def feature_names(self) -> List[str]:
        """Return list of feature names used by the dataset."""
        return self.dataset.feature_names_to_keep

    @property
    def scaling_tensors(self):
        """Return scaling tensors (subtract_tensor, divide_tensor) for reference."""
        return self.dataset.subtract_tensor, self.dataset.divide_tensor

    def get_raw_data_for_linear_probe(self):
        """
        Get all data including targets for linear probe evaluation.

        Returns:
            Tuple of (features, targets, ids) if data is preloaded, None otherwise.
        """
        if self.args.parquet_preload_data:
            return (
                self.dataset.all_features_tensor,
                self.dataset.all_target_tensor,
                self.dataset.all_ids
            )
        else:
            logger.warning(
                "Cannot get raw data for linear probe: data is not preloaded. "
                "Set --parquet_preload_data=True to enable."
            )
            return None


def create_parquet_dataset_from_args(args):
    """
    Factory function to create TJEPAParquetDataset from argparse arguments.

    Args:
        args: Argparse namespace with configuration

    Returns:
        TJEPAParquetDataset instance
    """
    return TJEPAParquetDataset(args)
