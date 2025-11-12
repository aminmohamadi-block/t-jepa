"""
ParquetBaseDataset - BaseDataset-compatible wrapper for parquet data.

This module provides a wrapper that makes parquet datasets compatible with
the OnlineDataset interface, enabling linear probe evaluation for T-JEPA.

Key differences from TJEPAParquetDataset:
- Inherits from BaseDataset (required by OnlineDataset)
- Always preloads data (linear probe needs all data in memory)
- Exposes X, y as attributes (not just via method)
- Has task_type attribute for supervised evaluation
"""

import logging
from typing import Optional, List
import numpy as np

from src.datasets.base import BaseDataset
from src.datasets.local_dataset import LocalFilesDataset
from src.datasets.parquet_dataset import load_feature_names_from_csv
from src.utils.models_utils import TASK_TYPE

logger = logging.getLogger(__name__)


class ParquetBaseDataset(BaseDataset):
    """
    BaseDataset-compatible wrapper for parquet datasets.

    This class enables linear probe evaluation by providing the interface
    expected by OnlineDataset while reusing LocalFilesDataset for data loading.

    Required BaseDataset attributes:
    - X: numpy array [N, D] of features
    - y: numpy array [N] of targets
    - N: number of samples
    - D: number of features
    - task_type: TASK_TYPE enum (BINARY_CLASS, MULTI_CLASS, REGRESSION)
    - cardinalities: list of (idx, cardinality) for categorical features
    - cat_features: list of categorical feature indices
    - num_features: list of numerical feature indices
    - num_or_cat: dict mapping feature idx to boolean (True=numerical)
    - name: dataset name string

    Args:
        args: Argparse namespace with parquet-related configuration.
              Must include all parquet_* arguments.
    """

    def __init__(self, args):
        # Initialize BaseDataset attributes
        super().__init__(args)

        self.args = args
        self.name = "parquet_dataset"

        # Map task_type string to TASK_TYPE enum
        task_type_map = {
            "binary_class": TASK_TYPE.BINARY_CLASS,
            "multi_class": TASK_TYPE.MULTI_CLASS,
            "regression": TASK_TYPE.REGRESSION,
        }

        # Get task type from args, default to binary classification
        task_type_str = getattr(args, "parquet_task_type", "binary_class")
        self.task_type = task_type_map.get(task_type_str, TASK_TYPE.BINARY_CLASS)

        logger.info(f"ParquetBaseDataset initialized with task_type: {task_type_str}")

        # Will be set in load()
        self._local_dataset = None

    def load(self):
        """
        Load parquet data using LocalFilesDataset.

        This method:
        1. Creates LocalFilesDataset with preload_all_data=True
        2. Extracts features and targets into X, y attributes
        3. Sets all BaseDataset attributes (N, D, cardinalities, etc.)
        """
        if self.is_data_loaded:
            logger.info("Data already loaded, skipping.")
            return

        logger.info("Loading parquet data via LocalFilesDataset...")

        # Validate required arguments
        self._validate_args()

        # Load feature names if provided
        external_feature_names = None
        if self.args.parquet_feature_names_file is not None:
            external_feature_names = load_feature_names_from_csv(
                self.args.parquet_feature_names_file
            )

        # Parse data_files if provided as comma-separated string
        data_files = None
        if self.args.parquet_data_files is not None:
            data_files = [f.strip() for f in self.args.parquet_data_files.split(",")]

        # Create feature_scaling configuration dict
        feature_scaling = {
            "method": self.args.parquet_scaling_method,
            "infill_value": self.args.parquet_infill_value,
            "transform": self.args.parquet_transform,
            "categorize_nan": self.args.parquet_categorize_nan,
            "clip_min": self.args.parquet_clip_min,
            "clip_max": self.args.parquet_clip_max,
        }

        # Initialize LocalFilesDataset with preload_all_data=True
        # Linear probe REQUIRES all data in memory
        logger.info("Creating LocalFilesDataset with preload_all_data=True...")
        self._local_dataset = LocalFilesDataset(
            data_dir=self.args.parquet_data_dir,
            batch_size=self.args.batch_size,
            scaling_stats_file=self.args.parquet_scaling_stats_file,
            target_col=self.args.parquet_target_col,
            data_files=data_files,
            id_col=self.args.parquet_id_col,
            shuffle=False,  # Don't shuffle for linear probe - need deterministic order
            external_feature_names=external_feature_names,
            num_features_to_keep=self.args.parquet_num_features,
            feature_scaling=feature_scaling,
            sample_rates=None,  # No sampling for linear probe
            preload_all_data=True,  # REQUIRED for linear probe
        )

        # Extract data from LocalFilesDataset
        features_tensor = self._local_dataset.all_features_tensor
        target_tensor = self._local_dataset.all_target_tensor

        # Convert to numpy arrays (BaseDataset expects numpy)
        self.X = features_tensor.cpu().numpy()  # [N, D]
        self.y = target_tensor.cpu().numpy()    # [N]

        # Set dimensions
        self.N, self.D = self.X.shape

        # All features are numerical after LocalFilesDataset preprocessing
        # (categoricals are embedded/scaled, NaNs are handled)
        self.num_features = list(range(self.D))
        self.cat_features = []
        self.cardinalities = []

        # Create num_or_cat mapping (all numerical)
        self.num_or_cat = {idx: True for idx in range(self.D)}

        self.is_data_loaded = True

        logger.info(f"ParquetBaseDataset loaded: N={self.N}, D={self.D}")
        logger.info(f"Task type: {self.task_type}")
        logger.info(f"Target shape: {self.y.shape}, dtype: {self.y.dtype}")
        logger.info(f"Features shape: {self.X.shape}, dtype: {self.X.dtype}")

        # Log target distribution for verification
        if self.task_type == TASK_TYPE.BINARY_CLASS:
            n_pos = np.sum(self.y == 1)
            n_neg = np.sum(self.y == 0)
            logger.info(f"Binary class distribution: pos={n_pos}, neg={n_neg}, ratio={n_pos/n_neg:.3f}")
        elif self.task_type == TASK_TYPE.MULTI_CLASS:
            unique_classes = np.unique(self.y)
            logger.info(f"Multi-class distribution: {len(unique_classes)} classes - {unique_classes}")
        else:
            logger.info(f"Regression target: min={self.y.min():.3f}, max={self.y.max():.3f}, mean={self.y.mean():.3f}")

    def _validate_args(self):
        """Validate that required parquet arguments are provided."""
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
                f"Missing required arguments for ParquetBaseDataset:\n" +
                "\n".join(f"  - {m}" for m in missing)
            )

        # Warn if preload_data is explicitly set to False
        if hasattr(self.args, 'parquet_preload_data') and not self.args.parquet_preload_data:
            logger.warning(
                "parquet_preload_data=False is not supported for linear probe evaluation. "
                "Forcing preload_all_data=True."
            )
