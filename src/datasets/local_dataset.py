from typing import Optional, List, Tuple, Dict
import numpy as np
import polars as pl
import pandas as pd
import torch
import logging
from pathlib import Path
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def raise_value_error(error_msg: str):
    logger.error(error_msg)
    raise ValueError(error_msg)


class LocalFilesDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        scaling_stats_file: str,
        target_col: str = 'target',
        data_files: Optional[List[str]] = None,
        id_col: str = 'ID',
        shuffle: bool = False,
        external_feature_names: Optional[List[str]] = None,
        num_features_to_keep: Optional[int] = None,
        feature_scaling: dict = None,
        sample_rates: Optional[Dict[str, float]] = None,
        preload_all_data: bool = False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.target_col = target_col
        self.id_col = id_col
        self.scaling_stats_file = scaling_stats_file
        self.shuffle = shuffle
        self.feature_scaling = {
            "method": "mean_std",
            "clip_min": None,
            "clip_max": None,
            "infill_value": "global_mean",
            "transform": None,
            "categorize_nan": False,
        }
        self.feature_scaling.update(feature_scaling or {})
        self.sample_rates = sample_rates

        # Validate all inputs and process everything in one place
        self.feature_names_to_keep, self.feature_stats = self._validate_and_process_all(
            external_feature_names, num_features_to_keep
        )

        # Discover data files
        self.batched_files = self._discover_data_files(data_dir, data_files)
        self.create_scaling_tensors()
        self.preload_all_data = preload_all_data
        if preload_all_data:
            self.all_features_tensor, self.all_target_tensor, self.all_ids = self.load_all_data()
            self.batched_length = len(self.all_features_tensor) // self.batch_size
        else:
            self.batched_length = None # Implement length later, I probably need to load all files and determine number of batches in each file


    def _validate_and_process_all(
        self,
        external_feature_names: Optional[List[str]],
        num_features_to_keep: Optional[int]
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Coordinate validation and processing of all inputs.

        Returns:
            Tuple of (feature_names_to_keep, mean_tensor, std_dev_tensor)
        """
        # Validate basic inputs
        self._validate_basic_inputs(external_feature_names, num_features_to_keep)

        # Load stats file
        loaded_feature_stats = pd.read_parquet(self.scaling_stats_file)

        # Validate inputs that require loaded stats
        self._validate_with_stats(external_feature_names, loaded_feature_stats.columns)

        # Process feature selection (no validation, just logic)
        feature_names_to_keep = self._select_features(
            external_feature_names, num_features_to_keep, loaded_feature_stats.columns
        )

        # Prepare final scaling arrays
        feature_stats_to_keep = loaded_feature_stats[feature_names_to_keep]

        return feature_names_to_keep, feature_stats_to_keep

    def _validate_basic_inputs(
        self,
        external_feature_names: Optional[List[str]],
        num_features_to_keep: Optional[int]
    ) -> None:
        """Validate basic input parameters without loading any files."""
        if external_feature_names is not None and not external_feature_names:
            raise_value_error("`external_feature_names` cannot be an empty list. Use None if you want to use all features from stats file.")

        if external_feature_names is None and num_features_to_keep is not None:
            raise_value_error(
                "`num_features_to_keep` can only be specified when `external_feature_names` is provided. "
                "When `external_feature_names` is None, all features from the scaling stats file are used."
            )

        if num_features_to_keep is not None and (not isinstance(num_features_to_keep, int) or num_features_to_keep <= 0):
            raise_value_error(f"`num_features_to_keep` was provided as '{num_features_to_keep}', but must be a positive integer or None.")

        if not self.scaling_stats_file or not Path(self.scaling_stats_file).exists():
            raise FileNotFoundError(f"Scaling stats file is required and was not found at: {self.scaling_stats_file}")


    def _validate_with_stats(
        self,
        external_feature_names: Optional[List[str]],
        all_feature_names_from_stats: List[str]
    ) -> None:
        """Validate inputs that require loaded stats data."""
        if external_feature_names is not None:
            # Validate external_feature_names against the stats file
            if not set(external_feature_names).issubset(set(all_feature_names_from_stats)):
                missing_from_stats = list(set(external_feature_names) - set(all_feature_names_from_stats))
                raise_value_error(
                    f"External feature names contain names not found in the stats file's 'feature_names' array: "
                    f"{missing_from_stats[:10]}... ({len(missing_from_stats)} total missing). "
                    f"Stats file 'feature_names' has {len(all_feature_names_from_stats)} entries."
                )

    def _select_features(
        self,
        external_feature_names: Optional[List[str]],
        num_features_to_keep: Optional[int],
        all_feature_names_from_stats: List[str]
    ) -> List[str]:
        """Select final features based on inputs (no validation, pure logic).

        Returns:
            List of feature names to keep
        """
        if external_feature_names is not None:
            logger.info(f"Dataset: Using externally provided list of {len(external_feature_names)} feature names.")
            candidate_feature_names = external_feature_names

            # Determine num_features_to_keep, defaulting to len(candidate_feature_names)
            if num_features_to_keep is None:
                num_features_to_keep = len(candidate_feature_names)
                logger.info(f"Dataset: `num_features_to_keep` not specified. Using all {num_features_to_keep} external features.")

            # Take the first min(num_features_to_keep, len(candidate_feature_names)) features
            actual_features_to_use = min(num_features_to_keep, len(candidate_feature_names))
            feature_names_to_keep = candidate_feature_names[:actual_features_to_use]

            if actual_features_to_use < num_features_to_keep:
                logger.warning(f"Requested {num_features_to_keep} features, but only {len(candidate_feature_names)} available. Using {actual_features_to_use} features.")
            else:
                logger.info(f"Dataset: Using {actual_features_to_use} features from the externally provided list.")

        else:
            # Use all features from stats file (external_feature_names is None)
            logger.info(f"Dataset: No external feature names provided. Using all {len(all_feature_names_from_stats)} features from stats file.")
            feature_names_to_keep = all_feature_names_from_stats

        logger.info(f"Dataset: Final list of features to keep has {len(feature_names_to_keep)} entries.")
        return feature_names_to_keep


    def _discover_data_files(self, data_dir: str, data_files: Optional[List[str]]) -> List[str]:
        """Discover parquet files in the data directory.

        Returns:
            List of parquet file paths
        """
        path = Path(data_dir)
        batched_files = [str(f) for f in path.glob("**/*.parquet")]

        if data_files is not None:
            #confirm filename (not including path) is in data_files
            batched_files = [f for f in batched_files if Path(f).name in data_files]

        if not batched_files:
            raise_value_error(f"No parquet files found in {data_dir}")
        batched_files = sorted(
            batched_files,
            key=lambda x: int(x.split("chunk_")[1].split(".")[0])
        ) # sort them by chink to preserve the temporal order if needed
        logger.info(f"Found {len(batched_files)} parquet files in {data_dir}")
        return batched_files

    def create_scaling_tensors(self) -> None:
        method = self.feature_scaling["method"]
        stats_df = self.feature_stats.T
        if method == "mean_std":
            subtract_tensor = stats_df["mean"]
            divide_tensor = stats_df["std"]
        elif method == "min_max":
            subtract_tensor = stats_df["min"]
            divide_tensor = stats_df["max"] - stats_df["min"]
        elif (method == "none") or (method is None):
            subtract_tensor = stats_df["mean"] * 0
            divide_tensor = stats_df["std"] * 0 + 1
        elif method == "1_percentile":
            subtract_tensor = stats_df["1%"]
            divide_tensor = stats_df["99%"] - stats_df["1%"]
            # Default to min / max if 1% == 99%
            invalid_mask = stats_df["1%"] == stats_df["99%"]
            subtract_tensor[invalid_mask] = stats_df["min"][invalid_mask]
            divide_tensor[invalid_mask] = stats_df["max"][invalid_mask] - stats_df["min"][invalid_mask]
        elif method == "5_percentile":
            subtract_tensor = stats_df["5%"]
            divide_tensor = stats_df["95%"] - stats_df["5%"]
            # Default to min / max if 5% == 95%
            invalid_mask = stats_df["5%"] == stats_df["95%"]
            subtract_tensor[invalid_mask] = stats_df["min"][invalid_mask]
            divide_tensor[invalid_mask] = stats_df["max"][invalid_mask] - stats_df["min"][invalid_mask]
        elif method == "IQR":
            # FIXED: Use median (Q50) for centering to match bin computation
            subtract_tensor = stats_df["50%"]
            divide_tensor = stats_df["75%"] - stats_df["25%"]
            # Default to min / max if IQR == 0
            invalid_mask = stats_df["25%"] == stats_df["75%"]
            subtract_tensor[invalid_mask] = stats_df["min"][invalid_mask]
            divide_tensor[invalid_mask] = stats_df["max"][invalid_mask] - stats_df["min"][invalid_mask]
        else:
            logger.warning(f"Unknown scaling method: {method}. Defaulting to none.")
            subtract_tensor = stats_df["mean"] * 0
            divide_tensor = stats_df["std"] * 0 + 1

        #Fallback for zero division
        divide_tensor[divide_tensor == 0] = 1
        # Convert to tensors
        self.subtract_tensor = torch.from_numpy(subtract_tensor.to_numpy()).to(torch.float32)
        self.divide_tensor = torch.from_numpy(divide_tensor.to_numpy()).to(torch.float32)
        self.feature_order = self.feature_stats.index.tolist()

    def get_infill_tensor(self, chunk_idx: int) -> torch.Tensor:
        #match case would be cleaner
        infill_value = self.feature_scaling["infill_value"]
        stat_df = self.feature_stats.T
        if infill_value == "global_mean":
            infill_tensor = stat_df["mean"]
        elif infill_value == "previous_mean":
            tensor_key = f"chunk_{max(0, chunk_idx - 1)}.parquet"
            infill_tensor = stat_df[tensor_key]
        elif "chunk" in ( infill_value or ""):
            infill_tensor = stat_df[infill_value]
        elif infill_value == "zero":
            infill_tensor = stat_df["mean"] * 0
        else:
            #When set to None we don't replace nan values
            infill_tensor = stat_df["mean"] * 0 + float("nan")

        infill_tensor = torch.from_numpy(infill_tensor.to_numpy()).to(torch.float32)
        return infill_tensor

    def load_all_data(self) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """
        Memory-safe preload:
        • count rows first (metadata / streaming) to preallocate exactly once
        • fill preallocated storage incrementally
        • avoid list accumulation and torch.cat
        • never shuffle here (shuffle happens lazily in __iter__)
        • on RAM OOM, fall back to disk-backed numpy.memmap so preloading still works
        """
        files = self.batched_files
        logger.info(f"Preloading {len(files)} files (memory-first, memmap fallback).")

        # We do NOT shuffle here — that would force allocating full-size copies.
        # Shuffling is performed lazily in __iter__ via index permutations.

        # Determine total rows we will keep (sampling aware).
        sampling = self.sample_rates is not None and (
            self.sample_rates.get('pos', 1.0) != 1.0 or self.sample_rates.get('neg', 1.0) != 1.0
        )
        if sampling:
            pos_rate = self.sample_rates.get('pos', 1.0)
            neg_rate = self.sample_rates.get('neg', 1.0)
            total_rows_est = 0
            per_file_expected = []
            for f in files:
                pos, neg = self._fast_pos_neg_counts(f)
                keep = int(pos * pos_rate + 0.5) + int(neg * neg_rate + 0.5)
                total_rows_est += keep
                per_file_expected.append(keep)
        else:
            total_rows_est = sum(self._fast_row_count(f) for f in files)
            per_file_expected = None

        n_feat = len(self.feature_names_to_keep)
        # Account for categorize_nan doubling the features
        if self.feature_scaling.get("categorize_nan", False):
            n_feat *= 2
        logger.info(f"Estimated rows to preload: {total_rows_est}, features: {n_feat} (categorize_nan={self.feature_scaling.get('categorize_nan', False)})")

        # Try RAM allocation first; fall back to memmap if needed
        use_memmap = False
        self._mm_features = None
        self._mm_targets = None
        self._mm_ids = None
        cache_dir = None

        try:
            all_features = torch.empty((total_rows_est, n_feat), dtype=torch.float32)
            all_targets = torch.empty((total_rows_est,), dtype=torch.float32)
            logger.info("Allocated preloaded tensors in RAM.")
        except Exception as e:
            logger.warning(f"RAM allocation failed ({e}). Falling back to on-disk memmap to avoid OOM.")
            use_memmap = True
            # Put memmaps next to the first data file under a cache folder.
            cache_dir = Path(files[0]).parent / ".preload_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            f_path = cache_dir / f"features_{total_rows_est}x{n_feat}.f32.mmap"
            t_path = cache_dir / f"targets_{total_rows_est}.f32.mmap"
            self._mm_features = np.memmap(f_path, dtype=np.float32, mode="w+", shape=(total_rows_est, n_feat))
            self._mm_targets  = np.memmap(t_path, dtype=np.float32, mode="w+", shape=(total_rows_est,))
            # Keep memmap objects alive on self; Torch tensors will share their storage.
            all_features = torch.from_numpy(self._mm_features)
            all_targets  = torch.from_numpy(self._mm_targets)
            logger.info(f"Memmap backing created at: {cache_dir}")

        # Defer id allocation until we know dtype from the first loaded chunk
        all_ids: Optional[np.ndarray] = None
        wrote = 0

        for i, file_path in enumerate(files):
            # Get infill tensor for this chunk
            file_idx = int(file_path.split("chunk_")[1].split(".")[0])
            infill_tensor = self.get_infill_tensor(file_idx)

            features_tensor, target_tensor, ids = process_file(
                file_path=file_path,
                target_col=self.target_col,
                id_col=self.id_col,
                subtract_tensor=self.subtract_tensor,
                divide_tensor=self.divide_tensor,
                infill_tensor=infill_tensor,
                features_to_keep=self.feature_names_to_keep,
                feature_scaling=self.feature_scaling,
                sample_rates=self.sample_rates
            )
            n = target_tensor.shape[0]
            if n == 0:
                continue

            # Initialize ids storage (RAM or memmap) lazily based on dtype of first chunk.
            if all_ids is None:
                id_dtype = ids.dtype
                try:
                    if use_memmap and id_dtype.kind in ("i", "u", "f", "b", "M", "m"):
                        # numeric/time dtypes -> safe for memmap
                        id_path = cache_dir / f"ids_{total_rows_est}.{id_dtype.str.replace('|','').replace('<','').replace('>','')}.mmap"
                        self._mm_ids = np.memmap(id_path, dtype=id_dtype, mode="w+", shape=(total_rows_est,))
                        all_ids = self._mm_ids
                        logger.info(f"IDs memmap backing created at: {id_path}")
                    else:
                        # RAM allocation; for string/object, memmap is not safe
                        all_ids = np.empty((total_rows_est,), dtype=id_dtype)
                except Exception as ex:
                    logger.warning(f"Falling back to in-RAM object array for ids due to dtype {id_dtype}: {ex}")
                    all_ids = np.empty((total_rows_est,), dtype=object)

            # In-place fill of the preallocated storage
            all_features[wrote:wrote+n].copy_(features_tensor)
            all_targets[wrote:wrote+n].copy_(target_tensor)
            all_ids[wrote:wrote+n] = ids
            wrote += n

            # release chunk memory ASAP
            del features_tensor, target_tensor, ids

        if wrote != total_rows_est:
            logger.info(f"Trim preload view from {total_rows_est} to actual {wrote} rows.")
            all_features = all_features[:wrote]
            all_targets  = all_targets[:wrote]
            all_ids      = all_ids[:wrote]

        # DO NOT SHUFFLE HERE. Shuffling is handled lazily in __iter__.
        return all_features, all_targets, all_ids

    def __len__(self):
        return self.batched_length


    def __iter__(self):
        if self.preload_all_data:
            # Lazy shuffle via a permutation of indices; never materialize full shuffled copies.
            N = len(self.all_features_tensor)
            if self.shuffle:
                indices = torch.randperm(N)
            else:
                indices = torch.arange(N)

            for j in range(0, N, self.batch_size):
                index_subset = indices[j:j + self.batch_size]
                if len(index_subset) < self.batch_size:
                    return
                # features/targets are torch; ids is numpy — convert index for ids
                ids_subset = self.all_ids[index_subset.cpu().numpy()]
                yield self.all_features_tensor[index_subset], self.all_target_tensor[index_subset], ids_subset
            return

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            this_workers_items = get_worker_items(worker_info.id, worker_info.num_workers, len(self.batched_files))
            files = np.array(self.batched_files)[this_workers_items].tolist()
        else:
            files = self.batched_files

        if self.shuffle:
            np.random.shuffle(files)


        for file_path in files:
            file_idx = int(file_path.split("chunk_")[1].split(".")[0])
            infill_tensor = self.get_infill_tensor(file_idx)
            logger.info(f"Loading file: {file_path}")
            features_tensor, target_tensor, ids = process_file(
                file_path=file_path,
                target_col=self.target_col,
                id_col=self.id_col,
                subtract_tensor=self.subtract_tensor,
                divide_tensor=self.divide_tensor,
                infill_tensor=infill_tensor,
                features_to_keep=self.feature_names_to_keep,
                feature_scaling=self.feature_scaling,
                sample_rates=self.sample_rates
            )

            # We want to shuffle each of these:
            if self.shuffle:
                num_rows = features_tensor.shape[0]
                indices = torch.randperm(num_rows)
                features_tensor = features_tensor[indices]
                target_tensor = target_tensor[indices]
                ids = ids[indices]
            num_rows = features_tensor.shape[0]
            for j in range(0, num_rows, self.batch_size):
                end_idx = min(j + self.batch_size, num_rows)
                if j == end_idx:
                    continue
                yield features_tensor[j:end_idx], target_tensor[j:end_idx], ids[j:end_idx]

    def _fast_row_count(self, file_path: str) -> int:
        """Return number of rows in a Parquet file without loading it."""
        # Try metadata-only path via pyarrow
        if pq is not None:
            try:
                return pq.ParquetFile(file_path).metadata.num_rows
            except Exception:
                pass
        # Fallback: streaming count via Polars (still memory-light)
        try:
            lf = pl.scan_parquet(file_path)
            out = lf.select(pl.len().alias("n")).collect(streaming=True)
            return int(out["n"][0])
        except Exception as e:
            logger.warning(f"Row count fallback failed for {file_path}: {e}; reading one column.")
            return int(pl.read_parquet(file_path, columns=[self.id_col]).height)

    def _fast_pos_neg_counts(self, file_path: str) -> Tuple[int, int]:
        """Return (pos_count, neg_count) using streaming to respect sample_rates."""
        lf = pl.scan_parquet(file_path)
        # Cast to int to handle bool targets robustly
        expr = pl.col(self.target_col).cast(pl.Int64)
        out = lf.select(
            pos=expr.sum().fill_null(0),
            total=pl.len()
        ).collect(streaming=True)
        pos = int(out["pos"][0] or 0)
        total = int(out["total"][0])
        neg = total - pos
        return pos, neg

def process_file(
    file_path: str,
    target_col: str,
    id_col: str,
    subtract_tensor: torch.Tensor,
    divide_tensor: torch.Tensor,
    infill_tensor: torch.Tensor,
    features_to_keep: List[str],
    feature_scaling: dict,
    sample_rates: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Reads a cleaned Parquet file, selects features based on features_to_keep,
    optionally scales them using provided stats, and extracts the target.

    MEMORY FIX: only read the columns we need from disk.
    """
    try:
        required_cols = list(dict.fromkeys(features_to_keep + [target_col, id_col]))
        logger.info(f"Reading file (cols={len(required_cols)}): {file_path}")
        df = pl.read_parquet(file_path, columns=required_cols)
    except Exception as e:
        raise_value_error(f"Error reading {file_path} with columns {required_cols}: {e}")

    if target_col not in df.columns:
        raise_value_error(f"Target column '{target_col}' not found in file {file_path}")

    if missing := set(features_to_keep) - set(df.columns):
        raise_value_error(f"File {file_path} is missing expected columns: {missing}")

    # Apply sampling at the dataframe level if specified (kept in-memory but with only necessary columns)
    if sample_rates is not None:
        pos_rate = sample_rates.get('pos', 1.0)
        neg_rate = sample_rates.get('neg', 1.0)
        seed = sample_rates.get('seed', 1)

        # Note: cast avoids dtype mismatch (bool/int) in comparisons
        t = pl.col(target_col).cast(pl.Int8)
        positives = df.filter(t == 1)
        negatives = df.filter(t == 0)

        pos_seed = seed + len(positives)
        neg_seed = seed + len(negatives)

        positives = positives.sample(n=int(len(positives) * pos_rate + 0.5), shuffle=True, seed=pos_seed)
        negatives = negatives.sample(n=int(len(negatives) * neg_rate + 0.5), shuffle=True, seed=neg_seed)

        df = pl.concat([positives, negatives])
        if len(positives) == 0:
            logger.warning(f"No positives found in {file_path} after sampling")
        if len(negatives) == 0:
            logger.warning(f"No negatives found in {file_path} after sampling")

    features = extract_features(file_path, subtract_tensor, divide_tensor, infill_tensor, features_to_keep, df, feature_scaling)
    target = extract_target(target_col, df)
    ids = extract_id(id_col, df)

    return features, target, ids

def extract_features(file_path, subtract_tensor, divide_tensor, infill_tensor, features_to_keep, df, feature_scaling):
    features_np = df.select(features_to_keep).to_numpy()
    # Copy is required as pytorch does not support non-writeable tensors
    features = torch.from_numpy(features_np.copy()).float()
    if features.shape[1] != len(subtract_tensor):
        raise_value_error(f"Dimension mismatch during scaling {file_path}! Features: {features.shape[1]}, Stats: {len(subtract_tensor)}")
    nan_mask = torch.isnan(features)
    #Nonzero infill is applied before scaling
    if nan_mask.any() and feature_scaling["infill_value"] != "zero":
        features[nan_mask] = infill_tensor.expand_as(features)[nan_mask]
    features.sub_(subtract_tensor).div_(divide_tensor)  # In-place operations
    if feature_scaling["clip_min"] is not None:
        features = torch.clamp(features, min=feature_scaling["clip_min"])
    if feature_scaling["clip_max"] is not None:
        features = torch.clamp(features, max=feature_scaling["clip_max"])
    if feature_scaling["transform"] is not None:
        if feature_scaling["transform"] == "asinh":
            # FIXED: Apply asinh directly to match bin computation
            features = torch.asinh(features)
        else:
            raise_value_error(f"Unknown transform: {feature_scaling['transform']}")
    #Zeroing out the nan values is done after the scaling
    if nan_mask.any() and feature_scaling["infill_value"] == "zero":
        features[nan_mask] = 0
    if feature_scaling["categorize_nan"]:
        nan_mask = nan_mask.float()
        features = torch.cat([features, nan_mask], dim=1)
    return features


def extract_target(target_col, df):
    target_np = df[target_col].to_numpy()
    # Copy is required as pytorch does not support non-writeable tensors
    target = torch.from_numpy(target_np.copy()).float()
    return target


def extract_id(id_col, df):
    return df[id_col].to_numpy()


def get_worker_items(worker_idx, num_workers, num_files):
    """Returns the indices of the files for the given worker."""
    if num_workers <= 0:
        return list(range(num_files))
    chunks = np.array_split(np.arange(num_files), num_workers)
    return chunks[worker_idx].tolist()

def downsample_negatives(all_features_tensor, all_target_tensor, all_ids):
    # find the number of positives
    # include all positives in the output
    # downsample the negatives to match the number of positives
    pos_indices = torch.where(all_target_tensor == 1)[0]
    pos_features = all_features_tensor[pos_indices]
    pos_targets = all_target_tensor[pos_indices]
    neg_indices = torch.where(all_target_tensor == 0)[0]
    neg_features = all_features_tensor[neg_indices]
    neg_targets = all_target_tensor[neg_indices]

    # if there are not enough negatives, raise an error
    if len(neg_indices) < len(pos_indices):
        raise ValueError(f"Not enough negatives to downsample. Positives: {len(pos_indices)}, Negatives: {len(neg_indices)}")

    # downsample the negatives to match the number of positives
    neg_indices = torch.randperm(len(neg_indices))[:len(pos_indices)]
    neg_features = neg_features[neg_indices]
    neg_targets = neg_targets[neg_indices]

    ids = np.concatenate([all_ids[pos_indices], all_ids[neg_indices]])

    return torch.cat([pos_features, neg_features]), torch.cat([pos_targets, neg_targets]), ids
