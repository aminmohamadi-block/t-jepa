# Parquet Dataset Integration for T-JEPA

This document describes the integration of large-scale parquet datasets from the `risk-tabular-slurm` project into T-JEPA's self-supervised learning framework.

## Overview

T-JEPA can now train on large-scale tabular datasets stored in parquet format, leveraging the sophisticated data loading and preprocessing infrastructure from `risk-tabular-slurm`. This enables training on datasets that are too large to fit in memory and supports advanced preprocessing techniques.

## What Was Changed

### 1. New Files Added

#### `src/datasets/local_dataset.py`
- **Direct copy** from `risk-tabular-slurm/training/local_dataset.py`
- Implements `LocalFilesDataset` class for efficient parquet loading
- Features:
  - Automatic parquet file discovery and chunking
  - Multiple scaling methods (mean_std, IQR, percentiles)
  - Sophisticated NaN handling (infill strategies)
  - Optional transforms (asinh)
  - Memory-efficient preloading with memmap fallback
  - Streaming mode for datasets larger than RAM

#### `src/datasets/parquet_dataset.py`
- **New wrapper class** `TJEPAParquetDataset` that adapts `LocalFilesDataset` for T-JEPA
- Key adaptations:
  - Simplified initialization from argparse arguments
  - Iterator yields only features (targets kept for linear probe)
  - Compatible with T-JEPA's encoder/predictor architecture
  - Handles feature metadata (D, num_features, cardinalities)

### 2. Modified Files

#### `src/configs.py`
- **Added new section**: "Parquet Dataset Config" (lines 67-169)
- New arguments:
  - `--use_parquet_dataset`: Enable parquet dataset mode
  - `--parquet_data_dir`: Directory containing chunk_*.parquet files
  - `--parquet_data_files`: Specific files to use (optional)
  - `--parquet_scaling_stats_file`: Path to stats_bootstrap.pq
  - `--parquet_feature_names_file`: Path to mi.csv for feature ranking
  - `--parquet_num_features`: Number of top features to use
  - `--parquet_scaling_method`: Scaling method (mean_std, IQR, etc.)
  - `--parquet_infill_value`: NaN handling strategy
  - `--parquet_transform`: Optional transform (asinh)
  - `--parquet_categorize_nan`: Add NaN indicator features
  - `--parquet_clip_min/max`: Value clipping

#### `run.py`
- **Lines 30**: Added import for `create_parquet_dataset_from_args`
- **Lines 106-165**: Conditional dataset loading based on `use_parquet_dataset` flag
  - If True: Creates `TJEPAParquetDataset`
  - If False: Uses existing benchmark dataset loading
- **Lines 255-280**: Custom `ParquetDataLoaderWrapper` class
  - Applies masking to pre-batched data from `LocalFilesDataset`
  - Handles device transfer and mask generation

#### `requirements.txt`
- **Added dependencies**:
  - `polars`: High-performance parquet reading
  - `pyarrow`: Parquet file support

## Installation

### 1. Install New Dependencies

```bash
pip install polars pyarrow
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

Run the test script to verify all components:

```bash
python test_parquet_integration.py
```

Expected output:
```
======================================================================
T-JEPA Parquet Dataset Integration Test
======================================================================

Testing imports...
  ✓ polars imported successfully
  ✓ pyarrow imported successfully
  ✓ LocalFilesDataset imported successfully
  ✓ parquet_dataset module imported successfully
  ✓ configs module imported successfully

✓ All imports successful!

...

✓ All tests passed! Parquet integration is ready.
```

## Usage

### Dataset Requirements

Your parquet dataset should have the following structure:

```
/path/to/dataset/
├── chunk_0.parquet       # Training data (oldest)
├── chunk_1.parquet       # Training data
├── ...
├── chunk_N.parquet       # Training/validation data (newest)
├── stats_bootstrap.pq    # Scaling statistics (required)
└── mi.csv               # Feature importance ranking (optional)
```

#### Required Files

1. **Parquet chunks** (`chunk_*.parquet`):
   - Columns: features + target + ID
   - Sorted by temporal order (chunk_0 is oldest)
   - Can have any number of chunks

2. **Scaling statistics** (`stats_bootstrap.pq`):
   - Parquet file with per-feature statistics
   - Columns: mean, std, min, max, percentiles (1%, 5%, 25%, 50%, 75%, 95%, 99%)
   - Per-chunk means for infill (chunk_0.parquet, chunk_1.parquet, ...)
   - Generate using `data_pipeline/stats_generation.py` from risk-tabular-slurm

3. **Feature importance** (`mi.csv`) [optional]:
   - CSV with columns: feature, importance, method
   - Sorted by descending importance
   - Used for feature selection (top-K features)

### Basic Training Command

```bash
python run.py \
  --use_parquet_dataset=True \
  --parquet_data_dir=/path/to/dataset \
  --parquet_scaling_stats_file=/path/to/dataset/stats_bootstrap.pq \
  --parquet_feature_names_file=/path/to/dataset/mi.csv \
  --parquet_num_features=512 \
  --batch_size=1024 \
  --exp_train_total_epochs=100
```

### Advanced Configuration

#### Feature Selection

Use top-K features by mutual information:

```bash
python run.py \
  --use_parquet_dataset=True \
  --parquet_feature_names_file=/path/to/mi.csv \
  --parquet_num_features=256  # Use top 256 features
```

Use all features (no selection):

```bash
python run.py \
  --use_parquet_dataset=True \
  --parquet_num_features=None  # Use all features from stats file
```

#### Preprocessing Options

**Scaling methods**:
- `mean_std`: Standard scaling (x - mean) / std
- `min_max`: Min-max scaling to [0, 1]
- `IQR`: Median-centered IQR scaling (robust to outliers)
- `1_percentile`: Scale using 1% and 99% percentiles
- `5_percentile`: Scale using 5% and 95% percentiles
- `none`: No scaling

```bash
--parquet_scaling_method=IQR
```

**NaN handling**:
- `zero`: Fill NaNs with 0 after scaling
- `global_mean`: Fill with global mean before scaling
- `previous_mean`: Fill with mean from previous chunk
- `chunk_N.parquet`: Fill with mean from specific chunk
- `None`: Keep NaNs as-is

```bash
--parquet_infill_value=zero
```

**Transforms**:
- `asinh`: Apply inverse hyperbolic sine (good for heavy-tailed distributions)
- `None`: No transform

```bash
--parquet_transform=asinh
```

**NaN categorization**:

Add binary indicator features for NaN values (doubles feature count):

```bash
--parquet_categorize_nan=True
```

**Value clipping**:

Clip values after scaling:

```bash
--parquet_clip_min=-3.0 \
--parquet_clip_max=3.0
```

#### Memory Management

**Preload data** (faster, requires RAM):

```bash
--parquet_preload_data=True  # Load entire dataset to RAM/memmap
```

**Stream data** (memory-efficient):

```bash
--parquet_preload_data=False  # Load chunks on-the-fly
```

### Complete Example

Train T-JEPA on large parquet dataset with optimal settings:

```bash
python run.py \
  --use_parquet_dataset=True \
  --parquet_data_dir=/projects/risk-tabular/ctw_payments/processed/train \
  --parquet_data_files=chunk_0.parquet,chunk_1.parquet,chunk_2.parquet \
  --parquet_scaling_stats_file=/projects/risk-tabular/ctw_payments/processed/train/stats_bootstrap.pq \
  --parquet_feature_names_file=/projects/risk-tabular/ctw_payments/processed/train/mi.csv \
  --parquet_num_features=512 \
  --parquet_target_col=target \
  --parquet_id_col=ID \
  --parquet_preload_data=True \
  --parquet_shuffle=True \
  --parquet_scaling_method=IQR \
  --parquet_infill_value=zero \
  --parquet_transform=asinh \
  --parquet_categorize_nan=False \
  --parquet_clip_min=-3.0 \
  --parquet_clip_max=3.0 \
  --batch_size=2048 \
  --exp_train_total_epochs=300 \
  --model_dim_hidden=256 \
  --model_num_layers=12 \
  --exp_lr=0.0003 \
  --mask_min_ctx_share=0.15 \
  --mask_max_ctx_share=0.40
```

## Architecture Details

### Data Flow

```
Parquet files
    ↓
LocalFilesDataset
  - Discovers chunk_*.parquet files
  - Loads scaling stats
  - Selects features (top-K by importance)
  - Applies preprocessing (scale, transform, clip)
  - Batches data
    ↓
TJEPAParquetDataset (wrapper)
  - Yields only features (no targets)
  - Exposes T-JEPA-compatible interface
    ↓
ParquetDataLoaderWrapper
  - Applies masking to pre-batched data
  - Generates masks_enc (context) and masks_pred (target)
    ↓
Trainer
  - Context encoder sees masked features
  - Target encoder sees all features
  - Predictor predicts target representations
```

### Key Design Decisions

1. **Direct copy of `local_dataset.py`**: Minimizes risk and ensures consistency with risk-tabular-slurm preprocessing

2. **Pre-batched data**: `LocalFilesDataset` yields batches, not individual samples. This is efficient for large datasets but requires a wrapper to apply masking.

3. **Unsupervised adaptation**: T-JEPA doesn't need targets during training, but we keep them in `LocalFilesDataset` for potential linear probe evaluation.

4. **All features as numerical**: After preprocessing, all features are treated as numerical (categorical features are pre-encoded in upstream pipeline).

5. **Argparse configuration**: Unlike risk-tabular-slurm's YAML configs, T-JEPA uses argparse. All preprocessing options are exposed as command-line arguments.

## Compatibility

### With Existing T-JEPA Code

- **Fully backward compatible**: Setting `--use_parquet_dataset=False` (default) uses existing CSV/ARFF datasets
- **Same encoder/predictor**: Parquet datasets work with existing T-JEPA architecture
- **Same training loop**: No changes to `src/train.py` required
- **Same masking**: Uses existing `MaskCollator` class

### With risk-tabular-slurm

- **Same preprocessing**: Parquet datasets use identical scaling and transformation logic
- **Same data format**: Can use datasets prepared for supervised training
- **Same statistics**: Uses the same `stats_bootstrap.pq` format

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'polars'`

**Solution**:
```bash
pip install polars pyarrow
```

### Memory Errors

**Problem**: `RAM allocation failed` when preloading

**Solution**: `LocalFilesDataset` automatically falls back to memmap. Check that disk space is available for `.preload_cache/` directory.

Or disable preloading:
```bash
--parquet_preload_data=False
```

### Feature Mismatch

**Problem**: `Features missing from stats file`

**Solution**: Ensure `mi.csv` features match `stats_bootstrap.pq` columns. Regenerate stats if needed.

### Empty Batches

**Problem**: No data yielded from iterator

**Solution**: Check that:
1. Parquet files exist in `--parquet_data_dir`
2. Files match pattern `chunk_*.parquet`
3. Files contain data (not empty)

## Performance Tips

1. **Feature selection**: Use `--parquet_num_features` to limit features and reduce memory/compute
2. **Preload data**: Set `--parquet_preload_data=True` for faster training (if RAM allows)
3. **Batch size**: Larger batches (2048-8192) work well with large datasets
4. **Scaling method**: IQR is robust to outliers and works well for fraud detection data
5. **Distributed training**: Use `--mp_distributed=True` for multi-GPU training

## Future Enhancements

Potential improvements (not yet implemented):

1. **Multi-GPU parquet loading**: Shard parquet files across GPUs for distributed training
2. **Online feature importance**: Update feature selection during training
3. **Temporal splits**: Support separate train/val/test parquet directories
4. **Custom sampling**: Support pos/neg sampling for class imbalance
5. **Linear probe integration**: Automatic linear probe evaluation using parquet targets

## References

- **risk-tabular-slurm**: Original implementation of `LocalFilesDataset`
- **T-JEPA paper**: Joint Embedding Predictive Architecture
- **Polars**: https://pola-rs.github.io/polars/
- **PyArrow**: https://arrow.apache.org/docs/python/

## Contact

For issues or questions about this integration, please refer to:
- T-JEPA documentation: `CLAUDE.md`
- risk-tabular-slurm documentation: `amin_slurm/ablation_prod_v0/README.md`
