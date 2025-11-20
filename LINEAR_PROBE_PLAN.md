# Linear Probe Integration for Parquet Datasets - Implementation Plan

**Date:** 2025-11-07
**Status:** In Progress
**Branch:** task-parquet-scaling

## Overview

Enable linear probe evaluation for parquet datasets by creating a BaseDataset-compatible wrapper that can be used by OnlineDataset.

## Current State Analysis

**Problem:**
- `TJEPAParquetDataset` is not registered in `DATASET_NAME_TO_DATASET_MAP`
- Doesn't inherit from `BaseDataset` (required by OnlineDataset)
- Missing required attributes: `X`, `y`, `task_type`
- Linear probe crashes with `KeyError: 'parquet_dataset'`

**Status from PARQUET_STATUS.md:**
> **Current Issue:** Linear probe has not been tested with parquet data yet.

## Design Decision

Create **ParquetBaseDataset** class that:
1. Inherits from `BaseDataset` (required interface)
2. Uses `LocalFilesDataset` internally (reuse existing logic)
3. Exposes `X`, `y`, `N`, `D`, `task_type` attributes
4. Registered in `DATASET_NAME_TO_DATASET_MAP`

## Implementation Tasks

### Task 1: Create ParquetBaseDataset Class
**File:** `src/datasets/parquet_base_dataset.py`

```python
class ParquetBaseDataset(BaseDataset):
    """
    BaseDataset-compatible wrapper for parquet data.

    Enables linear probe evaluation by:
    - Inheriting from BaseDataset
    - Always preloading data (linear probe needs all data)
    - Exposing X, y as attributes
    - Having task_type attribute
    """
```

**Key Requirements:**
- Force `preload_all_data=True` (linear probe needs all data in memory)
- Set `self.X` as numpy array [N, D]
- Set `self.y` as numpy array [N]
- Set `self.task_type` from args
- Set `self.N`, `self.D`, `self.cardinalities`, `self.num_features`, `self.cat_features`
- Reuse `LocalFilesDataset` internally

### Task 2: Add task_type Configuration
**File:** `src/configs.py`

```python
parser.add_argument(
    "--parquet_task_type",
    type=str,
    default="binary_class",
    choices=["binary_class", "multi_class", "regression"],
    help="Task type for parquet dataset (for linear probe evaluation)"
)
```

### Task 3: Register in Dataset Map
**File:** `src/datasets/dict_to_data.py`

```python
from src.datasets.parquet_base_dataset import ParquetBaseDataset

DATASET_NAME_TO_DATASET_MAP = {
    # ... existing entries ...
    "parquet_dataset": ParquetBaseDataset,
    "parquet_chargeback": ParquetBaseDataset,  # Alias
}
```

### Task 4: Ensure Args Passing
**File:** `src/train.py`

Modify `online_dataset_args` to include all parquet-specific args when using parquet dataset.

**Challenge:** `OnlineDatasetArgs` TypedDict doesn't include parquet args
**Solution:** Pass full args object or extend TypedDict

## Testing Strategy

### Test 1: Unit Test - Dataset Creation
```python
def test_parquet_base_dataset_creation():
    """Verify ParquetBaseDataset matches BaseDataset interface"""
    # Create with parquet args
    # Call load()
    # Assert X, y, N, D, task_type are set
```

### Test 2: Integration Test - OnlineDataset
```python
def test_online_dataset_with_parquet():
    """Verify OnlineDataset can load parquet"""
    # Create OnlineDataset with parquet_dataset name
    # Call load()
    # Assert embeddings generated
```

### Test 3: End-to-End - Linear Probe Training
```bash
python run.py \
  --use_parquet_dataset=True \
  --data_set=parquet_chargeback \
  --parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train \
  --parquet_data_files=chunk_10.parquet \
  --parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/stats_bootstrap.pq \
  --parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/feature_importance/xgboost_importances_seed_1_sample_0.5.csv \
  --parquet_num_features=30 \
  --parquet_preload_data=True \
  --parquet_task_type=binary_class \
  --probe_cadence=2 \
  --exp_train_total_epochs=5 \
  --batch_size=512 \
  --test=True
```

**Verification Checklist:**
- [ ] T-JEPA training starts successfully
- [ ] At epoch 2, linear probe evaluation triggers
- [ ] OnlineDataset loads parquet dataset without KeyError
- [ ] Embeddings are generated (check shape in logs)
- [ ] DataModule splits embeddings into train/val/test (80/10/10)
- [ ] Linear probe model trains without errors
- [ ] Validation metrics logged to MLflow
- [ ] Training continues after probe completes

### Test 4: Deterministic Splits
```python
def test_deterministic_splits():
    """Verify same random_state produces same splits"""
    # Run twice with same seed
    # Assert identical splits
```

## Potential Issues & Mitigations

| Issue | Mitigation |
|-------|-----------|
| Memory: Loading all chunks for linear probe | Use subset of chunks or document memory requirements |
| Args mismatch between training paths | Pass full args, ignore unknown keys in constructors |
| Different preprocessing in T-JEPA vs OnlineDataset | Ensure both use same LocalFilesDataset config |
| Small validation splits with few chunks | Document minimum 3+ chunks needed |
| Task type not properly detected | Require explicit --parquet_task_type arg |

## Documentation Updates

After successful implementation:
- [ ] Update PARQUET_STATUS.md (remove linear probe limitation)
- [ ] Add linear probe example to README
- [ ] Document required args for linear probe
- [ ] Update known limitations section

## Success Criteria

✅ Linear probe runs successfully on parquet dataset
✅ Embeddings generated correctly from target encoder
✅ Train/val/test splits are deterministic
✅ Metrics logged to MLflow
✅ No crashes or errors during probe evaluation
✅ T-JEPA training continues normally after probe

## Implementation Order

1. ✅ Write plan to file
2. Add --parquet_task_type config argument
3. Create ParquetBaseDataset class
4. Register in DATASET_NAME_TO_DATASET_MAP
5. Test dataset creation manually
6. Update train.py args passing if needed
7. Run end-to-end test
8. Debug any issues
9. Create comprehensive test script
10. Update documentation
