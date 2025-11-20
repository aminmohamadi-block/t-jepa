# Parquet Dataset Integration - Current Status & Next Steps

**Last Updated:** 2025-11-04
**Branch:** `task-parquet-scaling`
**Worktree:** `/home/aminmohamadi_squareup_com/projects/t-jepa-parquet-scaling`

## Current State

### Integration Complete ✅

The parquet dataset integration is **fully functional and tested** with real data. All critical bugs have been fixed.

### Recent Bug Fixes (Commit: fcde82c)

1. **encoder.py:295-344** - Critical T-JEPA bug fix
   - Moved `feature_index_embedding` before masking
   - Added zero padding for CLS and REG tokens
   - Issue: Was trying to add 30-dimensional embeddings to 32-dimensional tensor (30 features + CLS + REG)

2. **parquet_dataset.py:111-124** - Linear probe support
   - Added `dataset_name = "parquet_dataset"` attribute
   - Required for OnlineDataset to reload dataset for linear probe evaluation

3. **run.py:266-277** - MaskCollator compatibility
   - Fixed ParquetDataLoaderWrapper batch format
   - MaskCollator expects list of (features, targets) tuples
   - Converted batched tensor to proper format

4. **checkpointer.py:406-407** - Checkpoint robustness
   - Auto-create checkpoint directories if they don't exist
   - Prevents save failures when subdirectories missing

5. **requirements.txt** - System monitoring
   - Added `psutil` dependency for configs.py

### Test Results ✅

**Test Job:** 13609
**Configuration:**
- Dataset: 30 features from chargeback data
- Data files: chunk_10.parquet, chunk_11.parquet
- Preprocessing: IQR scaling, asinh transform
- Training: 2 epochs, batch_size=512
- Model: 4 layers, 4 heads, 64 hidden dim

**Results:**
- Training time: ~77 seconds/epoch
- Batches/epoch: 1171
- Loss progression: 190.12 → 155.66 (decreasing ✅)
- Checkpoints: Saved successfully for epochs 0, 1, 2
- MLflow: Logged to `/groups/block-aird-team/t-jepa-test`

## Data Locations

### Chargeback Dataset
**Base Path:** `/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/`

**Training Data:**
- `train/chunk_10.parquet` (~1.7GB, 600,000 rows)
- `train/chunk_11.parquet` (~1.7GB, 600,000 rows)
- Additional chunks available: chunk_0 through chunk_11

**Metadata Files:**
- `stats_bootstrap.pq` - Scaling statistics (4878 columns including metadata)
- `feature_importance/xgboost_importances_seed_1_sample_0.5.csv` - 1784 ranked features

**Data Characteristics:**
- Target column: `target` (binary classification)
- ID column: `ID` (excluded from features)
- Metadata columns in stats: `row_id`, `TARGET`, `target`, `row_id_hash`
- Actual features: 1784 after excluding metadata

## How to Run

### Basic Command

```bash
python run.py \
  --use_parquet_dataset=True \
  --data_set=parquet_chargeback \
  --parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train \
  --parquet_data_files=chunk_10.parquet,chunk_11.parquet \
  --parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/stats_bootstrap.pq \
  --parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/feature_importance/xgboost_importances_seed_1_sample_0.5.csv \
  --parquet_num_features=256 \
  --parquet_target_col=target \
  --parquet_id_col=ID \
  --parquet_scaling_method=iqr \
  --parquet_transform=asinh \
  --parquet_categorize_nan=False \
  --parquet_preload_data=True \
  --batch_size=2048 \
  --exp_train_total_epochs=100 \
  --project_name=tjepa-parquet-scaling
```

### SLURM Job Template

```bash
#!/bin/bash
#SBATCH --job-name=parquet_scaling
#SBATCH --output=logs/parquet_scaling_%j.out
#SBATCH --error=logs/parquet_scaling_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# Activate environment
source ../bin/activate-hermit

# Create checkpoint directory
mkdir -p checkpoints/parquet_chargeback

# Run training
python run.py \
  --use_parquet_dataset=True \
  --data_set=parquet_chargeback \
  --parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train \
  --parquet_data_files=chunk_10.parquet,chunk_11.parquet \
  --parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/stats_bootstrap.pq \
  --parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/feature_importance/xgboost_importances_seed_1_sample_0.5.csv \
  --parquet_num_features=256 \
  --parquet_preload_data=True \
  --batch_size=2048 \
  --exp_train_total_epochs=100 \
  --project_name=tjepa-parquet-scaling

# Submit with: sbatch scripts/parquet_scaling.sh
```

## Key Configuration Parameters

### Parquet-Specific

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_parquet_dataset` | False | Enable parquet dataset (required) |
| `--parquet_data_dir` | - | Directory containing parquet files |
| `--parquet_data_files` | - | Comma-separated list of parquet files |
| `--parquet_num_features` | - | Number of top features to use |
| `--parquet_feature_names_file` | - | CSV with feature importance rankings |
| `--parquet_scaling_stats_file` | - | Parquet file with scaling statistics |
| `--parquet_scaling_method` | iqr | Scaling method: iqr, standard, minmax |
| `--parquet_transform` | asinh | Transform: asinh, log1p, none |
| `--parquet_categorize_nan` | False | Doubles features with NaN indicators |
| `--parquet_preload_data` | False | Load all data to memory (recommended for small datasets) |
| `--parquet_target_col` | target | Name of target column |
| `--parquet_id_col` | ID | Name of ID column |
| `--parquet_clip_min` | -5.0 | Min clipping value after scaling |
| `--parquet_clip_max` | 5.0 | Max clipping value after scaling |
| `--parquet_infill_value` | 0.0 | Value for NaN infilling |
| `--parquet_shuffle` | True | Shuffle data during loading |

### T-JEPA Training

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `--batch_size` | 2048-4096 | Batch size (scale based on GPU memory) |
| `--exp_train_total_epochs` | 100-300 | Total training epochs |
| `--exp_lr` | 0.0003 | Learning rate |
| `--model_dim_hidden` | 64-256 | Hidden dimension (scale for larger models) |
| `--model_num_layers` | 4-8 | Transformer layers |
| `--mask_min_ctx_share` | 0.15 | Min context mask ratio |
| `--mask_max_ctx_share` | 0.40 | Max context mask ratio |
| `--mask_num_preds` | 2-4 | Number of prediction targets |
| `--probe_cadence` | 20 | Linear probe evaluation frequency |

## Next Steps

### 1. Scaling Study (Priority: High)

**Goal:** Understand how model performance scales with:
- Number of features (128, 256, 512, 1024, 1784)
- Model size (hidden_dim: 64, 128, 256, 512)
- Dataset size (number of chunks: 2, 4, 8, 12)
- Training epochs (50, 100, 200, 300)

**Experiments to Run:**

```bash
# Small model, few features (baseline)
--parquet_num_features=128 --model_dim_hidden=64 --model_num_layers=4

# Medium model, medium features
--parquet_num_features=256 --model_dim_hidden=128 --model_num_layers=6

# Large model, many features
--parquet_num_features=512 --model_dim_hidden=256 --model_num_layers=8

# Full feature set
--parquet_num_features=1784 --model_dim_hidden=512 --model_num_layers=12
```

**Metrics to Track:**
- Training loss curve
- Linear probe validation score
- Training time per epoch
- GPU memory usage
- Representation collapse metrics (KL divergence, variance)

### 2. Linear Probe Evaluation (Priority: High)

**Current Issue:** Linear probe has not been tested with parquet data yet.

**Tasks:**
- Set `--probe_cadence=20` to enable periodic evaluation
- Verify OnlineDataset creates correct embeddings from parquet
- Monitor linear probe validation score over training
- Compare to baseline models (XGBoost, TabNet, etc.)

**Expected Behavior:**
- Linear probe should train every 20 epochs
- Uses frozen target encoder embeddings as input
- Trains supervised model on downstream task
- Validation score should improve as representations improve

### 3. Multi-GPU Training (Priority: Medium)

**Goal:** Scale to multiple GPUs for faster training.

**Changes Needed:**
- Test with `--mp_distributed=True --mp_gpus=4`
- Verify DistributedDataParallel works with ParquetDataLoaderWrapper
- Adjust batch size for multi-GPU (batch_size * num_gpus)

**Expected Speedup:**
- Near-linear scaling with number of GPUs
- 4 GPUs → ~3.5x speedup

### 4. Hyperparameter Optimization (Priority: Medium)

**Use Optuna for systematic tuning:**

**Key Hyperparameters to Tune:**
- Learning rate (1e-4 to 1e-3)
- Model dimensions (64, 128, 256)
- Masking ratios (context: 0.1-0.5, target: 0.1-0.7)
- Number of prediction targets (2-6)
- Dropout probability (0.0-0.3)

**Run Optuna:**
```bash
python run_optuna.py \
  --config_file=configs/parquet_chargeback_optuna.yaml \
  --n_trials=100
```

### 5. Production Scaling (Priority: Low)

**For very large datasets:**

**Memory Optimization:**
- Use `--parquet_preload_data=False` for streaming
- Implement chunk-based training
- Profile memory usage with different batch sizes

**Data Pipeline:**
- Benchmark different scaling methods (IQR vs standard)
- Test impact of `--parquet_categorize_nan=True`
- Evaluate different transforms (asinh vs log1p)

**Checkpointing:**
- Implement resume from checkpoint
- Use `--load_from_checkpoint=True --load_path=<path>`

## Known Limitations

1. **Linear Probe Not Fully Tested:** Only basic training tested, not linear probe evaluation
2. **Single GPU Only:** Multi-GPU training not yet validated with parquet
3. **Memory Usage:** Large feature sets (>1000) may require streaming mode
4. **Feature Selection:** Currently uses top-N by importance, could explore other strategies

## MLflow Tracking

**Experiment Location:** `/groups/block-aird-team/t-jepa-test`

**Run Naming Convention:**
```
{data_set}__model_nlyrs_{num_layers}_nheads_{num_heads}_hdim_{hidden_dim}__pred_ovrlap_{overlap}_npreds_{num_preds}__nlyrs_{pred_layers}_activ_{activation}nenc_{num_encs}__lr_{lr}_start_{start_lr}_final_{final_lr}_{timestamp}
```

**Key Metrics Logged:**
- `tjepa_train_loss`: MSE reconstruction loss
- `tjepa_lr`: Current learning rate
- `tjepa_momentum`: EMA coefficient for target encoder
- `linear_probe_metric`: Downstream task performance
- `context_encoder_grad_*`: Gradient statistics
- System metrics: CPU, memory, disk, network

## Environment

**Python Environment:** Hermit (no venv needed)

**Key Dependencies:**
- PyTorch 2.9.0+cu128
- polars 1.35.1
- pyarrow 21.0.0
- psutil 7.1.3

**Activate:**
```bash
source ../bin/activate-hermit
```

## Troubleshooting

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'psutil'`
**Fix:** Already in requirements.txt, should be installed

**Issue:** `Division by zero in momentum scheduler`
**Fix:** Use `--parquet_preload_data=True` to ensure dataloader has length

**Issue:** `Tensor size mismatch in feature_index_embedding`
**Fix:** Already fixed in commit fcde82c

**Issue:** `'TJEPAParquetDataset' object has no attribute 'dataset_name'`
**Fix:** Already fixed in commit fcde82c

**Issue:** `Checkpoint directory doesn't exist`
**Fix:** Already fixed in commit fcde82c, auto-creates directories

### Performance Debugging

**Check GPU utilization:**
```bash
watch -n 1 nvidia-smi
```

**Monitor SLURM job:**
```bash
squeue -u $USER
tail -f logs/parquet_scaling_<job_id>.out
```

**Check MLflow:**
```bash
mlflow ui --backend-store-uri ./mlruns
# Open browser to http://localhost:5000
```

## References

- **CLAUDE.md**: T-JEPA architecture and training details
- **PARQUET_INTEGRATION.md**: Original integration documentation
- **test_parquet_real_data.py**: Working test example
- **scripts/test_parquet_slurm.sh**: Working SLURM job script

## Contact

For issues or questions about this worktree:
- Check git log for recent changes
- Review test results in `logs/test_parquet_13609.out`
- Refer to MLflow experiment for training metrics
