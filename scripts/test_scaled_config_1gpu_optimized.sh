#!/bin/bash
#SBATCH --job-name=scaled_1gpu_opt
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00

echo "========================================================================"
echo "Scaled Configuration Test - 1 GPU, 250 Epochs with Optimizations"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs Requested: 1"
echo "========================================================================"
echo "Configuration:"
echo "  - Data: chunk_10 to chunk_20 (11 files, 3.3M samples)"
echo "  - Features: 128 → 256 (after categorize_nan)"
echo "  - Batch size: 4096 per GPU"
echo "  - Learning rate: 1.0e-4 (single GPU baseline)"
echo "  - Scaling: IQR + asinh transform"
echo "  - Epochs: 250 T-JEPA epochs (patience: 250, no early stopping)"
echo "  - Linear probe: Every 25 epochs on last 10% (330K samples, 10 epochs, batch 4096)"
echo "  - Masking: Context 30-40%, Target 30-40% (narrower range, no overlap)"
echo "  - Architecture: No dropout, no feature embeddings, no reg tokens"
echo "  - Weight decay: 1e-5 → 1e-3 (lighter regularization)"
echo "  - Checkpointing: Every 25 epochs"
echo "  - OPTIMIZATIONS ENABLED:"
echo "    * Vectorized masking (16.9x faster)"
echo "    * Mixed precision training (1.84x faster)"
echo "    * Expected combined speedup: ~2x"
echo "========================================================================"

# Activate environment
source ../bin/activate-hermit

# Create logs directory
mkdir -p logs

# Check GPU
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""
nvidia-smi

echo ""
echo "========================================================================"
echo "Starting 1-GPU T-JEPA training with optimizations..."
echo "========================================================================"

# Run with 1 GPU with optimizations enabled
python run.py \
  --use_parquet_dataset=True \
  --data_set=parquet_dataset \
  --parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train \
  --parquet_data_files=chunk_10.parquet,chunk_11.parquet,chunk_12.parquet,chunk_13.parquet,chunk_14.parquet,chunk_15.parquet,chunk_16.parquet,chunk_17.parquet,chunk_18.parquet,chunk_19.parquet,chunk_20.parquet \
  --parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train/stats_bootstrap.pq \
  --parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv \
  --parquet_num_features=128 \
  --parquet_task_type=binary_class \
  --parquet_preload_data=True \
  --parquet_scaling_method=IQR \
  --parquet_infill_value=zero \
  --parquet_transform=asinh \
  --parquet_categorize_nan=True \
  --probe_cadence=25 \
  --exp_train_total_epochs=250 \
  --exp_patience=250 \
  --exp_cache_cadence=25 \
  --batch_size=4096 \
  --exp_start_lr=1.0e-4 \
  --exp_lr=1.0e-3 \
  --exp_final_lr=1.0e-5 \
  --exp_final_weight_decay=1e-3 \
  --model_dim_hidden=64 \
  --model_num_layers=4 \
  --model_num_heads=4 \
  --model_dropout_prob=0 \
  --model_feature_type_embedding=False \
  --model_feature_index_embedding=False \
  --n_cls_tokens=1 \
  --n_reg_tokens=0 \
  --pred_p_dropout=0 \
  --mask_min_ctx_share=0.3 \
  --mask_max_ctx_share=0.4 \
  --mask_min_trgt_share=0.3 \
  --mask_max_trgt_share=0.4 \
  --mp_distributed=False \
  --mp_gpus=1 \
  --use_vectorized_masking=True \
  --model_amp=True \
  --project_name=scaled_config_1gpu_optimized

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: 1-GPU scaled configuration with optimizations completed!"
    echo ""
    echo "Configuration summary:"
    echo "  - Dataset size: ~3.3M samples (11 chunks × 300K)"
    echo "  - Input features: 128 → 256 (after categorize_nan)"
    echo "  - Batch size: 4096 per GPU"
    echo "  - Learning rate: 1.0e-4"
    echo "  - Preprocessing: IQR scaling + asinh transform + NaN categorization"
    echo "  - T-JEPA epochs: 250 (patience: 250, no early stopping)"
    echo "  - Linear probe: 11 runs (epochs 0,25,50,...,250), 10 epochs each, batch 4096"
    echo "  - Masking: Context 30-40%, Target 30-40% (no overlap)"
    echo "  - Architecture: No dropout, no feature embeddings, no reg tokens"
    echo "  - Weight decay: 1e-5 → 1e-3"
    echo "  - Checkpointing: Every 25 epochs"
    echo "  - Optimizations: Vectorized masking + Mixed precision (AMP)"
    echo ""
    echo "Extracting timing information..."
    grep -E "time_per_epoch|Total training time|seconds/epoch|Running probe" slurm-${SLURM_JOB_ID}.out | tail -10
else
    echo "✗ FAILED: Training exited with error code $EXIT_CODE"
    echo ""
    echo "Checking for common issues..."
    echo "Last 50 lines of output:"
    tail -50 slurm-${SLURM_JOB_ID}.out
fi

exit $EXIT_CODE
