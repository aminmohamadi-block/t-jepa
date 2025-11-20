#!/bin/bash
#SBATCH --job-name=scaled_500ep
#SBATCH --partition=h100
#SBATCH --gpus=4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00

echo "========================================================================"
echo "Scaled Configuration Test - 4 GPUs, 500 Epochs with Linear Probe"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs Requested: 4"
echo "========================================================================"
echo "Configuration:"
echo "  - Data: chunk_10 to chunk_20 (11 files, 3.3M samples)"
echo "  - Features: 128 → 256 (after categorize_nan)"
echo "  - Batch size: 4096 per GPU (16384 total effective)"
echo "  - Learning rate: 2.828e-4 (scaled by sqrt(8))"
echo "  - Scaling: IQR + asinh transform"
echo "  - Epochs: 500 T-JEPA epochs (patience: 500, no early stopping)"
echo "  - Linear probe: Every 10 epochs on last 10% (330K samples, 10 epochs, batch 4096)"
echo "========================================================================"

# Activate environment
source ../bin/activate-hermit

# Create logs directory
mkdir -p logs

# Check GPUs
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""
nvidia-smi

echo ""
echo "========================================================================"
echo "SLURM Environment:"
echo "========================================================================"
echo "SLURM_GPUS: ${SLURM_GPUS}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS}"
echo "SLURM_GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE}"
echo "========================================================================"

echo ""
echo "========================================================================"
echo "Starting 4-GPU T-JEPA training with scaled configuration..."
echo "========================================================================"

# Run with 4 GPUs using torchrun
srun --mpi=pmi2 torchrun --standalone --nnodes=1 --nproc_per_node=4 run.py \
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
  --probe_cadence=10 \
  --exp_train_total_epochs=500 \
  --exp_patience=500 \
  --batch_size=4096 \
  --exp_lr=2.828e-4 \
  --model_dim_hidden=64 \
  --model_num_layers=4 \
  --model_num_heads=4 \
  --mp_distributed=True \
  --mp_gpus=4 \
  --project_name=scaled_config_4gpu_test

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: 4-GPU scaled configuration test completed!"
    echo ""
    echo "Configuration summary:"
    echo "  - Dataset size: ~3.3M samples (11 chunks × 300K)"
    echo "  - Input features: 128 → 256 (after categorize_nan)"
    echo "  - Batch size: 4096 per GPU × 4 GPUs = 16384 total"
    echo "  - Learning rate: 2.828e-4"
    echo "  - Preprocessing: IQR scaling + asinh transform + NaN categorization"
    echo "  - T-JEPA epochs: 500 (patience: 500, no early stopping)"
    echo "  - Linear probe: 51 runs (epochs 0,10,20,...,500), 10 epochs each, batch 4096"
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
