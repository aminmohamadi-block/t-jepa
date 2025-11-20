#!/bin/bash
#SBATCH --job-name=single_gpu_test
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00

echo "========================================================================"
echo "Single-GPU Training Test (1x H100 - Baseline)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs Requested: 1"
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
echo "Starting 1-GPU T-JEPA training (10 epochs, no linear probe)..."
echo "========================================================================"

# Run with 1 GPU
python run.py \
  --use_parquet_dataset=True \
  --data_set=parquet_dataset \
  --parquet_data_dir=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train \
  --parquet_data_files=chunk_10.parquet \
  --parquet_scaling_stats_file=/projects/risk-tabular/general_chargebacks/202207-202407-all-positives-negative-ratio-1.0-v0/train/stats_bootstrap.pq \
  --parquet_feature_names_file=/projects/risk-tabular/general_chargebacks/20240724_feature_selection/xgboost_importances_seed_1_sample_0.5.csv \
  --parquet_num_features=30 \
  --parquet_task_type=binary_class \
  --parquet_preload_data=True \
  --probe_cadence=0 \
  --exp_train_total_epochs=10 \
  --batch_size=512 \
  --model_dim_hidden=64 \
  --model_num_layers=4 \
  --model_num_heads=4 \
  --mp_distributed=False \
  --mp_gpus=1 \
  --project_name=single_gpu_test

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Single-GPU test completed!"
    echo ""
    echo "Extracting timing information..."
    grep -E "time_per_epoch|Total training time|seconds" slurm-${SLURM_JOB_ID}.out | tail -10
else
    echo "✗ FAILED: Training exited with error code $EXIT_CODE"
fi

exit $EXIT_CODE
