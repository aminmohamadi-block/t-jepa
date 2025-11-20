#!/bin/bash
#SBATCH --job-name=test_linear_probe
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "========================================================================"
echo "Linear Probe Integration Test - End-to-End"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"

# Activate environment
source ../bin/activate-hermit

# Create logs directory if it doesn't exist
mkdir -p logs

# Check GPU
nvidia-smi

echo ""
echo "========================================================================"
echo "Starting T-JEPA training with linear probe..."
echo "========================================================================"

# Run training with linear probe
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
  --probe_cadence=2 \
  --exp_train_total_epochs=5 \
  --batch_size=512 \
  --model_dim_hidden=64 \
  --model_num_layers=4 \
  --model_num_heads=4 \
  --test=True \
  --project_name=linear_probe_test

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Linear probe integration test passed!"
else
    echo "✗ FAILED: Training exited with error code $EXIT_CODE"
fi

exit $EXIT_CODE
