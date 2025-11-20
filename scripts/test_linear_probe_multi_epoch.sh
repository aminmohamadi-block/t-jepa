#!/bin/bash
#SBATCH --job-name=probe_10epochs
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00

echo "========================================================================"
echo "Linear Probe Multi-Epoch Test (10 epochs, probe at 0, 5, 10)"
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
echo "Starting T-JEPA training with linear probe at epochs 0, 5, 10..."
echo "========================================================================"

# Run training with 10 epochs and probe_cadence=5
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
  --probe_cadence=5 \
  --exp_train_total_epochs=10 \
  --batch_size=512 \
  --model_dim_hidden=64 \
  --model_num_layers=4 \
  --model_num_heads=4 \
  --project_name=linear_probe_multi_epoch_test

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Multi-epoch linear probe test passed!"
    echo ""
    echo "Verifying probe was triggered at expected epochs..."
    echo "Expected: epochs 0, 5, 10"
    grep -n "Running probe at epoch" logs/probe_10epochs_${SLURM_JOB_ID}.out || \
    grep -n "Running probe at epoch" slurm-${SLURM_JOB_ID}.out
else
    echo "✗ FAILED: Training exited with error code $EXIT_CODE"
fi

exit $EXIT_CODE
