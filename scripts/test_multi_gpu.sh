#!/bin/bash
#SBATCH --job-name=multi_gpu_test
#SBATCH --partition=h100
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=00:30:00

echo "========================================================================"
echo "Multi-GPU Training Test (2x H100)"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPUs Requested: 2"
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
echo "Starting 2-GPU T-JEPA training (10 epochs, no linear probe)..."
echo "========================================================================"

# Run with 2 GPUs using torchrun
# Note: launch_tjepa.sh will auto-detect SLURM_GPUS=2 and use torchrun
srun --mpi=pmi2 torchrun --standalone --nnodes=1 --nproc_per_node=2 run.py \
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
  --mp_distributed=True \
  --mp_gpus=2 \
  --project_name=multi_gpu_test

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Multi-GPU test completed!"
    echo ""
    echo "Extracting timing information..."
    grep -E "time_per_epoch|Total training time|seconds/epoch" slurm-${SLURM_JOB_ID}.out | tail -5
else
    echo "✗ FAILED: Training exited with error code $EXIT_CODE"
fi

exit $EXIT_CODE
