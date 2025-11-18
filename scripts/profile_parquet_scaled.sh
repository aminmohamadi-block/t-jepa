#!/bin/bash
#SBATCH --job-name=profile_parquet
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00

echo "========================================================================"
echo "Parquet Scaled Configuration Profiling Run"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"
echo "Configuration:"
echo "  - Data: chunk_10 to chunk_20 (11 files)"
echo "  - Features: 128 (up from 30)"
echo "  - Batch size: 4096"
echo "  - Epochs: 2 (profiling run)"
echo "  - Profiling level: DETAILED"
echo "  - Scaling: IQR + asinh transform"
echo "  - Categorize NaN: True (doubles features to 256)"
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
echo "Starting T-JEPA training with DETAILED profiling..."
echo "========================================================================"

# Run with profiling enabled - 2 epochs only for detailed analysis
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
  --probe_cadence=0 \
  --exp_train_total_epochs=2 \
  --batch_size=4096 \
  --exp_lr=2.828e-4 \
  --model_dim_hidden=64 \
  --model_num_layers=4 \
  --model_num_heads=4 \
  --mp_distributed=False \
  --mp_gpus=1 \
  --profiling_level=DETAILED \
  --profiling_output=profiling_parquet_scaled_2epoch.json \
  --project_name=profile_parquet_scaled

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Profiling run completed!"
    echo ""
    echo "Profiling output saved to: profiling_parquet_scaled_2epoch.json"
    echo ""
    echo "To analyze the profiling results:"
    echo "  python analyze_profiling.py summary profiling_parquet_scaled_2epoch.json"
    echo "  python analyze_profiling.py bottlenecks profiling_parquet_scaled_2epoch.json --threshold 0.05"
    echo ""
    echo "Extracting key profiling metrics..."
    if [ -f "profiling_parquet_scaled_2epoch.json" ]; then
        python -c "
import json
with open('profiling_parquet_scaled_2epoch.json', 'r') as f:
    data = json.load(f)

print('\\n=== Top 10 Operations by Mean Time ===')
items = [(k, v['mean']) for k, v in data.get('summary', {}).items() if 'mean' in v]
items.sort(key=lambda x: x[1], reverse=True)
for i, (op, time) in enumerate(items[:10], 1):
    print(f'{i:2d}. {op:40s} {time*1000:8.2f} ms')
"
    fi
else
    echo "✗ FAILED: Training exited with error code $EXIT_CODE"
    echo ""
    echo "Checking for common issues..."
    echo "Last 50 lines of output:"
    tail -50 slurm-${SLURM_JOB_ID}.out
fi

exit $EXIT_CODE
