#!/bin/bash
#SBATCH --job-name=prof_vec
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prof_vec_%j.out
#SBATCH --error=logs/prof_vec_%j.err

echo "========================================================================"
echo "Parquet Scaled Profiling Run - WITH VECTORIZED MASKING"
echo "========================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "========================================================================"
echo "Configuration:"
echo "  - Data: chunk_10 to chunk_20 (11 files)"
echo "  - Features: 128 → 256 (with categorize_nan)"
echo "  - Batch size: 4096"
echo "  - Epochs: 1 (profiling run)"
echo "  - Profiling level: DETAILED"
echo "  - Scaling: IQR + asinh transform"
echo "  - OPTIMIZATION: Vectorized masking ENABLED (16.9x faster)"
echo "========================================================================"

# Activate environment
source ../bin/activate-hermit

# Create logs directory
mkdir -p logs

# Check GPU
echo "Available GPUs:"
nvidia-smi --list-gpus
echo ""

echo ""
echo "========================================================================"
echo "Starting T-JEPA training with VECTORIZED masking and profiling..."
echo "========================================================================"

# Run with vectorized masking and profiling - 1 epoch for comparison
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
  --exp_train_total_epochs=1 \
  --batch_size=4096 \
  --exp_lr=2.828e-4 \
  --model_dim_hidden=64 \
  --model_num_layers=4 \
  --model_num_heads=4 \
  --mp_distributed=False \
  --mp_gpus=1 \
  --use_vectorized_masking=True \
  --profiling_level=DETAILED \
  --profiling_output=profiling_parquet_vectorized_1epoch.json \
  --project_name=profile_parquet_vectorized

EXIT_CODE=$?

echo ""
echo "========================================================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "========================================================================"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ SUCCESS: Profiling run with vectorized masking completed!"
    echo ""
    echo "Profiling output saved to: profiling_parquet_vectorized_1epoch.json"
    echo ""
    echo "To compare with original implementation:"
    echo "  Original (2 epochs): profiling_parquet_scaled_2epoch.json"
    echo "  Vectorized (1 epoch): profiling_parquet_vectorized_1epoch.json"
    echo ""
    echo "Extracting key profiling metrics..."
    if [ -f "profiling_parquet_vectorized_1epoch.json" ]; then
        python3 -c "
import json
import sys

try:
    with open('profiling_parquet_vectorized_1epoch.json', 'r') as f:
        data = json.load(f)

    summary = data.get('summary', [])
    if isinstance(summary, list):
        # Convert list to dict
        summary_dict = {entry.get('name', ''): entry for entry in summary}
    else:
        summary_dict = summary

    print('\\n=== Key Performance Metrics ===')

    # Get iteration time
    if 'iteration' in summary_dict:
        iter_data = summary_dict['iteration']
        iter_mean = iter_data.get('mean_time', 0) * 1000
        iter_count = iter_data.get('count', 0)
        print(f'Iteration time: {iter_mean:.2f} ms/iteration')
        print(f'Total iterations: {iter_count}')
        print(f'Throughput: {1000/iter_mean:.2f} iterations/second')

    print('\\n=== Top 15 Operations by Mean Time ===')

    # Get all operations with mean_time
    ops = []
    for name, entry in summary_dict.items():
        mean_time = entry.get('mean_time', 0)
        count = entry.get('count', 0)
        if mean_time > 0 and name != 'epoch':
            ops.append((name, mean_time * 1000, count))

    ops.sort(key=lambda x: x[1], reverse=True)

    for i, (op, time_ms, count) in enumerate(ops[:15], 1):
        print(f'{i:2d}. {op:35s} {time_ms:8.2f} ms  (count: {count})')

    # Highlight mask_collation improvement
    print('\\n=== Mask Generation Performance ===')
    if 'mask_collation' in summary_dict:
        mask_time = summary_dict['mask_collation'].get('mean_time', 0) * 1000
        print(f'Mask collation time: {mask_time:.2f} ms')
        print(f'Expected improvement: ~17x faster vs original')
        print(f'Original time was: ~293 ms')
        print(f'Expected new time: ~17 ms')
        print(f'Actual speedup: {293/mask_time:.1f}x')

except Exception as e:
    print(f'Error parsing profiling data: {e}')
    import traceback
    traceback.print_exc()
"
    fi
else
    echo "✗ FAILED: Training failed with exit code $EXIT_CODE"
    echo ""
    echo "Check the error log for details:"
    echo "  logs/prof_vec_${SLURM_JOB_ID}.err"
fi

echo ""
echo "========================================================================"
echo "End time: $(date)"
echo "========================================================================"