#!/bin/bash
#SBATCH --job-name=profile_higgs_detailed
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/profile_higgs_%j.out
#SBATCH --error=logs/profile_higgs_%j.err

# Detailed profiling run for HIGGS dataset
# 10 epochs with full configuration from launch_tjepa.sh
# Batch size 1024 for realistic throughput

echo "=================================================="
echo "T-JEPA DETAILED PROFILING - HIGGS DATASET"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=================================================="
echo "Configuration:"
echo "  Dataset: higgs"
echo "  Epochs: 10"
echo "  Batch size: 1024"
echo "  Model: 16 layers, 64 hidden dim (from launch_tjepa.sh defaults)"
echo "  Profiling: DETAILED with summary every epoch"
echo "=================================================="

# Activate environment
source ../bin/activate-hermit
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Create logs directory
mkdir -p logs

# Run with detailed profiling
# Using default config from launch_tjepa.sh but with higgs dataset
./scripts/launch_tjepa.sh \
  --data_path ./datasets \
  --data_set higgs \
  --batch_size 1024 \
  --exp_train_total_epochs 10 \
  --probe_cadence 0 \
  --tag higgs_baseline \
  --profiling_level DETAILED \
  --profiling_summary_every 1

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=================================================="
echo "PROFILING RUN COMPLETED"
echo "=================================================="
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"

    # Find profiling results
    PROF_FILE=$(ls -t profiling_higgs_baseline*.json 2>/dev/null | head -1)

    if [ -n "$PROF_FILE" ]; then
        echo "✓ Profiling results found: $PROF_FILE"
        echo ""
        echo "To analyze results, run:"
        echo "  python analyze_profiling.py summary $PROF_FILE"
        echo "  python analyze_profiling.py bottlenecks $PROF_FILE --threshold 0.05"
        echo ""

        # Show file size
        FILE_SIZE=$(ls -lh "$PROF_FILE" | awk '{print $5}')
        echo "Profiling data size: $FILE_SIZE"

        # Quick stats
        python3 << EOF
import json
with open('$PROF_FILE') as f:
    data = json.load(f)
print(f"Operations captured: {len(data['summary'])}")
print(f"Detailed traces: {len(data['detailed_traces'])}")
print(f"Profiling level: {data['profiling_level']}")

# Top 5 by total time
summary = sorted(data['summary'], key=lambda x: x['total_time'], reverse=True)
print("\nTop 5 operations by total time:")
for i, s in enumerate(summary[:5], 1):
    print(f"  {i}. {s['name']}: {s['total_time']:.2f}s (mean: {s['mean_time']*1000:.1f}ms, count: {s['count']})")
EOF
    else
        echo "⚠️  Warning: Profiling results file not found"
    fi
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "Check error log: logs/profile_higgs_${SLURM_JOB_ID}.err"
fi

echo "=================================================="
