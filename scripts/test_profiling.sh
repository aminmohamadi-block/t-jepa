#!/bin/bash
#SBATCH --job-name=test_profiling
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/profiling_test_%j.out
#SBATCH --error=logs/profiling_test_%j.err

# Test script for profiling integration
# This runs a quick 5-epoch training job with DETAILED profiling enabled
# to verify the profiling system works correctly

echo "=================================================="
echo "T-JEPA PROFILING INTEGRATION TEST"
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=================================================="

# Activate environment
source ../bin/activate-hermit
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Run with profiling enabled
./scripts/launch_tjepa.sh \
  --data_path ./datasets \
  --data_set jannis \
  --batch_size 256 \
  --exp_train_total_epochs 5 \
  --model_num_layers 4 \
  --model_dim_hidden 64 \
  --pred_num_layers 2 \
  --probe_cadence 0 \
  --tag profiling_test \
  --profiling_level DETAILED \
  --profiling_summary_every 1

# Check exit status
EXIT_CODE=$?

echo ""
echo "=================================================="
echo "PROFILING TEST COMPLETED"
echo "=================================================="
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"

    # Look for profiling results
    PROFILING_FILE=$(ls -t profiling_profiling_test_*.json 2>/dev/null | head -1)

    if [ -n "$PROFILING_FILE" ]; then
        echo "✓ Profiling results found: $PROFILING_FILE"

        # Run analysis
        echo ""
        echo "=================================================="
        echo "PROFILING ANALYSIS"
        echo "=================================================="

        python analyze_profiling.py summary "$PROFILING_FILE" | head -40

        echo ""
        echo "=================================================="
        echo "BOTTLENECK ANALYSIS"
        echo "=================================================="

        python analyze_profiling.py bottlenecks "$PROFILING_FILE" --threshold 0.05

        echo ""
        echo "✓ Profiling test PASSED"
        echo ""
        echo "Profiling results saved to: $PROFILING_FILE"
        echo ""
        echo "To view full summary:"
        echo "  python analyze_profiling.py summary $PROFILING_FILE"
        echo ""
        echo "To generate plots:"
        echo "  python analyze_profiling.py plot $PROFILING_FILE"

    else
        echo "⚠️  Warning: Profiling results file not found"
        echo "Expected file matching: profiling_profiling_test_*.json"
        echo ""
        echo "Files in directory:"
        ls -lh profiling_*.json 2>/dev/null || echo "  (none found)"
    fi
else
    echo "✗ Training failed with exit code: $EXIT_CODE"
    echo "Check logs for errors"
fi

echo "=================================================="
