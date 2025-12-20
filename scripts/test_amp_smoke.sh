#!/bin/bash
#SBATCH --job-name=amp_smoke
#SBATCH --output=logs/amp_smoke_%j.out
#SBATCH --error=logs/amp_smoke_%j.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00

echo "========================================"
echo "Phase 3 AMP Smoke Test"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

cd /home/aminmohamadi_squareup_com/projects/t-jepa-code-optimization
source ../bin/activate-hermit

timeout 120 python run.py \
    --data_set=jannis \
    --data_path=./datasets \
    --exp_train_total_epochs 1 \
    --batch_size 1024 \
    --model_amp=True \
    --exp_patience 999 \
    --exp_cache_cadence 999 \
    --probe_cadence 0 \
    --model_dim_hidden 64 \
    --model_num_layers 4 \
    --model_num_heads 4 \
    --pred_num_layers 2 \
    --use_vectorized_masking=True \
    --profiling_level=DISABLED

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "✅ AMP smoke test PASSED"
else
    echo "❌ AMP smoke test FAILED with exit code: $exit_code"
fi
echo "Completed at: $(date)"
echo "========================================"
