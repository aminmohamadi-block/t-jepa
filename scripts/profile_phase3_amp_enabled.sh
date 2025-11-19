#!/bin/bash
#SBATCH --job-name=phase3_amp
#SBATCH --output=logs/phase3_amp_%j.out
#SBATCH --error=logs/phase3_amp_%j.err
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

echo "========================================"
echo "Phase 3 AMP: FP16 (WITH AMP ENABLED)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo ""

cd /home/aminmohamadi_squareup_com/projects/t-jepa-code-optimization
source ../bin/activate-hermit

timeout 300 python run.py \
    --data_set=jannis \
    --data_path=./datasets \
    --exp_train_total_epochs 1 \
    --batch_size 4096 \
    --model_amp=True \
    --exp_patience 999 \
    --exp_cache_cadence 999 \
    --probe_cadence 0 \
    --model_dim_hidden 64 \
    --model_num_layers 4 \
    --model_num_heads 4 \
    --pred_num_layers 2 \
    --use_vectorized_masking=True \
    --profiling_level=DETAILED

exit_code=$?
echo ""
echo "Exit code: $exit_code"
echo "Completed at: $(date)"
exit $exit_code
