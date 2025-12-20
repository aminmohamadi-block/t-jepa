#!/bin/bash
#SBATCH --job-name=phase6_train
#SBATCH --output=logs/phase6_train_%j.out
#SBATCH --error=logs/phase6_train_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=01:00:00
#SBATCH --mem=64G

echo "========================================="
echo "Phase 6 Training Verification"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo ""

# Activate environment
source ../bin/activate-hermit

cd /home/aminmohamadi_squareup_com/projects/t-jepa-code-optimization

# Test 1: Train with MLP predictor (exercises vectorized MLP)
echo ""
echo "========================================="
echo "Test 1: MLP Predictor Training (Jannis)"
echo "========================================="
python run.py \
    --data_set jannis \
    --data_path ./datasets \
    --pred_type mlp \
    --exp_train_total_epochs 3 \
    --batch_size 2048 \
    --model_dim_hidden 64 \
    --model_num_layers 2 \
    --pred_num_layers 2 \
    --tag "phase6_mlp_jannis" \
    --profiling_level DETAILED \
    --profiling_output profiling_phase6_mlp_jannis.json

echo ""
echo "Test 1 completed at: $(date)"

# Test 2: Train with Adult dataset (exercises categorical encoding fix)
echo ""
echo "========================================="
echo "Test 2: Categorical Encoding (Adult)"
echo "========================================="
python run.py \
    --data_set adult \
    --data_path ./datasets \
    --pred_type transformer \
    --exp_train_total_epochs 3 \
    --batch_size 2048 \
    --model_dim_hidden 64 \
    --model_num_layers 2 \
    --tag "phase6_cat_adult" \
    --profiling_level DETAILED \
    --profiling_output profiling_phase6_cat_adult.json

echo ""
echo "Test 2 completed at: $(date)"

echo ""
echo "========================================="
echo "All tests completed at: $(date)"
echo "========================================="
echo ""
echo "Check profiling results:"
echo "  - profiling_phase6_mlp_jannis.json"
echo "  - profiling_phase6_cat_adult.json"
