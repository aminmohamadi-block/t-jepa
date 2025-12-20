#!/bin/bash
#SBATCH --job-name=phase6_verify
#SBATCH --output=logs/phase6_verify_%j.out
#SBATCH --error=logs/phase6_verify_%j.err
#SBATCH --gres=gpu:1
#SBATCH --partition=h100
#SBATCH --time=00:30:00
#SBATCH --mem=32G

echo "========================================="
echo "Phase 6 Optimization Verification"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Started at: $(date)"
echo ""

# Activate environment
source ../bin/activate-hermit

# Run verification script
cd /home/aminmohamadi_squareup_com/projects/t-jepa-code-optimization

python scripts/verify_phase6_optimizations.py

echo ""
echo "========================================="
echo "Verification completed at: $(date)"
echo "========================================="
