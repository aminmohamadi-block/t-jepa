#!/bin/bash
#SBATCH --job-name=mask_test
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:05:00
#SBATCH --output=logs/mask_test_%j.out
#SBATCH --error=logs/mask_test_%j.err

echo "========================================="
echo "SIMPLE MASK BENCHMARK TEST"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "========================================="

# Load environment
source ../bin/activate-hermit

# Run the test
python3 test_mask_benchmark.py

echo "========================================="
echo "Test completed!"
echo "End time: $(date)"
echo "========================================="