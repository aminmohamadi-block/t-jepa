#!/bin/bash
#SBATCH --job-name=gen_comp
#SBATCH --partition=h100
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=logs/gen_comp_%j.out
#SBATCH --error=logs/gen_comp_%j.err

echo "========================================="
echo "Generating Comparison Visualizations"
echo "========================================="

source ../bin/activate-hermit

python3 compare_profiling_results.py

echo "========================================="
echo "Done!"
echo "========================================="