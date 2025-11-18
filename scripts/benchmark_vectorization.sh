#!/bin/bash
#SBATCH --job-name=vec_bench
#SBATCH --partition=h100
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=logs/vec_bench_%j.out
#SBATCH --error=logs/vec_bench_%j.err

echo "========================================="
echo "VECTORIZATION BENCHMARK"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "========================================="

# Load environment
source ../bin/activate-hermit

# Run the vectorization benchmark
python3 -c "
from src.mask_vectorized import benchmark_vectorization
benchmark_vectorization()
"

echo "========================================="
echo "Benchmark completed!"
echo "End time: $(date)"
echo "========================================="