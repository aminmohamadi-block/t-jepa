#!/bin/bash
#SBATCH --job-name=mask_benchmark
#SBATCH --partition=h100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mask_benchmark_%j.out
#SBATCH --error=logs/mask_benchmark_%j.err

# Mask Optimization Benchmark Script
# Expected results:
# - Original: ~240-350ms per batch (CPU-bound NumPy operations)
# - Optimized: <10ms per batch (GPU-accelerated vectorized operations)
# - Expected speedup: 25-35x

echo "========================================="
echo "MASK OPTIMIZATION BENCHMARK"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS"
echo "Start time: $(date)"
echo "========================================="

# Load environment
source ../bin/activate-hermit

# Verify GPU availability
echo "Checking GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run the benchmark
echo ""
echo "Running mask optimization benchmark..."
echo "========================================="

python3 - << 'EOF'
import torch
import time
import numpy as np
from src.mask import MaskCollator
from src.mask_optimized import OptimizedMaskCollator

def benchmark_mask_collator(collator, collator_name, num_iterations=100, batch_size=4096, num_features=256):
    """Benchmark a mask collator"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {collator_name}")
    print(f"{'='*60}")
    print(f"Batch size: {batch_size}, Features: {num_features}, Iterations: {num_iterations}")

    # Create dummy batch
    dummy_batch = [(torch.randn(num_features),) for _ in range(batch_size)]

    # Warmup
    print("Warming up...")
    for _ in range(5):
        _ = collator(dummy_batch)

    # Timing
    print("Running benchmark...")
    times = []
    for i in range(num_iterations):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        batch_data, masks_ctx, masks_trgt = collator(dummy_batch)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            avg_so_far = np.mean(times)
            print(f"  Iteration {i+1}/{num_iterations}: Current={times[-1]:.2f}ms, Avg={avg_so_far:.2f}ms")

    # Statistics
    times = np.array(times)
    percentiles = np.percentile(times, [25, 50, 75, 90, 95, 99])

    print(f"\n{collator_name} Results:")
    print(f"  Mean:     {np.mean(times):.2f}ms")
    print(f"  Median:   {np.median(times):.2f}ms")
    print(f"  Std:      {np.std(times):.2f}ms")
    print(f"  Min:      {np.min(times):.2f}ms")
    print(f"  Max:      {np.max(times):.2f}ms")
    print(f"  P25:      {percentiles[0]:.2f}ms")
    print(f"  P75:      {percentiles[2]:.2f}ms")
    print(f"  P90:      {percentiles[3]:.2f}ms")
    print(f"  P95:      {percentiles[4]:.2f}ms")
    print(f"  P99:      {percentiles[5]:.2f}ms")

    return times

def main():
    print("\n" + "="*70)
    print("MASK GENERATION OPTIMIZATION BENCHMARK")
    print("="*70)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Parameters
    num_features = 256  # After categorize_nan
    batch_size = 4096   # Production batch size
    num_iterations = 100

    # Mask parameters from config
    min_context = 0.15
    max_context = 0.85
    min_target = 0.15
    max_target = 0.85
    num_encs = 1
    num_preds = 4

    print(f"\nParameters:")
    print(f"  Features: {num_features}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")
    print(f"  Context range: {min_context:.0%}-{max_context:.0%}")
    print(f"  Target range: {min_target:.0%}-{max_target:.0%}")
    print(f"  Encoders: {num_encs}, Predictors: {num_preds}")

    # Create original collator
    print("\nInitializing original collator...")
    original_collator = MaskCollator(
        allow_overlap=False,
        min_context_share=min_context,
        max_context_share=max_context,
        min_target_share=min_target,
        max_target_share=max_target,
        num_preds=num_preds,
        num_encs=num_encs,
        num_features=num_features,
        cardinalities=[],
        n_cls_tokens=1,
    )

    # Create optimized collator
    print("Initializing optimized collator...")
    optimized_collator = OptimizedMaskCollator(
        num_features=num_features,
        min_context=min_context,
        max_context=max_context,
        min_target=min_target,
        max_target=max_target,
        num_encs=num_encs,
        num_preds=num_preds,
        device=device,
        cache_size=10000,
    )

    # Benchmark original
    original_times = benchmark_mask_collator(
        original_collator,
        "Original MaskCollator",
        num_iterations=num_iterations,
        batch_size=batch_size,
        num_features=num_features
    )

    # Benchmark optimized
    optimized_times = benchmark_mask_collator(
        optimized_collator,
        "Optimized MaskCollator",
        num_iterations=num_iterations,
        batch_size=batch_size,
        num_features=num_features
    )

    # Comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)

    original_mean = np.mean(original_times)
    original_median = np.median(original_times)
    optimized_mean = np.mean(optimized_times)
    optimized_median = np.median(optimized_times)

    speedup_mean = original_mean / optimized_mean
    speedup_median = original_median / optimized_median

    print(f"\nOriginal Implementation:")
    print(f"  Mean time:   {original_mean:.2f}ms")
    print(f"  Median time: {original_median:.2f}ms")
    print(f"  Expected from profiling: ~241.93ms (mask_creation only)")
    print(f"  Note: Full mask_collation was ~293.05ms")

    print(f"\nOptimized Implementation:")
    print(f"  Mean time:   {optimized_mean:.2f}ms")
    print(f"  Median time: {optimized_median:.2f}ms")

    print(f"\nSpeedup:")
    print(f"  Mean-based:   {speedup_mean:.1f}x faster")
    print(f"  Median-based: {speedup_median:.1f}x faster")

    # Impact on training
    print("\n" + "="*70)
    print("PROJECTED IMPACT ON TRAINING")
    print("="*70)

    # From profiling results
    iteration_time_original = 487.97  # ms
    mask_collation_original = 293.05  # ms
    mask_creation_original = 241.93   # ms (subset of collation)

    # New times
    mask_collation_new = optimized_mean
    iteration_time_new = iteration_time_original - mask_collation_original + mask_collation_new

    print(f"\nCurrent Performance (from profiling):")
    print(f"  Iteration time:    {iteration_time_original:.2f}ms")
    print(f"  Mask collation:    {mask_collation_original:.2f}ms ({mask_collation_original/iteration_time_original*100:.1f}%)")
    print(f"  - mask_creation:   {mask_creation_original:.2f}ms")
    print(f"  - batch_collation: 49.83ms")
    print(f"  - preprocessing:   0.27ms")
    print(f"  - sampling:        0.20ms")

    print(f"\nProjected Performance with Optimization:")
    print(f"  New mask time:     {mask_collation_new:.2f}ms ({mask_collation_new/iteration_time_new*100:.1f}%)")
    print(f"  New iteration:     {iteration_time_new:.2f}ms")
    print(f"  Speedup:           {iteration_time_original/iteration_time_new:.2f}x")

    print(f"\nThroughput Improvement:")
    throughput_original = 1000 / iteration_time_original  # iter/sec
    throughput_new = 1000 / iteration_time_new
    print(f"  Original:          {throughput_original:.2f} iter/sec")
    print(f"  Optimized:         {throughput_new:.2f} iter/sec")
    print(f"  Improvement:       {(throughput_new - throughput_original)/throughput_original*100:.1f}%")

    # Training time estimates
    print(f"\nTraining Time Estimates (for 100 epochs, 805 iter/epoch):")
    total_iterations = 100 * 805
    time_original = total_iterations * iteration_time_original / 1000 / 60  # minutes
    time_new = total_iterations * iteration_time_new / 1000 / 60
    time_saved = time_original - time_new

    print(f"  Original:          {time_original:.1f} minutes")
    print(f"  Optimized:         {time_new:.1f} minutes")
    print(f"  Time saved:        {time_saved:.1f} minutes ({time_saved/time_original*100:.1f}%)")

    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
EOF

echo ""
echo "========================================="
echo "Benchmark completed successfully!"
echo "End time: $(date)"
echo "========================================="