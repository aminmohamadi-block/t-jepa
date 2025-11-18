#!/usr/bin/env python3
"""
Benchmark script to compare original vs optimized mask generation.

Expected results:
- Original: ~241.93ms per batch (based on profiling)
- Optimized: <10ms per batch (25x+ speedup)
"""

import torch
import time
import numpy as np
from src.mask import MaskCollator
from src.mask_optimized import OptimizedMaskCollator, AsyncMaskCollator


def benchmark_mask_collator(collator, num_iterations=100, batch_size=4096, num_features=256):
    """Benchmark a mask collator"""
    print(f"\nBenchmarking {collator.__class__.__name__}...")
    print(f"Batch size: {batch_size}, Features: {num_features}, Iterations: {num_iterations}")

    # Create dummy batch
    dummy_batch = [(torch.randn(num_features),) for _ in range(batch_size)]

    # Warmup
    for _ in range(5):
        _ = collator(dummy_batch)

    # Timing
    times = []
    for i in range(num_iterations):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.perf_counter()

        batch_data, masks_ctx, masks_trgt = collator(dummy_batch)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.perf_counter()

        times.append((end - start) * 1000)  # Convert to ms

        if (i + 1) % 20 == 0:
            print(f"  Iteration {i+1}/{num_iterations}: {times[-1]:.2f}ms")

    # Statistics
    times = np.array(times)
    print(f"\nResults for {collator.__class__.__name__}:")
    print(f"  Mean:   {np.mean(times):.2f}ms")
    print(f"  Median: {np.median(times):.2f}ms")
    print(f"  Std:    {np.std(times):.2f}ms")
    print(f"  Min:    {np.min(times):.2f}ms")
    print(f"  Max:    {np.max(times):.2f}ms")

    return times


def main():
    """Main benchmark function"""
    print("=" * 70)
    print("MASK GENERATION OPTIMIZATION BENCHMARK")
    print("=" * 70)

    # Parameters based on profiling
    num_features = 256  # After categorize_nan
    batch_size = 4096
    num_iterations = 50

    # Mask parameters from config
    min_context = 0.15
    max_context = 0.85
    min_target = 0.15
    max_target = 0.85
    num_encs = 1
    num_preds = 4

    # Create collators
    print("\nInitializing collators...")

    # Original collator
    original_collator = MaskCollator(
        allow_overlap=False,
        min_context_share=min_context,
        max_context_share=max_context,
        min_target_share=min_target,
        max_target_share=max_target,
        num_preds=num_preds,
        num_encs=num_encs,
        num_features=num_features,
        cardinalities=[],  # Not used in masking
        n_cls_tokens=1,
    )

    # Optimized collator (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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
        num_iterations=num_iterations,
        batch_size=batch_size,
        num_features=num_features
    )

    # Benchmark optimized
    optimized_times = benchmark_mask_collator(
        optimized_collator,
        num_iterations=num_iterations,
        batch_size=batch_size,
        num_features=num_features
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    original_mean = np.mean(original_times)
    optimized_mean = np.mean(optimized_times)
    speedup = original_mean / optimized_mean

    print(f"\nOriginal Implementation:")
    print(f"  Mean time: {original_mean:.2f}ms")
    print(f"  Expected (from profiling): ~241.93ms")

    print(f"\nOptimized Implementation:")
    print(f"  Mean time: {optimized_mean:.2f}ms")
    print(f"  Speedup: {speedup:.1f}x")

    print(f"\nProjected Impact on Training:")
    iteration_time = 487.97  # ms from profiling
    mask_overhead = 293.05  # ms from profiling
    new_mask_overhead = optimized_mean
    new_iteration_time = iteration_time - mask_overhead + new_mask_overhead

    print(f"  Current iteration time: {iteration_time:.2f}ms")
    print(f"  Current mask overhead: {mask_overhead:.2f}ms ({mask_overhead/iteration_time*100:.1f}%)")
    print(f"  New mask overhead: {new_mask_overhead:.2f}ms ({new_mask_overhead/new_iteration_time*100:.1f}%)")
    print(f"  New iteration time: {new_iteration_time:.2f}ms")
    print(f"  Overall speedup: {iteration_time/new_iteration_time:.2f}x")
    print(f"  Throughput improvement: {2.05 * iteration_time/new_iteration_time:.2f} iter/s (from 2.05)")

    # Test async version briefly
    print("\n" + "=" * 70)
    print("BONUS: Testing Async Mask Generation")
    print("=" * 70)

    async_collator = AsyncMaskCollator(optimized_collator, buffer_size=3)
    async_times = benchmark_mask_collator(
        async_collator,
        num_iterations=20,  # Fewer iterations for quick test
        batch_size=batch_size,
        num_features=num_features
    )

    async_mean = np.mean(async_times)
    print(f"\nAsync Implementation:")
    print(f"  Mean time: {async_mean:.2f}ms")
    print(f"  Additional speedup vs optimized: {optimized_mean/async_mean:.1f}x")
    print(f"  Total speedup vs original: {original_mean/async_mean:.1f}x")


if __name__ == "__main__":
    main()