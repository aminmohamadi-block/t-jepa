#!/usr/bin/env python3
"""
Simple test script to benchmark mask generation optimization.
Tests both CPU and GPU implementations.
"""

import torch
import time
import numpy as np
from src.mask import MaskCollator
from src.mask_optimized import OptimizedMaskCollator


def simple_benchmark():
    """Run a simple benchmark to test the optimization"""

    print("=" * 70)
    print("SIMPLE MASK BENCHMARK TEST")
    print("=" * 70)

    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Parameters
    num_features = 256
    batch_size = 512  # Smaller for quick test
    num_iterations = 10

    print(f"\nTest Parameters:")
    print(f"  Features: {num_features}")
    print(f"  Batch size: {batch_size}")
    print(f"  Iterations: {num_iterations}")

    # Create dummy batch
    dummy_batch = [(torch.randn(num_features),) for _ in range(batch_size)]

    # Test original collator
    print("\n" + "-" * 50)
    print("Testing Original MaskCollator...")
    try:
        original_collator = MaskCollator(
            allow_overlap=False,
            min_context_share=0.15,
            max_context_share=0.85,
            min_target_share=0.15,
            max_target_share=0.85,
            num_preds=4,
            num_encs=1,
            num_features=num_features,
            cardinalities=[],
            n_cls_tokens=1,
        )

        # Warmup
        _ = original_collator(dummy_batch)

        # Time it
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = original_collator(dummy_batch)
        end = time.perf_counter()

        original_time = (end - start) / num_iterations * 1000
        print(f"✓ Original: {original_time:.2f}ms per batch")
    except Exception as e:
        print(f"✗ Original failed: {e}")
        original_time = None

    # Test optimized collator
    print("\n" + "-" * 50)
    print("Testing Optimized MaskCollator...")
    try:
        optimized_collator = OptimizedMaskCollator(
            num_features=num_features,
            min_context=0.15,
            max_context=0.85,
            min_target=0.15,
            max_target=0.85,
            num_encs=1,
            num_preds=4,
            device=device,
        )

        # Warmup
        _ = optimized_collator(dummy_batch)

        # Time it
        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = optimized_collator(dummy_batch)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()

        optimized_time = (end - start) / num_iterations * 1000
        print(f"✓ Optimized: {optimized_time:.2f}ms per batch")
    except Exception as e:
        print(f"✗ Optimized failed: {e}")
        import traceback
        traceback.print_exc()
        optimized_time = None

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    if original_time and optimized_time:
        speedup = original_time / optimized_time
        print(f"\nOriginal:  {original_time:.2f}ms")
        print(f"Optimized: {optimized_time:.2f}ms")
        print(f"Speedup:   {speedup:.1f}x")

        # Project to full batch size
        scale_factor = 4096 / batch_size
        print(f"\nProjected for batch_size=4096:")
        print(f"  Original:  {original_time * scale_factor:.2f}ms")
        print(f"  Optimized: {optimized_time * scale_factor:.2f}ms")
    else:
        print("\nCould not compute speedup due to errors")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    simple_benchmark()