#!/usr/bin/env python3
"""
Comprehensive iteration-level profiling analysis for T-JEPA on parquet datasets.

This script runs training for multiple epochs and provides detailed breakdown of:
- Data loading time
- Forward pass breakdown (encoder, predictor, loss)
- Backward pass time
- Optimizer step time
- EMA update time
- Gradient logging overhead
- Mask collation time

Usage:
    python scripts/profile_iteration_breakdown.py --epochs 3 --batch_size 4096
"""

import argparse
import json
import sys
from collections import defaultdict


def analyze_profiling_json(filepath):
    """Analyze profiling JSON and produce detailed breakdown."""
    with open(filepath) as f:
        data = json.load(f)

    summary = data.get('summary', [])

    # Convert to dict for easier access
    timing_dict = {}
    for item in summary:
        name = item.get('name', '')
        timing_dict[name] = {
            'mean_ms': item.get('mean_time', 0) * 1000,
            'total_ms': item.get('total_time', 0) * 1000,
            'count': item.get('count', 0),
            'std_ms': item.get('std_time', 0) * 1000,
            'min_ms': item.get('min_time', 0) * 1000,
            'max_ms': item.get('max_time', 0) * 1000,
        }

    return timing_dict


def print_iteration_breakdown(timing_dict):
    """Print detailed iteration breakdown."""
    print("=" * 80)
    print("ITERATION TIME BREAKDOWN (Parquet Dataset, AMP/bfloat16)")
    print("=" * 80)
    print()

    iteration = timing_dict.get('iteration', {})
    num_iterations = iteration.get('count', 0)
    iter_mean = iteration.get('mean_ms', 0)

    print(f"Total iterations profiled: {num_iterations}")
    print(f"Mean iteration time: {iter_mean:.2f} ms")
    print(f"Std deviation: {iteration.get('std_ms', 0):.2f} ms")
    print(f"Min: {iteration.get('min_ms', 0):.2f} ms, Max: {iteration.get('max_ms', 0):.2f} ms")
    print()

    # Calculate throughput
    batch_size = 4096  # Default, could be parameterized
    samples_per_sec = (batch_size / iter_mean) * 1000 if iter_mean > 0 else 0
    print(f"Throughput: ~{samples_per_sec:.0f} samples/second (batch_size={batch_size})")
    print()

    print("=" * 80)
    print("CATEGORY BREAKDOWN")
    print("=" * 80)
    print()

    # Define categories and their components
    categories = {
        'Data Loading & Transfer': ['data_transfer', 'batch_collation'],
        'Mask Generation': ['mask_collation', 'mask_creation'],
        'Forward Pass': ['forward_pass'],
        '  - Target Encoder': ['target_encoder'],
        '  - Context Encoder': ['context_encoder'],
        '  - Predictor': ['predictor'],
        '  - Loss Computation': ['loss_computation'],
        'Backward Pass': ['backward_pass'],
        'Optimizer Step': ['optimizer_step'],
        'EMA Update': ['ema_update'],
        'Gradient Logging': ['gradient_logging'],
    }

    print(f"{'Category':<35} {'Mean (ms)':<12} {'% of Iter':<12} {'Calls':<10}")
    print("-" * 69)

    accounted_time = 0
    for category, keys in categories.items():
        total_mean = 0
        total_count = 0
        for key in keys:
            if key in timing_dict:
                # Only count if called per iteration
                t = timing_dict[key]
                if t['count'] >= num_iterations * 0.5:  # Called at least half the iterations
                    total_mean += t['mean_ms']
                    total_count = max(total_count, t['count'])

        if total_mean > 0:
            pct = (total_mean / iter_mean * 100) if iter_mean > 0 else 0
            print(f"{category:<35} {total_mean:<12.2f} {pct:<12.1f} {total_count:<10}")
            if not category.startswith('  '):
                accounted_time += total_mean

    print("-" * 69)
    unaccounted = iter_mean - accounted_time
    unaccounted_pct = (unaccounted / iter_mean * 100) if iter_mean > 0 else 0
    print(f"{'Unaccounted/Overhead':<35} {unaccounted:<12.2f} {unaccounted_pct:<12.1f}")
    print()

    # Identify biggest bottlenecks
    print("=" * 80)
    print("TOP 10 BOTTLENECKS (sorted by mean time)")
    print("=" * 80)
    print()

    sorted_items = sorted(
        [(k, v) for k, v in timing_dict.items() if v['count'] > 1],
        key=lambda x: x[1]['mean_ms'],
        reverse=True
    )

    print(f"{'Operation':<40} {'Mean (ms)':<12} {'% of Iter':<12} {'Calls':<10}")
    print("-" * 74)
    for name, t in sorted_items[:10]:
        pct = (t['mean_ms'] / iter_mean * 100) if iter_mean > 0 else 0
        print(f"{name:<40} {t['mean_ms']:<12.2f} {pct:<12.1f} {t['count']:<10}")

    print()

    # Optimization opportunities
    print("=" * 80)
    print("OPTIMIZATION OPPORTUNITIES")
    print("=" * 80)
    print()

    opportunities = []

    # Check backward pass ratio
    backward = timing_dict.get('backward_pass', {}).get('mean_ms', 0)
    forward = timing_dict.get('forward_pass', {}).get('mean_ms', 0)
    if backward > 0 and forward > 0:
        ratio = backward / forward
        if ratio > 2.0:
            opportunities.append(f"Backward pass is {ratio:.1f}x forward pass - consider gradient checkpointing")

    # Check mask collation
    mask_time = timing_dict.get('mask_collation', {}).get('mean_ms', 0)
    if mask_time > 10:
        mask_pct = (mask_time / iter_mean * 100) if iter_mean > 0 else 0
        opportunities.append(f"Mask collation takes {mask_time:.1f}ms ({mask_pct:.1f}%) - consider GPU-based masking")

    # Check gradient logging
    grad_log = timing_dict.get('gradient_logging', {})
    if grad_log.get('mean_ms', 0) > 50 and grad_log.get('count', 0) > 1:
        opportunities.append(f"Gradient logging takes {grad_log['mean_ms']:.1f}ms - disable for production")

    # Check predictor time
    predictor_time = timing_dict.get('predictor', {}).get('mean_ms', 0)
    if predictor_time > 30:
        pred_pct = (predictor_time / iter_mean * 100) if iter_mean > 0 else 0
        opportunities.append(f"Predictor takes {predictor_time:.1f}ms ({pred_pct:.1f}%) - consider torch.compile")

    # Check data transfer
    data_transfer = timing_dict.get('data_transfer', {}).get('mean_ms', 0)
    if data_transfer > 5:
        opportunities.append(f"Data transfer takes {data_transfer:.1f}ms - ensure pin_memory=True")

    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp}")

    if not opportunities:
        print("No major optimization opportunities identified.")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile_file', type=str, required=True,
                        help='Path to profiling JSON file')
    args = parser.parse_args()

    timing_dict = analyze_profiling_json(args.profile_file)
    print_iteration_breakdown(timing_dict)


if __name__ == '__main__':
    main()
