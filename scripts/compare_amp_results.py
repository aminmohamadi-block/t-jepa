#!/usr/bin/env python3
"""
Compare FP32 baseline vs AMP enabled profiling results.
Usage: python3 scripts/compare_amp_results.py <fp32_json> <amp_json>
"""

import json
import sys
from pathlib import Path


def load_profiling(json_path):
    """Load profiling JSON and extract key metrics."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def extract_iteration_stats(data):
    """Extract iteration-level statistics."""
    for op in data['summary']:
        if op['name'] == 'iteration' and op['parent'] == 'epoch':
            return op
    return None


def extract_component_stats(data):
    """Extract component-level statistics."""
    components = {}
    for op in data['summary']:
        if op['parent'] == 'iteration':
            components[op['name']] = {
                'mean_time': op['mean_time'] * 1000,  # Convert to ms
                'total_time': op['total_time'],
                'count': op['count']
            }
    return components


def print_comparison(fp32_data, amp_data):
    """Print detailed comparison between FP32 and AMP."""

    print("=" * 80)
    print("PHASE 3: FP32 vs AMP COMPARISON")
    print("=" * 80)
    print()

    # Iteration-level comparison
    fp32_iter = extract_iteration_stats(fp32_data)
    amp_iter = extract_iteration_stats(amp_data)

    if fp32_iter and amp_iter:
        fp32_time = fp32_iter['mean_time'] * 1000
        amp_time = amp_iter['mean_time'] * 1000
        speedup = fp32_time / amp_time
        time_saved = fp32_time - amp_time

        print("OVERALL ITERATION TIME:")
        print("-" * 80)
        print(f"  FP32 (baseline):  {fp32_time:7.2f}ms per iteration ({fp32_iter['count']} iters)")
        print(f"  AMP (FP16):       {amp_time:7.2f}ms per iteration ({amp_iter['count']} iters)")
        print()
        print(f"  Speedup:          {speedup:.3f}x")
        print(f"  Time saved:       {time_saved:.2f}ms per iteration")
        print(f"  Improvement:      {(speedup-1)*100:+.2f}%")
        print()

    # Component-level comparison
    fp32_components = extract_component_stats(fp32_data)
    amp_components = extract_component_stats(amp_data)

    if fp32_components and amp_components:
        print("COMPONENT BREAKDOWN:")
        print("-" * 80)
        print(f"{'Component':<25} {'FP32 (ms)':<12} {'AMP (ms)':<12} {'Speedup':<10} {'Impact'}")
        print("-" * 80)

        # Sort by FP32 time (descending) to show biggest components first
        sorted_components = sorted(
            fp32_components.items(),
            key=lambda x: x[1]['mean_time'],
            reverse=True
        )

        total_speedup_contribution = 0
        for name, fp32_stats in sorted_components:
            if name in amp_components:
                amp_stats = amp_components[name]
                fp32_t = fp32_stats['mean_time']
                amp_t = amp_stats['mean_time']
                comp_speedup = fp32_t / amp_t if amp_t > 0 else 1.0
                time_saved_comp = fp32_t - amp_t

                # Calculate impact: percentage of total speedup contributed by this component
                impact = (time_saved_comp / fp32_time) * 100 if fp32_time > 0 else 0

                print(f"{name:<25} {fp32_t:>10.2f}ms  {amp_t:>10.2f}ms  {comp_speedup:>8.3f}x  {impact:>6.2f}%")

        print("-" * 80)

    # Training time projection
    if fp32_iter and amp_iter:
        epochs = 100
        fp32_total_hours = (fp32_time * fp32_iter['count'] * epochs) / 3600 / 1000
        amp_total_hours = (amp_time * amp_iter['count'] * epochs) / 3600 / 1000
        hours_saved = fp32_total_hours - amp_total_hours

        print()
        print("TRAINING TIME PROJECTION (100 epochs):")
        print("-" * 80)
        print(f"  FP32:       {fp32_total_hours:6.2f} hours")
        print(f"  AMP:        {amp_total_hours:6.2f} hours")
        print(f"  Time saved: {hours_saved:6.2f} hours ({hours_saved/fp32_total_hours*100:.1f}% faster)")
        print()

    # Success criteria check
    print("=" * 80)
    print("SUCCESS CRITERIA CHECK:")
    print("=" * 80)

    if speedup >= 1.4:
        print(f"✅ Speedup >= 1.4x: {speedup:.3f}x (PASSED)")
    elif speedup >= 1.3:
        print(f"⚠️  Speedup >= 1.3x: {speedup:.3f}x (ACCEPTABLE)")
    else:
        print(f"❌ Speedup < 1.3x: {speedup:.3f}x (BELOW EXPECTATION)")

    print()
    print("Next steps:")
    print("  1. Check loss convergence (compare training logs)")
    print("  2. Verify no NaN/Inf values in gradients")
    print("  3. Run longer training (50+ epochs) for stability test")
    print("=" * 80)


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 scripts/compare_amp_results.py <fp32_json> <amp_json>")
        print()
        print("Example:")
        print("  python3 scripts/compare_amp_results.py \\")
        print("    profiling_phase3_fp32_h100_15752.json \\")
        print("    profiling_phase3_amp_h100_15753.json")
        sys.exit(1)

    fp32_path = Path(sys.argv[1])
    amp_path = Path(sys.argv[2])

    if not fp32_path.exists():
        print(f"Error: FP32 profiling file not found: {fp32_path}")
        sys.exit(1)

    if not amp_path.exists():
        print(f"Error: AMP profiling file not found: {amp_path}")
        sys.exit(1)

    fp32_data = load_profiling(fp32_path)
    amp_data = load_profiling(amp_path)

    print_comparison(fp32_data, amp_data)


if __name__ == '__main__':
    main()
