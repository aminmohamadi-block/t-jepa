#!/usr/bin/env python
"""
Analyze profiling results from T-JEPA training.

Usage:
    # View summary
    python analyze_profiling.py summary profiling_results.json

    # Compare two runs
    python analyze_profiling.py compare baseline.json optimized.json

    # Generate visualization
    python analyze_profiling.py plot profiling_results.json --output profiling.png

    # Extract bottlenecks
    python analyze_profiling.py bottlenecks profiling_results.json --threshold 0.05
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd


def load_profiling_results(filepath: str) -> Dict:
    """Load profiling results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def print_summary(filepath: str, top_k: int = 30, sort_by: str = "total_time"):
    """Print summary of profiling results."""
    data = load_profiling_results(filepath)

    summary = data['summary']
    summary.sort(key=lambda x: x.get(sort_by, 0), reverse=True)
    summary = summary[:top_k]

    print("\n" + "=" * 120)
    print(f"PERFORMANCE PROFILING SUMMARY")
    print(f"File: {filepath}")
    print(f"Rank: {data['rank']}, World Size: {data['world_size']}")
    print("=" * 120)
    print(f"{'Operation':<40} {'Count':>8} {'Total (s)':>12} {'Mean (ms)':>12} "
          f"{'Std (ms)':>12} {'Min (ms)':>12} {'Max (ms)':>12} {'GPU Mem (MB)':>15}")
    print("-" * 120)

    for stat in summary:
        indent = "  " * stat['level']
        name = indent + stat['name']

        print(f"{name:<40} "
              f"{stat['count']:>8} "
              f"{stat['total_time']:>12.3f} "
              f"{stat['mean_time']*1000:>12.2f} "
              f"{stat['std_time']*1000:>12.2f} "
              f"{stat['min_time']*1000:>12.2f} "
              f"{stat['max_time']*1000:>12.2f} "
              f"{stat['mean_gpu_mem_allocated_mb']:>15.1f}")

    print("=" * 120)

    # Find iteration stats
    iteration_stats = [s for s in summary if 'iteration' in s['name'].lower()]
    if iteration_stats:
        iter_stat = iteration_stats[0]
        print(f"\nAverage iteration time: {iter_stat['mean_time']*1000:.2f} ms "
              f"(±{iter_stat['std_time']*1000:.2f} ms)")

    print()


def print_breakdown(filepath: str, parent: str = None):
    """Print time breakdown for operations."""
    data = load_profiling_results(filepath)
    summary = data['summary']

    # Find children of parent
    children = [s for s in summary if s.get('parent') == parent]

    if not children:
        print(f"No operations found under parent: {parent}")
        return

    total_time = sum(s['total_time'] for s in children)

    if total_time == 0:
        print("No time recorded for these operations")
        return

    parent_name = parent or "TOP-LEVEL"
    print(f"\n{'='*60}")
    print(f"TIME BREAKDOWN: {parent_name}")
    print(f"{'='*60}")

    for s in sorted(children, key=lambda x: x['total_time'], reverse=True):
        percentage = (s['total_time'] / total_time) * 100
        bar_length = int(percentage / 2)
        bar = '█' * bar_length
        print(f"{s['name']:<30} {percentage:>6.2f}% {bar}")

    print(f"{'='*60}\n")


def compare_results(baseline_file: str, optimized_file: str, output_file: str = None):
    """Compare two profiling runs."""
    baseline = load_profiling_results(baseline_file)
    optimized = load_profiling_results(optimized_file)

    baseline_summary = {s['name']: s for s in baseline['summary']}
    optimized_summary = {s['name']: s for s in optimized['summary']}

    common_ops = set(baseline_summary.keys()) & set(optimized_summary.keys())

    print("\n" + "=" * 140)
    print("PERFORMANCE COMPARISON")
    print(f"Baseline: {baseline_file}")
    print(f"Optimized: {optimized_file}")
    print("=" * 140)
    print(f"{'Operation':<40} {'Before (ms)':>15} {'After (ms)':>15} {'Speedup':>12} {'Change':>12}")
    print("-" * 140)

    comparisons = []
    for op_name in sorted(common_ops):
        before_stat = baseline_summary[op_name]
        after_stat = optimized_summary[op_name]

        before_mean = before_stat['mean_time'] * 1000
        after_mean = after_stat['mean_time'] * 1000

        if before_mean > 0:
            speedup = before_mean / after_mean if after_mean > 0 else float('inf')
            change_pct = ((after_mean - before_mean) / before_mean) * 100
        else:
            speedup = 1.0
            change_pct = 0.0

        comparisons.append({
            'operation': op_name,
            'before_ms': before_mean,
            'after_ms': after_mean,
            'speedup': speedup,
            'change_pct': change_pct,
            'level': before_stat['level'],
        })

    # Sort by absolute improvement (ms saved)
    comparisons.sort(key=lambda x: abs(x['after_ms'] - x['before_ms']), reverse=True)

    for comp in comparisons:
        indent = "  " * comp['level']
        display_name = indent + comp['operation'].split('/')[-1]

        change_str = f"{comp['change_pct']:+.1f}%"

        # Color code improvements/regressions
        if comp['change_pct'] < -5:  # Improvement
            change_str = f"\033[92m{change_str}\033[0m"  # Green
        elif comp['change_pct'] > 5:  # Regression
            change_str = f"\033[91m{change_str}\033[0m"  # Red

        print(f"{display_name:<40} {comp['before_ms']:>15.2f} {comp['after_ms']:>15.2f} "
              f"{comp['speedup']:>12.2f}x {change_str:>12}")

    print("=" * 140)

    # Calculate overall speedup (top-level operations only)
    baseline_top = [s for s in baseline['summary'] if s['level'] == 0]
    optimized_top = [s for s in optimized['summary'] if s['level'] == 0]

    before_total = sum(s['total_time'] for s in baseline_top)
    after_total = sum(s['total_time'] for s in optimized_top)

    if before_total > 0 and after_total > 0:
        overall_speedup = before_total / after_total
        improvement_pct = (overall_speedup - 1) * 100
        print(f"\nOverall speedup: {overall_speedup:.2f}x ({improvement_pct:+.1f}%)")
        print(f"Time saved per iteration: {(before_total - after_total) / baseline_top[0]['count'] * 1000:.2f} ms")

    if output_file:
        comparison_data = {
            'baseline_file': baseline_file,
            'optimized_file': optimized_file,
            'comparisons': comparisons,
            'before_total_s': before_total,
            'after_total_s': after_total,
            'overall_speedup': overall_speedup if before_total > 0 and after_total > 0 else None,
        }

        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"\nComparison saved to: {output_file}")


def identify_bottlenecks(filepath: str, threshold: float = 0.05, output_file: str = None):
    """
    Identify operations taking more than threshold% of total time.

    Args:
        filepath: Path to profiling results
        threshold: Minimum percentage of total time (default: 5%)
        output_file: Optional output file for bottleneck report
    """
    data = load_profiling_results(filepath)
    summary = data['summary']

    # Find iteration time
    iteration_stats = [s for s in summary if 'iteration' in s['name'].lower() and s['level'] == 0]
    if not iteration_stats:
        print("No iteration timing found in results")
        return

    iteration_time = iteration_stats[0]['mean_time']

    # Find bottlenecks
    bottlenecks = [
        s for s in summary
        if s['mean_time'] > iteration_time * threshold and s['level'] > 0  # Exclude top-level
    ]

    bottlenecks.sort(key=lambda x: x['mean_time'], reverse=True)

    print("\n" + "=" * 100)
    print(f"BOTTLENECK ANALYSIS (Operations > {threshold*100:.1f}% of iteration time)")
    print(f"File: {filepath}")
    print(f"Iteration time: {iteration_time*1000:.2f} ms")
    print("=" * 100)
    print(f"{'Operation':<40} {'Time (ms)':>12} {'% of Iter':>12} {'Count':>8} {'Priority':>10}")
    print("-" * 100)

    for i, b in enumerate(bottlenecks, 1):
        percentage = (b['mean_time'] / iteration_time) * 100
        indent = "  " * b['level']
        name = indent + b['name']

        # Assign priority
        if percentage > 20:
            priority = "🔴 CRITICAL"
        elif percentage > 10:
            priority = "🟠 HIGH"
        elif percentage > 5:
            priority = "🟡 MEDIUM"
        else:
            priority = "🟢 LOW"

        print(f"{name:<40} {b['mean_time']*1000:>12.2f} {percentage:>12.1f} {b['count']:>8} {priority:>10}")

    print("=" * 100)
    print(f"\nFound {len(bottlenecks)} bottlenecks accounting for "
          f"{sum(b['mean_time'] for b in bottlenecks)/iteration_time*100:.1f}% of iteration time")

    if output_file:
        report = {
            'filepath': filepath,
            'iteration_time_ms': iteration_time * 1000,
            'threshold_pct': threshold * 100,
            'bottlenecks': [
                {
                    'operation': b['name'],
                    'time_ms': b['mean_time'] * 1000,
                    'percentage': (b['mean_time'] / iteration_time) * 100,
                    'count': b['count'],
                    'parent': b['parent'],
                }
                for b in bottlenecks
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Bottleneck report saved to: {output_file}")


def plot_results(filepath: str, output_file: str = "profiling_plot.png", top_k: int = 15):
    """Generate visualization of profiling results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    data = load_profiling_results(filepath)
    summary = data['summary']

    # Get top operations by total time
    top_ops = sorted(summary, key=lambda x: x['total_time'], reverse=True)[:top_k]

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Total time bar chart
    names = [s['name'] for s in top_ops]
    times = [s['total_time'] for s in top_ops]
    colors = ['red' if s['level'] == 0 else 'blue' if s['level'] == 1 else 'green' for s in top_ops]

    ax1.barh(names, times, color=colors, alpha=0.7)
    ax1.set_xlabel('Total Time (seconds)')
    ax1.set_title(f'Top {top_k} Operations by Total Time')
    ax1.invert_yaxis()

    # Legend for hierarchy
    red_patch = mpatches.Patch(color='red', label='Top-level', alpha=0.7)
    blue_patch = mpatches.Patch(color='blue', label='Level 1', alpha=0.7)
    green_patch = mpatches.Patch(color='green', label='Level 2+', alpha=0.7)
    ax1.legend(handles=[red_patch, blue_patch, green_patch])

    # Plot 2: Mean time with error bars
    names2 = [s['name'] for s in top_ops]
    means = [s['mean_time'] * 1000 for s in top_ops]
    stds = [s['std_time'] * 1000 for s in top_ops]

    ax2.barh(names2, means, xerr=stds, color=colors, alpha=0.7, capsize=3)
    ax2.set_xlabel('Mean Time (milliseconds)')
    ax2.set_title(f'Top {top_k} Operations by Mean Time (with std dev)')
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")


def export_to_csv(filepath: str, output_file: str = "profiling_results.csv"):
    """Export profiling results to CSV for external analysis."""
    data = load_profiling_results(filepath)
    df = pd.DataFrame(data['summary'])

    # Flatten for easier analysis
    df.to_csv(output_file, index=False)
    print(f"Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze T-JEPA profiling results")
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Print summary of results')
    summary_parser.add_argument('filepath', help='Path to profiling results JSON')
    summary_parser.add_argument('--top-k', type=int, default=30, help='Number of top operations to show')
    summary_parser.add_argument('--sort-by', default='total_time',
                               choices=['total_time', 'mean_time', 'count', 'max_time'],
                               help='Metric to sort by')

    # Breakdown command
    breakdown_parser = subparsers.add_parser('breakdown', help='Show time breakdown')
    breakdown_parser.add_argument('filepath', help='Path to profiling results JSON')
    breakdown_parser.add_argument('--parent', default=None, help='Parent operation (None for top-level)')

    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare two profiling runs')
    compare_parser.add_argument('baseline', help='Baseline profiling results')
    compare_parser.add_argument('optimized', help='Optimized profiling results')
    compare_parser.add_argument('--output', help='Output comparison JSON')

    # Bottlenecks command
    bottleneck_parser = subparsers.add_parser('bottlenecks', help='Identify bottlenecks')
    bottleneck_parser.add_argument('filepath', help='Path to profiling results JSON')
    bottleneck_parser.add_argument('--threshold', type=float, default=0.05,
                                   help='Minimum percentage of iteration time (default: 0.05 = 5%%)')
    bottleneck_parser.add_argument('--output', help='Output bottleneck report JSON')

    # Plot command
    plot_parser = subparsers.add_parser('plot', help='Generate visualization')
    plot_parser.add_argument('filepath', help='Path to profiling results JSON')
    plot_parser.add_argument('--output', default='profiling_plot.png', help='Output image file')
    plot_parser.add_argument('--top-k', type=int, default=15, help='Number of top operations to plot')

    # Export command
    export_parser = subparsers.add_parser('export', help='Export to CSV')
    export_parser.add_argument('filepath', help='Path to profiling results JSON')
    export_parser.add_argument('--output', default='profiling_results.csv', help='Output CSV file')

    args = parser.parse_args()

    if args.command == 'summary':
        print_summary(args.filepath, top_k=args.top_k, sort_by=args.sort_by)
    elif args.command == 'breakdown':
        print_breakdown(args.filepath, parent=args.parent)
    elif args.command == 'compare':
        compare_results(args.baseline, args.optimized, output_file=args.output)
    elif args.command == 'bottlenecks':
        identify_bottlenecks(args.filepath, threshold=args.threshold, output_file=args.output)
    elif args.command == 'plot':
        plot_results(args.filepath, output_file=args.output, top_k=args.top_k)
    elif args.command == 'export':
        export_to_csv(args.filepath, output_file=args.output)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
