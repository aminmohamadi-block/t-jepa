#!/usr/bin/env python3
"""
Comprehensive profiling visualization script for T-JEPA profiling data.
Creates detailed matplotlib visualizations and saves them to results/profiling/visualizations/.
"""

import json
import argparse
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import defaultdict
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_profiling_data(json_path):
    """Load profiling JSON data and convert to dict format."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Convert summary list to dictionary format
    if isinstance(data.get('summary'), list):
        summary_dict = {}
        for entry in data['summary']:
            name = entry.get('name', 'unknown')
            summary_dict[name] = {
                'mean': entry.get('mean_time', 0),
                'std': entry.get('std_time', 0),
                'min': entry.get('min_time', 0),
                'max': entry.get('max_time', 0),
                'count': entry.get('count', 0),
                'total': entry.get('total_time', 0),
                'parent': entry.get('parent'),
                'level': entry.get('level', 0)
            }
        data['summary'] = summary_dict

    return data


def create_output_dir(base_dir="results/profiling/visualizations"):
    """Create output directory for visualizations."""
    Path(base_dir).mkdir(parents=True, exist_ok=True)
    return base_dir


def plot_overall_timing_breakdown(data, output_dir):
    """Create pie chart and bar chart of overall timing breakdown."""
    summary = data.get('summary', {})

    # Get iteration count for reference
    iteration_count = summary.get('iteration', {}).get('count', 1610)

    # Filter for operations that happen per-iteration:
    # 1. Level 2 operations (direct children of iteration)
    # 2. Level 1 operations that have same count as iterations (like mask_collation)
    operations = {}
    for op_name, op_data in summary.items():
        if 'mean' not in op_data or op_name == 'epoch':
            continue

        level = op_data.get('level', 0)
        parent = op_data.get('parent', '')
        count = op_data.get('count', 0)

        # Include if it's a child of iteration (level 2)
        if level == 2 and 'iteration' in parent and count == iteration_count:
            operations[op_name] = op_data['mean'] * 1000

        # Include if it's level 1 but runs same number of times as iteration
        # (e.g., mask_collation happens in DataLoader before each iteration)
        elif level == 1 and count == iteration_count and op_name != 'iteration':
            operations[op_name] = op_data['mean'] * 1000

    if not operations:
        print("⚠ No iteration-level operations found, skipping overall breakdown")
        return

    # Sort by time
    sorted_ops = sorted(operations.items(), key=lambda x: x[1], reverse=True)

    # Take top 15 operations
    top_ops = dict(sorted_ops[:15])
    labels = list(top_ops.keys())
    times = list(top_ops.values())

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax1.pie(times, labels=labels, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax1.set_title('Per-Iteration Operations Breakdown', fontsize=16, fontweight='bold')

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    # Bar chart with iteration time reference
    y_pos = np.arange(len(labels))
    ax2.barh(y_pos, times, color=colors)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Time (ms per iteration)', fontsize=12)
    ax2.set_title('Per-Iteration Operations', fontsize=16, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    # Add iteration time reference line
    iteration_time = summary.get('iteration', {}).get('mean', 0) * 1000
    if iteration_time > 0:
        ax2.axvline(x=iteration_time, color='red', linestyle='--', linewidth=2,
                   label=f'Total iteration: {iteration_time:.2f}ms')
        ax2.legend()

    # Add value labels on bars
    for i, v in enumerate(times):
        ax2.text(v, i, f' {v:.2f}ms', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_overall_timing_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: 01_overall_timing_breakdown.png")


def plot_hierarchical_breakdown(data, output_dir):
    """Create hierarchical breakdown showing parent-child relationships."""
    summary = data.get('summary', {})

    # Build hierarchy
    hierarchy = defaultdict(list)
    for op_name, op_data in summary.items():
        if 'mean' in op_data:
            parts = op_name.split('.')
            if len(parts) > 1:
                parent = '.'.join(parts[:-1])
                hierarchy[parent].append((parts[-1], op_data['mean'] * 1000))

    # Find most interesting hierarchies (those with children)
    interesting_parents = [(parent, sum(c[1] for c in children))
                          for parent, children in hierarchy.items()
                          if len(children) > 1]
    interesting_parents.sort(key=lambda x: x[1], reverse=True)

    # Create subplots for top hierarchies
    num_plots = min(6, len(interesting_parents))
    if num_plots == 0:
        print("⚠ No hierarchical data found, skipping hierarchical breakdown")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (parent, total_time) in enumerate(interesting_parents[:num_plots]):
        children = hierarchy[parent]
        children.sort(key=lambda x: x[1], reverse=True)

        labels = [c[0] for c in children]
        times = [c[1] for c in children]

        ax = axes[idx]
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(labels)))
        ax.barh(labels, times, color=colors)
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_title(f'{parent}\n(Total: {total_time:.2f}ms)', fontsize=11, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, v in enumerate(times):
            ax.text(v, i, f' {v:.2f}', va='center', fontsize=9)

    # Hide unused subplots
    for idx in range(num_plots, 6):
        axes[idx].axis('off')

    plt.suptitle('Hierarchical Operation Breakdown', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_hierarchical_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: 02_hierarchical_breakdown.png")


def plot_iteration_timeline(data, output_dir):
    """Create timeline showing iteration times across epochs."""
    detailed_traces = data.get('detailed_traces', [])

    # Extract iteration-level entries (operation == 'epoch/iteration')
    iteration_data = [entry for entry in detailed_traces
                     if entry.get('operation') == 'epoch/iteration']

    if not iteration_data:
        print("⚠ No iteration data found, skipping timeline")
        return

    iterations = []
    times = []
    epochs = []

    for entry in iteration_data:
        iterations.append(entry.get('iteration', 0))
        times.append(entry.get('elapsed_ms', 0))  # Already in ms
        epochs.append(entry.get('epoch', 0))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Timeline plot
    colors = ['blue' if e == 0 else 'green' for e in epochs]
    ax1.scatter(iterations, times, c=colors, alpha=0.5, s=10)
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Iteration Time Timeline', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add legend
    blue_patch = mpatches.Patch(color='blue', label='Epoch 1')
    green_patch = mpatches.Patch(color='green', label='Epoch 2')
    ax1.legend(handles=[blue_patch, green_patch], loc='upper right')

    # Add mean line
    mean_time = np.mean(times)
    ax1.axhline(y=mean_time, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_time:.2f}ms')
    ax1.legend(loc='upper right')

    # Histogram of iteration times
    ax2.hist(times, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.axvline(x=mean_time, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {mean_time:.2f}ms')
    ax2.axvline(x=np.median(times), color='orange', linestyle='--', linewidth=2,
                label=f'Median: {np.median(times):.2f}ms')
    ax2.set_xlabel('Iteration Time (ms)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Iteration Time Distribution', fontsize=16, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_iteration_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: 03_iteration_timeline.png")


def plot_encoder_breakdown(data, output_dir):
    """Create detailed breakdown of encoder operations."""
    summary = data.get('summary', {})

    # Extract encoder-related operations
    target_encoder_ops = {}
    context_encoder_ops = {}
    predictor_ops = {}

    for op_name, op_data in summary.items():
        if 'mean' not in op_data:
            continue

        time_ms = op_data['mean'] * 1000

        if 'target_encoder' in op_name:
            target_encoder_ops[op_name] = time_ms
        elif 'context_encoder' in op_name:
            context_encoder_ops[op_name] = time_ms
        elif 'predictor' in op_name:
            predictor_ops[op_name] = time_ms

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Target Encoder
    if target_encoder_ops:
        sorted_ops = sorted(target_encoder_ops.items(), key=lambda x: x[1], reverse=True)
        labels = [op.split('.')[-1][:30] for op, _ in sorted_ops[:10]]
        times = [time for _, time in sorted_ops[:10]]

        axes[0].barh(labels, times, color='coral')
        axes[0].set_xlabel('Time (ms)', fontsize=12)
        axes[0].set_title('Target Encoder Operations', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)

        for i, v in enumerate(times):
            axes[0].text(v, i, f' {v:.2f}', va='center', fontsize=9)
    else:
        axes[0].text(0.5, 0.5, 'No target encoder data', ha='center', va='center',
                     transform=axes[0].transAxes)
        axes[0].set_title('Target Encoder Operations', fontsize=14, fontweight='bold')

    # Context Encoder
    if context_encoder_ops:
        sorted_ops = sorted(context_encoder_ops.items(), key=lambda x: x[1], reverse=True)
        labels = [op.split('.')[-1][:30] for op, _ in sorted_ops[:10]]
        times = [time for _, time in sorted_ops[:10]]

        axes[1].barh(labels, times, color='skyblue')
        axes[1].set_xlabel('Time (ms)', fontsize=12)
        axes[1].set_title('Context Encoder Operations', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)

        for i, v in enumerate(times):
            axes[1].text(v, i, f' {v:.2f}', va='center', fontsize=9)
    else:
        axes[1].text(0.5, 0.5, 'No context encoder data', ha='center', va='center',
                     transform=axes[1].transAxes)
        axes[1].set_title('Context Encoder Operations', fontsize=14, fontweight='bold')

    # Predictor
    if predictor_ops:
        sorted_ops = sorted(predictor_ops.items(), key=lambda x: x[1], reverse=True)
        labels = [op.split('.')[-1][:30] for op, _ in sorted_ops[:10]]
        times = [time for _, time in sorted_ops[:10]]

        axes[2].barh(labels, times, color='lightgreen')
        axes[2].set_xlabel('Time (ms)', fontsize=12)
        axes[2].set_title('Predictor Operations', fontsize=14, fontweight='bold')
        axes[2].invert_yaxis()
        axes[2].grid(axis='x', alpha=0.3)

        for i, v in enumerate(times):
            axes[2].text(v, i, f' {v:.2f}', va='center', fontsize=9)
    else:
        axes[2].text(0.5, 0.5, 'No predictor data', ha='center', va='center',
                     transform=axes[2].transAxes)
        axes[2].set_title('Predictor Operations', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_encoder_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: 04_encoder_breakdown.png")


def plot_bottleneck_analysis(data, output_dir, threshold=0.05):
    """Identify and visualize bottlenecks (operations taking >threshold of iteration time)."""
    summary = data.get('summary', {})

    # Find iteration time and count
    iteration_time = summary.get('iteration', {}).get('mean', 1.0)
    iteration_count = summary.get('iteration', {}).get('count', 1610)

    # Find bottlenecks - operations that run per-iteration
    bottlenecks = {}
    for op_name, op_data in summary.items():
        if 'mean' not in op_data or op_name in ['iteration', 'epoch']:
            continue

        parent = op_data.get('parent', '')
        level = op_data.get('level', 0)
        count = op_data.get('count', 0)

        # Include per-iteration operations (level 2) or level 1 with same count
        include = False
        if level == 2 and 'iteration' in parent and count == iteration_count:
            include = True
        elif level == 1 and count == iteration_count:
            include = True

        if include:
            time = op_data['mean']
            percentage = (time / iteration_time) * 100
            if percentage >= threshold * 100:
                bottlenecks[op_name] = {
                    'time_ms': time * 1000,
                    'percentage': percentage,
                    'count': count
                }

    if not bottlenecks:
        print(f"⚠ No bottlenecks found above {threshold*100}% threshold")
        return

    # Sort by percentage
    sorted_bottlenecks = sorted(bottlenecks.items(), key=lambda x: x[1]['percentage'], reverse=True)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Bar chart of bottlenecks by percentage
    labels = [op[:40] for op, _ in sorted_bottlenecks[:15]]
    percentages = [data['percentage'] for _, data in sorted_bottlenecks[:15]]
    times = [data['time_ms'] for _, data in sorted_bottlenecks[:15]]

    y_pos = np.arange(len(labels))
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(labels)))

    bars = ax1.barh(y_pos, percentages, color=colors)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel('% of Iteration Time', fontsize=12)
    ax1.set_title(f'Top Bottlenecks (>{threshold*100}% of iteration time)',
                  fontsize=14, fontweight='bold')
    ax1.axvline(x=threshold*100, color='red', linestyle='--', linewidth=2,
                label=f'Threshold: {threshold*100}%')
    ax1.legend()
    ax1.grid(axis='x', alpha=0.3)

    # Add percentage labels
    for i, (pct, time) in enumerate(zip(percentages, times)):
        ax1.text(pct, i, f' {pct:.1f}% ({time:.2f}ms)', va='center', fontsize=9)

    # Cumulative time chart
    cumulative_times = np.cumsum([data['time_ms'] for _, data in sorted_bottlenecks[:15]])
    cumulative_percentages = (cumulative_times / (iteration_time * 1000)) * 100

    ax2.plot(range(len(cumulative_percentages)), cumulative_percentages,
             marker='o', linewidth=2, markersize=8, color='darkred')
    ax2.fill_between(range(len(cumulative_percentages)), cumulative_percentages,
                      alpha=0.3, color='red')
    ax2.set_xlabel('Number of Operations', fontsize=12)
    ax2.set_ylabel('Cumulative % of Iteration Time', fontsize=12)
    ax2.set_title('Cumulative Impact of Top Bottlenecks', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(range(len(cumulative_percentages)))
    ax2.set_xticklabels(range(1, len(cumulative_percentages) + 1))

    # Add 80% line
    ax2.axhline(y=80, color='orange', linestyle='--', linewidth=2,
                label='80% threshold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_bottleneck_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: 05_bottleneck_analysis.png")


def plot_data_movement_analysis(data, output_dir):
    """Analyze data movement and transfer operations."""
    summary = data.get('summary', {})

    # Find data movement operations
    data_ops = {}
    keywords = ['transfer', 'mask', 'collation', 'preparation', 'separation', 'encoding']

    for op_name, op_data in summary.items():
        if 'mean' in op_data:
            for keyword in keywords:
                if keyword in op_name.lower():
                    data_ops[op_name] = op_data['mean'] * 1000
                    break

    if not data_ops:
        print("⚠ No data movement operations found")
        return

    # Sort by time
    sorted_ops = sorted(data_ops.items(), key=lambda x: x[1], reverse=True)

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 8))

    labels = [op.split('.')[-1][:50] for op, _ in sorted_ops[:15]]
    times = [time for _, time in sorted_ops[:15]]

    y_pos = np.arange(len(labels))
    colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(labels)))

    ax.barh(y_pos, times, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_title('Data Movement & Transfer Operations', fontsize=16, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, v in enumerate(times):
        ax.text(v, i, f' {v:.2f}ms', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_data_movement_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: 06_data_movement_analysis.png")


def plot_operation_statistics(data, output_dir):
    """Plot statistical analysis of operations (mean, std, min, max)."""
    summary = data.get('summary', {})

    # Get top 15 operations by mean time
    ops_with_stats = {}
    for op_name, op_data in summary.items():
        if all(k in op_data for k in ['mean', 'std', 'min', 'max']):
            ops_with_stats[op_name] = {
                'mean': op_data['mean'] * 1000,
                'std': op_data['std'] * 1000,
                'min': op_data['min'] * 1000,
                'max': op_data['max'] * 1000
            }

    if not ops_with_stats:
        print("⚠ No operations with statistics found")
        return

    sorted_ops = sorted(ops_with_stats.items(), key=lambda x: x[1]['mean'], reverse=True)[:15]

    # Create visualization
    fig, ax = plt.subplots(figsize=(14, 10))

    labels = [op.split('.')[-1][:40] for op, _ in sorted_ops]
    means = [data['mean'] for _, data in sorted_ops]
    stds = [data['std'] for _, data in sorted_ops]
    mins = [data['min'] for _, data in sorted_ops]
    maxs = [data['max'] for _, data in sorted_ops]

    y_pos = np.arange(len(labels))

    # Plot error bars (min to max range)
    for i, (mean, std, min_val, max_val) in enumerate(zip(means, stds, mins, maxs)):
        ax.plot([min_val, max_val], [i, i], 'gray', linewidth=1, alpha=0.5)
        ax.plot(min_val, i, 'o', color='blue', markersize=6, label='Min' if i == 0 else '')
        ax.plot(max_val, i, 'o', color='red', markersize=6, label='Max' if i == 0 else '')
        ax.plot(mean, i, 's', color='green', markersize=8, label='Mean' if i == 0 else '')

        # Error bar for std
        ax.plot([mean - std, mean + std], [i, i], 'green', linewidth=3, alpha=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_title('Operation Statistics (Mean ± Std, Min-Max Range)',
                 fontsize=16, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_operation_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Created: 07_operation_statistics.png")


def create_summary_report(data, output_dir):
    """Create a text summary report."""
    summary = data.get('summary', {})

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("T-JEPA PROFILING SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Overall statistics
    if 'iteration' in summary and 'mean' in summary['iteration']:
        iter_time = summary['iteration']['mean'] * 1000
        iter_count = summary['iteration'].get('count', 0)
        report_lines.append(f"Iteration Time: {iter_time:.2f} ms/iteration")
        report_lines.append(f"Throughput: {1000/iter_time:.2f} iterations/second")
        report_lines.append(f"Total Iterations: {iter_count}")
        report_lines.append("")

    # Top 20 iteration-level operations
    report_lines.append("TOP 20 OPERATIONS PER ITERATION:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Rank':<6} {'Operation':<45} {'Time (ms)':<12} {'% of Iter':<12} {'Count':<8}")
    report_lines.append("-" * 80)

    # Get iteration count
    iteration_count = summary.get('iteration', {}).get('count', 1610)

    # Filter for per-iteration operations
    iteration_ops = []
    for k, v in summary.items():
        if 'mean' not in v or k in ['iteration', 'epoch']:
            continue

        level = v.get('level', 0)
        parent = v.get('parent', '')
        count = v.get('count', 0)

        # Include if level 2 (child of iteration) or level 1 with same count
        if (level == 2 and 'iteration' in parent and count == iteration_count) or \
           (level == 1 and count == iteration_count):
            iteration_ops.append((k, v))

    sorted_ops = sorted(iteration_ops, key=lambda x: x[1]['mean'], reverse=True)[:20]

    iteration_time = summary.get('iteration', {}).get('mean', 1.0)

    for rank, (op_name, op_data) in enumerate(sorted_ops, 1):
        time_ms = op_data['mean'] * 1000
        percentage = (op_data['mean'] / iteration_time) * 100
        count = op_data.get('count', 0)
        report_lines.append(f"{rank:<6} {op_name[:45]:<45} {time_ms:<12.2f} {percentage:<12.1f} {count:<8}")

    report_lines.append("")
    report_lines.append("=" * 80)

    # Write report
    report_path = f'{output_dir}/profiling_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Created: profiling_summary_report.txt")

    # Also print to console
    print("\n" + "\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description='Create comprehensive profiling visualizations')
    parser.add_argument('profiling_json', help='Path to profiling JSON file')
    parser.add_argument('--output-dir', default='results/profiling/visualizations',
                       help='Output directory for visualizations')
    parser.add_argument('--threshold', type=float, default=0.05,
                       help='Threshold for bottleneck detection (default: 0.05 = 5%%)')

    args = parser.parse_args()

    print("=" * 80)
    print("T-JEPA PROFILING VISUALIZATION")
    print("=" * 80)
    print(f"Input: {args.profiling_json}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    print("")

    # Load data
    print("Loading profiling data...")
    data = load_profiling_data(args.profiling_json)
    print(f"✓ Loaded {len(data.get('trace', []))} profiling entries")
    print("")

    # Create output directory
    output_dir = create_output_dir(args.output_dir)
    print(f"✓ Created output directory: {output_dir}")
    print("")

    # Generate visualizations
    print("Generating visualizations...")
    print("-" * 80)

    plot_overall_timing_breakdown(data, output_dir)
    plot_hierarchical_breakdown(data, output_dir)
    plot_iteration_timeline(data, output_dir)
    plot_encoder_breakdown(data, output_dir)
    plot_bottleneck_analysis(data, output_dir, threshold=args.threshold)
    plot_data_movement_analysis(data, output_dir)
    plot_operation_statistics(data, output_dir)

    print("-" * 80)
    print("")

    # Create summary report
    print("Creating summary report...")
    create_summary_report(data, output_dir)
    print("")

    print("=" * 80)
    print(f"✓ All visualizations saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
