#!/bin/bash
# Generate PNG plots using matplotlib in hermit environment
# This script activates the environment where matplotlib is available

echo "=================================================="
echo "Generating PNG Profiling Plots"
echo "=================================================="

# Activate environment
source ../bin/activate-hermit 2>/dev/null || echo "Hermit not found, trying without"

# Check if matplotlib is available
python << 'CHECKEOF'
try:
    import matplotlib
    print("✓ matplotlib available")
    exit(0)
except ImportError:
    print("✗ matplotlib not available")
    print("Installing matplotlib...")
    exit(1)
CHECKEOF

if [ $? -ne 0 ]; then
    pip install matplotlib --quiet
fi

# Generate plots
python << 'EOF'
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

print("\n📊 Loading profiling data...")

# Load data
higgs_file = 'profiling_higgs_baseline-higgs__model_nlyrs_16_nheads_2_hdim_64__pred_ovrlap_F_npreds_4__nlyrs_16_activ_relunenc_1__lr_0.0003658682841082736_start_0.0_final_0.0_20251107_192429.json'
jannis_file = 'profiling_profiling_test-jannis__model_nlyrs_4_nheads_2_hdim_64__pred_ovrlap_F_npreds_4__nlyrs_2_activ_relunenc_1__lr_0.0003658682841082736_start_0.0_final_0.0_20251107_185418.json'

with open(higgs_file) as f:
    higgs_data = json.load(f)

with open(jannis_file) as f:
    jannis_data = json.load(f)

output_dir = Path('profiling_plots')
output_dir.mkdir(exist_ok=True)

h_iter = [s for s in higgs_data['summary'] if s['name'] == 'iteration'][0]
j_iter = [s for s in jannis_data['summary'] if s['name'] == 'iteration'][0]
h_iter_time = h_iter['mean_time']
j_iter_time = j_iter['mean_time']

print(f"✓ HIGGS: {len(higgs_data['summary'])} operations")
print(f"✓ Jannis: {len(jannis_data['summary'])} operations")
print(f"\n📊 Creating PNG plots...\n")

# ===== Plot 1: HIGGS Breakdown =====
fig, ax = plt.subplots(figsize=(14, 10))

components = [s for s in higgs_data['summary'] if s['level'] == 2]
components.sort(key=lambda x: x['mean_time'], reverse=True)

names = [c['name'].replace('_', ' ').title() for c in components[:10]]
times = [c['mean_time'] * 1000 for c in components[:10]]
colors = ['#d62728' if t > 40 else '#ff7f0e' if t > 20 else '#2ca02c' for t in times]

bars = ax.barh(names, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.set_xlabel('Time (milliseconds)', fontsize=14, fontweight='bold')
ax.set_title('HIGGS: Iteration Component Breakdown\n174.9ms average iteration time',
            fontsize=16, fontweight='bold', pad=20)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)

for i, (bar, time) in enumerate(zip(bars, times)):
    pct = (components[i]['mean_time'] / h_iter_time) * 100
    ax.text(time + 3, bar.get_y() + bar.get_height()/2,
            f'{time:.1f}ms ({pct:.1f}%)',
            va='center', fontsize=11, fontweight='bold')

legend_elements = [
    mpatches.Patch(facecolor='#d62728', edgecolor='black', label='Critical (>40ms)', alpha=0.8),
    mpatches.Patch(facecolor='#ff7f0e', edgecolor='black', label='High (20-40ms)', alpha=0.8),
    mpatches.Patch(facecolor='#2ca02c', edgecolor='black', label='Medium (<20ms)', alpha=0.8)
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

plt.tight_layout()
plot1 = output_dir / '1_higgs_iteration_breakdown.png'
plt.savefig(plot1, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ {plot1.name}")
plt.close()

# ===== Plot 2: Comparison =====
fig, ax = plt.subplots(figsize=(14, 9))

comp_names = ['backward_pass', 'predictor', 'context_encoder', 'target_encoder',
              'ema_update', 'optimizer_step', 'loss_computation', 'data_transfer']
display_names = ['Backward\nPass', 'Predictor', 'Context\nEncoder', 'Target\nEncoder',
                 'EMA\nUpdate', 'Optimizer\nStep', 'Loss\nComputation', 'Data\nTransfer']

j_times = []
h_times = []

for comp in comp_names:
    j_stat = [s for s in jannis_data['summary'] if s['name'] == comp]
    h_stat = [s for s in higgs_data['summary'] if s['name'] == comp]
    j_times.append(j_stat[0]['mean_time'] * 1000 if j_stat else 0)
    h_times.append(h_stat[0]['mean_time'] * 1000 if h_stat else 0)

x = np.arange(len(display_names))
width = 0.38

bars1 = ax.bar(x - width/2, j_times, width, label='Jannis (4 layers, 256 batch)',
              alpha=0.85, color='#3498db', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, h_times, width, label='HIGGS (16 layers, 1024 batch)',
              alpha=0.85, color='#e67e22', edgecolor='black', linewidth=1.5)

ax.set_ylabel('Time (milliseconds)', fontsize=14, fontweight='bold')
ax.set_title('Component Comparison: Jannis (4L) vs HIGGS (16L)\nScaling from Small to Production Config',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=11, fontweight='bold')
ax.legend(fontsize=12, loc='upper left', framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)

# Add ratio labels
for i, (j_t, h_t) in enumerate(zip(j_times, h_times)):
    if j_t > 0:
        ratio = h_t / j_t
        y_pos = max(j_t, h_t) + 3
        ax.text(i, y_pos, f'{ratio:.1f}×',
                ha='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plot2 = output_dir / '2_jannis_vs_higgs_comparison.png'
plt.savefig(plot2, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ {plot2.name}")
plt.close()

# ===== Plot 3: Pie Chart =====
fig, ax = plt.subplots(figsize=(12, 12))

level2 = [s for s in higgs_data['summary'] if s['level'] == 2]
level2.sort(key=lambda x: x['mean_time'], reverse=True)

pie_names = [c['name'].replace('_', ' ').title() for c in level2[:6]]
pie_values = [c['mean_time'] * 1000 for c in level2[:6]]

other_time = (h_iter_time * 1000) - sum(pie_values)
if other_time > 1:
    pie_names.append('Other')
    pie_values.append(other_time)

colors_pie = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6', '#1abc9c', '#95a5a6']
explode = [0.05 if i < 2 else 0 for i in range(len(pie_values))]

wedges, texts, autotexts = ax.pie(pie_values, labels=pie_names, autopct='%1.1f%%',
                                   colors=colors_pie, startangle=90, explode=explode,
                                   textprops={'fontsize': 12, 'fontweight': 'bold'},
                                   pctdistance=0.85)

# Make percentage text white
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)

ax.set_title('HIGGS: Time Distribution by Component\n174.9ms iteration time',
            fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plot3 = output_dir / '3_time_distribution_pie.png'
plt.savefig(plot3, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ {plot3.name}")
plt.close()

# ===== Plot 4: Scaling Analysis =====
fig, ax = plt.subplots(figsize=(12, 9))

comp_data = []
for comp in comp_names:
    j_stat = [s for s in jannis_data['summary'] if s['name'] == comp]
    h_stat = [s for s in higgs_data['summary'] if s['name'] == comp]

    if j_stat and h_stat:
        ratio = h_stat[0]['mean_time'] / j_stat[0]['mean_time']
        comp_data.append((comp, ratio))

comp_data.sort(key=lambda x: x[1], reverse=True)

comp_labels = [c[0].replace('_', ' ').title() for c in comp_data]
ratios = [c[1] for c in comp_data]

expected_vals = []
for comp, ratio in comp_data:
    if comp == 'predictor':
        expected_vals.append(8.0)
    elif comp in ['backward_pass', 'context_encoder', 'target_encoder']:
        expected_vals.append(4.0)
    else:
        expected_vals.append(1.0)

colors_scale = ['#d62728' if r > 8 else '#ff7f0e' if r > 4 else '#2ca02c' for r in ratios]

bars = ax.barh(comp_labels, ratios, color=colors_scale, alpha=0.8, edgecolor='black', linewidth=1.5)
ax.scatter(expected_vals, range(len(expected_vals)), color='black', s=150,
          marker='D', label='Expected Scaling', zorder=5, edgecolor='white', linewidth=2)

ax.set_xlabel('Scaling Factor (HIGGS / Jannis)', fontsize=14, fontweight='bold')
ax.set_title('Component Scaling Analysis: 4 Layers → 16 Layers\nActual vs Expected Slowdown',
            fontsize=16, fontweight='bold', pad=20)
ax.axvline(x=4.0, color='gray', linestyle='--', alpha=0.8, linewidth=2, label='4× Baseline (16/4 layers)')
ax.invert_yaxis()
ax.legend(fontsize=12, loc='lower right', framealpha=0.9)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)

for i, (bar, ratio, expected) in enumerate(zip(bars, ratios, expected_vals)):
    efficiency = (expected / ratio) * 100 if ratio > 0 else 0
    color_text = '#27ae60' if efficiency > 100 else '#f39c12' if efficiency > 80 else '#e74c3c'
    ax.text(ratio + 0.3, bar.get_y() + bar.get_height()/2,
            f'{ratio:.2f}× ({efficiency:.0f}%)',
            va='center', fontsize=10, fontweight='bold', color=color_text)

plt.tight_layout()
plot4 = output_dir / '4_scaling_factor_analysis.png'
plt.savefig(plot4, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ {plot4.name}")
plt.close()

# ===== Plot 5: Bottleneck Priority Scatter =====
fig, ax = plt.subplots(figsize=(14, 11))

bottlenecks_data = [
    ('gradient_logging', 139.31, 79.7, 1),
    ('backward_pass', 60.59, 34.7, 4),
    ('predictor', 38.55, 22.0, 3),
    ('context_encoder', 27.56, 15.8, 5),
    ('target_encoder', 24.55, 14.0, 5),
    ('categorical_encoding', 6.4, 3.6, 2),
    ('ema_update', 8.58, 4.9, 5),
    ('optimizer_step', 7.33, 4.2, 3),
]

efforts = [b[3] for b in bottlenecks_data]
impacts = [b[2] for b in bottlenecks_data]
sizes = [b[1] * 5 for b in bottlenecks_data]
names = [b[0].replace('_', ' ').title() for b in bottlenecks_data]

priorities = [imp / eff for imp, eff in zip(impacts, efforts)]

scatter = ax.scatter(efforts, impacts, s=sizes, alpha=0.6, c=priorities,
                    cmap='RdYlGn_r', edgecolor='black', linewidth=2)

for i, name in enumerate(names):
    ax.annotate(name, (efforts[i], impacts[i]),
               xytext=(12, 8), textcoords='offset points',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5, edgecolor='black'))

ax.set_xlabel('Implementation Effort (1=Easy, 5=Hard)', fontsize=14, fontweight='bold')
ax.set_ylabel('Impact (% of Iteration Time)', fontsize=14, fontweight='bold')
ax.set_title('Bottleneck Priority Matrix\nBubble size = Absolute time, Color = Priority score',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 6)
ax.set_ylim(0, 90)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1)

ax.axhline(y=20, color='gray', linestyle='--', alpha=0.6, linewidth=2)
ax.axvline(x=3, color='gray', linestyle='--', alpha=0.6, linewidth=2)

ax.text(1.5, 85, 'HIGH IMPACT\nEASY FIX', ha='center', fontsize=12,
       fontweight='bold', color='green',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=2))
ax.text(4.5, 85, 'HIGH IMPACT\nHARD FIX', ha='center', fontsize=12,
       fontweight='bold', color='orange',
       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8, edgecolor='orange', linewidth=2))

cbar = plt.colorbar(scatter, ax=ax, label='Priority Score (Impact/Effort)')
cbar.set_label('Priority Score', fontsize=12, fontweight='bold')

plt.tight_layout()
plot5 = output_dir / '5_bottleneck_priority_matrix.png'
plt.savefig(plot5, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ {plot5.name}")
plt.close()

# ===== Plot 6: Stacked Percentage Bar =====
fig, ax = plt.subplots(figsize=(14, 7))

categories = ['Jannis (4 layers)', 'HIGGS (16 layers)']

comp_list = ['backward_pass', 'predictor', 'context_encoder', 'target_encoder',
            'ema_update', 'optimizer_step', 'loss_computation']
colors_stack = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6', '#e67e22', '#1abc9c']
labels_stack = [c.replace('_', ' ').title() for c in comp_list]

jannis_pcts = []
higgs_pcts = []

for comp in comp_list:
    j_stat = [s for s in jannis_data['summary'] if s['name'] == comp]
    h_stat = [s for s in higgs_data['summary'] if s['name'] == comp]

    j_pct = (j_stat[0]['mean_time'] / j_iter_time) * 100 if j_stat else 0
    h_pct = (h_stat[0]['mean_time'] / h_iter_time) * 100 if h_stat else 0

    jannis_pcts.append(j_pct)
    higgs_pcts.append(h_pct)

y_pos = np.arange(len(categories))
bottom_j = 0
bottom_h = 0

for i, (comp, color, label) in enumerate(zip(comp_list, colors_stack, labels_stack)):
    ax.barh(0, jannis_pcts[i], left=bottom_j, color=color, alpha=0.85,
           edgecolor='black', linewidth=1, label=label if i < len(comp_list) else '')
    bottom_j += jannis_pcts[i]

    ax.barh(1, higgs_pcts[i], left=bottom_h, color=color, alpha=0.85,
           edgecolor='black', linewidth=1)
    bottom_h += higgs_pcts[i]

ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=13, fontweight='bold')
ax.set_xlabel('Percentage of Iteration Time (%)', fontsize=14, fontweight='bold')
ax.set_title('Component Distribution: Percentage Breakdown\nHow iteration time is allocated',
            fontsize=16, fontweight='bold', pad=20)
ax.set_xlim(0, 100)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, framealpha=0.9)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)

plt.tight_layout()
plot6 = output_dir / '6_percentage_distribution.png'
plt.savefig(plot6, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ {plot6.name}")
plt.close()

# ===== Plot 7: Categorical Encoding Focus =====
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Left: Categorical encoding breakdown
cat_ops = [s for s in higgs_data['summary'] if 'categorical' in s['name'].lower()]
cat_names = [c['name'] for c in cat_ops]
cat_times = [c['mean_time'] * 1000 for c in cat_ops]

ax1.barh(cat_names, cat_times, color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Time (milliseconds)', fontsize=13, fontweight='bold')
ax1.set_title('Categorical Encoding Bottleneck\nCPU-GPU Transfer Issue',
             fontsize=14, fontweight='bold')
ax1.invert_yaxis()
ax1.grid(axis='x', alpha=0.3, linestyle='--')

for i, (name, time) in enumerate(zip(cat_names, cat_times)):
    pct = (cat_ops[i]['mean_time'] / h_iter_time) * 100
    ax1.text(time + 0.2, i, f'{time:.2f}ms ({pct:.1f}%)',
            va='center', fontsize=10, fontweight='bold')

# Right: Before/After potential
current_total = sum(cat_times)
optimized_total = current_total / 16  # 15-20x speedup expected

categories_opt = ['Current\n(CPU-GPU)', 'Optimized\n(GPU-only)']
times_opt = [current_total, optimized_total]
colors_opt = ['#e74c3c', '#27ae60']

bars = ax2.bar(categories_opt, times_opt, color=colors_opt, alpha=0.8,
              edgecolor='black', linewidth=2, width=0.6)
ax2.set_ylabel('Total Time (milliseconds)', fontsize=13, fontweight='bold')
ax2.set_title('Categorical Encoding: Optimization Potential\n15-20× Speedup Expected',
             fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

for bar, time in zip(bars, times_opt):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.3,
            f'{time:.2f}ms',
            ha='center', fontsize=12, fontweight='bold')

# Add speedup annotation
ax2.annotate('', xy=(1, optimized_total), xytext=(0, current_total),
            arrowprops=dict(arrowstyle='->', lw=3, color='green'))
ax2.text(0.5, (current_total + optimized_total) / 2,
        f'{current_total/optimized_total:.1f}× faster',
        ha='center', fontsize=13, fontweight='bold', color='green',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plot7 = output_dir / '7_categorical_encoding_analysis.png'
plt.savefig(plot7, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ {plot7.name}")
plt.close()

print(f"\n✅ All plots created successfully!")
print(f"📁 Location: profiling_plots/")
print(f"\nFiles created:")
print(f"  • 1_higgs_iteration_breakdown.png")
print(f"  • 2_jannis_vs_higgs_comparison.png")
print(f"  • 3_time_distribution_pie.png")
print(f"  • 4_scaling_factor_analysis.png")
print(f"  • 5_bottleneck_priority_matrix.png")
print(f"  • 6_percentage_distribution.png")
print(f"  • 7_categorical_encoding_analysis.png")

EOF

echo ""
echo "=================================================="
echo "✅ PNG plots generated successfully"
echo "=================================================="
echo "📁 Location: profiling_plots/"
ls -lh profiling_plots/*.png 2>/dev/null || echo "Checking..."
