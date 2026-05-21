#!/usr/bin/env python3
"""
Constitutive vs Variable Residue Classification Analysis
 vs analysis

Purpose: Classify residues into functional categories:
- Constitutive: High importance across both classes (low difference)
- Class-Specific: High importance AND class-discriminative (high difference)
- Variable: Low overall importance but class-discriminative
- Silent: Low importance, low difference

This analysis may help explain R283's role even if it's not a top differential residue.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Paths
csv_file = Path('result/class_attention_analysis_pdb/attention_analysis_pdb_numbering_CORRECTED.csv')
output_dir = Path('result/class_attention_analysis_pdb')
output_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("Constitutive vs Variable Residue Classification Analysis")
print("="*80)

# Load data
print(f"\nLoading data from: {csv_file}")
df = pd.read_csv(csv_file)
print(f"Total residues: {len(df)}")

# Calculate overall importance and absolute difference
df['Overall_Importance'] = (df['Class_0_Mean'] + df['Class_1_Mean']) / 2
df['Abs_Difference'] = abs(df['Difference'])

print("\nCalculating classification metrics...")
print(f"  Overall_Importance = (Class_0_Mean + Class_1_Mean) / 2")
print(f"  Abs_Difference = |Difference|")

# Classification function
def classify_residue(row):
    """
    Classify residues based on overall importance and differential

    Categories:
    - Constitutive: High importance (>1.5), low difference (<0.1)
      → Important for both classes, not discriminative
    - Class-Specific: High importance (>1.5), high difference (>0.2)
      → Important AND discriminative (like E124)
    - Variable: Low importance (<1.5), high difference (>0.2)
      → Not generally important but class-discriminative
    - Silent: Low importance (<1.5), low difference (<0.1)
      → Not important, not discriminative
    - Moderate: Everything else
    """
    imp = row['Overall_Importance']
    diff = row['Abs_Difference']

    if imp > 1.5 and diff < 0.1:
        return 'Constitutive'
    elif imp > 1.5 and diff > 0.2:
        return 'Class-Specific'
    elif imp < 1.5 and diff > 0.2:
        return 'Variable'
    elif imp < 1.5 and diff < 0.1:
        return 'Silent'
    else:
        return 'Moderate'

df['Residue_Type'] = df.apply(classify_residue, axis=1)

# Statistics
print("\n" + "="*80)
print("Classification Results")
print("="*80)
type_counts = df['Residue_Type'].value_counts()
print(f"\nTotal residues classified: {len(df)}")
for res_type, count in type_counts.items():
    pct = count / len(df) * 100
    print(f"  {res_type:20s}: {count:3d} ({pct:5.1f}%)")

# Find E124 and R283
print("\n" + "="*80)
print("Key Residues: E124 and R283")
print("="*80)

e124_data = df[df['PDB_Residue'] == 124]
r283_data = df[df['PDB_Residue'] == 283]

if not e124_data.empty:
    row = e124_data.iloc[0]
    print(f"\nE124:")
    print(f"  Residue Type: {row['Residue_Type']}")
    print(f"  Overall Importance: {row['Overall_Importance']:.4f}")
    print(f"  Abs Difference: {row['Abs_Difference']:.4f}")
    print(f"  Class 0 Mean: {row['Class_0_Mean']:.4f}")
    print(f"  Class 1 Mean: {row['Class_1_Mean']:.4f}")
    print(f"  Difference: {row['Difference']:.4f}")
    print(f"  P-value: {row['P_value']:.2e}")
    print(f"  Significance: {'***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*' if row['P_value'] < 0.05 else 'ns'}")

if not r283_data.empty:
    row = r283_data.iloc[0]
    print(f"\nR283:")
    print(f"  Residue Type: {row['Residue_Type']}")
    print(f"  Overall Importance: {row['Overall_Importance']:.4f}")
    print(f"  Abs Difference: {row['Abs_Difference']:.4f}")
    print(f"  Class 0 Mean: {row['Class_0_Mean']:.4f}")
    print(f"  Class 1 Mean: {row['Class_1_Mean']:.4f}")
    print(f"  Difference: {row['Difference']:.4f}")
    print(f"  P-value: {row['P_value']:.2e}")
    print(f"  Significance: {'***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*' if row['P_value'] < 0.05 else 'ns'}")

# Top residues in each category
print("\n" + "="*80)
print("Top Residues by Category")
print("="*80)

for res_type in ['Constitutive', 'Class-Specific', 'Variable', 'Silent', 'Moderate']:
    df_type = df[df['Residue_Type'] == res_type]
    if len(df_type) > 0:
        print(f"\n{res_type} (top 10 by overall importance):")
        df_top = df_type.nlargest(10, 'Overall_Importance')
        for idx, row in df_top.iterrows():
            sig = '***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*' if row['P_value'] < 0.05 else 'ns'
            print(f"  {row['Residue']}{int(row['PDB_Residue']):3d}  "
                  f"Imp={row['Overall_Importance']:6.3f}  "
                  f"Diff={row['Difference']:7.4f}  "
                  f"p={row['P_value']:.2e}  {sig}")

# Save classification results
output_csv = output_dir / 'constitutive_variable_classification.csv'
df_save = df[['PDB_Residue', 'Residue', 'Overall_Importance', 'Difference', 'Abs_Difference',
              'Class_0_Mean', 'Class_1_Mean', 'P_value', 'Significant', 'Residue_Type']]
df_save = df_save.sort_values('Overall_Importance', ascending=False)
df_save.to_csv(output_csv, index=False)
print(f"\n✓ Classification results saved: {output_csv}")

# Create scatter plot: Overall Importance vs Abs Difference
print("\nGenerating visualization...")

fig, ax = plt.subplots(figsize=(14, 10))

# Color map for categories
color_map = {
    'Constitutive': '#27AE60',    # Green
    'Class-Specific': '#E74C3C',  # Red
    'Variable': '#F39C12',        # Orange
    'Silent': '#95A5A6',          # Gray
    'Moderate': '#3498DB'         # Blue
}

# Plot each category
for res_type, color in color_map.items():
    df_type = df[df['Residue_Type'] == res_type]
    ax.scatter(df_type['Overall_Importance'], df_type['Abs_Difference'],
               c=color, label=f'{res_type} (n={len(df_type)})',
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

# Add decision boundaries
ax.axhline(y=0.1, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=0.2, color='gray', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=1.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)

# Annotate E124 and R283
if not e124_data.empty:
    row = e124_data.iloc[0]
    ax.scatter(row['Overall_Importance'], row['Abs_Difference'],
               c='red', s=200, marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax.annotate('E124', xy=(row['Overall_Importance'], row['Abs_Difference']),
                xytext=(10, 10), textcoords='offset points', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

if not r283_data.empty:
    row = r283_data.iloc[0]
    ax.scatter(row['Overall_Importance'], row['Abs_Difference'],
               c='blue', s=200, marker='*', edgecolors='black', linewidth=2, zorder=10)
    ax.annotate('R283', xy=(row['Overall_Importance'], row['Abs_Difference']),
                xytext=(10, -20), textcoords='offset points', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=2))

# Add category labels
ax.text(0.5, 0.25, 'Variable', fontsize=11, alpha=0.5, ha='center')
ax.text(0.5, 0.05, 'Silent', fontsize=11, alpha=0.5, ha='center')
ax.text(3.5, 0.05, 'Constitutive', fontsize=11, alpha=0.5, ha='center')
ax.text(3.5, 0.35, 'Class-Specific', fontsize=11, alpha=0.5, ha='center')

ax.set_xlabel('Overall Importance (Mean Attention Across Both Classes)', fontsize=13, fontweight='bold')
ax.set_ylabel('Absolute Difference |Δ|', fontsize=13, fontweight='bold')
ax.set_title('Constitutive vs Variable Residue Classification\n'
             'Overall Importance vs Class Discrimination',
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax.grid(True, alpha=0.3)

plt.tight_layout()
output_plot = output_dir / 'constitutive_variable_scatter.png'
plt.savefig(output_plot, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Scatter plot saved: {output_plot}")

output_pdf = output_dir / 'constitutive_variable_scatter.pdf'
plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
print(f"✓ PDF version saved: {output_pdf}")
plt.close()

# Create bar chart showing category distribution
fig, ax = plt.subplots(figsize=(10, 6))

categories = ['Constitutive', 'Class-Specific', 'Variable', 'Moderate', 'Silent']
counts = [type_counts.get(cat, 0) for cat in categories]
colors_bar = [color_map[cat] for cat in categories]

bars = ax.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add count labels on bars
for bar, count in zip(bars, counts):
    height = bar.get_height()
    pct = count / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{count}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Number of Residues', fontsize=13, fontweight='bold')
ax.set_title('Distribution of Residue Types', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0, max(counts) * 1.15)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_bar = output_dir / 'constitutive_variable_distribution.png'
plt.savefig(output_bar, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Distribution bar chart saved: {output_bar}")

output_bar_pdf = output_dir / 'constitutive_variable_distribution.pdf'
plt.savefig(output_bar_pdf, format='pdf', bbox_inches='tight', facecolor='white')
print(f"✓ PDF version saved: {output_bar_pdf}")
plt.close()

# Create coefficient of variation analysis
print("\n" + "="*80)
print("Coefficient of Variation (CV) Analysis")
print("="*80)

# Calculate CV for each class
df['Class_0_CV'] = (df['Class_0_Std'] / df['Class_0_Mean']) * 100
df['Class_1_CV'] = (df['Class_1_Std'] / df['Class_1_Mean']) * 100
df['Overall_CV'] = (df['Class_0_CV'] + df['Class_1_CV']) / 2

print("\nCV interpretation:")
print("  Low CV (<30%): Stable attention across compounds")
print("  Medium CV (30-60%): Moderate variability")
print("  High CV (>60%): High variability")

# Find low CV residues with high overall importance (constitutive anchors)
df['Constitutive_Anchor'] = (df['Overall_CV'] < 30) & (df['Overall_Importance'] > 1.5)
constitutive_anchors = df[df['Constitutive_Anchor']].sort_values('Overall_Importance', ascending=False)

print(f"\nConstitutive Anchors (CV < 30%, Importance > 1.5): {len(constitutive_anchors)}")
if len(constitutive_anchors) > 0:
    print("\nTop 10 Constitutive Anchors:")
    for idx, row in constitutive_anchors.head(10).iterrows():
        print(f"  {row['Residue']}{int(row['PDB_Residue']):3d}  "
              f"Imp={row['Overall_Importance']:6.3f}  "
              f"CV={row['Overall_CV']:5.1f}%  "
              f"Diff={row['Difference']:7.4f}")

# Check R283 CV (need to get from the main df, not the filtered subset)
r283_idx = df[df['PDB_Residue'] == 283].index
if len(r283_idx) > 0:
    row = df.loc[r283_idx[0]]
    print(f"\nR283 CV Analysis:")
    print(f"  Class 0 CV: {row['Class_0_CV']:.1f}%")
    print(f"  Class 1 CV: {row['Class_1_CV']:.1f}%")
    print(f"  Overall CV: {row['Overall_CV']:.1f}%")
    print(f"  Constitutive Anchor: {'YES' if row['Constitutive_Anchor'] else 'NO'}")

# Save CV analysis
output_cv = output_dir / 'coefficient_variation_analysis.csv'
df_cv = df[['PDB_Residue', 'Residue', 'Overall_Importance', 'Overall_CV',
            'Class_0_CV', 'Class_1_CV', 'Difference', 'Constitutive_Anchor']]
df_cv = df_cv.sort_values('Overall_Importance', ascending=False)
df_cv.to_csv(output_cv, index=False)
print(f"\n✓ CV analysis saved: {output_cv}")

print("\n" + "="*80)
print("Analysis Complete!")
print("="*80)
print("\nKey Findings:")
print(f"  Total residues: {len(df)}")
print(f"  Constitutive: {type_counts.get('Constitutive', 0)} ({type_counts.get('Constitutive', 0)/len(df)*100:.1f}%)")
print(f"  Class-Specific: {type_counts.get('Class-Specific', 0)} ({type_counts.get('Class-Specific', 0)/len(df)*100:.1f}%)")
print(f"  Variable: {type_counts.get('Variable', 0)} ({type_counts.get('Variable', 0)/len(df)*100:.1f}%)")
print(f"  Constitutive Anchors (CV<30%, Imp>1.5): {len(constitutive_anchors)}")

if not e124_data.empty:
    print(f"\n  E124 Classification: {e124_data.iloc[0]['Residue_Type']}")
if not r283_data.empty:
    print(f"  R283 Classification: {r283_data.iloc[0]['Residue_Type']}")

print("\nFiles generated:")
print(f"  • {output_csv}")
print(f"  • {output_cv}")
print(f"  • {output_plot}")
print(f"  • {output_bar}")
print("="*80)
