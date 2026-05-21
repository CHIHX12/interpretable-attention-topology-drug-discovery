#!/usr/bin/env python3
"""
bidirectionalanalysisActive-Preferred vs Inactive-Preferred
analysis Constitutive residue

Purpose: - active inactive
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load data
csv_file = Path('result/class_attention_analysis_pdb/attention_analysis_pdb_numbering_CORRECTED.csv')
df = pd.read_csv(csv_file)

# Calculate metrics
df['Overall_Importance'] = (df['Class_0_Mean'] + df['Class_1_Mean']) / 2
df['Abs_Difference'] = abs(df['Difference'])

# Classify
def classify_residue(row):
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

# groupanalysis
print("="*80)
print("bidirectionalanalysisConstitutive residue")
print("="*80)

# 1. Constitutive - Active-Preferred (Difference > 0.01)
df_const = df[df['Residue_Type'] == 'Constitutive'].copy()
df_const_active = df_const[df_const['Difference'] > 0.01].sort_values('Overall_Importance', ascending=False)
df_const_inactive = df_const[df_const['Difference'] < -0.01].sort_values('Overall_Importance', ascending=False)
df_const_neutral = df_const[df_const['Abs_Difference'] <= 0.01].sort_values('Overall_Importance', ascending=False)

print(f"\nConstitutive residue: {len(df_const)}")
print(f"  • Active-Preferred (Diff > 0.01):   {len(df_const_active):3d} ({len(df_const_active)/len(df_const)*100:5.1f}%)")
print(f"  • Inactive-Preferred (Diff < -0.01): {len(df_const_inactive):3d} ({len(df_const_inactive)/len(df_const)*100:5.1f}%)")
print(f"  • Neutral (|Diff| ≤ 0.01):           {len(df_const_neutral):3d} ({len(df_const_neutral)/len(df_const)*100:5.1f}%)")

# Table 1: Active-Preferred Constitutive
print("\n" + "="*80)
print("Table 1: Active-Preferred Constitutive Residues (Top 10)")
print("="*80)
print(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'Class_0':<10}{'Class_1':<10}")
print("-"*80)
for i, (idx, row) in enumerate(df_const_active.head(10).iterrows(), 1):
    print(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
          f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
          f"{row['Class_0_Mean']:<10.4f}{row['Class_1_Mean']:<10.4f}")

# Find R283
r283_in_active = df_const_active[df_const_active['PDB_Residue'] == 283]
if not r283_in_active.empty:
    rank = list(df_const_active.index).index(r283_in_active.index[0]) + 1
    print(f"\n*** R283 Active-Preferred Constitutive: Rank #{rank}/{len(df_const_active)}")

# Table 2: Inactive-Preferred Constitutive
print("\n" + "="*80)
print("Table 2: Inactive-Preferred Constitutive Residues (Top 10)")
print("="*80)
print(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'Class_0':<10}{'Class_1':<10}")
print("-"*80)
for i, (idx, row) in enumerate(df_const_inactive.head(10).iterrows(), 1):
    print(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
          f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
          f"{row['Class_0_Mean']:<10.4f}{row['Class_1_Mean']:<10.4f}")

# Table 3: Neutral Constitutive (Perfect Anchors)
print("\n" + "="*80)
print("Table 3: Neutral Constitutive Residues (|Diff| ≤ 0.01) - Perfect Anchors")
print("="*80)
print(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'Class_0':<10}{'Class_1':<10}")
print("-"*80)
for i, (idx, row) in enumerate(df_const_neutral.head(15).iterrows(), 1):
    print(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
          f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
          f"{row['Class_0_Mean']:<10.4f}{row['Class_1_Mean']:<10.4f}")

# Find R283 in neutral
r283_in_neutral = df_const_neutral[df_const_neutral['PDB_Residue'] == 283]
if not r283_in_neutral.empty:
    rank = list(df_const_neutral.index).index(r283_in_neutral.index[0]) + 1
    print(f"\n*** R283 Neutral Constitutive: Rank #{rank}/{len(df_const_neutral)}")

# Class-Specific analysis
print("\n" + "="*80)
print("Class-Specific residuebidirectionalanalysis")
print("="*80)

df_class_spec = df[df['Residue_Type'] == 'Class-Specific'].copy()
df_class_active = df_class_spec[df_class_spec['Difference'] > 0].sort_values('Difference', ascending=False)
df_class_inactive = df_class_spec[df_class_spec['Difference'] < 0].sort_values('Difference', ascending=True)

print(f"\nClass-Specific residue: {len(df_class_spec)}")
print(f"  • Active-Preferred (Diff > 0):   {len(df_class_active):3d} ({len(df_class_active)/len(df_class_spec)*100:5.1f}%)")
print(f"  • Inactive-Preferred (Diff < 0): {len(df_class_inactive):3d} ({len(df_class_inactive)/len(df_class_spec)*100:5.1f}%)")

# Table 4: Active-Preferred Class-Specific (E124 should be here)
print("\n" + "="*80)
print("Table 4: Active-Preferred Class-Specific Residues (Top 10)")
print("="*80)
print(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'P-value':<12}")
print("-"*80)
for i, (idx, row) in enumerate(df_class_active.head(10).iterrows(), 1):
    sig = '***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*'
    print(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
          f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
          f"{row['P_value']:<12.2e} {sig}")

# Find E124
e124_in_active = df_class_active[df_class_active['PDB_Residue'] == 124]
if not e124_in_active.empty:
    rank = list(df_class_active.index).index(e124_in_active.index[0]) + 1
    print(f"\n*** E124 Active-Preferred Class-Specific: Rank #{rank}/{len(df_class_active)}")

# Table 5: Inactive-Preferred Class-Specific
print("\n" + "="*80)
print("Table 5: Inactive-Preferred Class-Specific Residues (Top 10)")
print("="*80)
print(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'P-value':<12}")
print("-"*80)
for i, (idx, row) in enumerate(df_class_inactive.head(10).iterrows(), 1):
    sig = '***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*'
    print(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
          f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
          f"{row['P_value']:<12.2e} {sig}")

# Summary comparison
print("\n" + "="*80)
print("summarypair (Summary Comparison)")
print("="*80)

summary_data = {
    'Category': [
        'Constitutive (Active-Pref)',
        'Constitutive (Inactive-Pref)',
        'Constitutive (Neutral)',
        'Class-Specific (Active-Pref)',
        'Class-Specific (Inactive-Pref)',
    ],
    'Count': [
        len(df_const_active),
        len(df_const_inactive),
        len(df_const_neutral),
        len(df_class_active),
        len(df_class_inactive),
    ],
    'Mean_Importance': [
        df_const_active['Overall_Importance'].mean() if len(df_const_active) > 0 else 0,
        df_const_inactive['Overall_Importance'].mean() if len(df_const_inactive) > 0 else 0,
        df_const_neutral['Overall_Importance'].mean() if len(df_const_neutral) > 0 else 0,
        df_class_active['Overall_Importance'].mean() if len(df_class_active) > 0 else 0,
        df_class_inactive['Overall_Importance'].mean() if len(df_class_inactive) > 0 else 0,
    ],
    'Mean_AbsDiff': [
        df_const_active['Abs_Difference'].mean() if len(df_const_active) > 0 else 0,
        df_const_inactive['Abs_Difference'].mean() if len(df_const_inactive) > 0 else 0,
        df_const_neutral['Abs_Difference'].mean() if len(df_const_neutral) > 0 else 0,
        df_class_active['Abs_Difference'].mean() if len(df_class_active) > 0 else 0,
        df_class_inactive['Abs_Difference'].mean() if len(df_class_inactive) > 0 else 0,
    ]
}

summary_df = pd.DataFrame(summary_data)
print(f"\n{'Category':<35}{'Count':<10}{'Mean_Imp':<12}{'Mean_|Δ|':<12}")
print("-"*80)
for idx, row in summary_df.iterrows():
    print(f"{row['Category']:<35}{row['Count']:<10}{row['Mean_Importance']:<12.4f}{row['Mean_AbsDiff']:<12.4f}")

# Save detailed results
output_dir = Path('result/class_attention_analysis_pdb')
output_dir.mkdir(parents=True, exist_ok=True)

# Save all tables
with open(output_dir / 'bidirectional_constitutive_analysis.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("bidirectionalanalysisConstitutive Class-Specific residue\n")
    f.write("="*80 + "\n\n")

    # Table 1
    f.write("Table 1: Active-Preferred Constitutive Residues (Top 10)\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'Class_0':<10}{'Class_1':<10}\n")
    for i, (idx, row) in enumerate(df_const_active.head(10).iterrows(), 1):
        f.write(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
                f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
                f"{row['Class_0_Mean']:<10.4f}{row['Class_1_Mean']:<10.4f}\n")

    # Table 2
    f.write("\n" + "="*80 + "\n")
    f.write("Table 2: Inactive-Preferred Constitutive Residues (Top 10)\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'Class_0':<10}{'Class_1':<10}\n")
    for i, (idx, row) in enumerate(df_const_inactive.head(10).iterrows(), 1):
        f.write(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
                f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
                f"{row['Class_0_Mean']:<10.4f}{row['Class_1_Mean']:<10.4f}\n")

    # Table 3
    f.write("\n" + "="*80 + "\n")
    f.write("Table 3: Neutral Constitutive Residues (Top 15)\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Rank':<6}{'Residue':<10}{'PDB#':<8}{'Importance':<12}{'Difference':<12}{'Class_0':<10}{'Class_1':<10}\n")
    for i, (idx, row) in enumerate(df_const_neutral.head(15).iterrows(), 1):
        f.write(f"{i:<6}{row['Residue']}{int(row['PDB_Residue']):<9}{int(row['PDB_Residue']):<8}"
                f"{row['Overall_Importance']:<12.4f}{row['Difference']:<12.4f}"
                f"{row['Class_0_Mean']:<10.4f}{row['Class_1_Mean']:<10.4f}\n")

# Save CSV
df_const['Direction'] = df_const.apply(
    lambda x: 'Active-Pref' if x['Difference'] > 0.01 else 'Inactive-Pref' if x['Difference'] < -0.01 else 'Neutral',
    axis=1
)
df_const_sorted = df_const.sort_values(['Direction', 'Overall_Importance'], ascending=[True, False])
df_const_sorted.to_csv(output_dir / 'constitutive_bidirectional.csv', index=False)

print(f"\n✓ analysissave:")
print(f"  • {output_dir / 'bidirectional_constitutive_analysis.txt'}")
print(f"  • {output_dir / 'constitutive_bidirectional.csv'}")
print("\n" + "="*80)
