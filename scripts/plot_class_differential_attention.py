#!/usr/bin/env python3
"""
classdifferenceattentionanalysisClass 0 vs Class 1
 Nature Machine Intelligence Figure 4
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from pathlib import Path
from scipy.stats import mannwhitneyu
import pandas as pd


def compute_class_statistics(results):
    """
 compute Class 0 Class 1 attentionstatistics

    Returns:
        class0_data: dict with mean, std, samples
        class1_data: dict with mean, std, samples
    """
    y_true = np.array(results['y_true'])

 # Class 0 Class 1
    class0_indices = np.where(y_true == 0)[0]
    class1_indices = np.where(y_true == 1)[0]

    print(f"\n📊 Class Distribution:")
    print(f"   Class 0 (Low affinity): {len(class0_indices)} samples")
    print(f"   Class 1 (High affinity): {len(class1_indices)} samples")

 # attentionmatrix
    class0_attentions = [results['attentions'][i] for i in class0_indices]
    class1_attentions = [results['attentions'][i] for i in class1_indices]

 # computeaverageattentionresidue
 # proteinlengthminlength
    min_protein_len = min([att.shape[1] for att in results['attentions']])

    class0_avg_per_sample = []
    for att in class0_attentions:
 # averagesampleattentiondrugdimension
        avg_att = att[:, :min_protein_len].mean(axis=0)
        class0_avg_per_sample.append(avg_att)

    class1_avg_per_sample = []
    for att in class1_attentions:
        avg_att = att[:, :min_protein_len].mean(axis=0)
        class1_avg_per_sample.append(avg_att)

    class0_matrix = np.array(class0_avg_per_sample)  # [N0, protein_len]
    class1_matrix = np.array(class1_avg_per_sample)  # [N1, protein_len]

    class0_data = {
        'mean': class0_matrix.mean(axis=0),
        'std': class0_matrix.std(axis=0),
        'matrix': class0_matrix,
        'n_samples': len(class0_indices)
    }

    class1_data = {
        'mean': class1_matrix.mean(axis=0),
        'std': class1_matrix.std(axis=0),
        'matrix': class1_matrix,
        'n_samples': len(class1_indices)
    }

    return class0_data, class1_data, min_protein_len


def statistical_test(class0_data, class1_data, alpha=0.01):
    """
 pairresiduerowstatistics

    Returns:
        significant_residues: list of residue indices with p < alpha
        p_values: array of p-values for each residue
    """
    n_residues = class0_data['matrix'].shape[1]
    p_values = []

    for i in range(n_residues):
        class0_values = class0_data['matrix'][:, i]
        class1_values = class1_data['matrix'][:, i]

        # Mann-Whitney U test (non-parametric)
        _, p_value = mannwhitneyu(class0_values, class1_values, alternative='two-sided')
        p_values.append(p_value)

    p_values = np.array(p_values)

    # Bonferroni correction
    alpha_corrected = alpha / n_residues
    significant_residues = np.where(p_values < alpha_corrected)[0]

    print(f"\n🔬 Statistical Testing:")
    print(f"   Total residues tested: {n_residues}")
    print(f"   Significance threshold: {alpha} (Bonferroni corrected: {alpha_corrected:.2e})")
    print(f"   Significant residues: {len(significant_residues)} ({len(significant_residues)/n_residues*100:.1f}%)")

    return significant_residues, p_values


def plot_class_differential_attention(class0_data, class1_data, significant_residues,
                                      p_values, save_path, protein_len):
    """
 classdifferenceattentionanalysis

 3Panel
 - Panel A: Class 0 averageattention + 
 - Panel B: Class 1 averageattention + 
 - Panel C: differenceattention (Class1 - Class0) + statisticssignificance
    """
    fig, axes = plt.subplots(3, 1, figsize=(20, 14))

    residue_pos = np.arange(protein_len)
    class0_mean = class0_data['mean']
    class0_std = class0_data['std']
    class1_mean = class1_data['mean']
    class1_std = class1_data['std']

    # Panel A: Class 0 (Low affinity) attention profile
    axes[0].plot(residue_pos, class0_mean, color='#2E86AB', linewidth=2.5,
                 label=f'Class 0 Mean (n={class0_data["n_samples"]})', zorder=3)
    axes[0].fill_between(residue_pos,
                         class0_mean - class0_std,
                         class0_mean + class0_std,
                         alpha=0.25, color='#2E86AB', label='±1 SD', zorder=2)

    axes[0].set_ylabel('Attention Weight', fontsize=14, fontweight='bold')
    axes[0].set_title('Class 0 (Low Affinity) - Attention Profile',
                     fontsize=16, fontweight='bold', pad=15, color='#2E86AB')
    axes[0].legend(fontsize=12, loc='upper right')
    axes[0].grid(alpha=0.3, linestyle='--', linewidth=0.5)
    axes[0].set_xlim(0, protein_len)
    axes[0].set_ylim(0, max(class0_mean + class0_std) * 1.1)

 # Top-10 residueClass 0
    top10_class0 = np.argsort(-class0_mean)[:10]
    axes[0].scatter(top10_class0, class0_mean[top10_class0],
                   color='red', s=80, zorder=5, marker='o', edgecolor='black', linewidth=1.5)

    # Panel B: Class 1 (High affinity) attention profile
    axes[1].plot(residue_pos, class1_mean, color='#A23B72', linewidth=2.5,
                 label=f'Class 1 Mean (n={class1_data["n_samples"]})', zorder=3)
    axes[1].fill_between(residue_pos,
                         class1_mean - class1_std,
                         class1_mean + class1_std,
                         alpha=0.25, color='#A23B72', label='±1 SD', zorder=2)

    axes[1].set_ylabel('Attention Weight', fontsize=14, fontweight='bold')
    axes[1].set_title('Class 1 (High Affinity) - Attention Profile',
                     fontsize=16, fontweight='bold', pad=15, color='#A23B72')
    axes[1].legend(fontsize=12, loc='upper right')
    axes[1].grid(alpha=0.3, linestyle='--', linewidth=0.5)
    axes[1].set_xlim(0, protein_len)
    axes[1].set_ylim(0, max(class1_mean + class1_std) * 1.1)

 # Top-10 residueClass 1
    top10_class1 = np.argsort(-class1_mean)[:10]
    axes[1].scatter(top10_class1, class1_mean[top10_class1],
                   color='red', s=80, zorder=5, marker='o', edgecolor='black', linewidth=1.5)

    # Panel C: Differential attention (Class1 - Class0)
    diff = class1_mean - class0_mean
    colors = ['#A23B72' if d > 0 else '#2E86AB' for d in diff]

    axes[2].bar(residue_pos, diff, color=colors, alpha=0.7, edgecolor='none', width=1.0)
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=1)

    axes[2].set_xlabel('Protein Residue Position', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Differential Attention\n(Class 1 - Class 0)', fontsize=14, fontweight='bold')
    axes[2].set_title('Differential Attention Analysis (Class-Specific Preferences)',
                     fontsize=16, fontweight='bold', pad=15)
    axes[2].grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    axes[2].set_xlim(0, protein_len)

 # statisticsresidue
    if len(significant_residues) > 0:
        axes[2].scatter(significant_residues, diff[significant_residues],
                       color='gold', s=150, zorder=10, marker='*',
                       edgecolor='black', linewidth=0.8,
                       label=f'Significant (p<0.01, n={len(significant_residues)})')

 # 5residue
        top5_sig = significant_residues[np.argsort(np.abs(diff[significant_residues]))[-5:]]
        for res_idx in top5_sig:
            axes[2].annotate(f'{res_idx}',
                           xy=(res_idx, diff[res_idx]),
                           xytext=(0, 15 if diff[res_idx] > 0 else -15),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                         color='black', lw=1.5))

    axes[2].legend(fontsize=12, loc='upper right')

 # plot
    legend_text = f"Positive values (red): Class 1 prefers these residues\n" \
                 f"Negative values (blue): Class 0 prefers these residues\n" \
                 f"Gold stars: Statistically significant differences"

    axes[2].text(0.02, 0.98, legend_text,
                transform=axes[2].transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close()

    print(f"\n✅ Class differential plot saved:")
    print(f"   PNG: {save_path}")
    print(f"   PDF: {save_path.with_suffix('.pdf')}")


def save_significant_residues_report(class0_data, class1_data, significant_residues,
                                    p_values, output_path):
    """
 saveresiduereportCSV
    """
    diff = class1_data['mean'] - class0_data['mean']

    # create DataFrame
    report_data = []
    for i in range(len(class0_data['mean'])):
        report_data.append({
            'Residue_Position': i,
            'Class0_Mean': class0_data['mean'][i],
            'Class0_Std': class0_data['std'][i],
            'Class1_Mean': class1_data['mean'][i],
            'Class1_Std': class1_data['std'][i],
            'Differential': diff[i],
            'P_Value': p_values[i],
            'Significant': 'Yes' if i in significant_residues else 'No',
            'Preferred_By': 'Class1' if diff[i] > 0 else 'Class0'
        })

    df = pd.DataFrame(report_data)

 # differencepairvaluesort
    df['Abs_Differential'] = df['Differential'].abs()
    df_sorted = df.sort_values('Abs_Differential', ascending=False)

    # save
    df_sorted.to_csv(output_path, index=False, float_format='%.6f')

    print(f"\n📄 Residue report saved: {output_path}")
    print(f"   Total residues: {len(df)}")
    print(f"   Significant residues: {(df['Significant'] == 'Yes').sum()}")

 # 10residue
    print(f"\n🔝 Top 10 Most Differential Residues:")
    print(df_sorted[['Residue_Position', 'Differential', 'P_Value', 'Preferred_By']].head(10).to_string(index=False))

    return df_sorted


def main():
    parser = argparse.ArgumentParser(description='Plot class differential attention')
    parser.add_argument('--input', type=str, required=True,
                       help='Input pickle file from extraction')
    parser.add_argument('--output-dir', type=str, default='nature_mi_figures/class_differential',
                       help='Output directory')
    parser.add_argument('--alpha', type=float, default=0.01,
                       help='Significance threshold for statistical test')

    args = parser.parse_args()

    # Load data
    print("\n" + "="*80)
    print("📊 Class Differential Attention Analysis")
    print("="*80)
    print(f"Input: {args.input}")

    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded {len(results['attentions'])} samples")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute class statistics
    print("\n🔄 Computing class statistics...")
    class0_data, class1_data, protein_len = compute_class_statistics(results)

    # Statistical test
    print("\n🔬 Performing statistical tests...")
    significant_residues, p_values = statistical_test(class0_data, class1_data, args.alpha)

    # Plot
    print("\n📈 Generating plots...")
    plot_path = output_dir / 'class_differential_attention.png'
    plot_class_differential_attention(class0_data, class1_data, significant_residues,
                                     p_values, plot_path, protein_len)

    # Save report
    print("\n📝 Generating report...")
    report_path = output_dir / 'significant_residues_report.csv'
    df_report = save_significant_residues_report(class0_data, class1_data, significant_residues,
                                                 p_values, report_path)

    print("\n" + "="*80)
    print("✅ Analysis complete!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Files generated:")
    print(f"  - {plot_path.name}")
    print(f"  - {plot_path.with_suffix('.pdf').name}")
    print(f"  - {report_path.name}")


if __name__ == "__main__":
    main()
