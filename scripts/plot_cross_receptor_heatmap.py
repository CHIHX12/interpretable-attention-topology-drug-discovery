#!/usr/bin/env python3
"""
receptorplot
12x12 matrixreceptormodelreceptor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def plot_heatmap(matrix_df, metric_name, output_file, vmin=None, vmax=None):
    """
    plot

    Args:
        matrix_df: Performance matrix (DataFrame)
        metric_name: Metric name (AUROC, AUPRC, R²)
        output_file: Output file path
        vmin, vmax: Value range for colormap
    """
    plt.figure(figsize=(14, 12))

    # computepairpairvalue
    diagonal = np.diag(matrix_df.values)
    off_diagonal = matrix_df.values[~np.eye(len(matrix_df), dtype=bool)]

    diagonal_mean = np.nanmean(diagonal)
    off_diagonal_mean = np.nanmean(off_diagonal)

    # plot
    mask = np.isnan(matrix_df.values)

    ax = sns.heatmap(
        matrix_df,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        center=(vmin + vmax) / 2 if vmin and vmax else None,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor='gray',
        cbar_kws={'label': metric_name},
        square=True,
        mask=mask
    )

    # label
    ax.set_xlabel('Test Receptor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Receptor', fontsize=14, fontweight='bold')
    ax.set_title(f'Cross-Receptor Generalization: {metric_name}\n' +
                f'Diagonal (same): {diagonal_mean:.3f} | Off-diagonal (cross): {off_diagonal_mean:.3f}',
                fontsize=16, fontweight='bold', pad=20)

                # pair
    for i in range(len(matrix_df)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='blue', lw=3))

        # label
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Heatmap saved to: {output_file}")


def plot_performance_comparison(stats_df, output_file):
    """
    pair vs pairpair
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = stats_df['metric'].values
    diagonal_means = stats_df['diagonal_mean'].values
    diagonal_stds = stats_df['diagonal_std'].values
    off_diagonal_means = stats_df['off_diagonal_mean'].values
    off_diagonal_stds = stats_df['off_diagonal_std'].values

    x = np.arange(len(metrics))
    width = 0.35

    # pair
    bars1 = ax.bar(x - width/2, diagonal_means, width,
                  label='Same Receptor (Diagonal)',
                  color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(x - width/2, diagonal_means, yerr=diagonal_stds,
               fmt='none', color='black', capsize=5, linewidth=2)

                # pair
    bars2 = ax.bar(x + width/2, off_diagonal_means, width,
                  label='Cross Receptor (Off-diagonal)',
                  color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.errorbar(x + width/2, off_diagonal_means, yerr=off_diagonal_stds,
               fmt='none', color='black', capsize=5, linewidth=2)

                # valuelabel
    for i, (bars, means) in enumerate([(bars1, diagonal_means), (bars2, off_diagonal_means)]):
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Same Receptor vs Cross-Receptor Performance',
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Performance comparison saved to: {output_file}")


def plot_receptor_similarity_matrix(matrix_df, metric_name, output_file):
    """
    receptormatrix
    receptorcomputereceptor
    """
    # computebidirectionalaverage
    n = len(matrix_df)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                similarity_matrix[i, j] = 1.0 # diagonal: self-similarity
            else:
                # bidirectionalaverageitrainingj + jtrainingi
                val1 = matrix_df.iloc[i, j]
                val2 = matrix_df.iloc[j, i]
                if not np.isnan(val1) and not np.isnan(val2):
                    similarity_matrix[i, j] = (val1 + val2) / 2
                else:
                    similarity_matrix[i, j] = np.nan

    # create DataFrame
    receptor_ids = [int(name.split('_')[1]) for name in matrix_df.index]
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=[f'R{i}' for i in receptor_ids],
        columns=[f'R{i}' for i in receptor_ids]
    )

    # plot
    plt.figure(figsize=(12, 10))

    mask = np.isnan(similarity_df.values)

    ax = sns.heatmap(
        similarity_df,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': f'Similarity ({metric_name})'},
        square=True,
        mask=mask
    )

    ax.set_xlabel('Receptor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Receptor', fontsize=14, fontweight='bold')
    ax.set_title(f'Receptor Similarity Matrix (Based on {metric_name})\n' +
                'Bidirectional average of cross-receptor performance',
                fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Similarity matrix saved to: {output_file}")

    return similarity_df


def plot_transferability_ranking(matrix_df, metric_name, output_file):
    """
    receptor
    receptormodelreceptor
    """
    # computetrainingreceptorreceptoraverage
    transferability = []

    for i, train_receptor in enumerate(matrix_df.index):
        # rowpair
        row = matrix_df.iloc[i].values
        off_diagonal = np.concatenate([row[:i], row[i+1:]])
        mean_perf = np.nanmean(off_diagonal)

        receptor_id = int(train_receptor.split('_')[1])
        transferability.append({
            'receptor': receptor_id,
            'mean_cross_performance': mean_perf,
            'std_cross_performance': np.nanstd(off_diagonal)
        })

    df_transfer = pd.DataFrame(transferability).sort_values('mean_cross_performance', ascending=False)

    # 
    fig, ax = plt.subplots(figsize=(12, 8))

    receptors = df_transfer['receptor'].values
    means = df_transfer['mean_cross_performance'].values
    stds = df_transfer['std_cross_performance'].values

    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(receptors)))

    bars = ax.barh(receptors, means, xerr=stds, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1.5, capsize=5)

                    # valuelabel
    for bar, mean, std in zip(bars, means, stds):
        width = bar.get_width()
        ax.text(width + std + 0.01, bar.get_y() + bar.get_height()/2,
               f'{mean:.3f}±{std:.3f}',
               ha='left', va='center', fontweight='bold', fontsize=9)

    ax.set_xlabel(f'Mean {metric_name} on Other Receptors', fontsize=14, fontweight='bold')
    ax.set_ylabel('Training Receptor', fontsize=14, fontweight='bold')
    ax.set_title(f'Receptor Model Transferability Ranking\n' +
                'Which receptor models generalize best to other receptors?',
                fontsize=16, fontweight='bold')
    ax.set_yticks(receptors)
    ax.set_yticklabels([f'Receptor {r}' for r in receptors])
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Transferability ranking saved to: {output_file}")

    return df_transfer


def main():
    parser = argparse.ArgumentParser(description='Plot cross-receptor heatmaps')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing matrix CSV files')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: same as input)')

    args = parser.parse_args()

    input_dir = Path(args.input)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_dir / 'figures'

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("📊 Plotting Cross-Receptor Heatmaps")
    print("="*80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print()

    # readdata
    auroc_file = input_dir / 'cross_receptor_auroc_matrix.csv'
    auprc_file = input_dir / 'cross_receptor_auprc_matrix.csv'
    r2_file = input_dir / 'cross_receptor_r2_matrix.csv'
    stats_file = input_dir / 'cross_receptor_statistics.csv'

    if not auroc_file.exists():
        print(f"❌ AUROC matrix not found: {auroc_file}")
        return

    # readmatrix
    df_auroc = pd.read_csv(auroc_file, index_col=0)
    df_auprc = pd.read_csv(auprc_file, index_col=0)
    df_stats = pd.read_csv(stats_file)

    print("✅ Loaded matrices")
    print(f"   AUROC matrix: {df_auroc.shape}")
    print(f"   AUPRC matrix: {df_auprc.shape}")
    print()

    # 1. AUROC plot
    print("Plotting AUROC heatmap...")
    plot_heatmap(df_auroc, 'AUROC', output_dir / 'cross_receptor_auroc_heatmap.png',
                vmin=0.5, vmax=1.0)

                # 2. AUPRC plot
    print("Plotting AUPRC heatmap...")
    plot_heatmap(df_auprc, 'AUPRC', output_dir / 'cross_receptor_auprc_heatmap.png',
                vmin=0.5, vmax=1.0)

                # 3. R² plot
    if r2_file.exists():
        df_r2 = pd.read_csv(r2_file, index_col=0)
        if not df_r2.isnull().all().all():
            print("Plotting R² heatmap...")
            plot_heatmap(df_r2, 'R²', output_dir / 'cross_receptor_r2_heatmap.png',
                        vmin=-0.5, vmax=1.0)

                        # 4. pair
    print("Plotting performance comparison...")
    plot_performance_comparison(df_stats, output_dir / 'cross_receptor_performance_comparison.png')

    # 5. receptormatrix
    print("Plotting receptor similarity matrix...")
    similarity_df = plot_receptor_similarity_matrix(
        df_auroc, 'AUROC',
        output_dir / 'receptor_similarity_matrix.png'
    )
    similarity_df.to_csv(output_dir / 'receptor_similarity_matrix.csv')

    # 6. 
    print("Plotting transferability ranking...")
    transferability_df = plot_transferability_ranking(
        df_auroc, 'AUROC',
        output_dir / 'receptor_transferability_ranking.png'
    )
    transferability_df.to_csv(output_dir / 'receptor_transferability_ranking.csv')

    print()
    print("="*80)
    print("✅ All plots generated!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated figures:")
    print("  - cross_receptor_auroc_heatmap.png")
    print("  - cross_receptor_auprc_heatmap.png")
    if r2_file.exists():
        print("  - cross_receptor_r2_heatmap.png")
    print("  - cross_receptor_performance_comparison.png")
    print("  - receptor_similarity_matrix.png")
    print("  - receptor_transferability_ranking.png")
    print()


if __name__ == '__main__':
    main()
