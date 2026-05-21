#!/usr/bin/env python3
"""
    LORO result
generate Nature MI visualizationplot
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def plot_loro_boxplot(df, output_dir):
    """
    LORO plot
    AUROC, AUPRC, R² 12 folds 
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 
    metrics = ['auroc', 'auprc']
    if 'r2' in df.columns:
        metrics.append('r2')

        # plot
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16

    # createplot
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        data = df[metric].dropna()

        # plot
        bp = ax.boxplot([data], positions=[0], widths=0.6,
                        patch_artist=True,
                        boxprops=dict(facecolor=colors[idx], alpha=0.7),
                        medianprops=dict(color='black', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))

                        # fold value
        x = np.random.normal(0, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.6, s=100, color=colors[idx], edgecolors='black', linewidth=1, zorder=3)

        # averagevalue
        mean_val = data.mean()
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')

        # label
        metric_name = metric.upper().replace('_', ' ')
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_name} across 12 Folds', fontsize=16, fontweight='bold')
        ax.set_xticks([])
        ax.legend(loc='lower right', fontsize=12)
        ax.grid(True, alpha=0.3)

        # y axis
        y_min, y_max = data.min(), data.max()
        y_range = y_max - y_min
        ax.set_ylim([y_min - 0.1*y_range, y_max + 0.1*y_range])

    plt.tight_layout()
    output_file = output_dir / 'loro_performance_boxplot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Boxplot saved to: {output_file}")


def plot_per_fold_performance(df, output_dir):
    """
    fold plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 
    metrics = ['auroc', 'auprc']
    if 'r2' in df.columns:
        metrics.append('r2')

        # data
    fold_ids = df['fold'].values
    n_folds = len(fold_ids)

    # plot
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        values = df[metric].values
        mean_val = np.mean(values)
        std_val = np.std(values)

        # plot
        bars = ax.bar(fold_ids, values, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=1.5)

        # fold
        best_idx = np.argmax(values)
        worst_idx = np.argmin(values)
        bars[best_idx].set_color('#2ecc71') # 
        bars[worst_idx].set_color('#e74c3c') # 

        # averagevalue
        ax.axhline(mean_val, color='black', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f} ± {std_val:.4f}')

                    # label
        metric_name = metric.upper().replace('_', ' ')
        ax.set_xlabel('Test Receptor (Fold ID)', fontsize=14, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=14, fontweight='bold')
        ax.set_title(f'{metric_name} for Each Fold (Green=Best, Red=Worst)',
                    fontsize=16, fontweight='bold')
        ax.set_xticks(fold_ids)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # 
        ax.text(fold_ids[best_idx], values[best_idx] + 0.01,
               f'{values[best_idx]:.4f}', ha='center', va='bottom',
               fontweight='bold', fontsize=10)
        ax.text(fold_ids[worst_idx], values[worst_idx] - 0.01,
               f'{values[worst_idx]:.4f}', ha='center', va='top',
               fontweight='bold', fontsize=10)

    plt.tight_layout()
    output_file = output_dir / 'loro_per_fold_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Per-fold performance plot saved to: {output_file}")


def plot_metric_distribution(df, output_dir):
    """
    plot
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ['auroc', 'auprc']
    if 'r2' in df.columns:
        metrics.append('r2')

        # data
    plot_data = []
    for metric in metrics:
        for val in df[metric].dropna():
            plot_data.append({
                'Metric': metric.upper(),
                'Value': val
            })

    plot_df = pd.DataFrame(plot_data)

    # plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=plot_df, x='Metric', y='Value', palette='Set2')
    plt.title('Distribution of Performance Metrics across 12 Folds',
             fontsize=16, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.xlabel('Metric', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    output_file = output_dir / 'loro_metric_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Metric distribution plot saved to: {output_file}")


def plot_receptor_difficulty(df, output_dir):
    """
    receptoranalysis
    pairreceptorrowsort
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # AUROC sort
    df_sorted = df.sort_values('auroc', ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    # createplot
    y_pos = np.arange(len(df_sorted))
    auroc_vals = df_sorted['auroc'].values
    auprc_vals = df_sorted['auprc'].values

    # AUROC 
    ax.scatter(auroc_vals, y_pos, s=200, alpha=0.7, color='#3498db',
              label='AUROC', marker='o', edgecolors='black', linewidth=1.5)

                # AUPRC 
    ax.scatter(auprc_vals, y_pos, s=200, alpha=0.7, color='#e74c3c',
              label='AUPRC', marker='s', edgecolors='black', linewidth=1.5)

                # 
    for i, (auroc, auprc) in enumerate(zip(auroc_vals, auprc_vals)):
        ax.plot([auroc, auprc], [i, i], color='gray', alpha=0.3, linewidth=1)

        # label
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"Receptor {fold}" for fold in df_sorted['fold']])
    ax.set_xlabel('Score', fontsize=14, fontweight='bold')
    ax.set_ylabel('Test Receptor', fontsize=14, fontweight='bold')
    ax.set_title('Receptor Difficulty Ranking (Sorted by AUROC)',
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    output_file = output_dir / 'loro_receptor_difficulty.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Receptor difficulty plot saved to: {output_file}")


def plot_classification_vs_regression(df, output_dir):
    """
    classification vs regressionpair R²
    """
    if 'r2' not in df.columns:
        print("⚠️  No R² data, skipping classification vs regression plot")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # AUROC vs R²
    ax1 = axes[0]
    scatter1 = ax1.scatter(df['auroc'], df['r2'], s=200, alpha=0.6,
                          c=df['fold'], cmap='tab20', edgecolors='black', linewidth=1.5)

                            # pair
    from scipy.stats import pearsonr
    corr, p_val = pearsonr(df['auroc'].dropna(), df['r2'].dropna())

    ax1.set_xlabel('AUROC (Classification)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('R² (Regression)', fontsize=14, fontweight='bold')
    ax1.set_title(f'Classification vs Regression Performance\n(Pearson r={corr:.3f}, p={p_val:.3e})',
                 fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Fold ID')

    # AUPRC vs R²
    ax2 = axes[1]
    scatter2 = ax2.scatter(df['auprc'], df['r2'], s=200, alpha=0.6,
                          c=df['fold'], cmap='tab20', edgecolors='black', linewidth=1.5)

    corr2, p_val2 = pearsonr(df['auprc'].dropna(), df['r2'].dropna())

    ax2.set_xlabel('AUPRC (Classification)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R² (Regression)', fontsize=14, fontweight='bold')
    ax2.set_title(f'Classification vs Regression Performance\n(Pearson r={corr2:.3f}, p={p_val2:.3e})',
                 fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Fold ID')

    plt.tight_layout()
    output_file = output_dir / 'loro_classification_vs_regression.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Classification vs Regression plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Plot LORO results')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file (loro_results_summary.csv)')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for figures')

    args = parser.parse_args()

    # readdata
    print("="*80)
    print("📊 Plotting LORO Results")
    print("="*80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print()

    df = pd.read_csv(args.input)
    print(f"✅ Loaded {len(df)} folds")
    print()

    # createplot
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating plots...")
    print()

    # 1. plot
    plot_loro_boxplot(df, output_dir)

    # 2. fold 
    plot_per_fold_performance(df, output_dir)

    # 3. 
    plot_metric_distribution(df, output_dir)

    # 4. receptorsort
    plot_receptor_difficulty(df, output_dir)

    # 5. classification vs regression
    plot_classification_vs_regression(df, output_dir)

    print()
    print("="*80)
    print("✅ All plots generated!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated figures:")
    print("  - loro_performance_boxplot.png")
    print("  - loro_per_fold_performance.png")
    print("  - loro_metric_distribution.png")
    print("  - loro_receptor_difficulty.png")
    if 'r2' in df.columns:
        print("  - loro_classification_vs_regression.png")
    print()


if __name__ == '__main__':
    main()
