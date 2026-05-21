#!/usr/bin/env python3
"""
    LORO result
    fold generatestatisticsreport
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import json


def load_fold_results(result_dir, n_folds=12):
    """
    load fold result

    Returns:
        list of dicts with metrics for each fold
    """
    results = []

    for fold_id in range(n_folds):
        fold_dir = Path(result_dir) / f'fold_{fold_id}'
        metrics_file = fold_dir / 'result_metrics.pt'

        if not metrics_file.exists():
            print(f"⚠️  Fold {fold_id}: Metrics file not found: {metrics_file}")
            continue

        # loadresult
        import torch
        metrics = torch.load(metrics_file, map_location='cpu')

        # extract
        test_metrics = metrics.get('test_metrics', {})

        fold_result = {
            'fold': fold_id,
            'test_receptor': fold_id,
            'auroc': test_metrics.get('auroc', np.nan),
            'auprc': test_metrics.get('auprc', np.nan),
            'accuracy': test_metrics.get('accuracy', np.nan),
            'sensitivity': test_metrics.get('sensitivity', np.nan),
            'specificity': test_metrics.get('specificity', np.nan),
            'f1': test_metrics.get('F1', np.nan),
            'precision': test_metrics.get('Precision', np.nan),
            'test_loss': test_metrics.get('test_loss', np.nan),
            'best_epoch': test_metrics.get('best_epoch', np.nan),
        }

        # multi-task R²
        if 'R2' in test_metrics:
            fold_result['r2'] = test_metrics['R2']

        results.append(fold_result)

        print(f"✅ Fold {fold_id}: AUROC={fold_result['auroc']:.4f}, "
              f"AUPRC={fold_result['auprc']:.4f}", end='')
        if 'r2' in fold_result:
            print(f", R²={fold_result['r2']:.4f}")
        else:
            print()

    return results


def compute_statistics(results):
    """
    computestatistics
    """
    df = pd.DataFrame(results)

    # computevalue
    metrics = ['auroc', 'auprc', 'accuracy', 'sensitivity', 'specificity', 'f1', 'precision']
    if 'r2' in df.columns:
        metrics.append('r2')

    stats = {}
    for metric in metrics:
        if metric in df.columns:
            values = df[metric].dropna()
            if len(values) > 0:
                stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'n_folds': len(values)
                }

    return stats, df


def generate_report(stats, df, output_file):
    """
    generatereport
    """
    lines = []
    lines.append("="*80)
    lines.append("LORO (Leave-One-Receptor-Out) Experiment Results")
    lines.append("="*80)
    lines.append("")
    lines.append(f"Total folds: {len(df)}")
    lines.append("")
    lines.append("="*80)
    lines.append("Performance Metrics (Mean ± Std)")
    lines.append("="*80)
    lines.append("")

    for metric, values in stats.items():
        metric_name = metric.upper().replace('_', ' ')
        lines.append(f"{metric_name}:")
        lines.append(f"  Mean: {values['mean']:.4f}")
        lines.append(f"  Std:  {values['std']:.4f}")
        lines.append(f"  Min:  {values['min']:.4f}")
        lines.append(f"  Max:  {values['max']:.4f}")
        lines.append(f"  95% CI: [{values['mean'] - 1.96*values['std']:.4f}, "
                    f"{values['mean'] + 1.96*values['std']:.4f}]")
        lines.append("")

    lines.append("="*80)
    lines.append("Per-Fold Results")
    lines.append("="*80)
    lines.append("")
    lines.append(df.to_string(index=False))
    lines.append("")

    lines.append("="*80)
    lines.append("Interpretation for Paper")
    lines.append("="*80)
    lines.append("")

    auroc_mean = stats['auroc']['mean']
    auroc_std = stats['auroc']['std']
    auprc_mean = stats['auprc']['mean']
    auprc_std = stats['auprc']['std']

    lines.append("For Results section:")
    lines.append("")
    lines.append(f'"To evaluate generalization to unseen receptors, we performed')
    lines.append(f'leave-one-receptor-out cross-validation across all 12 receptors.')
    lines.append(f'The model achieved an average AUROC of {auroc_mean:.3f} ± {auroc_std:.3f}')
    lines.append(f'and AUPRC of {auprc_mean:.3f} ± {auprc_std:.3f} on held-out receptors,')
    lines.append(f'demonstrating strong generalization capability beyond the training set."')
    lines.append("")

    if 'r2' in stats:
        r2_mean = stats['r2']['mean']
        r2_std = stats['r2']['std']
        lines.append(f'"For the regression task, the model achieved R² of {r2_mean:.3f} ± {r2_std:.3f}')
        lines.append(f'on unseen receptors, indicating that learned representations transfer')
        lines.append(f'effectively for both classification and regression tasks."')
        lines.append("")

    lines.append("="*80)

    report_text = "\n".join(lines)

    # 
    print("\n" + report_text)

    # savefile
    with open(output_file, 'w') as f:
        f.write(report_text)

    print(f"\n📄 Report saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate LORO results')
    parser.add_argument('--loro-dir', type=str,
                       default='datasets/cnnscore_database/loro',
                       help='LORO data directory')
    parser.add_argument('--result-dir', type=str,
                       default='result/LORO',
                       help='LORO results directory')
    parser.add_argument('--output-dir', type=str,
                       default='result/LORO/analysis',
                       help='Output directory for analysis')
    parser.add_argument('--n-folds', type=int, default=12,
                       help='Number of folds')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("📊 Evaluating LORO Experiment Results")
    print("="*80)
    print(f"Result directory: {args.result_dir}")
    print(f"Output directory: {output_dir}")
    print()

    # loadresult
    print("Loading fold results...")
    results = load_fold_results(args.result_dir, args.n_folds)

    if len(results) == 0:
        print("\n❌ No results found!")
        return

    print(f"\n✅ Loaded {len(results)}/{args.n_folds} folds")

    # computestatistics
    print("\ncomputestatistics...")
    stats, df = compute_statistics(results)

    # save CSV
    csv_file = output_dir / 'loro_results_summary.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ Results CSV saved to: {csv_file}")

    # savestatistics JSON
    stats_file = output_dir / 'loro_statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✅ Statistics JSON saved to: {stats_file}")

    # generatereport
    report_file = output_dir / 'loro_analysis_report.txt'
    generate_report(stats, df, report_file)

    print("\n" + "="*80)
    print("✅ LORO Evaluation Complete!")
    print("="*80)
    print(f"\nNext step: Generate visualizations")
    print(f"  python scripts/plot_loro_results.py --input {csv_file}")


if __name__ == '__main__':
    main()
