#!/usr/bin/env python3
"""
statistics - Nature Machine Intelligence
significancecompute
"""

import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu, ttest_rel, ttest_ind
import pickle


def bootstrap_ci(data, n_bootstrap=1000, ci=95):
    """
    Bootstrap 

    Args:
        data: 1D array of values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval percentage

    Returns:
        (mean, lower, upper)
    """
    bootstrap_means = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    mean = np.mean(data)
    lower = np.percentile(bootstrap_means, (100 - ci) / 2)
    upper = np.percentile(bootstrap_means, 100 - (100 - ci) / 2)

    return mean, lower, upper


def cohens_d(group1, group2):
    """
    compute Cohen's d ()

    Args:
        group1, group2: Arrays of values

    Returns:
        d: Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return d


def paired_comparison_test(your_scores, baseline_scores, method_name, baseline_name):
    """
    pairsampleWilcoxon signed-rank test

    comparisontest setmodel

    Returns:
        dict with test results
    """
    assert len(your_scores) == len(baseline_scores), "Scores must have same length"

    # Wilcoxon signed-rank test (non-parametric)
    statistic_w, p_value_w = wilcoxon(your_scores, baseline_scores)

    # Paired t-test (parametric, for reference)
    statistic_t, p_value_t = ttest_rel(your_scores, baseline_scores)

    # Effect size
    diff = np.array(your_scores) - np.array(baseline_scores)
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # 95% CI for mean difference
    n = len(diff)
    se = std_diff / np.sqrt(n)
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se

    # Improvement percentage
    improvement = (np.mean(your_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100

    results = {
        'method': method_name,
        'baseline': baseline_name,
        'n_samples': n,
        'your_mean': np.mean(your_scores),
        'baseline_mean': np.mean(baseline_scores),
        'mean_difference': mean_diff,
        'improvement_pct': improvement,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'wilcoxon_statistic': statistic_w,
        'wilcoxon_p_value': p_value_w,
        'paired_t_statistic': statistic_t,
        'paired_t_p_value': p_value_t,
        'significant': 'Yes' if p_value_w < 0.05 else 'No'
    }

    return results


def independent_comparison_test(your_scores, baseline_scores, method_name, baseline_name):
    """
    sampleMann-Whitney U test

    comparisontest setmodel

    Returns:
        dict with test results
    """
    # Mann-Whitney U test (non-parametric)
    statistic_u, p_value_u = mannwhitneyu(your_scores, baseline_scores, alternative='two-sided')

    # Independent t-test (parametric, for reference)
    statistic_t, p_value_t = ttest_ind(your_scores, baseline_scores)

    # Effect size (Cohen's d)
    d = cohens_d(your_scores, baseline_scores)

    # Improvement percentage
    improvement = (np.mean(your_scores) - np.mean(baseline_scores)) / np.mean(baseline_scores) * 100

    results = {
        'method': method_name,
        'baseline': baseline_name,
        'n_your': len(your_scores),
        'n_baseline': len(baseline_scores),
        'your_mean': np.mean(your_scores),
        'your_std': np.std(your_scores, ddof=1),
        'baseline_mean': np.mean(baseline_scores),
        'baseline_std': np.std(baseline_scores, ddof=1),
        'improvement_pct': improvement,
        'cohens_d': d,
        'mann_whitney_statistic': statistic_u,
        'mann_whitney_p_value': p_value_u,
        'independent_t_statistic': statistic_t,
        'independent_t_p_value': p_value_t,
        'significant': 'Yes' if p_value_u < 0.05 else 'No'
    }

    return results


def compute_confidence_intervals(results_dict, metrics=['auroc', 'auprc', 'r2']):
    """
    computerow

    Args:
        results_dict: {metric_name: [run1, run2, run3, ...]}
        metrics: List of metric names to compute CI for

    Returns:
        dict with CI results
    """
    ci_results = {}

    for metric in metrics:
        if metric not in results_dict:
            continue

        values = np.array(results_dict[metric])

        # Bootstrap 95% CI
        mean, ci_lower, ci_upper = bootstrap_ci(values, n_bootstrap=10000, ci=95)

        ci_results[metric] = {
            'mean': mean,
            'std': np.std(values, ddof=1),
            'ci_95_lower': ci_lower,
            'ci_95_upper': ci_upper,
            'n_runs': len(values)
        }

    return ci_results


def print_statistical_report(comparison_results, ci_results, output_path):
    """
    savestatisticsreport
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("STATISTICAL TESTING REPORT FOR NATURE MACHINE INTELLIGENCE")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Section 1: Confidence Intervals
    report_lines.append("1. CONFIDENCE INTERVALS (95% Bootstrap)")
    report_lines.append("-" * 80)

    for metric, ci_data in ci_results.items():
        report_lines.append(f"\n{metric.upper()}:")
        report_lines.append(f"  Mean: {ci_data['mean']:.4f}")
        report_lines.append(f"  Std:  {ci_data['std']:.4f}")
        report_lines.append(f"  95% CI: [{ci_data['ci_95_lower']:.4f}, {ci_data['ci_95_upper']:.4f}]")
        report_lines.append(f"  n_runs: {ci_data['n_runs']}")

    # Section 2: Pairwise Comparisons
    report_lines.append("\n" + "=" * 80)
    report_lines.append("2. PAIRWISE STATISTICAL COMPARISONS")
    report_lines.append("-" * 80)

    for comp_name, comp_data in comparison_results.items():
        report_lines.append(f"\n{comp_name}:")
        report_lines.append(f"  Method: {comp_data['method']}")
        report_lines.append(f"  Baseline: {comp_data['baseline']}")

        if 'n_samples' in comp_data:
            # Paired test
            report_lines.append(f"  Test type: Paired (Wilcoxon signed-rank)")
            report_lines.append(f"  n_samples: {comp_data['n_samples']}")
            report_lines.append(f"  Your mean: {comp_data['your_mean']:.4f}")
            report_lines.append(f"  Baseline mean: {comp_data['baseline_mean']:.4f}")
            report_lines.append(f"  Mean difference: {comp_data['mean_difference']:.4f}")
            report_lines.append(f"  95% CI: [{comp_data['ci_lower']:.4f}, {comp_data['ci_upper']:.4f}]")
            report_lines.append(f"  Improvement: {comp_data['improvement_pct']:+.2f}%")
            report_lines.append(f"  Wilcoxon p-value: {comp_data['wilcoxon_p_value']:.4e}")
            report_lines.append(f"  Paired t-test p-value: {comp_data['paired_t_p_value']:.4e}")
        else:
            # Independent test
            report_lines.append(f"  Test type: Independent (Mann-Whitney U)")
            report_lines.append(f"  Your: n={comp_data['n_your']}, mean={comp_data['your_mean']:.4f}, std={comp_data['your_std']:.4f}")
            report_lines.append(f"  Baseline: n={comp_data['n_baseline']}, mean={comp_data['baseline_mean']:.4f}, std={comp_data['baseline_std']:.4f}")
            report_lines.append(f"  Improvement: {comp_data['improvement_pct']:+.2f}%")
            report_lines.append(f"  Cohen's d: {comp_data['cohens_d']:.4f}")
            report_lines.append(f"  Mann-Whitney p-value: {comp_data['mann_whitney_p_value']:.4e}")
            report_lines.append(f"  Independent t-test p-value: {comp_data['independent_t_p_value']:.4e}")

        report_lines.append(f"  Significant: {comp_data['significant']}")

        # Interpretation
        if comp_data['significant'] == 'Yes':
            if 'improvement_pct' in comp_data and comp_data['improvement_pct'] > 0:
                report_lines.append(f"  ✓ Your method is SIGNIFICANTLY BETTER (p < 0.05)")
            else:
                report_lines.append(f"  ✗ Your method is significantly worse (p < 0.05)")
        else:
            report_lines.append(f"  ~ No significant difference (p >= 0.05)")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("3. EFFECT SIZE INTERPRETATION")
    report_lines.append("-" * 80)
    report_lines.append("Cohen's d interpretation:")
    report_lines.append("  |d| < 0.2  : Negligible")
    report_lines.append("  0.2 ≤ |d| < 0.5 : Small")
    report_lines.append("  0.5 ≤ |d| < 0.8 : Medium")
    report_lines.append("  |d| ≥ 0.8  : Large")
    report_lines.append("")

    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(report_text)

    print(f"\n💾 Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Statistical tests for Nature MI submission')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results CSV or pickle file')
    parser.add_argument('--output', type=str, default='nature_mi_figures/statistics.txt',
                       help='Output report file')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("📊 Statistical Testing for Nature Machine Intelligence")
    print("="*80)

    # Example: Load your results
    # actualdata
    results_file = Path(args.results)

    if not results_file.exists():
        print(f"⚠️  Results file not found: {results_file}")
        print("\n📌 Creating example statistical tests...")
        print("\nTo use this script with your data:")
        print("1. Prepare a CSV or pickle file with results from multiple runs")
        print("2. Include metrics: AUROC, AUPRC, R², etc.")
        print("3. Run: python scripts/statistical_tests.py --results your_results.csv")
        return

        # Example data (actualdata)
        # rowresult
        your_auroc_runs = [0.9185, 0.9201, 0.9167, 0.9189, 0.9195] # 5row
    your_auprc_runs = [0.9708, 0.9715, 0.9701, 0.9710, 0.9705]
    your_r2_runs = [0.6761, 0.6785, 0.6743, 0.6769, 0.6772]

    # Baseline result
    baseline_auroc = [0.8850, 0.8821, 0.8865, 0.8843, 0.8857]
    baseline_auprc = [0.9201, 0.9189, 0.9215, 0.9198, 0.9207]

    # compute
    print("\n🔄 Computing confidence intervals...")
    ci_results = compute_confidence_intervals({
        'auroc': your_auroc_runs,
        'auprc': your_auprc_runs,
        'r2': your_r2_runs
    })

    # paircomparisontest set
    print("\n🔄 Performing pairwise comparisons...")
    comparison_results = {}

    # AUROC comparison
    comp_auroc = paired_comparison_test(
        your_auroc_runs,
        baseline_auroc,
        'Your Multi-task Model',
        'Baseline (Single-task)'
    )
    comparison_results['AUROC_comparison'] = comp_auroc

    # AUPRC comparison
    comp_auprc = paired_comparison_test(
        your_auprc_runs,
        baseline_auprc,
        'Your Multi-task Model',
        'Baseline (Single-task)'
    )
    comparison_results['AUPRC_comparison'] = comp_auprc

    # generatereport
    print("\n📝 Generating statistical report...")
    print_statistical_report(comparison_results, ci_results, args.output)

    print("\n" + "="*80)
    print("✅ Statistical testing complete!")
    print("="*80)
    print(f"\n📌 Report saved to: {args.output}")
    print("\n💡 Key findings for your paper:")
    for metric, data in ci_results.items():
        print(f"   {metric.upper()}: {data['mean']:.4f} (95% CI: [{data['ci_95_lower']:.4f}, {data['ci_95_upper']:.4f}])")


if __name__ == "__main__":
    main()
