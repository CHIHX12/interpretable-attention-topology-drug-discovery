#!/usr/bin/env python3
"""
aggregate DrugBAN result
 12 model
"""

import os
import re
import json
import pandas as pd
from pathlib import Path

# 12 config
EXPERIMENTS = {
    # Random Split (6)
    'random': [
        {
            'name': 'DrugBAN',
            'drug': 'GCN',
            'protein': 'CNN',
            'result_dir': 'result/DrugBAN',
            'label': 'GCN + CNN'
        },
        {
            'name': 'DrugBAN_BiLSTM',
            'drug': 'GCN',
            'protein': 'BiLSTM(4 feat)',
            'result_dir': 'result/DrugBAN_BiLSTM',
            'label': 'GCN + BiLSTM(4f)'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM',
            'drug': 'BiLSTM(no feat)',
            'protein': 'CNN',
            'result_dir': 'result/DrugBAN_DrugBiLSTM',
            'label': 'BiLSTM + CNN'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM_Features',
            'drug': 'BiLSTM(8 feat)',
            'protein': 'CNN',
            'result_dir': 'result/DrugBAN_DrugBiLSTM_Features',
            'label': 'BiLSTM(8f) + CNN'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM_BothFeatures',
            'drug': 'BiLSTM(8 feat)',
            'protein': 'BiLSTM(4 feat)',
            'result_dir': 'result/DrugBAN_DrugBiLSTM_BothFeatures',
            'label': 'BiLSTM(8f) + BiLSTM(4f)'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures',
            'drug': 'BiLSTM(no feat)',
            'protein': 'BiLSTM(4 feat)',
            'result_dir': 'result/DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures',
            'label': 'BiLSTM + BiLSTM(4f)'
        },
    ],
    # DA (Cluster Split) (6)
    'da': [
        {
            'name': 'DrugBAN_DA',
            'drug': 'GCN',
            'protein': 'CNN',
            'result_dir': 'result/DrugBAN_DA',
            'label': 'GCN + CNN'
        },
        {
            'name': 'DrugBAN_BiLSTM_DA',
            'drug': 'GCN',
            'protein': 'BiLSTM(4 feat)',
            'result_dir': 'result/DrugBAN_BiLSTM_DA',
            'label': 'GCN + BiLSTM(4f)'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM_DA',
            'drug': 'BiLSTM(no feat)',
            'protein': 'CNN',
            'result_dir': 'result/DrugBAN_DrugBiLSTM_DA',
            'label': 'BiLSTM + CNN'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM_Features_DA',
            'drug': 'BiLSTM(8 feat)',
            'protein': 'CNN',
            'result_dir': 'result/DrugBAN_DrugBiLSTM_Features_DA',
            'label': 'BiLSTM(8f) + CNN'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM_BothFeatures_DA',
            'drug': 'BiLSTM(8 feat)',
            'protein': 'BiLSTM(4 feat)',
            'result_dir': 'result/DrugBAN_DrugBiLSTM_BothFeatures_DA',
            'label': 'BiLSTM(8f) + BiLSTM(4f)'
        },
        {
            'name': 'DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures_DA',
            'drug': 'BiLSTM(no feat)',
            'protein': 'BiLSTM(4 feat)',
            'result_dir': 'result/DrugBAN_DrugBiLSTM_ProteinBiLSTM_ProteinFeatures_DA',
            'label': 'BiLSTM + BiLSTM(4f)'
        },
    ]
}


def parse_test_results(result_file):
    """ test_markdowntable.txt result"""
    try:
        with open(result_file, 'r') as f:
            content = f.read()

        # extractvalueuse
        # format: |   epoch XX   | AUROC  | AUPRC  |   F1   | ...
        match = re.search(r'\|\s+epoch\s+(\d+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|\s+([\d.]+)\s+\|', content)

        if match:
            return {
                'epoch': int(match.group(1)),
                'auroc': float(match.group(2)),
                'auprc': float(match.group(3)),
                'f1': float(match.group(4)),
                'sensitivity': float(match.group(5)),
                'specificity': float(match.group(6)),
                'accuracy': float(match.group(7))
            }
        else:
            print(f"Warning: Could not parse {result_file}")
            return None
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
        return None


def aggregate_all_results():
    """aggregateresult"""
    all_results = []

    for split_type, experiments in EXPERIMENTS.items():
        for exp in experiments:
            result_file = os.path.join(exp['result_dir'], 'test_markdowntable.txt')

            if not os.path.exists(result_file):
                print(f"⚠️  Missing: {result_file}")
                continue

            metrics = parse_test_results(result_file)
            if metrics:
                result = {
                    'name': exp['name'],
                    'label': exp['label'],
                    'drug_encoder': exp['drug'],
                    'protein_encoder': exp['protein'],
                    'split': 'Random' if split_type == 'random' else 'DA (Cluster)',
                    **metrics
                }
                all_results.append(result)
                print(f"✅ {exp['name']}: AUROC {metrics['auroc']:.4f}")

    # convert DataFrame
    df = pd.DataFrame(all_results)

    # save CSV
    output_file = 'all_results_summary.csv'
    df.to_csv(output_file, index=False)
    print(f"\n📊 Results saved to: {output_file}")

    # save JSON
    json_file = 'all_results_summary.json'
    with open(json_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"📊 Results saved to: {json_file}")

    # statistics
    print("\n" + "="*80)
    print("📈 Performance Summary")
    print("="*80)

    for split in ['Random', 'DA (Cluster)']:
        split_df = df[df['split'] == split]
        print(f"\n{split} Split:")
        print(f"  Mean AUROC: {split_df['auroc'].mean():.4f} ± {split_df['auroc'].std():.4f}")
        print(f"  Best AUROC: {split_df['auroc'].max():.4f} ({split_df.loc[split_df['auroc'].idxmax(), 'label']})")
        print(f"  Worst AUROC: {split_df['auroc'].min():.4f} ({split_df.loc[split_df['auroc'].idxmin(), 'label']})")

    return df


if __name__ == '__main__':
    print("🔍 Aggregating DrugBAN experiment results...")
    print("="*80)
    df = aggregate_all_results()
    print("\n✅ Done!")
