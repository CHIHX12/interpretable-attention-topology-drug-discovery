#!/usr/bin/env python3
"""
create Leave-One-Receptor-Out (LORO) data
modelreceptor

input random split datatrain.csv, val.csv, test.csv
output12 fold fold 
 - train.csv: 11 receptordata
 - val.csv: 11 receptorvalidation set
 - test.csv: 12 receptordata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_loro_splits(data_dir, output_dir, val_ratio=0.1, seed=42):
    """
 create LORO data

    Args:
 data_dir: datadirectory
        output_dir: outputdirectory
 val_ratio: trainingreceptorvalidation set
 seed: 
    """
    np.random.seed(seed)

 # readdata
    print("="*80)
    print("Creating Leave-One-Receptor-Out (LORO) Data Splits")
    print("="*80)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

 # train, val, test
    print("\n📂 Loading data...")
    all_data = []
    for split in ['train.csv', 'val.csv', 'test.csv']:
        filepath = data_dir / split
        if filepath.exists():
            df = pd.read_csv(filepath)
            print(f"   {split}: {len(df)} samples")
            all_data.append(df)
        else:
            print(f"   ⚠️  {split} not found, skipping")

    df_all = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total samples: {len(df_all)}")

 # column
    required_cols = ['SMILES', 'Protein', 'Y', 'Z', 'target_cluster']
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"Missing required column: {col}")

 # receptor ID
    receptor_ids = sorted(df_all['target_cluster'].unique())
    n_receptors = len(receptor_ids)

    print(f"\n🧬 Found {n_receptors} receptors: {receptor_ids}")
    print("\nSamples per receptor:")
    receptor_counts = df_all['target_cluster'].value_counts().sort_index()
    for receptor_id, count in receptor_counts.items():
        print(f"   Receptor {receptor_id}: {count:,} samples")

 # receptorcreate fold
    print("\n" + "="*80)
    print("Creating LORO Folds")
    print("="*80)

    fold_stats = []

    for test_receptor in receptor_ids:
        print(f"\n📊 Fold {test_receptor}: Testing on Receptor {test_receptor}")

 # test setheld-out receptor
        test_mask = df_all['target_cluster'] == test_receptor
        df_test = df_all[test_mask].copy()

 # training 11 receptor
        train_mask = ~test_mask
        df_train_all = df_all[train_mask].copy()

 # trainingvalidation setreceptor
        val_indices = []
        for receptor in receptor_ids:
            if receptor == test_receptor:
                continue

            receptor_mask = df_train_all['target_cluster'] == receptor
            receptor_indices = df_train_all[receptor_mask].index.tolist()

 # receptor val_ratio 
            n_val = int(len(receptor_indices) * val_ratio)
            val_idx = np.random.choice(receptor_indices, size=n_val, replace=False)
            val_indices.extend(val_idx)

 # trainingvalidation set
        df_val = df_train_all.loc[val_indices].copy()
        df_train = df_train_all.drop(val_indices).copy()

 # statistics
        n_train_receptors = df_train['target_cluster'].nunique()
        n_val_receptors = df_val['target_cluster'].nunique()

        print(f"   Train: {len(df_train):,} samples from {n_train_receptors} receptors")
        print(f"   Val:   {len(df_val):,} samples from {n_val_receptors} receptors")
        print(f"   Test:  {len(df_test):,} samples from Receptor {test_receptor}")

 # class
        train_y_dist = df_train['Y'].value_counts(normalize=True)
        val_y_dist = df_val['Y'].value_counts(normalize=True)
        test_y_dist = df_test['Y'].value_counts(normalize=True)

        print(f"   Class distribution:")
        print(f"     Train: Y=0: {train_y_dist.get(0, 0):.2%}, Y=1: {train_y_dist.get(1, 0):.2%}")
        print(f"     Val:   Y=0: {val_y_dist.get(0, 0):.2%}, Y=1: {val_y_dist.get(1, 0):.2%}")
        print(f"     Test:  Y=0: {test_y_dist.get(0, 0):.2%}, Y=1: {test_y_dist.get(1, 0):.2%}")

 # savefile
        fold_dir = output_dir / f'fold_{test_receptor}'
        fold_dir.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(fold_dir / 'train.csv', index=False)
        df_val.to_csv(fold_dir / 'val.csv', index=False)
        df_test.to_csv(fold_dir / 'test.csv', index=False)

        print(f"   ✅ Saved to: {fold_dir}")

 # statistics
        fold_stats.append({
            'fold': test_receptor,
            'test_receptor': test_receptor,
            'n_train': len(df_train),
            'n_val': len(df_val),
            'n_test': len(df_test),
            'train_receptors': n_train_receptors,
            'test_y0_ratio': test_y_dist.get(0, 0),
            'test_y1_ratio': test_y_dist.get(1, 0)
        })

 # savestatistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    df_stats = pd.DataFrame(fold_stats)
    stats_file = output_dir / 'loro_split_summary.csv'
    df_stats.to_csv(stats_file, index=False)

    print("\n📊 LORO Split Summary:")
    print(df_stats.to_string(index=False))

    print(f"\n✅ Summary saved to: {stats_file}")

    # create README
    readme_file = output_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write("# Leave-One-Receptor-Out (LORO) Data Splits\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Total receptors: {n_receptors}\n")
        f.write(f"- Total samples: {len(df_all):,}\n")
        f.write(f"- Validation ratio: {val_ratio:.1%}\n\n")
        f.write("## Fold Structure\n\n")
        f.write("Each fold contains:\n")
        f.write(f"- `train.csv`: ~{df_stats['n_train'].mean():,.0f} samples from 11 receptors\n")
        f.write(f"- `val.csv`: ~{df_stats['n_val'].mean():,.0f} samples from 11 receptors\n")
        f.write(f"- `test.csv`: ~{df_stats['n_test'].mean():,.0f} samples from 1 held-out receptor\n\n")
        f.write("## Purpose\n\n")
        f.write("Test model generalization to unseen receptors (Cold-Start scenario).\n\n")
        f.write("## Training Protocol\n\n")
        f.write("1. Train model on `train.csv` (11 receptors)\n")
        f.write("2. Validate on `val.csv` (same 11 receptors)\n")
        f.write("3. Test on `test.csv` (held-out receptor)\n")
        f.write("4. Repeat for all 12 folds\n")
        f.write("5. Report average performance ± std\n\n")
        f.write("## Files\n\n")
        for fold_id in receptor_ids:
            f.write(f"- `fold_{fold_id}/`: Testing on Receptor {fold_id}\n")
        f.write(f"\n- `loro_split_summary.csv`: Statistics for all folds\n")

    print(f"\n📄 README saved to: {readme_file}")

    print("\n" + "="*80)
    print("✅ LORO Data Splits Created Successfully!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total folds: {n_receptors}")
    print(f"\nNext steps:")
    print(f"1. Train model for each fold:")
    print(f"   python scripts/train_loro_fold.py --fold 0")
    print(f"2. Evaluate all folds:")
    print(f"   python scripts/evaluate_loro_results.py")


def main():
    parser = argparse.ArgumentParser(description='Create LORO data splits')
    parser.add_argument('--data-dir', type=str,
                       default='datasets/cnnscore_database/random',
                       help='Directory containing original data')
    parser.add_argument('--output-dir', type=str,
                       default='datasets/cnnscore_database/loro',
                       help='Output directory for LORO splits')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation ratio (default: 0.1 = 10%%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    create_loro_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
