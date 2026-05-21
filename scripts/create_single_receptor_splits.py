#!/usr/bin/env python3
"""
createreceptortrainingdata
receptoranalysis

pairreceptor
- train.csv: receptor 80% data
- val.csv: receptor 10% data
- test.csv: receptor 10% data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_single_receptor_splits(data_dir, output_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    """
 receptorcreatetraining//

    Args:
 data_dir: datadirectory
        output_dir: outputdirectory
 train_ratio: training (default: 0.8)
 val_ratio: validation set (default: 0.1)
 seed: 
    """
    np.random.seed(seed)

    print("="*80)
    print("Creating Single-Receptor Training Splits")
    print("="*80)

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

 # readdata
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

 # receptorcreate
    print("\n" + "="*80)
    print("Creating Single-Receptor Splits")
    print("="*80)

    split_stats = []

    for receptor_id in receptor_ids:
        print(f"\n📊 Receptor {receptor_id}:")

 # extractreceptordata
        receptor_mask = df_all['target_cluster'] == receptor_id
        df_receptor = df_all[receptor_mask].copy()

        n_samples = len(df_receptor)
        print(f"   Total samples: {n_samples:,}")

 # data
        df_receptor = df_receptor.sample(frac=1, random_state=seed).reset_index(drop=True)

 # data
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)

        df_train = df_receptor.iloc[:n_train].copy()
        df_val = df_receptor.iloc[n_train:n_train+n_val].copy()
        df_test = df_receptor.iloc[n_train+n_val:].copy()

        print(f"   Train: {len(df_train):,} ({len(df_train)/n_samples:.1%})")
        print(f"   Val:   {len(df_val):,} ({len(df_val)/n_samples:.1%})")
        print(f"   Test:  {len(df_test):,} ({len(df_test)/n_samples:.1%})")

 # class
        train_y_dist = df_train['Y'].value_counts(normalize=True)
        val_y_dist = df_val['Y'].value_counts(normalize=True)
        test_y_dist = df_test['Y'].value_counts(normalize=True)

        print(f"   Class distribution:")
        print(f"     Train: Y=0: {train_y_dist.get(0, 0):.2%}, Y=1: {train_y_dist.get(1, 0):.2%}")
        print(f"     Val:   Y=0: {val_y_dist.get(0, 0):.2%}, Y=1: {val_y_dist.get(1, 0):.2%}")
        print(f"     Test:  Y=0: {test_y_dist.get(0, 0):.2%}, Y=1: {test_y_dist.get(1, 0):.2%}")

 # savefile
        receptor_dir = output_dir / f'receptor_{receptor_id}'
        receptor_dir.mkdir(parents=True, exist_ok=True)

        df_train.to_csv(receptor_dir / 'train.csv', index=False)
        df_val.to_csv(receptor_dir / 'val.csv', index=False)
        df_test.to_csv(receptor_dir / 'test.csv', index=False)

        print(f"   ✅ Saved to: {receptor_dir}")

 # statistics
        split_stats.append({
            'receptor': receptor_id,
            'n_total': n_samples,
            'n_train': len(df_train),
            'n_val': len(df_val),
            'n_test': len(df_test),
            'train_y0_ratio': train_y_dist.get(0, 0),
            'train_y1_ratio': train_y_dist.get(1, 0)
        })

 # savestatistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    df_stats = pd.DataFrame(split_stats)
    stats_file = output_dir / 'single_receptor_split_summary.csv'
    df_stats.to_csv(stats_file, index=False)

    print("\n📊 Single-Receptor Split Summary:")
    print(df_stats.to_string(index=False))

    print(f"\n✅ Summary saved to: {stats_file}")

    # create README
    readme_file = output_dir / 'README.md'
    with open(readme_file, 'w') as f:
        f.write("# Single-Receptor Training Splits\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Total receptors: {n_receptors}\n")
        f.write(f"- Total samples: {len(df_all):,}\n")
        f.write(f"- Train ratio: {train_ratio:.1%}\n")
        f.write(f"- Val ratio: {val_ratio:.1%}\n")
        f.write(f"- Test ratio: {1-train_ratio-val_ratio:.1%}\n\n")
        f.write("## Receptor Structure\n\n")
        f.write("Each receptor has its own directory:\n")
        f.write("- `train.csv`: Training data (80% of receptor samples)\n")
        f.write("- `val.csv`: Validation data (10% of receptor samples)\n")
        f.write("- `test.csv`: Test data (10% of receptor samples)\n\n")
        f.write("## Purpose\n\n")
        f.write("Test cross-receptor generalization:\n")
        f.write("1. Train a model on Receptor A's data\n")
        f.write("2. Test on Receptor B's test set\n")
        f.write("3. Create 12x12 performance matrix\n")
        f.write("4. Identify receptor similarity/transferability\n\n")
        f.write("## Usage\n\n")
        f.write("```bash\n")
        f.write("# Train model for receptor 0\n")
        f.write("python scripts/train_single_receptor.py --receptor 0\n\n")
        f.write("# Test all receptors' models on all receptors\n")
        f.write("python scripts/cross_receptor_evaluation.py\n")
        f.write("```\n\n")
        f.write("## Directories\n\n")
        for receptor_id in receptor_ids:
            f.write(f"- `receptor_{receptor_id}/`: Receptor {receptor_id} data\n")
        f.write(f"\n- `single_receptor_split_summary.csv`: Statistics\n")

    print(f"\n📄 README saved to: {readme_file}")

    print("\n" + "="*80)
    print("✅ Single-Receptor Splits Created Successfully!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Total receptors: {n_receptors}")
    print(f"\nNext steps:")
    print(f"1. Train model for each receptor:")
    print(f"   bash run_single_receptor_training.sh")
    print(f"2. Evaluate cross-receptor performance:")
    print(f"   python scripts/cross_receptor_evaluation.py")


def main():
    parser = argparse.ArgumentParser(description='Create single-receptor splits')
    parser.add_argument('--data-dir', type=str,
                       default='datasets/cnnscore_database/random',
                       help='Directory containing original data')
    parser.add_argument('--output-dir', type=str,
                       default='datasets/cnnscore_database/single_receptor',
                       help='Output directory for single-receptor splits')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Training ratio (default: 0.8 = 80%)')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation ratio (default: 0.1 = 10%)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    create_single_receptor_splits(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
