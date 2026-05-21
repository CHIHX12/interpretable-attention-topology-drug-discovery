#!/usr/bin/env python3
"""
    GHSR train/val/test
Split GHSR data into random train/val/test sets for transfer learning


1. read GHSR_training_data.csv
2. 70% train, 10% val, 20% test
3. save datasets/GPCR_resarch/random/ directory
4. transfer learningfine-tuning DrugBAN_BiLSTM model
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Split GHSR data into train/val/test")
    parser.add_argument('--input', type=str,
                        default='datasets/GPCR_resarch/GHSR_training_data.csv',
                        help='Input CSV file')
    parser.add_argument('--output_dir', type=str,
                        default='datasets/GPCR_resarch/random',
                        help='Output directory for split files')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Training set ratio (default: 0.7)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--stratify', action='store_true',
                        help='Use stratified split based on Y labels')
    return parser.parse_args()


def split_data(df, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2,
               seed=42, stratify=False):
    """
 

    Args:
        df: Input DataFrame
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
        stratify: Whether to stratify by Y labels

    Returns:
        train_df, val_df, test_df
    """
        # 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

        # 
    np.random.seed(seed)

    # 
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    n_total = len(df_shuffled)

    if stratify and 'Y' in df.columns:
    # layeractive/active
        print(f" layer Y=0/Y=1 ")

        df_active = df_shuffled[df_shuffled['Y'] == 1]
        df_inactive = df_shuffled[df_shuffled['Y'] == 0]

        n_active = len(df_active)
        n_inactive = len(df_inactive)

        # pairactiveactiverow
        def split_by_ratio(df_sub):
            n = len(df_sub)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train = df_sub.iloc[:n_train]
            val = df_sub.iloc[n_train:n_train + n_val]
            test = df_sub.iloc[n_train + n_val:]

            return train, val, test

        train_active, val_active, test_active = split_by_ratio(df_active)
        train_inactive, val_inactive, test_inactive = split_by_ratio(df_inactive)

        # 
        train_df = pd.concat([train_active, train_inactive], ignore_index=True).sample(frac=1, random_state=seed)
        val_df = pd.concat([val_active, val_inactive], ignore_index=True).sample(frac=1, random_state=seed)
        test_df = pd.concat([test_active, test_inactive], ignore_index=True).sample(frac=1, random_state=seed)

    else:
    # 
        print(f" ")

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_df = df_shuffled.iloc[:n_train].reset_index(drop=True)
        val_df = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
        test_df = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)

    return train_df, val_df, test_df


def print_statistics(df, name):
    """statistics"""
    print(f"\n   {name}:")
    print(f" : {len(df)}")
    if 'Y' in df.columns:
        n_active = (df['Y'] == 1).sum()
        n_inactive = (df['Y'] == 0).sum()
        print(f"      active (Y=1): {n_active} ({n_active/len(df)*100:.1f}%)")
        print(f" active (Y=0): {n_inactive} ({n_inactive/len(df)*100:.1f}%)")


def main():
    args = parse_args()

    print("=" * 80)
    print("🧬 GHSR - transfer learning")
    print("   GHSR Data Splitting for Transfer Learning")
    print("=" * 80)
    print()

    # read
    print(f"📊 read: {args.input}")
    df = pd.read_csv(args.input)

    print(f" : {len(df)}")
    print(f" : {list(df.columns)}")

    if 'Y' in df.columns:
        n_active = (df['Y'] == 1).sum()
        n_inactive = (df['Y'] == 0).sum()
        print(f"   active (Y=1): {n_active} ({n_active/len(df)*100:.1f}%)")
        print(f" active (Y=0): {n_inactive} ({n_inactive/len(df)*100:.1f}%)")
    print()

    # 
    print(f"✂️ : Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}, Test={args.test_ratio:.1%}")
    print(f"   Random seed: {args.seed}")

    train_df, val_df, test_df = split_data(
        df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify=args.stratify
    )

    # statistics
    print("\n📈 result:")
    print_statistics(train_df, "Train")
    print_statistics(val_df, "Validation")
    print_statistics(test_df, "Test")
    print()

    # createoutputdirectory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save
    print(f"💾 save: {output_dir}/")

    train_file = output_dir / "train.csv"
    val_file = output_dir / "val.csv"
    test_file = output_dir / "test.csv"

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"   ✓ {train_file}")
    print(f"   ✓ {val_file}")
    print(f"   ✓ {test_file}")
    print()

    # 
    total_saved = len(train_df) + len(val_df) + len(test_df)
    print(f"✅ : {len(df)} → {total_saved} ()")
    print()

    print("=" * 80)
    print("✅ done")
    print("=" * 80)
    print()

    print("📝 transfer learningfine-tuning")
    print()
    print(" trainingmodelrowfine-tuning")
    print(f"   python main.py \\")
    print(f"       --cfg configs/DrugBAN_BiLSTM.yaml \\")
    print(f" --data GPCR_resarch \\") # random 
    print(f"       --split random \\")
    print(f"       --pretrained result/DrugBAN_BiLSTM/model_epoch_94.pth")
    print()
    print(" filetrainingmodelpath")
    print()


if __name__ == "__main__":
    main()
