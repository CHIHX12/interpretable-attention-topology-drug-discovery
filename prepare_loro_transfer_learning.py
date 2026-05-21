#!/usr/bin/env python
"""
Prepare LORO data for Transfer Learning experiment.

Experiment Design:
1. Use LORO pretrained model (trained on 11 receptors)
2. Split test receptor data into:
   - finetune_train.csv: For fine-tuning the pretrained model
   - finetune_val.csv: For validation during fine-tuning
   - finetune_test.csv: For final evaluation
3. Fine-tune on target receptor, then test

This tests whether transfer learning can help model adapt to new receptors.

Usage:
  python prepare_loro_transfer_learning.py --fold fold_1 --finetune_ratio 0.3
"""

import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_transfer_learning_data(loro_dir, output_dir, finetune_ratio=0.3, val_ratio=0.1, seed=42):
    """
    Prepare data for transfer learning experiment.

    Args:
        loro_dir: Directory containing LORO fold data
        output_dir: Output directory for transfer learning data
        finetune_ratio: Ratio of test data to use for fine-tuning (default: 0.3 = 30%)
        val_ratio: Ratio of finetune data to use for validation (default: 0.1 = 10%)
        seed: Random seed
    """
    # Read LORO test data (the held-out receptor)
    test_path = os.path.join(loro_dir, 'test.csv')
    df_test = pd.read_csv(test_path)

    print(f"Original test data size: {len(df_test)}")
    print(f"Class distribution: Y=0: {(df_test['Y']==0).sum()}, Y=1: {(df_test['Y']==1).sum()}")

    # Split test data into finetune and final_test
    # finetune_ratio of data for fine-tuning, rest for testing
    df_finetune, df_final_test = train_test_split(
        df_test,
        train_size=finetune_ratio,
        stratify=df_test['Y'],
        random_state=seed
    )

    # Further split finetune into train and val
    df_finetune_train, df_finetune_val = train_test_split(
        df_finetune,
        test_size=val_ratio,
        stratify=df_finetune['Y'],
        random_state=seed
    )

    print(f"\nSplit results:")
    print(f"  Fine-tune train: {len(df_finetune_train)} samples ({len(df_finetune_train)/len(df_test)*100:.1f}%)")
    print(f"  Fine-tune val:   {len(df_finetune_val)} samples ({len(df_finetune_val)/len(df_test)*100:.1f}%)")
    print(f"  Final test:      {len(df_final_test)} samples ({len(df_final_test)/len(df_test)*100:.1f}%)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save files
    df_finetune_train.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    df_finetune_val.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
    df_final_test.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

    print(f"\nSaved to {output_dir}:")
    print(f"  train.csv: {len(df_finetune_train)} samples (for fine-tuning)")
    print(f"  val.csv:   {len(df_finetune_val)} samples (for validation)")
    print(f"  test.csv:  {len(df_final_test)} samples (for final evaluation)")

    # Print class distributions
    print(f"\nClass distributions:")
    print(f"  Train - Y=0: {(df_finetune_train['Y']==0).sum()}, Y=1: {(df_finetune_train['Y']==1).sum()}")
    print(f"  Val   - Y=0: {(df_finetune_val['Y']==0).sum()}, Y=1: {(df_finetune_val['Y']==1).sum()}")
    print(f"  Test  - Y=0: {(df_final_test['Y']==0).sum()}, Y=1: {(df_final_test['Y']==1).sum()}")

    return df_finetune_train, df_finetune_val, df_final_test


def main():
    parser = argparse.ArgumentParser(description="Prepare LORO data for transfer learning")
    parser.add_argument('--fold', type=str, required=True, help="LORO fold (e.g., fold_1)")
    parser.add_argument('--data_dir', type=str, default='./datasets/cnnscore_database/loro',
                        help="Base directory for LORO data")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Output directory (default: loro_transfer/<fold>)")
    parser.add_argument('--finetune_ratio', type=float, default=0.3,
                        help="Ratio of test data to use for fine-tuning (default: 0.3)")
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help="Ratio of finetune data to use for validation (default: 0.1)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    loro_dir = os.path.join(args.data_dir, args.fold)

    if args.output_dir is None:
        output_dir = f'./datasets/cnnscore_database/loro_transfer/{args.fold}'
    else:
        output_dir = args.output_dir

    print("=" * 60)
    print("LORO Transfer Learning Data Preparation")
    print("=" * 60)
    print(f"Source: {loro_dir}")
    print(f"Output: {output_dir}")
    print(f"Fine-tune ratio: {args.finetune_ratio} ({args.finetune_ratio*100}%)")
    print(f"Validation ratio: {args.val_ratio} ({args.val_ratio*100}%)")
    print(f"Random seed: {args.seed}")
    print("=" * 60)

    prepare_transfer_learning_data(
        loro_dir=loro_dir,
        output_dir=output_dir,
        finetune_ratio=args.finetune_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. Run transfer learning with pretrained model:")
    print(f"   CUDA_VISIBLE_DEVICES=0 python main.py \\")
    print(f"       --cfg configs/DrugBAN_BiLSTM_CNNScore_Multitask_Transfer.yaml \\")
    print(f"       --data cnnscore_database/loro_transfer \\")
    print(f"       --split {args.fold}")
    print("=" * 60)


if __name__ == '__main__':
    main()
