#!/usr/bin/env python3
"""
training LORO fold


    python scripts/train_loro_fold.py --fold 0 --config configs/DrugBAN_BiLSTM_CNNScore_Multitask.yaml
"""

import torch
import argparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

# main.py 
# datapath
import os


def train_loro_fold(fold_id, config_file, loro_dir, output_base_dir, epochs=50):
    """
 training LORO fold

    Args:
        fold_id: Fold ID (0-11)
 config_file: filepath
        loro_dir: LORO datadirectory
 output_base_dir: outputdirectory
 epochs: training epoch 
    """
    fold_dir = Path(loro_dir) / f'fold_{fold_id}'
    output_dir = Path(output_base_dir) / f'fold_{fold_id}'

    if not fold_dir.exists():
        raise ValueError(f"Fold directory not found: {fold_dir}")

 # datafile
    train_file = fold_dir / 'train.csv'
    val_file = fold_dir / 'val.csv'
    test_file = fold_dir / 'test.csv'

    for f in [train_file, val_file, test_file]:
        if not f.exists():
            raise ValueError(f"Data file not found: {f}")

    print("="*80)
    print(f"🚀 Training LORO Fold {fold_id}")
    print("="*80)
    print(f"Config: {config_file}")
    print(f"Data:   {fold_dir}")
    print(f"Output: {output_dir}")
    print("="*80)

 # training
 # main.py LORO datapath
    cmd = f"""python main.py \\
        --cfg {config_file} \\
        --data loro_fold_{fold_id} \\
        --split loro \\
        --data-dir {fold_dir} \\
        --output-dir {output_dir} \\
        --epochs {epochs}"""

    print(f"\n🔧 Running command:")
    print(cmd)
    print()

 # rowtraining
 # 
    env = os.environ.copy()
    env['LORO_FOLD'] = str(fold_id)
    env['LORO_DATA_DIR'] = str(fold_dir)

 # main.py
 # main.py datapath
 # training

 # bash 
    import subprocess
    result = subprocess.run(
        f"python main.py --cfg {config_file} "
 f"--data cnnscore_database " # data dataloader
 f"--split loro " # loro split
        f"--data-dir {fold_dir} "  # LORO datadirectory
        f"--output-dir {output_dir}",
        shell=True,
        env=env
    )

    if result.returncode == 0:
        print(f"\n✅ Fold {fold_id} training completed successfully!")
        print(f"   Output: {output_dir}")
        return True
    else:
        print(f"\n❌ Fold {fold_id} training failed!")
        return False


def main():
    parser = argparse.ArgumentParser(description='Train single LORO fold')
    parser.add_argument('--fold', type=int, required=True,
                       help='Fold ID (0-11)')
    parser.add_argument('--config', type=str,
                       default='configs/DrugBAN_BiLSTM_CNNScore_Multitask.yaml',
                       help='Config file')
    parser.add_argument('--loro-dir', type=str,
                       default='datasets/cnnscore_database/loro',
                       help='LORO data directory')
    parser.add_argument('--output-dir', type=str,
                       default='result/LORO',
                       help='Output base directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')

    args = parser.parse_args()

 # fold ID
    if args.fold < 0 or args.fold > 11:
        raise ValueError(f"Fold ID must be 0-11, got {args.fold}")

    success = train_loro_fold(
        fold_id=args.fold,
        config_file=args.config,
        loro_dir=args.loro_dir,
        output_base_dir=args.output_dir,
        epochs=args.epochs
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
