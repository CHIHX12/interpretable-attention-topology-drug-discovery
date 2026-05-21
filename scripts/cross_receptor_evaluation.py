#!/usr/bin/env python3
"""
receptor
receptor A trainingmodelreceptor B data

generate 12x12 matrix
- row：trainingreceptor
- columnreceptor
- value：AUROC / AUPRC / R²
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs import get_cfg_defaults
from models import DrugBAN
from dataloader import DTIDataset, collate_selfies_fn
from torch.utils.data import DataLoader
from utils import set_seed, graph_collate_func
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score
from tqdm import tqdm


def load_model(config_path, model_path, device):
    """loadtrainingmodel"""
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)

    model = DrugBAN(**cfg).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model, cfg


def create_dataloader(data_path, cfg, batch_size=32):
    """createdataload"""
    df = pd.read_csv(data_path)

    use_selfies = cfg.DRUG.get("USE_BILSTM", False)
    use_protein_features = cfg.PROTEIN.get("USE_BILSTM", False)

    # SELFIES vocab
    selfies_vocab = None
    if use_selfies:
        vocab_path = Path("datasets/selfies_vocab.pkl")
        if vocab_path.exists():
            import pickle
            with open(vocab_path, 'rb') as f:
                selfies_vocab = pickle.load(f)

    list_IDs = list(range(len(df)))
    dataset = DTIDataset(
        list_IDs=list_IDs,
        df=df,
        max_drug_nodes=cfg.DRUG.MAX_NODES,
        max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
        use_features=use_protein_features,
        use_selfies=use_selfies,
        selfies_vocab=selfies_vocab,
        max_drug_length=cfg.DRUG.MAX_DRUG_LENGTH,
        use_drug_features=cfg.DRUG.get("USE_FEATURES", False)
    )

    collate_fn = collate_selfies_fn if use_selfies else graph_collate_func

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    return dataloader


def evaluate_model(model, dataloader, device, cfg):
    """
    model

    Returns:
        dict with metrics: AUROC, AUPRC, R² (if multitask)
    """
    model.eval()
    use_multitask = cfg.get("MULTITASK", {}).get("ENABLED", False)

    y_true_list = []
    y_pred_list = []
    z_true_list = []
    z_pred_list = []

    with torch.no_grad():
        for batch_data in dataloader:
            v_d, v_p, y, z = batch_data

            # Move to device
            if isinstance(v_d, tuple):
                v_d = tuple(item.to(device) for item in v_d)
            elif hasattr(v_d, 'to'):
                v_d = v_d.to(device)

            if isinstance(v_p, tuple):
                v_p = tuple(item.to(device) for item in v_p)
            else:
                v_p = v_p.to(device)

            y = y.to(device)
            z = z.to(device)

            # Forward pass
            try:
                forward_output = model(v_d, v_p, mode="eval")

                if use_multitask:
                    v_d_out, v_p_encoded, score, att, reg_output = forward_output
                else:
                    v_d_out, v_p_encoded, score, att = forward_output
                    reg_output = None

                # Collect predictions
                y_pred = torch.sigmoid(score).cpu().numpy()
                y_true = y.cpu().numpy()

                y_true_list.extend(y_true)
                y_pred_list.extend(y_pred)

                if use_multitask and reg_output is not None:
                    z_pred = reg_output.cpu().numpy()
                    z_true = z.cpu().numpy()
                    z_true_list.extend(z_true)
                    z_pred_list.extend(z_pred)

            except Exception as e:
                print(f"⚠️  Error in batch: {e}")
                continue

    # Calculate metrics
    metrics = {}

    if len(y_true_list) > 0:
        metrics['auroc'] = roc_auc_score(y_true_list, y_pred_list)
        metrics['auprc'] = average_precision_score(y_true_list, y_pred_list)

    if use_multitask and len(z_true_list) > 0:
        metrics['r2'] = r2_score(z_true_list, z_pred_list)

    return metrics


def cross_receptor_evaluation(
    model_base_dir,
    data_base_dir,
    config_path,
    output_dir,
    n_receptors=12,
    device='cuda'
):
    """
    rowreceptor

    Args:
    model_base_dir: modeldirectory receptor_0, receptor_1, ...
    data_base_dir: datadirectory receptor_0, receptor_1, ...
        config_path: file
        output_dir: outputdirectory
        n_receptors: receptor
        device: device
    """
    model_base_dir = Path(model_base_dir)
    data_base_dir = Path(data_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("🔬 Cross-Receptor Generalization Evaluation")
    print("="*80)
    print(f"Model directory: {model_base_dir}")
    print(f"Data directory: {data_base_dir}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Receptors: {n_receptors}")
    print()

    receptor_ids = list(range(n_receptors))

    # resultmatrix
    auroc_matrix = np.zeros((n_receptors, n_receptors))
    auprc_matrix = np.zeros((n_receptors, n_receptors))
    r2_matrix = np.zeros((n_receptors, n_receptors))

    # pairtrainingreceptor
    for train_receptor in tqdm(receptor_ids, desc="Training receptors"):
        print(f"\n{'='*80}")
        print(f"Train Receptor {train_receptor}")
        print(f"{'='*80}")

        # loadmodel
        model_path = model_base_dir / f'receptor_{train_receptor}' / 'best_model.pth'

        if not model_path.exists():
            print(f"⚠️  Model not found: {model_path}")
            print(f"   Skipping train receptor {train_receptor}")
            auroc_matrix[train_receptor, :] = np.nan
            auprc_matrix[train_receptor, :] = np.nan
            r2_matrix[train_receptor, :] = np.nan
            continue

        try:
            model, cfg = load_model(config_path, model_path, device)
            print(f"✅ Loaded model: {model_path}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            auroc_matrix[train_receptor, :] = np.nan
            auprc_matrix[train_receptor, :] = np.nan
            r2_matrix[train_receptor, :] = np.nan
            continue

            # receptor
        for test_receptor in receptor_ids:
            print(f"\n  Testing on Receptor {test_receptor}...", end=' ')

            # loaddata
            test_data_path = data_base_dir / f'receptor_{test_receptor}' / 'test.csv'

            if not test_data_path.exists():
                print(f"⚠️  Data not found: {test_data_path}")
                auroc_matrix[train_receptor, test_receptor] = np.nan
                auprc_matrix[train_receptor, test_receptor] = np.nan
                r2_matrix[train_receptor, test_receptor] = np.nan
                continue

            try:
                dataloader = create_dataloader(test_data_path, cfg, batch_size=32)
                metrics = evaluate_model(model, dataloader, device, cfg)

                auroc_matrix[train_receptor, test_receptor] = metrics.get('auroc', np.nan)
                auprc_matrix[train_receptor, test_receptor] = metrics.get('auprc', np.nan)
                r2_matrix[train_receptor, test_receptor] = metrics.get('r2', np.nan)

                print(f"AUROC={metrics.get('auroc', np.nan):.4f}, AUPRC={metrics.get('auprc', np.nan):.4f}", end='')
                if 'r2' in metrics:
                    print(f", R²={metrics['r2']:.4f}")
                else:
                    print()

            except Exception as e:
                print(f"❌ Error: {e}")
                auroc_matrix[train_receptor, test_receptor] = np.nan
                auprc_matrix[train_receptor, test_receptor] = np.nan
                r2_matrix[train_receptor, test_receptor] = np.nan

    # saveresultmatrix
    print("\n" + "="*80)
    print("💾 Saving Results")
    print("="*80)

    # create DataFrame
    df_auroc = pd.DataFrame(
        auroc_matrix,
        index=[f'Train_{i}' for i in receptor_ids],
        columns=[f'Test_{i}' for i in receptor_ids]
    )
    df_auprc = pd.DataFrame(
        auprc_matrix,
        index=[f'Train_{i}' for i in receptor_ids],
        columns=[f'Test_{i}' for i in receptor_ids]
    )
    df_r2 = pd.DataFrame(
        r2_matrix,
        index=[f'Train_{i}' for i in receptor_ids],
        columns=[f'Test_{i}' for i in receptor_ids]
    )

    # save CSV
    df_auroc.to_csv(output_dir / 'cross_receptor_auroc_matrix.csv')
    df_auprc.to_csv(output_dir / 'cross_receptor_auprc_matrix.csv')
    df_r2.to_csv(output_dir / 'cross_receptor_r2_matrix.csv')

    print(f"✅ AUROC matrix saved to: {output_dir / 'cross_receptor_auroc_matrix.csv'}")
    print(f"✅ AUPRC matrix saved to: {output_dir / 'cross_receptor_auprc_matrix.csv'}")
    print(f"✅ R² matrix saved to: {output_dir / 'cross_receptor_r2_matrix.csv'}")

    # computestatistics
    print("\n" + "="*80)
    print("📊 Statistics")
    print("="*80)

    # pair vs pair
    diagonal_auroc = np.diag(auroc_matrix)
    off_diagonal_auroc = auroc_matrix[~np.eye(n_receptors, dtype=bool)]

    print("\nAUROC:")
    print(f"  Diagonal (same receptor):   {np.nanmean(diagonal_auroc):.4f} ± {np.nanstd(diagonal_auroc):.4f}")
    print(f"  Off-diagonal (cross receptor): {np.nanmean(off_diagonal_auroc):.4f} ± {np.nanstd(off_diagonal_auroc):.4f}")
    print(f"  Performance drop: {np.nanmean(diagonal_auroc) - np.nanmean(off_diagonal_auroc):.4f}")

    diagonal_auprc = np.diag(auprc_matrix)
    off_diagonal_auprc = auprc_matrix[~np.eye(n_receptors, dtype=bool)]

    print("\nAUPRC:")
    print(f"  Diagonal (same receptor):   {np.nanmean(diagonal_auprc):.4f} ± {np.nanstd(diagonal_auprc):.4f}")
    print(f"  Off-diagonal (cross receptor): {np.nanmean(off_diagonal_auprc):.4f} ± {np.nanstd(off_diagonal_auprc):.4f}")
    print(f"  Performance drop: {np.nanmean(diagonal_auprc) - np.nanmean(off_diagonal_auprc):.4f}")

    if not np.all(np.isnan(r2_matrix)):
        diagonal_r2 = np.diag(r2_matrix)
        off_diagonal_r2 = r2_matrix[~np.eye(n_receptors, dtype=bool)]

        print("\nR²:")
        print(f"  Diagonal (same receptor):   {np.nanmean(diagonal_r2):.4f} ± {np.nanstd(diagonal_r2):.4f}")
        print(f"  Off-diagonal (cross receptor): {np.nanmean(off_diagonal_r2):.4f} ± {np.nanstd(off_diagonal_r2):.4f}")
        print(f"  Performance drop: {np.nanmean(diagonal_r2) - np.nanmean(off_diagonal_r2):.4f}")

        # savestatistics
    stats = {
        'metric': [],
        'diagonal_mean': [],
        'diagonal_std': [],
        'off_diagonal_mean': [],
        'off_diagonal_std': [],
        'performance_drop': []
    }

    stats['metric'].append('AUROC')
    stats['diagonal_mean'].append(np.nanmean(diagonal_auroc))
    stats['diagonal_std'].append(np.nanstd(diagonal_auroc))
    stats['off_diagonal_mean'].append(np.nanmean(off_diagonal_auroc))
    stats['off_diagonal_std'].append(np.nanstd(off_diagonal_auroc))
    stats['performance_drop'].append(np.nanmean(diagonal_auroc) - np.nanmean(off_diagonal_auroc))

    stats['metric'].append('AUPRC')
    stats['diagonal_mean'].append(np.nanmean(diagonal_auprc))
    stats['diagonal_std'].append(np.nanstd(diagonal_auprc))
    stats['off_diagonal_mean'].append(np.nanmean(off_diagonal_auprc))
    stats['off_diagonal_std'].append(np.nanstd(off_diagonal_auprc))
    stats['performance_drop'].append(np.nanmean(diagonal_auprc) - np.nanmean(off_diagonal_auprc))

    if not np.all(np.isnan(r2_matrix)):
        stats['metric'].append('R2')
        stats['diagonal_mean'].append(np.nanmean(diagonal_r2))
        stats['diagonal_std'].append(np.nanstd(diagonal_r2))
        stats['off_diagonal_mean'].append(np.nanmean(off_diagonal_r2))
        stats['off_diagonal_std'].append(np.nanstd(off_diagonal_r2))
        stats['performance_drop'].append(np.nanmean(diagonal_r2) - np.nanmean(off_diagonal_r2))

    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(output_dir / 'cross_receptor_statistics.csv', index=False)

    print(f"\n✅ Statistics saved to: {output_dir / 'cross_receptor_statistics.csv'}")

    print("\n" + "="*80)
    print("✅ Cross-Receptor Evaluation Complete!")
    print("="*80)
    print(f"\nNext step: Generate heatmap visualization")
    print(f"  python scripts/plot_cross_receptor_heatmap.py --input {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Cross-receptor evaluation')
    parser.add_argument('--model-dir', type=str,
                       default='result/SINGLE_RECEPTOR',
                       help='Base directory containing trained models')
    parser.add_argument('--data-dir', type=str,
                       default='datasets/cnnscore_database/single_receptor',
                       help='Base directory containing test data')
    parser.add_argument('--config', type=str,
                       default='configs/DrugBAN_BiLSTM_CNNScore_Multitask.yaml',
                       help='Config file')
    parser.add_argument('--output-dir', type=str,
                       default='result/CROSS_RECEPTOR',
                       help='Output directory')
    parser.add_argument('--n-receptors', type=int, default=12,
                       help='Number of receptors')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    device = torch.device(args.device)

    cross_receptor_evaluation(
        model_base_dir=args.model_dir,
        data_base_dir=args.data_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        n_receptors=args.n_receptors,
        device=device
    )


if __name__ == '__main__':
    main()
