#!/usr/bin/env python3
"""
attentionstatisticsanalysis
- computesampleaverageattention
- Z-score analysis
- frequencystatisticsresidue
- consensusbinding siteprediction
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# Import from existing codebase
from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_selfies_fn
from models import DrugBAN
from utils import graph_collate_func

try:
    import pickle
except ImportError as exc:
    raise RuntimeError("pickle module is required") from exc


def load_selfies_vocab(cfg):
    """ SELFIES BiLSTM"""
    if not cfg.DRUG.get("USE_BILSTM", False):
        return None
    vocab_path = Path(cfg.RESULT.OUTPUT_DIR) / "selfies_vocab.pkl"
    if not vocab_path.exists():
        raise FileNotFoundError(f"SELFIES vocab not found: {vocab_path}")
    with vocab_path.open("rb") as fp:
        vocab = pickle.load(fp)
    return vocab


def build_dataset(df, cfg, selfies_vocab):
    """data"""
    return DTIDataset(
        df.index.values,
        df,
        max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
        use_features=cfg.PROTEIN.get("USE_BILSTM", False),
        use_selfies=cfg.DRUG.get("USE_BILSTM", False),
        selfies_vocab=selfies_vocab,
        max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
        use_drug_features=cfg.DRUG.get("USE_FEATURES", False),
    )


def collect_attention(model, v_d, v_p, labels, device):
    """modelextractattention"""
    # processinginput
    if isinstance(v_d, tuple):
        v_d = tuple(item.to(device) for item in v_d)
    else:
        v_d = v_d.to(device)

    if isinstance(v_p, tuple):
        v_p = tuple(item.to(device) for item in v_p)
        prot_tokens = v_p[0].squeeze(0).cpu().numpy()
    else:
        v_p = v_p.to(device)
        prot_tokens = v_p.squeeze(0).cpu().numpy()

    labels = labels.to(device)

    # model
    _, _, score_tensor, attn = model(v_d, v_p, mode="eval")
    score = torch.sigmoid(score_tensor).squeeze().item()

    # processingattentiontensor
    attn = attn.squeeze(0)
    if attn.dim() == 3:
        attn = attn.mean(dim=0)
    if attn.dim() != 2:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

        # attention
    att_vec = attn.sum(dim=0).cpu().numpy()

    # [0, 1]
    att_min = att_vec.min()
    att_max = att_vec.max()
    if att_max > att_min:
        att_vec = (att_vec - att_min) / (att_max - att_min)
    else:
        att_vec = np.zeros_like(att_vec)

        # computesequencelength
    seq_len = int((prot_tokens > 0).sum())
    att_vec = att_vec[:seq_len]

    mean = float(att_vec.mean())
    std = float(att_vec.std())
    return att_vec, score, mean, std

def analyze_all_samples(model, cfg, data_loader, df, device, std_mult=1.0):
    """analysissampleattention"""

    all_attention_maps = []
    all_high_attention_residues = []
    all_pocket_masks = []
    sample_info = []

    print(f"\nanalysis {len(data_loader.dataset)} sample...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            # dataloader returns 4 values (v_d, v_p, labels, z) when features enabled
            if len(batch) == 4:
                v_d, v_p, labels, _ = batch
            else:
                v_d, v_p, labels = batch

            # attention
            att_vec, score, mean_att, std_att = collect_attention(model, v_d, v_p, labels, device)

            # attentionresiduethreshold
            threshold = mean_att + std_mult * std_att
            high_attention_mask = att_vec > threshold
            high_attention_indices = np.where(high_attention_mask)[0]

            # pocket_mask
            row_idx = data_loader.dataset.list_IDs[batch_idx]
            row = df.iloc[row_idx]
            pocket_mask = None
            if "pocket_mask" in row and isinstance(row["pocket_mask"], str) and row["pocket_mask"]:
                try:
                    pocket_mask = json.loads(row["pocket_mask"])
                except json.JSONDecodeError:
                    pocket_mask = None

            # saveresult
            all_attention_maps.append(att_vec)
            all_high_attention_residues.append(high_attention_indices)
            if pocket_mask is not None:
                all_pocket_masks.append(np.array(pocket_mask))

            sample_info.append({
                'index': batch_idx,
                'attention': att_vec,
                'high_attention': high_attention_indices,
                'pocket_mask': np.array(pocket_mask) if pocket_mask else None,
                'threshold': threshold,
                'pred': score,
                'label': float(row.get("Y", 0.0))
            })

    return {
        'attention_maps': all_attention_maps,
        'high_attention_residues': all_high_attention_residues,
        'pocket_masks': all_pocket_masks,
        'sample_info': sample_info
    }

def compute_aggregate_statistics(results):
    """computestatistics"""

    attention_maps = results['attention_maps']
    high_attention_residues = results['high_attention_residues']
    pocket_masks = results['pocket_masks']

    # maxsequencelength
    max_len = max(len(att) for att in attention_maps)
    n_samples = len(attention_maps)

    print(f"\ncomputestatistics{n_samples} samplemaxlength {max_len}...")

    # 1. averageattention
    attention_matrix = np.zeros((n_samples, max_len))
    for i, att in enumerate(attention_maps):
        attention_matrix[i, :len(att)] = att

    mean_attention = np.mean(attention_matrix, axis=0)
    std_attention = np.std(attention_matrix, axis=0)

    # 2. Z-score
    z_scores = np.zeros_like(mean_attention)
    valid_positions = std_attention > 0
    z_scores[valid_positions] = (mean_attention[valid_positions] - mean_attention[valid_positions].mean()) / std_attention[valid_positions]

    # 3. frequencystatisticsresiduesampleattention
    residue_frequency = Counter()
    for high_res in high_attention_residues:
        residue_frequency.update(high_res)

        # 4. Pocket statistics pocket_mask
    if pocket_masks:
        pocket_coverage = []
        for i, high_res in enumerate(high_attention_residues):
            if i < len(pocket_masks) and pocket_masks[i] is not None:
                pocket = np.where(pocket_masks[i] > 0)[0]
                if len(high_res) > 0:
                    coverage = len(set(high_res) & set(pocket)) / len(high_res)
                else:
                    coverage = 0.0
                pocket_coverage.append(coverage)

        mean_coverage = np.mean(pocket_coverage)
        std_coverage = np.std(pocket_coverage)
    else:
        mean_coverage = None
        std_coverage = None

    return {
        'mean_attention': mean_attention,
        'std_attention': std_attention,
        'z_scores': z_scores,
        'residue_frequency': residue_frequency,
        'max_len': max_len,
        'n_samples': n_samples,
        'pocket_coverage_mean': mean_coverage,
        'pocket_coverage_std': std_coverage
    }

def visualize_aggregate_results(stats, output_dir, top_k=50):
    """result"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_attention = stats['mean_attention']
    z_scores = stats['z_scores']
    residue_frequency = stats['residue_frequency']
    n_samples = stats['n_samples']

    # plot 1: averageattention + Z-score
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1a. averageattention
    ax = axes[0]
    residue_positions = np.arange(len(mean_attention))
    ax.plot(residue_positions, mean_attention, linewidth=1.5, color='steelblue', label='Mean Attention')
    ax.fill_between(residue_positions,
                     mean_attention - stats['std_attention'],
                     mean_attention + stats['std_attention'],
                     alpha=0.3, color='steelblue', label='±1 STD')
    ax.axhline(mean_attention.mean(), color='red', linestyle='--', alpha=0.5, label='Overall Mean')
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Mean Attention Score', fontsize=12)
    ax.set_title(f'Average Attention Across {n_samples} Samples', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1b. Z-score
    ax = axes[1]
    ax.plot(residue_positions, z_scores, linewidth=1.5, color='darkgreen')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='Z=2 threshold')
    ax.axhline(-2, color='red', linestyle='--', alpha=0.5)
    ax.fill_between(residue_positions, 0, z_scores, where=(z_scores > 2),
                     alpha=0.3, color='red', label='High Z-score (>2)')
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.set_title('Attention Z-scores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 1c. frequencystatistics
    ax = axes[2]
    if residue_frequency:
        # top_k residue
        top_residues = sorted(residue_frequency.items(), key=lambda x: x[1], reverse=True)[:top_k]
        positions = [r[0] for r in top_residues]
        frequencies = [r[1] for r in top_residues]
        frequency_pct = [f / n_samples * 100 for f in frequencies]

        colors = plt.cm.Reds(np.array(frequency_pct) / 100)
        ax.bar(range(len(positions)), frequency_pct, color=colors)
        ax.set_xlabel(f'Top {len(positions)} Residues (by frequency)', fontsize=12)
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title(f'Residue Selection Frequency (appears in X% of {n_samples} samples)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(positions)))
        ax.set_xticklabels([str(p) for p in positions], rotation=90, fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_attention_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'aggregate_attention_analysis.svg', bbox_inches='tight')
    print(f"✓ save: {output_dir / 'aggregate_attention_analysis.png'}")
    plt.close()

    # plot 2: consensusbinding site（Top residues by frequency）
    if residue_frequency:
        fig, ax = plt.subplots(figsize=(16, 6))

        # createvector
        consensus_vector = np.zeros(len(mean_attention))
        for res, freq in residue_frequency.items():
            if res < len(consensus_vector):
                consensus_vector[res] = freq / n_samples * 100

                # 
        colors = np.where(consensus_vector > 50, 'red',
                         np.where(consensus_vector > 25, 'orange', 'lightblue'))
        ax.bar(residue_positions, consensus_vector, color=colors, width=1.0)

        ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50% consensus')
        ax.axhline(25, color='orange', linestyle='--', alpha=0.7, label='25% consensus')

        ax.set_xlabel('Residue Position', fontsize=12)
        ax.set_ylabel('Selection Frequency (%)', fontsize=12)
        ax.set_title(f'Consensus Binding Site Prediction (n={n_samples} samples)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'consensus_binding_site.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'consensus_binding_site.svg', bbox_inches='tight')
        print(f"✓ save: {output_dir / 'consensus_binding_site.png'}")
        plt.close()

        # plot 3: plotsample x residue
    fig, ax = plt.subplots(figsize=(20, min(12, max(6, n_samples * 0.3))))

    # createattentionmatrix 200 residue
    max_display = min(200, len(mean_attention))
    attention_matrix = np.zeros((n_samples, max_display))

    # fill attention matrix (up to 50 samples)
    for i in range(min(n_samples, 50)):
        attention_matrix[i, :] = mean_attention[:max_display]

    sns.heatmap(attention_matrix[:min(50, n_samples), :],
                cmap='YlOrRd',
                cbar_kws={'label': 'Attention Score'},
                xticklabels=20,
                yticklabels=5,
                ax=ax)
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title(f'Attention Heatmap (first {min(50, n_samples)} samples, first {max_display} residues)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'attention_heatmap_aggregate.png', dpi=300, bbox_inches='tight')
    print(f"✓ save: {output_dir / 'attention_heatmap_aggregate.png'}")
    plt.close()

def print_summary_statistics(stats):
    """statistics"""

    print("\n" + "="*60)
    print("📊 statisticsresult")
    print("="*60)

    print(f"\nsample: {stats['n_samples']}")
    print(f"maxsequencelength: {stats['max_len']}")

    mean_att = stats['mean_attention']
    print(f"\naverageattention:")
    print(f"  Mean: {mean_att.mean():.6f}")
    print(f"  STD:  {mean_att.std():.6f}")
    print(f"  Max:  {mean_att.max():.6f}")
    print(f"  Min:  {mean_att.min():.6f}")

    z_scores = stats['z_scores']
    high_z = np.sum(z_scores > 2)
    print(f"\nZ-score analysis:")
    print(f" Z-score residue (>2): {high_z} ")
    print(f"  max Z-score: {z_scores.max():.2f}")

    freq = stats['residue_frequency']
    if freq:
        print(f"\nfrequencystatistics:")
        print(f" residue: {len(freq)} ")
        top_10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f" 10 residue:")
        for res, count in top_10:
            pct = count / stats['n_samples'] * 100
            print(f" residue {res:4d}: {count:3d} ({pct:5.1f}%)")

    if stats['pocket_coverage_mean'] is not None:
        print(f"\nPocket :")
        print(f"  average: {stats['pocket_coverage_mean']*100:.1f}% ± {stats['pocket_coverage_std']*100:.1f}%")

    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='attentionstatisticsanalysis')
    parser.add_argument('--cfg', type=str, required=True, help='filepath')
    parser.add_argument('--ckpt', type=str, required=True, help='modelweightpath')
    parser.add_argument('--data', type=str, default='bindingdb', help='data')
    parser.add_argument('--data-split', type=str, default='overlap_with_mask', help='data')
    parser.add_argument('--split', type=str, default='test', help='train/val/test')
    parser.add_argument('--std-mult', type=float, default=1.0, help='threshold')
    parser.add_argument('--output-dir', type=str, default='aggregate_analysis', help='outputdirectory')
    parser.add_argument('--top-k', type=int, default=50, help=' K high-frequencyresidue')

    args = parser.parse_args()

    # 
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # data
    csv_path = Path("datasets") / args.data / args.data_split / f"{args.split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {csv_path}")
        print(f"data: {csv_path}")
    df = pd.read_csv(csv_path)

    # SELFIES 
    selfies_vocab = load_selfies_vocab(cfg)

    # data
    dataset = build_dataset(df, cfg, selfies_vocab)

    # create DataLoader
    collate_fn = collate_selfies_fn if cfg.DRUG.get("USE_BILSTM", False) else graph_collate_func
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # model
    print(f"model: {args.ckpt}")
    model = DrugBAN(**cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # analysissample
    results = analyze_all_samples(model, cfg, data_loader, df, device, args.std_mult)

    # computestatistics
    stats = compute_aggregate_statistics(results)

    # 
    print_summary_statistics(stats)

    # 
    print(f"\ngenerateplot...")
    visualize_aggregate_results(stats, args.output_dir, args.top_k)

    print(f"\n✓ doneresultsave: {args.output_dir}/")

if __name__ == '__main__':
    main()
