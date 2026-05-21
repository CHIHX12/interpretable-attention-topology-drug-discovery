#!/usr/bin/env python3
"""
proteingroupattentionstatisticsanalysis
============================

eachproteinPDB IDanalysis
- proteinconsensusbinding sitedrugaverage
- Z-score analysisprotein
- frequencystatisticsresidue
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

from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_selfies_fn
from models import DrugBAN
from utils import graph_collate_func

try:
    import pickle
except ImportError as exc:
    raise RuntimeError("pickle module is required") from exc


def load_selfies_vocab(cfg):
    """ SELFIES """
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
    """extractattentionscore"""
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

    _, _, score_tensor, attn = model(v_d, v_p, mode="eval")
    score = torch.sigmoid(score_tensor).squeeze().item()

    attn = attn.squeeze(0)
    if attn.dim() == 3:
        attn = attn.mean(dim=0)
    if attn.dim() != 2:
        raise ValueError(f"Unexpected attention shape: {attn.shape}")

    att_vec = attn.sum(dim=0).cpu().numpy()

    att_min = att_vec.min()
    att_max = att_vec.max()
    if att_max > att_min:
        att_vec = (att_vec - att_min) / (att_max - att_min)
    else:
        att_vec = np.zeros_like(att_vec)

    seq_len = int((prot_tokens > 0).sum())
    att_vec = att_vec[:seq_len]

    mean = float(att_vec.mean())
    std = float(att_vec.std())
    return att_vec, score, mean, std


def analyze_all_and_group_by_protein(model, data_loader, df, device, std_mult=1.0):
    """analysissampleproteingroup"""

    protein_data = defaultdict(lambda: {
        'attention_maps': [],
        'high_attention_residues': [],
        'pocket_masks': [],
        'sample_info': []
    })

    print(f"\nanalysissample PDB_ID group...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Processing samples")):
            # data
            v_d, v_p, labels = batch

            # attention
            att_vec, score, mean_att, std_att = collect_attention(model, v_d, v_p, labels, device)

            # attentionresidue
            threshold = mean_att + std_mult * std_att
            high_attention_indices = np.where(att_vec > threshold)[0]

            # data
            row_idx = data_loader.dataset.list_IDs[batch_idx]
            row = df.iloc[row_idx]
            pdb_id = str(row.get("PDB_ID", "")).strip().upper()

            # PDB_ID
            if not pdb_id or pdb_id == "NAN":
                continue

            # Pocket mask
            pocket_mask = None
            if "pocket_mask" in row and isinstance(row["pocket_mask"], str) and row["pocket_mask"]:
                try:
                    pocket_mask = json.loads(row["pocket_mask"])
                except json.JSONDecodeError:
                    pocket_mask = None

            # saveproteingroup
            protein_data[pdb_id]['attention_maps'].append(att_vec)
            protein_data[pdb_id]['high_attention_residues'].append(high_attention_indices)
            if pocket_mask is not None:
                protein_data[pdb_id]['pocket_masks'].append(np.array(pocket_mask))

            protein_data[pdb_id]['sample_info'].append({
                'batch_idx': batch_idx,
                'attention': att_vec,
                'high_attention': high_attention_indices,
                'pocket_mask': np.array(pocket_mask) if pocket_mask else None,
                'threshold': threshold,
                'pred': score,
                'label': float(row.get("Y", 0.0)),
                'smiles': row.get("SMILES", "")
            })

    # convertfinalformat
    protein_results = {}
    for pdb_id, data in protein_data.items():
        protein_results[pdb_id] = {
            'pdb_id': pdb_id,
            'n_samples': len(data['attention_maps']),
            'attention_maps': data['attention_maps'],
            'high_attention_residues': data['high_attention_residues'],
            'pocket_masks': data['pocket_masks'],
            'sample_info': data['sample_info']
        }

    return protein_results


def compute_protein_statistics(protein_results):
    """computeproteinstatistics"""
    attention_maps = protein_results['attention_maps']
    high_attention_residues = protein_results['high_attention_residues']
    pocket_masks = protein_results['pocket_masks']
    n_samples = protein_results['n_samples']

    if not attention_maps:
        return None

    # maxlength
    max_len = max(len(att) for att in attention_maps)

    # attentionmatrix
    attention_matrix = np.zeros((n_samples, max_len))
    for i, att in enumerate(attention_maps):
        attention_matrix[i, :len(att)] = att

    # statistics
    mean_attention = np.mean(attention_matrix, axis=0)
    std_attention = np.std(attention_matrix, axis=0)

    # Z-score
    z_scores = np.zeros_like(mean_attention)
    valid_positions = std_attention > 0
    if valid_positions.any():
        z_scores[valid_positions] = (mean_attention[valid_positions] - mean_attention[valid_positions].mean()) / std_attention[valid_positions]

    # frequency
    residue_frequency = Counter()
    for high_res in high_attention_residues:
        residue_frequency.update(high_res)

    # Pocket 
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

        mean_coverage = np.mean(pocket_coverage) if pocket_coverage else None
        std_coverage = np.std(pocket_coverage) if pocket_coverage else None
    else:
        mean_coverage = None
        std_coverage = None

    return {
        'pdb_id': protein_results['pdb_id'],
        'n_samples': n_samples,
        'seq_len': max_len,
        'mean_attention': mean_attention,
        'std_attention': std_attention,
        'z_scores': z_scores,
        'residue_frequency': residue_frequency,
        'pocket_coverage_mean': mean_coverage,
        'pocket_coverage_std': std_coverage
    }


def visualize_protein_results(stats, output_dir, top_k=30):
    """proteinresult"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_id = stats['pdb_id']
    n_samples = stats['n_samples']
    mean_attention = stats['mean_attention']
    z_scores = stats['z_scores']
    residue_frequency = stats['residue_frequency']

    # 
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))

    # 1. averageattention
    ax = axes[0]
    positions = np.arange(len(mean_attention))
    ax.plot(positions, mean_attention, linewidth=1.5, color='steelblue', label='Mean Attention')
    ax.fill_between(positions,
                     mean_attention - stats['std_attention'],
                     mean_attention + stats['std_attention'],
                     alpha=0.3, color='steelblue', label='±1 STD')
    ax.axhline(mean_attention.mean(), color='red', linestyle='--', alpha=0.5, label='Overall Mean')
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Mean Attention Score', fontsize=12)
    ax.set_title(f'{pdb_id}: Average Attention Across {n_samples} Drugs', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Z-score
    ax = axes[1]
    ax.plot(positions, z_scores, linewidth=1.5, color='darkgreen')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5, label='Z=2 threshold')
    ax.fill_between(positions, 0, z_scores, where=(z_scores > 2),
                     alpha=0.3, color='red', label='High Z-score (>2)')
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Z-score', fontsize=12)
    ax.set_title(f'{pdb_id}: Attention Z-scores', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. frequencystatistics
    ax = axes[2]
    if residue_frequency:
        top_residues = sorted(residue_frequency.items(), key=lambda x: x[1], reverse=True)[:top_k]
        positions_top = [r[0] for r in top_residues]
        frequencies = [r[1] for r in top_residues]
        frequency_pct = [f / n_samples * 100 for f in frequencies]

        colors = plt.cm.Reds(np.array(frequency_pct) / 100)
        ax.bar(range(len(positions_top)), frequency_pct, color=colors)
        ax.set_xlabel(f'Top {len(positions_top)} Residues (by frequency)', fontsize=12)
        ax.set_ylabel('Frequency (%)', fontsize=12)
        ax.set_title(f'{pdb_id}: Residue Selection Frequency (n={n_samples} drugs)',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(positions_top)))
        ax.set_xticklabels([str(p) for p in positions_top], rotation=90, fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{pdb_id}_aggregate_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / f'{pdb_id}_aggregate_analysis.svg', bbox_inches='tight')
    print(f" ✓ save: {pdb_id}_aggregate_analysis.png")
    plt.close()

    # consensusbinding site
    if residue_frequency:
        fig, ax = plt.subplots(figsize=(16, 6))

        consensus_vector = np.zeros(len(mean_attention))
        for res, freq in residue_frequency.items():
            if res < len(consensus_vector):
                consensus_vector[res] = freq / n_samples * 100

        colors = np.where(consensus_vector > 50, 'red',
                         np.where(consensus_vector > 25, 'orange', 'lightblue'))
        ax.bar(np.arange(len(consensus_vector)), consensus_vector, color=colors, width=1.0)

        ax.axhline(50, color='red', linestyle='--', alpha=0.7, label='50% consensus')
        ax.axhline(25, color='orange', linestyle='--', alpha=0.7, label='25% consensus')

        ax.set_xlabel('Residue Position', fontsize=12)
        ax.set_ylabel('Selection Frequency (%)', fontsize=12)
        ax.set_title(f'{pdb_id}: Consensus Binding Site (n={n_samples} drugs)',
                     fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / f'{pdb_id}_consensus_binding_site.png', dpi=300, bbox_inches='tight')
        print(f" ✓ save: {pdb_id}_consensus_binding_site.png")
        plt.close()


def generate_pymol_script(stats, pdb_root, output_dir):
    """generate PyMOL consensusbinding site"""
    pdb_id = stats['pdb_id']
    n_samples = stats['n_samples']
    residue_frequency = stats['residue_frequency']

    if not residue_frequency:
        return None

    # PDB file
    pdb_root = Path(pdb_root)
    pdb_lower = pdb_id.lower()
    pattern = f"**/{pdb_lower}/{pdb_lower}_protein.pdb"
    matches = list(pdb_root.glob(pattern))

    if not matches:
        print(f" ⚠️ PDB file: {pdb_id}")
        return None

    pdb_path = matches[0]

    # ligandfile
    ligand_path = None
    sdf_candidate = pdb_path.with_name(pdb_path.name.replace("_protein.pdb", "_ligand.sdf"))
    mol2_candidate = pdb_path.with_name(pdb_path.name.replace("_protein.pdb", "_ligand.mol2"))

    if sdf_candidate.exists():
        ligand_path = sdf_candidate
    elif mol2_candidate.exists():
        ligand_path = mol2_candidate

    # frequencyclassificationresidue
    very_high = []  # >75%
    high = []       # 50-75%
    moderate = []   # 25-50%

    for res, count in residue_frequency.items():
        freq_pct = count / n_samples * 100
        if freq_pct > 75:
            very_high.append(res)
        elif freq_pct > 50:
            high.append(res)
        elif freq_pct > 25:
            moderate.append(res)

    # generate PyMOL 
    output_dir = Path(output_dir)
    script_path = output_dir / f"{pdb_id}_consensus.pml"

    obj_name = f"{pdb_id}_protein"

    with script_path.open('w', encoding='ascii', errors='replace') as f:
        f.write("# PyMOL Script - Consensus Binding Site Visualization\n")
        f.write(f"# Protein: {pdb_id}\n")
        f.write(f"# Number of drugs analyzed: {n_samples}\n")
        f.write("#\n")
        f.write("# Color Scheme:\n")
        f.write("#   RED (>75%):     Very high frequency - most drugs focus here\n")
        f.write("#   ORANGE (50-75%): High frequency - many drugs focus here\n")
        f.write("#   YELLOW (25-50%): Moderate frequency - some drugs focus here\n")
        f.write("#   CYAN:           Ligand molecule (if available)\n")
        f.write("#   GREY:           Other residues (background)\n")
        f.write("#\n")

        f.write("reinitialize\n")
        f.write(f"load {pdb_path.resolve()}, {obj_name}\n")
        f.write("hide everything, all\n")
        f.write(f"show cartoon, {obj_name}\n")
        f.write(f"color grey70, {obj_name}\n")
        f.write("\n")

        # ligand
        if ligand_path:
            ligand_obj = f"{pdb_id}_ligand"
            f.write(f"# Load ligand\n")
            f.write(f"load {ligand_path.resolve()}, {ligand_obj}\n")
            f.write(f"show sticks, {ligand_obj}\n")
            f.write(f"color cyan, {ligand_obj}\n")
            f.write("\n")

        # RED - Very high frequency
        if very_high:
            resi_list = "+".join(map(str, sorted(very_high)))
            f.write(f"# Very high frequency residues (>75%, n={len(very_high)})\n")
            f.write(f"select {pdb_id}_very_high, ({obj_name} and resi {resi_list})\n")
            f.write(f"color red, {pdb_id}_very_high\n")
            f.write(f"show sticks, {pdb_id}_very_high\n")
            f.write("\n")

        # ORANGE - High frequency
        if high:
            resi_list = "+".join(map(str, sorted(high)))
            f.write(f"# High frequency residues (50-75%, n={len(high)})\n")
            f.write(f"select {pdb_id}_high, ({obj_name} and resi {resi_list})\n")
            f.write(f"color orange, {pdb_id}_high\n")
            f.write(f"show sticks, {pdb_id}_high\n")
            f.write("\n")

        # YELLOW - Moderate frequency
        if moderate:
            resi_list = "+".join(map(str, sorted(moderate)))
            f.write(f"# Moderate frequency residues (25-50%, n={len(moderate)})\n")
            f.write(f"select {pdb_id}_moderate, ({obj_name} and resi {resi_list})\n")
            f.write(f"color yellow, {pdb_id}_moderate\n")
            f.write(f"show sticks, {pdb_id}_moderate\n")
            f.write("\n")

        f.write("zoom\n")
        f.write("\n")
        f.write(f"# Summary Statistics:\n")
        f.write(f"#   RED residues:    {len(very_high)}\n")
        f.write(f"#   ORANGE residues: {len(high)}\n")
        f.write(f"#   YELLOW residues: {len(moderate)}\n")
        f.write(f"#   Total consensus residues: {len(very_high) + len(high) + len(moderate)}\n")

    return script_path


def print_protein_summary(stats):
    """proteinstatistics"""
    pdb_id = stats['pdb_id']
    n_samples = stats['n_samples']

    print(f"\n{'='*60}")
    print(f"📊 {pdb_id} statistics (n={n_samples} drugs)")
    print(f"{'='*60}")

    mean_att = stats['mean_attention']
    print(f"sequencelength: {stats['seq_len']}")
    print(f"\naverageattention:")
    print(f"  Mean: {mean_att.mean():.6f}")
    print(f"  STD:  {mean_att.std():.6f}")
    print(f"  Max:  {mean_att.max():.6f}")

    z_scores = stats['z_scores']
    high_z = np.sum(z_scores > 2)
    print(f"\nZ-score analysis:")
    print(f" Z-score residue (>2): {high_z} ")
    if high_z > 0:
        high_z_residues = np.where(z_scores > 2)[0]
        print(f" : {list(high_z_residues[:10])}{'...' if high_z > 10 else ''}")

    freq = stats['residue_frequency']
    if freq:
        print(f"\nfrequencystatistics:")
        print(f" residue: {len(freq)} ")
        top_10 = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f" 10 residue:")
        for res, count in top_10:
            pct = count / n_samples * 100
            print(f"    residue {res:4d}: {count:3d}/{n_samples} ({pct:5.1f}%)")

    if stats['pocket_coverage_mean'] is not None:
        print(f"\nPocket :")
        print(f"  average: {stats['pocket_coverage_mean']*100:.1f}% ± {stats['pocket_coverage_std']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='proteingroupattentionstatisticsanalysis')
    parser.add_argument('--cfg', type=str, required=True, help='configfilepath')
    parser.add_argument('--ckpt', type=str, required=True, help='modelweightpath')
    parser.add_argument('--data', type=str, default='bindingdb', help='data')
    parser.add_argument('--data-split', type=str, default='overlap_with_mask', help='data')
    parser.add_argument('--split', type=str, nargs='+', default=['test'],
                        help='train/val/test--split train val test')
    parser.add_argument('--std-mult', type=float, default=1.0, help='stdthreshold')
    parser.add_argument('--output-dir', type=str, default='aggregate_by_protein', help='outputdirectory')
    parser.add_argument('--top-k', type=int, default=30, help='display K high-frequencyresidue')
    parser.add_argument('--min-samples', type=int, default=2, help='sampleprotein')
    parser.add_argument('--pdb-root', type=str, default='P-L', help='PDB filedirectory')
    parser.add_argument('--generate-pymol', action='store_true', help='generate PyMOL ')
    parser.add_argument('--affinity-threshold', type=float, default=None,
                        help='affinitythresholdfilteranalysis Y >= threshold sample')

    args = parser.parse_args()

    # config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"usedevice: {device}")

    # data
    dfs = []
    for split in args.split:
        csv_path = Path("datasets") / args.data / args.data_split / f"{split}.csv"
        if not csv_path.exists():
            print(f"⚠️ warning: {csv_path} ")
            continue
        print(f"data: {csv_path}")
        df_split = pd.read_csv(csv_path)
        df_split['split_source'] = split # 
        dfs.append(df_split)

    if not dfs:
        raise FileNotFoundError("data")

    # data
    df = pd.concat(dfs, ignore_index=True)
    print(f"\n: {len(df)} sample")
    for split in args.split:
        count = (df['split_source'] == split).sum()
        print(f" {split}: {count} sample")

    # affinityfilter
    if args.affinity_threshold is not None:
        if 'Y' in df.columns:
            before_count = len(df)
            df = df[df['Y'] >= args.affinity_threshold].reset_index(drop=True)
            after_count = len(df)
            print(f"\naffinityfilter (Y >= {args.affinity_threshold}):")
            print(f" filter: {before_count} sample")
            print(f" filter: {after_count} sample")
            print(f" : {before_count - after_count} affinitysample ({(before_count-after_count)/before_count*100:.1f}%)")
        else:
            print(f"\n⚠️ warning: data 'Y' rowaffinityfilter")

    # 
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

    # analysissampleproteingroup
    protein_results_dict = analyze_all_and_group_by_protein(model, data_loader, df, device, args.std_mult)

    # min_samples sample
    protein_results_dict = {k: v for k, v in protein_results_dict.items() if v['n_samples'] >= args.min_samples}

    print(f"\n {len(protein_results_dict)} protein {args.min_samples} drugsample")
    for pdb_id in sorted(protein_results_dict.keys()):
        print(f" {pdb_id}: {protein_results_dict[pdb_id]['n_samples']} drug")

    # analysiseachprotein
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_protein_stats = []

    pymol_scripts_generated = []

    for pdb_id in sorted(protein_results_dict.keys()):
        protein_results = protein_results_dict[pdb_id]

        # computestatistics
        stats = compute_protein_statistics(protein_results)

        if stats:
            # 
            print_protein_summary(stats)

            # 
            visualize_protein_results(stats, output_dir, args.top_k)

            # generate PyMOL 
            if args.generate_pymol:
                pymol_script = generate_pymol_script(stats, args.pdb_root, output_dir)
                if pymol_script:
                    pymol_scripts_generated.append(pymol_script)
                    print(f" ✓ PyMOL : {pymol_script.name}")

            all_protein_stats.append(stats)

    print(f"\n{'='*60}")
    print(f"✓ doneanalysis {len(all_protein_stats)} protein")
    print(f"{'='*60}")
    print(f"\nresultsave: {output_dir}/")
    print(f"\neachproteingenerate：")
    print(f"  - {'{PDB_ID}'}_aggregate_analysis.png (averageattention + Z-score + frequency)")
    print(f"  - {'{PDB_ID}'}_consensus_binding_site.png (consensusbinding site)")
    if args.generate_pymol:
        print(f" - {'{PDB_ID}'}_consensus.pml (PyMOL consensus)")
        print(f"\ngenerate {len(pymol_scripts_generated)} PyMOL ")
        print(f"\nuse")
        print(f"  pymol aggregate_by_protein/{{PDB_ID}}_consensus.pml")


if __name__ == '__main__':
    main()
