#!/usr/bin/env python3
"""
analysis Class 1 (active) vs Class 0 (active) attentiondifference
Analyze Attention Differences between Active and Inactive Compounds


1. trainingmodelprediction GHSR 
2. extracteachdrugattentionscore
3. Class (Y=0/1) groupcomputeaverageattention
4. difference
5. keyamino acidresidue
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import project modules
from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_selfies_fn
from models import DrugBAN
from utils import set_seed, graph_collate_func, build_selfies_vocab

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze attention differences between active and inactive compounds")
    parser.add_argument('--config', type=str, default='configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str,
                        default='result/DrugBAN_BiLSTM_GHSR_Seed42/best_model_epoch_44.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_file', type=str,
                        default='datasets/GPCR_resarch/GHSR_training_data.csv',
                        help='Path to GHSR data CSV')
    parser.add_argument('--output_dir', type=str,
                        default='result/class_attention_analysis',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    return parser.parse_args()


def load_model(config_path, model_path, device):
    """model"""
    print(f"📦 config: {config_path}")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    print(f"🏗️ model...")
    model = DrugBAN(**cfg).to(device)

    print(f"⚙️ weight: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"✅ modelsuccess")
    return model, cfg


def extract_attention_by_class(model, dataloader, device, protein_len, use_drug_bilstm=False):
    """
    extractattentionscore class group

    Returns:
        class_0_attentions: list of attention arrays for Y=0
        class_1_attentions: list of attention arrays for Y=1
    """
    class_0_attentions = []
    class_1_attentions = []

    print(f"\n🔬 extractattentionscore...")

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Processing batches"):
            # dataloader returns 4 values (v_d, v_p, label, z) when features enabled
            if len(batch_data) == 4:
                v_d, v_p, label, _ = batch_data
            else:
                v_d, v_p, label = batch_data

            # Move to device
            v_d = v_d.to(device) if not isinstance(v_d, tuple) else tuple(x.to(device) for x in v_d)
            v_p = v_p.to(device) if not isinstance(v_p, tuple) else tuple(x.to(device) for x in v_p)
            label = label.to(device)

            # Forward pass
            _, _, score, att = model(v_d, v_p, mode="eval")

            batch_size = score.size(0)

            # Aggregate attention: [batch, heads, drug_len, protein_len] -> [batch, protein_len]
            # mean over heads, then mean over drug atoms (preserves E124/S125 signal)
            if len(att.shape) == 4:
                att_avg_heads = att.mean(dim=1)      # [batch, drug_len, protein_len]
                att_protein = att_avg_heads.mean(dim=1)  # [batch, protein_len]
            elif len(att.shape) == 3:
                att_protein = att.mean(dim=1)
            else:
                att_protein = att

            for i in range(batch_size):
                # actualproteinlength padding
                att_i = att_protein[i, :protein_len].cpu().numpy()
                label_i = label[i].item()

                # class group
                if label_i == 0:
                    class_0_attentions.append(att_i)
                else:
                    class_1_attentions.append(att_i)

    print(f"✅ done！")
    print(f" Class 0 (active): {len(class_0_attentions)} drug")
    print(f" Class 1 (active): {len(class_1_attentions)} drug")

    return class_0_attentions, class_1_attentions


def analyze_and_visualize(class_0_att, class_1_att, protein_seq, output_dir):
    """analysisvisualizationattentiondifference"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # convert numpy arrays
    class_0_matrix = np.array(class_0_att)  # [n_inactive, protein_len]
    class_1_matrix = np.array(class_1_att)  # [n_active, protein_len]

    protein_len = class_0_matrix.shape[1]

    # computeaverageattention
    class_0_mean = class_0_matrix.mean(axis=0)
    class_1_mean = class_1_matrix.mean(axis=0)

    # computestd
    class_0_std = class_0_matrix.std(axis=0)
    class_1_std = class_1_matrix.std(axis=0)

    # computedifference
    difference = class_1_mean - class_0_mean

    # statisticst-test
    p_values = np.zeros(protein_len)
    for i in range(protein_len):
        _, p_values[i] = stats.ttest_ind(class_1_matrix[:, i], class_0_matrix[:, i])

    # significantdifferenceresidue (p < 0.05)
    significant_indices = np.where(p_values < 0.05)[0]

    print(f"\n📊 statisticsanalysis:")
    print(f" significantdifferenceresidue: {len(significant_indices)} / {protein_len}")
    print(f"   significancethreshold: p < 0.05")

    # === 1: comparisonaverageattention ===
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))

    # 1: Class 0 vs Class 1 averageattention
    ax = axes[0]
    x = np.arange(protein_len)

    ax.plot(x, class_0_mean, label='Class 0 (active)', color='blue', linewidth=1.5, alpha=0.7)
    ax.fill_between(x, class_0_mean - class_0_std, class_0_mean + class_0_std,
                     color='blue', alpha=0.2)

    ax.plot(x, class_1_mean, label='Class 1 (active)', color='red', linewidth=1.5, alpha=0.7)
    ax.fill_between(x, class_1_mean - class_1_std, class_1_mean + class_1_std,
                     color='red', alpha=0.2)

    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Mean Attention Score', fontsize=12)
    ax.set_title('Average Attention: Active vs Inactive Compounds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2: difference (Class 1 - Class 0)
    ax = axes[1]
    colors = ['red' if d > 0 else 'blue' for d in difference]
    ax.bar(x, difference, color=colors, alpha=0.6, width=1.0)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # significantdifferenceresidue
    for idx in significant_indices:
        ax.axvline(x=idx, color='orange', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('Attention Difference (Active - Inactive)', fontsize=12)
    ax.set_title('Attention Difference (Red: Active > Inactive, Blue: Inactive > Active)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 3: P-values (significance)
    ax = axes[2]
    ax.bar(x, -np.log10(p_values), color='green', alpha=0.6, width=1.0)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2,
               label='Significance threshold (p=0.05)')
    ax.set_xlabel('Residue Position', fontsize=12)
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title('Statistical Significance of Differences', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    comparison_file = output_dir / 'attention_comparison.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ comparisonsave: {comparison_file}")
    plt.close()

    # === 2: ===
    fig, axes = plt.subplots(3, 1, figsize=(20, 8))

    # data
    heatmap_data = np.vstack([
        class_0_mean,
        class_1_mean,
        difference
    ])

    # 1: Class 0
    sns.heatmap(class_0_mean.reshape(1, -1), cmap='Blues', cbar_kws={'label': 'Attention'},
                ax=axes[0], vmin=0, vmax=max(class_0_mean.max(), class_1_mean.max()))
    axes[0].set_ylabel('Class 0\n(active)', fontsize=12)
    axes[0].set_title('Class 0 (Inactive Compounds) - Average Attention', fontsize=14, fontweight='bold')
    axes[0].set_xticks([])

    # 2: Class 1
    sns.heatmap(class_1_mean.reshape(1, -1), cmap='Reds', cbar_kws={'label': 'Attention'},
                ax=axes[1], vmin=0, vmax=max(class_0_mean.max(), class_1_mean.max()))
    axes[1].set_ylabel('Class 1\n(active)', fontsize=12)
    axes[1].set_title('Class 1 (Active Compounds) - Average Attention', fontsize=14, fontweight='bold')
    axes[1].set_xticks([])

    # 3: difference
    vmax_diff = max(abs(difference.min()), abs(difference.max()))
    sns.heatmap(difference.reshape(1, -1), cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Difference'}, ax=axes[2],
                vmin=-vmax_diff, vmax=vmax_diff)
    axes[2].set_ylabel('Difference\n(1 - 0)', fontsize=12)
    axes[2].set_xlabel('Residue Position', fontsize=12)
    axes[2].set_title('Attention Difference (Active - Inactive)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    heatmap_file = output_dir / 'attention_heatmap.png'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"✓ save: {heatmap_file}")
    plt.close()

    # === 3: Top significantresidue ===
    # differencemax 20 residue
    top_indices = np.argsort(np.abs(difference))[-20:][::-1]

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(top_indices))
    colors_top = ['red' if difference[i] > 0 else 'blue' for i in top_indices]

    bars = ax.barh(y_pos, difference[top_indices], color=colors_top, alpha=0.7)

    # residue
    labels = []
    for idx in top_indices:
        aa = protein_seq[idx] if idx < len(protein_seq) else 'X'
        p_val = p_values[idx]
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
        labels.append(f'{aa}{idx+1} {sig}')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Attention Difference (Active - Inactive)', fontsize=12)
    ax.set_title('Top 20 Residues with Largest Attention Differences\n(Red: More important for active, Blue: More important for inactive)',
                 fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    top_residues_file = output_dir / 'top_differential_residues.png'
    plt.savefig(top_residues_file, dpi=300, bbox_inches='tight')
    print(f"✓ Top residuesave: {top_residues_file}")
    plt.close()

    # === savedata ===
    # savedata
    results_df = pd.DataFrame({
        'Position': np.arange(1, protein_len + 1),
        'Residue': list(protein_seq[:protein_len]),
        'Class_0_Mean': class_0_mean,
        'Class_0_Std': class_0_std,
        'Class_1_Mean': class_1_mean,
        'Class_1_Std': class_1_std,
        'Difference': difference,
        'P_value': p_values,
        'Significant': p_values < 0.05
    })

    results_file = output_dir / 'attention_analysis_results.csv'
    results_df.to_csv(results_file, index=False)
    print(f"✓ resultsave: {results_file}")

    # savesignificantresidue
    significant_df = results_df[results_df['Significant']]
    significant_df = significant_df.sort_values('Difference', key=abs, ascending=False)

    sig_file = output_dir / 'significant_residues.csv'
    significant_df.to_csv(sig_file, index=False)
    print(f"✓ significantresiduesave: {sig_file}")

    # === generatereport ===
    report_file = output_dir / 'ANALYSIS_REPORT.md'
    with open(report_file, 'w') as f:
        f.write("# Class-specific Attention Analysis Report\n")
        f.write("# active vs activeattentionanalysisreport\n\n")
        f.write(f"**analysis**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## 📊 \n\n")
        f.write(f"- **Class 0 (active)**: {len(class_0_att)} \n")
        f.write(f"- **Class 1 (active)**: {len(class_1_att)} \n")
        f.write(f"- **proteinlength**: {protein_len} amino acid\n")
        f.write(f"- **significantdifferenceresidue**: {len(significant_indices)} (p < 0.05)\n\n")

        f.write("---\n\n")

        f.write("## 🎯 Top 10 activeresidue\n\n")
        f.write("(Attention: Active > Inactive)\n\n")
        f.write("| Rank | Position | Residue | Difference | P-value | Significance |\n")
        f.write("|------|----------|---------|------------|---------|-------------|\n")

        active_preferred = significant_df[significant_df['Difference'] > 0].head(10)
        for rank, (_, row) in enumerate(active_preferred.iterrows(), 1):
            sig_stars = '***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*'
            f.write(f"| {rank} | {int(row['Position'])} | {row['Residue']}{int(row['Position'])} | "
                   f"{row['Difference']:.4f} | {row['P_value']:.2e} | {sig_stars} |\n")

        f.write("\n---\n\n")

        f.write("## 🎯 Top 10 activeresidue\n\n")
        f.write("(Attention: Inactive > Active)\n\n")
        f.write("| Rank | Position | Residue | Difference | P-value | Significance |\n")
        f.write("|------|----------|---------|------------|---------|-------------|\n")

        inactive_preferred = significant_df[significant_df['Difference'] < 0].head(10)
        for rank, (_, row) in enumerate(inactive_preferred.iterrows(), 1):
            sig_stars = '***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*'
            f.write(f"| {rank} | {int(row['Position'])} | {row['Residue']}{int(row['Position'])} | "
                   f"{row['Difference']:.4f} | {row['P_value']:.2e} | {sig_stars} |\n")

        f.write("\n---\n\n")

        f.write("## 📈 statisticsanalysis\n\n")
        f.write(f"- **significantdifferenceresidueratio**: {len(significant_indices)/protein_len*100:.1f}%\n")
        f.write(f"- **average p-value**: {p_values.mean():.4f}\n")
        f.write(f"- **min p-value**: {p_values.min():.2e}\n\n")

        f.write("---\n\n")

        f.write("## 📁 generatefile\n\n")
        f.write("- `attention_comparison.png` - comparisonvalue±stddifferencep-values\n")
        f.write("- `attention_heatmap.png` - Class 0, Class 1, Difference\n")
        f.write("- `top_differential_residues.png` - Top 20 differencemaxresidue\n")
        f.write("- `attention_analysis_results.csv` - data\n")
        f.write("- `significant_residues.csv` - significantresiduecolumn\n\n")

    print(f"✓ analysisreportsave: {report_file}")

    return results_df, significant_df


def main():
    args = parse_args()

    print("=" * 80)
    print("🧬 GHSR Class-specific Attention Analysis")
    print(" analysisactiveactiveattentiondifference")
    print("=" * 80)
    print()

    # setdevice
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  device: {device}")
    print()

    # model
    model, cfg = load_model(args.config, args.model_path, device)
    print()

    # readdata
    print(f"📊 readdata: {args.data_file}")
    df = pd.read_csv(args.data_file)

    print(f" : {len(df)}")
    print(f" Class 0 (active): {len(df[df['Y']==0])}")
    print(f"   Class 1 (active): {len(df[df['Y']==1])}")

    # protein sequence
    protein_seq = df['Protein'].iloc[0]
    protein_len = len(protein_seq)
    print(f"   proteinlength: {protein_len} aa")
    print()

    # createdata
    use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
    use_drug_features = cfg.DRUG.get("USE_FEATURES", False)
    use_protein_features = cfg.PROTEIN.get("USE_BILSTM", False)
    max_drug_length = cfg.DRUG.get("MAX_DRUG_LENGTH", 200)

    selfies_vocab = None
    if use_drug_bilstm:
        print(f"🔧 SELFIES vocabulary...")
        smiles_list = df['SMILES'].tolist()
        selfies_vocab = build_selfies_vocab(smiles_list, max_vocab_size=cfg.DRUG.get("VOCAB_SIZE", 100))
        print(f"   ✓ Vocabulary size: {len(selfies_vocab)}")

    dataset = DTIDataset(
        df.index.values,
        df,
        use_features=use_protein_features,
        use_selfies=use_drug_bilstm,
        selfies_vocab=selfies_vocab,
        max_drug_length=max_drug_length,
        use_drug_features=use_drug_features
    )

    collate_fn = collate_selfies_fn if use_drug_bilstm else graph_collate_func
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )

    # extractattentionscore
    class_0_att, class_1_att = extract_attention_by_class(
        model, dataloader, device, protein_len, use_drug_bilstm
    )

    # analysisvisualization
    print()
    results_df, significant_df = analyze_and_visualize(
        class_0_att, class_1_att, protein_seq, args.output_dir
    )

    print()
    print("=" * 80)
    print("✅ analysisdone！")
    print("=" * 80)
    print()
    print(f"📁 outputdirectory: {args.output_dir}/")
    print()
    print("📊 significantresidue:")
    print()
    print(significant_df.head(10).to_string(index=False))
    print()


if __name__ == "__main__":
    main()
