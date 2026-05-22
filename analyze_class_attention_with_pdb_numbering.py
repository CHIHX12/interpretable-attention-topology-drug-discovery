#!/usr/bin/env python3
"""
Analyze Attention Differences between Active and Inactive Compounds
IMPROVED VERSION with PDB Residue Numbering on X-axis

Key improvements:
1. Pure English labels (no Chinese/Unicode)
2. X-axis shows PDB residue numbers (37-338) - ACTUAL crystal structure numbers
3. Only plots verifiable residues (within PDB range)
4. All residue labels use PDB numbering for easy verification
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
    parser = argparse.ArgumentParser(description="Analyze attention differences with PDB numbering")
    parser.add_argument('--config', type=str, default='configs/DrugBAN_BiLSTM_GHSR_TransferLearning.yaml',
                        help='Path to config file')
    parser.add_argument('--model_path', type=str,
                        default='result/DrugBAN_BiLSTM_GHSR_Seed42/best_model_epoch_44.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--data_file', type=str,
                        default='datasets/GPCR_resarch/GHSR_training_data.csv',
                        help='Path to GHSR data CSV')
    parser.add_argument('--output_dir', type=str,
                        default='result/class_attention_analysis_pdb',
                        help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--no_protein_features', action='store_true',
                        help='Disable protein physicochemical features (for diagnostic/reproducibility)')
    return parser.parse_args()


def load_model(config_path, model_path, device):
    """Load model"""
    print(f"Loading config: {config_path}")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    print(f"Building model...")
    model = DrugBAN(**cfg).to(device)

    print(f"Loading weights: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"Model loaded successfully")
    return model, cfg


def extract_attention_by_class(model, dataloader, device, protein_len, use_drug_bilstm=False):
    """
    Extract attention scores and group by class

    Returns:
        class_0_attentions: list of attention arrays for Y=0
        class_1_attentions: list of attention arrays for Y=1
    """
    class_0_attentions = []
    class_1_attentions = []

    print(f"\nExtracting attention scores...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
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

            # Debug: Print attention shape for first batch
            if batch_idx == 0:
                print(f"\nDEBUG: Attention tensor shape: {att.shape}")

            # Aggregate BAN attention to per-residue protein attention
            # att shape: [batch, heads, drug_atoms, protein_len] (4D)
            #         or [batch, drug_atoms, protein_len] (3D)
            #         or [batch, protein_len] (2D)
            if att.dim() == 4:
                # Average over attention heads → [batch, drug_atoms, protein_len]
                att_heads_avg = att.mean(dim=1)
                # Max over drug atoms → [batch, protein_len]
                att_protein = att_heads_avg.max(dim=1)[0]
            elif att.dim() == 3:
                # Max over drug atoms → [batch, protein_len]
                att_protein = att.max(dim=1)[0]
            else:
                # Already [batch, protein_len]
                att_protein = att

            # Slice to actual protein length (discard padding)
            att_np = att_protein[:, :protein_len].cpu().numpy()  # [batch, protein_len]

            label_np = label.cpu().numpy()

            # Group by class
            for i in range(len(label_np)):
                att_i = att_np[i]  # [protein_len]

                if len(att_i) != protein_len:
                    continue  # Skip malformed samples

                label_i = label_np[i]

                if label_i == 0:
                    class_0_attentions.append(att_i)
                else:
                    class_1_attentions.append(att_i)

    print(f"\nExtraction complete:")
    print(f"   Class 0 (Inactive): {len(class_0_attentions)} compounds")
    print(f"   Class 1 (Active): {len(class_1_attentions)} compounds")

    return class_0_attentions, class_1_attentions


def analyze_and_visualize(class_0_attentions, class_1_attentions, protein_len,
                         protein_seq, output_dir):
    """
    Analyze differences and create visualizations with PDB numbering
    """

    # Convert to numpy arrays with proper shape handling
    # Each element in the list should be a 1D array of length protein_len
    print(f"\n{'='*80}")
    print(f"Preparing Data")
    print(f"{'='*80}")

    # Check and convert to proper numpy arrays
    class_0_list = []
    for i, att in enumerate(class_0_attentions):
        att_array = np.array(att).flatten()  # Ensure 1D
        if len(att_array) != protein_len:
            print(f"Warning: Class 0 sample {i} has length {len(att_array)}, expected {protein_len}")
            continue
        class_0_list.append(att_array)

    class_1_list = []
    for i, att in enumerate(class_1_attentions):
        att_array = np.array(att).flatten()  # Ensure 1D
        if len(att_array) != protein_len:
            print(f"Warning: Class 1 sample {i} has length {len(att_array)}, expected {protein_len}")
            continue
        class_1_list.append(att_array)

    class_0_matrix = np.array(class_0_list)  # (n_samples, protein_len)
    class_1_matrix = np.array(class_1_list)

    print(f"Class 0 matrix shape: {class_0_matrix.shape}")
    print(f"Class 1 matrix shape: {class_1_matrix.shape}")

    if class_0_matrix.ndim != 2 or class_1_matrix.ndim != 2:
        raise ValueError(f"Matrix dimensions incorrect! Class 0: {class_0_matrix.shape}, Class 1: {class_1_matrix.shape}")

    print(f"\n{'='*80}")
    print(f"Statistical Analysis")
    print(f"{'='*80}")

    # Calculate statistics
    class_0_mean = class_0_matrix.mean(axis=0)
    class_0_std = class_0_matrix.std(axis=0)
    class_1_mean = class_1_matrix.mean(axis=0)
    class_1_std = class_1_matrix.std(axis=0)

    # Calculate difference
    difference = class_1_mean - class_0_mean

    # Statistical test (t-test)
    p_values = np.zeros(protein_len)
    for i in range(protein_len):
        try:
            result = stats.ttest_ind(class_1_matrix[:, i], class_0_matrix[:, i], equal_var=False)
            # Handle both old and new scipy versions
            if hasattr(result, 'pvalue'):
                p_values[i] = result.pvalue
            else:
                p_values[i] = result[1]
        except Exception as e:
            # If t-test fails (e.g., identical values), use p=1.0
            p_values[i] = 1.0

    # ========================================================================
    # FILTER TO PDB RANGE ONLY (Dataset positions 36-337 → PDB 37-338)
    # ========================================================================
    pdb_start_dataset = 36  # Dataset position 36 → PDB residue 37
    pdb_end_dataset = 337   # Dataset position 337 → PDB residue 338

    # Create PDB residue numbers array
    pdb_residue_numbers = np.arange(pdb_start_dataset, pdb_end_dataset + 1) + 1  # +1 for PDB numbering

    # Filter all arrays to PDB range
    class_0_mean_pdb = class_0_mean[pdb_start_dataset:pdb_end_dataset+1]
    class_0_std_pdb = class_0_std[pdb_start_dataset:pdb_end_dataset+1]
    class_1_mean_pdb = class_1_mean[pdb_start_dataset:pdb_end_dataset+1]
    class_1_std_pdb = class_1_std[pdb_start_dataset:pdb_end_dataset+1]
    difference_pdb = difference[pdb_start_dataset:pdb_end_dataset+1]
    p_values_pdb = p_values[pdb_start_dataset:pdb_end_dataset+1]

    # Find significantly different residues in PDB range
    significant_mask = p_values_pdb < 0.05
    significant_indices_pdb = pdb_residue_numbers[significant_mask]

    print(f"\nPDB Structure Coverage:")
    print(f"   PDB residue range: {pdb_residue_numbers[0]} - {pdb_residue_numbers[-1]}")
    print(f"   Total verifiable residues: {len(pdb_residue_numbers)}")
    print(f"   Significantly different: {len(significant_indices_pdb)} residues")
    print(f"   Significance threshold: p < 0.05")

    # ========================================================================
    # Figure 1: Comparison plot with PDB numbering
    # ========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))

    # Subplot 1: Class 0 vs Class 1 average attention
    ax = axes[0]

    ax.plot(pdb_residue_numbers, class_0_mean_pdb, label='Class 0 (Inactive)',
            color='blue', linewidth=1.5, alpha=0.7)
    ax.fill_between(pdb_residue_numbers,
                     class_0_mean_pdb - class_0_std_pdb,
                     class_0_mean_pdb + class_0_std_pdb,
                     color='blue', alpha=0.2)

    ax.plot(pdb_residue_numbers, class_1_mean_pdb, label='Class 1 (Active)',
            color='red', linewidth=1.5, alpha=0.7)
    ax.fill_between(pdb_residue_numbers,
                     class_1_mean_pdb - class_1_std_pdb,
                     class_1_mean_pdb + class_1_std_pdb,
                     color='red', alpha=0.2)

    ax.set_xlabel('PDB Residue Number (8JSR Chain R)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Mean Attention Score', fontsize=12)
    ax.set_title('Average Attention: Active vs Inactive Compounds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(pdb_residue_numbers[0], pdb_residue_numbers[-1])

    # Subplot 2: Difference (Class 1 - Class 0)
    ax = axes[1]
    colors = ['red' if d > 0 else 'blue' for d in difference_pdb]
    ax.bar(pdb_residue_numbers, difference_pdb, color=colors, alpha=0.6, width=1.0)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)

    # Mark significantly different residues
    for pdb_res in significant_indices_pdb:
        ax.axvline(x=pdb_res, color='orange', alpha=0.3, linewidth=0.5)

    ax.set_xlabel('PDB Residue Number (8JSR Chain R)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Attention Difference\n(Active - Inactive)', fontsize=12)
    ax.set_title('Attention Difference (Red: Active > Inactive, Blue: Inactive > Active)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(pdb_residue_numbers[0], pdb_residue_numbers[-1])

    # Subplot 3: P-values (significance)
    ax = axes[2]
    ax.bar(pdb_residue_numbers, -np.log10(p_values_pdb), color='green', alpha=0.6, width=1.0)
    ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', linewidth=2,
               label='Significance threshold (p=0.05)')
    ax.set_xlabel('PDB Residue Number (8JSR Chain R)', fontsize=14, fontweight='bold')
    ax.set_ylabel('-log10(p-value)', fontsize=12)
    ax.set_title('Statistical Significance of Differences', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim(pdb_residue_numbers[0], pdb_residue_numbers[-1])

    plt.tight_layout()
    comparison_file = output_dir / 'attention_comparison_pdb_numbering.png'
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {comparison_file}")
    plt.close()

    # ========================================================================
    # Figure 2: Heatmap with PDB numbering
    # ========================================================================
    fig, axes = plt.subplots(3, 1, figsize=(20, 8))

    # Subplot 1: Class 0
    sns.heatmap(class_0_mean_pdb.reshape(1, -1), cmap='Blues', cbar_kws={'label': 'Attention'},
                ax=axes[0], vmin=0, vmax=max(class_0_mean_pdb.max(), class_1_mean_pdb.max()),
                xticklabels=False)
    axes[0].set_ylabel('Class 0\n(Inactive)', fontsize=12)
    axes[0].set_title(f'Class 0 (Inactive) - Average Attention (PDB Residues {pdb_residue_numbers[0]}-{pdb_residue_numbers[-1]})',
                      fontsize=14, fontweight='bold')

    # Subplot 2: Class 1
    sns.heatmap(class_1_mean_pdb.reshape(1, -1), cmap='Reds', cbar_kws={'label': 'Attention'},
                ax=axes[1], vmin=0, vmax=max(class_0_mean_pdb.max(), class_1_mean_pdb.max()),
                xticklabels=False)
    axes[1].set_ylabel('Class 1\n(Active)', fontsize=12)
    axes[1].set_title(f'Class 1 (Active) - Average Attention (PDB Residues {pdb_residue_numbers[0]}-{pdb_residue_numbers[-1]})',
                      fontsize=14, fontweight='bold')

    # Subplot 3: Difference
    vmax_diff = max(abs(difference_pdb.min()), abs(difference_pdb.max()))
    sns.heatmap(difference_pdb.reshape(1, -1), cmap='RdBu_r', center=0,
                cbar_kws={'label': 'Difference'}, ax=axes[2],
                vmin=-vmax_diff, vmax=vmax_diff, xticklabels=False)
    axes[2].set_ylabel('Difference\n(Active - Inactive)', fontsize=12)
    axes[2].set_xlabel(f'PDB Residue Number (8JSR Chain R: {pdb_residue_numbers[0]}-{pdb_residue_numbers[-1]})',
                       fontsize=14, fontweight='bold')
    axes[2].set_title('Attention Difference (Active - Inactive)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    heatmap_file = output_dir / 'attention_heatmap_pdb_numbering.png'
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved: {heatmap_file}")
    plt.close()

    # ========================================================================
    # Figure 3: Top differential residues with PDB numbering
    # ========================================================================
    # Create dataframe for all PDB residues
    df_results_pdb = pd.DataFrame({
        'PDB_Residue': pdb_residue_numbers,
        'Dataset_Position': np.arange(pdb_start_dataset, pdb_end_dataset + 1),
        'Residue': [protein_seq[i] for i in range(pdb_start_dataset, pdb_end_dataset + 1)],
        'Class_0_Mean': class_0_mean_pdb,
        'Class_0_Std': class_0_std_pdb,
        'Class_1_Mean': class_1_mean_pdb,
        'Class_1_Std': class_1_std_pdb,
        'Difference': difference_pdb,
        'P_value': p_values_pdb,
        'Significant': p_values_pdb < 0.05
    })

    # Get top 10 SIGNIFICANT residues by absolute difference
    # CRITICAL: Only include statistically significant residues (p < 0.05)
    df_significant = df_results_pdb[df_results_pdb['Significant']]

    # Select top 10 from significant residues only
    df_top10 = df_significant.nlargest(10, 'Difference', keep='all')
    df_bottom10 = df_significant.nsmallest(10, 'Difference', keep='all')
    df_top_differential = pd.concat([df_bottom10, df_top10]).drop_duplicates()
    df_top_differential = df_top_differential.sort_values('Difference')

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    y_pos = np.arange(len(df_top_differential))
    colors = ['blue' if d < 0 else 'red' for d in df_top_differential['Difference']]

    bars = ax.barh(y_pos, df_top_differential['Difference'], color=colors, alpha=0.7)

    # Add significance stars
    for i, (idx, row) in enumerate(df_top_differential.iterrows()):
        if row['P_value'] < 0.001:
            marker = '***'
        elif row['P_value'] < 0.01:
            marker = '**'
        elif row['P_value'] < 0.05:
            marker = '*'
        else:
            marker = ''

        x_pos = row['Difference'] + (0.02 if row['Difference'] > 0 else -0.02)
        ax.text(x_pos, i, marker, va='center', ha='left' if row['Difference'] > 0 else 'right',
                fontsize=10, fontweight='bold')

    # Y-axis labels with PDB numbering
    labels = [f"{row['Residue']}{row['PDB_Residue']}"
              for idx, row in df_top_differential.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)

    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Difference (Class 1 - Class 0)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Residue (PDB Numbering)', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Active-Preferred vs Top 10 Inactive-Preferred Residues\n(Only Statistically Significant, p < 0.05)  *** p<0.001, ** p<0.01, * p<0.05',
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Active-preferred'),
        Patch(facecolor='blue', alpha=0.7, label='Inactive-preferred')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    top_file = output_dir / 'top_differential_residues_pdb_numbering.png'
    plt.savefig(top_file, dpi=300, bbox_inches='tight')
    print(f"Top differential residues plot saved: {top_file}")
    plt.close()

    # ========================================================================
    # Save results
    # ========================================================================
    # Save all results with PDB numbering
    results_file = output_dir / 'attention_analysis_pdb_numbering.csv'
    df_results_pdb.to_csv(results_file, index=False)
    print(f"\nResults saved: {results_file}")

    # Save only significant residues
    df_significant = df_results_pdb[df_results_pdb['Significant']]
    sig_file = output_dir / 'significant_residues_pdb_numbering.csv'
    df_significant.to_csv(sig_file, index=False)
    print(f"Significant residues saved: {sig_file}")

    # ========================================================================
    # Generate report
    # ========================================================================
    report_file = output_dir / 'ANALYSIS_REPORT_PDB_NUMBERING.md'
    with open(report_file, 'w') as f:
        f.write("# Class-specific Attention Analysis Report\n")
        f.write("# (With PDB Residue Numbering)\n\n")
        f.write(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")

        f.write("## Data Summary\n\n")
        f.write(f"- **Class 0 (Inactive)**: {len(class_0_attentions)} compounds\n")
        f.write(f"- **Class 1 (Active)**: {len(class_1_attentions)} compounds\n")
        f.write(f"- **PDB Structure Range**: Residues {pdb_residue_numbers[0]}-{pdb_residue_numbers[-1]}\n")
        f.write(f"- **Verifiable Residues**: {len(pdb_residue_numbers)}\n\n")

        f.write("---\n\n")
        f.write("## Statistical Results\n\n")
        f.write(f"- **Significantly different residues**: {len(significant_indices_pdb)} / {len(pdb_residue_numbers)}\n")
        f.write(f"- **Percentage**: {len(significant_indices_pdb)/len(pdb_residue_numbers)*100:.1f}%\n")
        f.write(f"- **Significance threshold**: p < 0.05\n\n")

        f.write("---\n\n")
        f.write("## Top 10 Inactive-Preferred Residues (PDB Numbering)\n\n")
        f.write("| PDB Res | AA | Difference | P-value | Significance |\n")
        f.write("|---------|----|-----------|---------|--------------|\n")
        top10_inactive = df_results_pdb.nsmallest(10, 'Difference')
        for idx, row in top10_inactive.iterrows():
            sig_str = '***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*' if row['P_value'] < 0.05 else ''
            f.write(f"| {row['Residue']}{int(row['PDB_Residue'])} | {row['Residue']} | {row['Difference']:.3f} | {row['P_value']:.2e} | {sig_str} |\n")

        f.write("\n---\n\n")
        f.write("## Top 10 Active-Preferred Residues (PDB Numbering)\n\n")
        f.write("| PDB Res | AA | Difference | P-value | Significance |\n")
        f.write("|---------|----|-----------|---------|--------------|\n")
        top10_active = df_results_pdb.nlargest(10, 'Difference')
        for idx, row in top10_active.iterrows():
            sig_str = '***' if row['P_value'] < 0.001 else '**' if row['P_value'] < 0.01 else '*' if row['P_value'] < 0.05 else ''
            f.write(f"| {row['Residue']}{int(row['PDB_Residue'])} | {row['Residue']} | {row['Difference']:.3f} | {row['P_value']:.2e} | {sig_str} |\n")

        f.write("\n---\n\n")
        f.write("## Notes\n\n")
        f.write("- All residue numbers are **PDB residue numbers** from 8JSR Chain R\n")
        f.write("- These numbers can be directly verified in the crystal structure\n")
        f.write("- No conversion needed - what you see is what's in the PDB file\n")
        f.write("- Negative difference: Inactive compounds have higher attention\n")
        f.write("- Positive difference: Active compounds have higher attention\n")
        f.write("- Significance levels: *** p<0.001, ** p<0.01, * p<0.05\n")

    print(f"Report saved: {report_file}")

    print(f"\n{'='*80}")
    print(f"Analysis Complete!")
    print(f"{'='*80}")
    print(f"\nAll plots use PDB residue numbers on the X-axis")
    print(f"Range: PDB {pdb_residue_numbers[0]} - {pdb_residue_numbers[-1]}")
    print(f"These are the ACTUAL residue numbers from crystal structure 8JSR Chain R")
    print(f"No conversion needed for verification!")


def main():
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*80}")
    print(f"Class-Specific Attention Analysis with PDB Numbering")
    print(f"{'='*80}\n")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    model, cfg = load_model(args.config, args.model_path, device)

    # Check encoder types from config
    use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
    use_protein_features = cfg.PROTEIN.get("USE_BILSTM", False)
    # Allow override via --no_protein_features flag
    if getattr(args, 'no_protein_features', False):
        use_protein_features = False
        print("   [OVERRIDE] use_protein_features = False")

    # Load data
    print(f"\nLoading data: {args.data_file}")
    df = pd.read_csv(args.data_file)
    print(f"   Total samples: {len(df)}")
    print(f"   Class 0 (Inactive): {(df['Y'] == 0).sum()}")
    print(f"   Class 1 (Active): {(df['Y'] == 1).sum()}")

    # Get protein sequence
    protein_seq = df['Protein'].iloc[0]
    protein_len = len(protein_seq)
    print(f"   Protein length: {protein_len} amino acids")

    # Create dataset
    selfies_vocab = None
    if use_drug_bilstm:
        # Load SELFIES vocab from model directory, or build from data
        import pickle, os
        model_dir = os.path.dirname(args.model_path)
        vocab_path = os.path.join(model_dir, 'selfies_vocab.pkl')
        if os.path.exists(vocab_path):
            print(f"   Loading SELFIES vocab from {vocab_path}")
            with open(vocab_path, 'rb') as f:
                selfies_vocab = pickle.load(f)
        else:
            print("   Building SELFIES vocab from data...")
            selfies_vocab = build_selfies_vocab(df['SMILES'].tolist())

    dataset = DTIDataset(
        df.index.values,
        df,
        use_features=use_protein_features,
        use_selfies=use_drug_bilstm,
        selfies_vocab=selfies_vocab,
        max_drug_nodes=290,
        max_protein_length=cfg.PROTEIN.get("MAX_PROTEIN_LENGTH", 1200)
    )

    if use_drug_bilstm:
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                               collate_fn=collate_selfies_fn, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                               collate_fn=graph_collate_func, shuffle=False)

    # Extract attention by class
    class_0_attentions, class_1_attentions = extract_attention_by_class(
        model, dataloader, device, protein_len, use_drug_bilstm
    )

    # Analyze and visualize
    analyze_and_visualize(class_0_attentions, class_1_attentions,
                         protein_len, protein_seq, output_dir)


if __name__ == "__main__":
    main()
