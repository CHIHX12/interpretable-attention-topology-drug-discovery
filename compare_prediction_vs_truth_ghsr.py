#!/usr/bin/env python3
"""
comparisonmodelprediction vs. pocket
Compare Model Predictions vs. True Binding Pocket


validate DrugBAN_BiLSTM modelattentionpocketresidue


1. Precision: predictionpocketpocketratio
2. Recall: pocketpredictionratio
3. F1-score: Precision Recall average
4. Enrichment Factor: 
5. Top-K Accuracy: Top K predictionpocketresidue
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.stats import fisher_exact, hypergeom

def parse_args():
    parser = argparse.ArgumentParser(description="Compare predicted vs. true binding pocket")
    parser.add_argument('--predicted', type=str,
                        default='datasets/GPCR_resarch/consensus_results/GHSR_consensus_residues.csv',
                        help='Path to predicted consensus residues CSV')
    parser.add_argument('--true_pocket', type=str,
                        default='datasets/GPCR_resarch/validation_results/GHSR_true_pocket.txt',
                        help='Path to true pocket residues file')
    parser.add_argument('--output_dir', type=str,
                        default='datasets/GPCR_resarch/validation_results',
                        help='Output directory for comparison results')
    parser.add_argument('--protein_length', type=int, default=523,
                        help='Protein sequence length')
    parser.add_argument('--top_k', type=int, nargs='+', default=[10, 20, 30, 50],
                        help='Top K values to evaluate')
    return parser.parse_args()

def load_true_pocket(pocket_file):
    """pocketresidue"""
    print(f"📖 pocket: {pocket_file}")

    true_pocket = []

    with open(pocket_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line or line.startswith('='):
                continue

            # residuerow
            if 'pocketresidue:' in line or 'Residues:' in line:
                # rowresidue
                continue
            elif ',' in line and not line.startswith('PDB'):
                # residue
                parts = line.split(',')
                for part in parts:
                    part = part.strip()
                    if part.isdigit():
                        true_pocket.append(int(part))

    # successmethod
    if len(true_pocket) == 0:
        with open(pocket_file, 'r') as f:
            content = f.read()
            # sequence
            import re
            numbers = re.findall(r'\b\d{1,3}\b', content)
            # filterresiduepercentage
            for num in numbers:
                num_int = int(num)
                if 1 <= num_int <= 1000: # residuerange
                    true_pocket.append(num_int)

    true_pocket = sorted(list(set(true_pocket)))

    print(f" ✓ pocketresidue: {len(true_pocket)}")
    print(f" ✓ pocketresidue: {true_pocket}")

    return true_pocket

def load_predicted_residues(predicted_file):
    """predictionconsensusresidue"""
    print(f"\n📖 predictionresidue: {predicted_file}")

    df = pd.read_csv(predicted_file)

    print(f" ✓ predictionresidue: {len(df)}")
    print(f"   ✓ frequencyrange: [{df['Frequency'].min():.4f}, {df['Frequency'].max():.4f}]")

    # frequencysortsort
    df = df.sort_values('Frequency', ascending=False).reset_index(drop=True)

    return df

def calculate_metrics(predicted_set, true_set, protein_len):
    """
    compute

    Args:
        predicted_set: set of predicted residues
        true_set: set of true pocket residues
        protein_len: total protein length

    Returns:
        dict with metrics
    """
    # 
    tp = len(predicted_set & true_set)  # True Positive
    fp = len(predicted_set - true_set)  # False Positive
    fn = len(true_set - predicted_set)  # False Negative
    tn = protein_len - tp - fp - fn     # True Negative

    # Precision = TP / (TP + FP)
    precision = tp / len(predicted_set) if len(predicted_set) > 0 else 0

    # Recall = TP / (TP + FN)
    recall = tp / len(true_set) if len(true_set) > 0 else 0

    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Enrichment factor
    # 
    expected_overlap = (len(predicted_set) * len(true_set)) / protein_len
    enrichment = tp / expected_overlap if expected_overlap > 0 else 0

    # statisticssignificance (Fisher's exact test)
    # 2x2 column
    # [[TP, FN],
    #  [FP, TN]]
    oddsratio, pvalue = fisher_exact([[tp, fn], [fp, tn]], alternative='greater')

    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'enrichment': enrichment,
        'p_value': pvalue,
        'n_predicted': len(predicted_set),
        'n_true': len(true_set),
        'n_overlap': tp
    }

def evaluate_top_k(predicted_df, true_pocket, protein_len, top_k_values):
    """ Top-K """

    print(f"\n📊 Top-K :")
    print(f" K value: {top_k_values}")
    print()

    results = []

    for k in top_k_values:
        if k > len(predicted_df):
            k = len(predicted_df)

        top_k_positions = set(predicted_df.iloc[:k]['Position'].astype(int).tolist())
        true_set = set(true_pocket)

        metrics = calculate_metrics(top_k_positions, true_set, protein_len)

        results.append({
            'K': k,
            **metrics
        })

        print(f"   Top-{k:3d}:")
        print(f"      Overlap: {metrics['n_overlap']:2d}/{metrics['n_true']} ({metrics['recall']*100:5.1f}% recall)")
        print(f"      Precision: {metrics['precision']*100:5.1f}%  |  Recall: {metrics['recall']*100:5.1f}%")
        print(f"      F1-score: {metrics['f1_score']:.4f}  |  Enrichment: {metrics['enrichment']:.2f}x")
        print(f"      P-value: {metrics['p_value']:.2e}")
        print()

    return pd.DataFrame(results)

def plot_performance_vs_k(results_df, output_dir):
    """ K """

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('GHSR Pocket Prediction Performance vs. Top-K',
                 fontsize=16, fontweight='bold')

    k_values = results_df['K'].values

    # Precision
    ax = axes[0, 0]
    ax.plot(k_values, results_df['precision'] * 100, marker='o', linewidth=2,
            markersize=8, color='#3498DB')
    ax.set_xlabel('Top-K Predictions', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
    ax.set_title('Precision vs. K', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # Recall
    ax = axes[0, 1]
    ax.plot(k_values, results_df['recall'] * 100, marker='s', linewidth=2,
            markersize=8, color='#E74C3C')
    ax.set_xlabel('Top-K Predictions', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recall (%)', fontsize=11, fontweight='bold')
    ax.set_title('Recall vs. K', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 100)

    # F1-score
    ax = axes[1, 0]
    ax.plot(k_values, results_df['f1_score'], marker='^', linewidth=2,
            markersize=8, color='#2ECC71')
    ax.set_xlabel('Top-K Predictions', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('F1-Score vs. K', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1)

    # Enrichment
    ax = axes[1, 1]
    ax.plot(k_values, results_df['enrichment'], marker='D', linewidth=2,
            markersize=8, color='#F39C12')
    ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Random (1x)')
    ax.set_xlabel('Top-K Predictions', fontsize=11, fontweight='bold')
    ax.set_ylabel('Enrichment Factor', fontsize=11, fontweight='bold')
    ax.set_title('Enrichment vs. K', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)

    plt.tight_layout()

    plot_file = Path(output_dir) / "GHSR_performance_vs_k.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / "GHSR_performance_vs_k.pdf", bbox_inches='tight')
    plt.close()

    print(f" ✓ : {plot_file}")

def visualize_overlap(predicted_df, true_pocket, protein_len, output_dir, top_k=30):
    """visualizationprediction vs. pocketoverlap"""

    top_k_positions = set(predicted_df.iloc[:top_k]['Position'].astype(int).tolist())
    true_set = set(true_pocket)

    # createvector
    pred_vector = np.zeros(protein_len)
    true_vector = np.zeros(protein_len)

    for pos in top_k_positions:
        if pos < protein_len:
            pred_vector[pos] = 1

    for pos in true_set:
        if pos < protein_len:
            true_vector[pos] = 1

    # createclassificationvector：TP, FP, FN
    overlap_vector = np.zeros(protein_len)
    for i in range(protein_len):
        if pred_vector[i] == 1 and true_vector[i] == 1:
            overlap_vector[i] = 3 # TP - 
        elif pred_vector[i] == 1 and true_vector[i] == 0:
            overlap_vector[i] = 2 # FP - 
        elif pred_vector[i] == 0 and true_vector[i] == 1:
            overlap_vector[i] = 1 # FN - 

    fig, axes = plt.subplots(3, 1, figsize=(16, 8))
    fig.suptitle(f'GHSR Binding Pocket: Predicted (Top-{top_k}) vs. True',
                 fontsize=16, fontweight='bold')

    # pocket
    ax = axes[0]
    true_matrix = true_vector.reshape(1, -1)
    ax.imshow(true_matrix, aspect='auto', cmap='Reds',
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_ylabel('True Pocket\n(PDB 8JSR)', fontsize=10, fontweight='bold',
                  rotation=0, ha='right', va='center')
    ax.set_yticks([])
    ax.set_xticklabels([])

    # predictionpocket
    ax = axes[1]
    pred_matrix = pred_vector.reshape(1, -1)
    ax.imshow(pred_matrix, aspect='auto', cmap='Blues',
              vmin=0, vmax=1, interpolation='nearest')
    ax.set_ylabel(f'Predicted\n(Top-{top_k})', fontsize=10, fontweight='bold',
                  rotation=0, ha='right', va='center')
    ax.set_yticks([])
    ax.set_xticklabels([])

    # overlapclassification
    ax = axes[2]
    from matplotlib.colors import ListedColormap
    colors = ['white', 'lightgray', 'yellow', 'green']  # Background, FN, FP, TP
    cmap = ListedColormap(colors)

    overlap_matrix = overlap_vector.reshape(1, -1)
    im = ax.imshow(overlap_matrix, aspect='auto', cmap=cmap,
                   vmin=0, vmax=3, interpolation='nearest')
    ax.set_ylabel('Overlap', fontsize=10, fontweight='bold',
                  rotation=0, ha='right', va='center')
    ax.set_yticks([])
    ax.set_xlabel('Residue Position', fontsize=12, fontweight='bold')

    # 
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='True Positive (TP)'),
        Patch(facecolor='yellow', edgecolor='black', label='False Positive (FP)'),
        Patch(facecolor='lightgray', edgecolor='black', label='False Negative (FN)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, ncol=3)

    plt.tight_layout()

    overlap_file = Path(output_dir) / f"GHSR_overlap_top{top_k}.png"
    plt.savefig(overlap_file, dpi=300, bbox_inches='tight')
    plt.savefig(Path(output_dir) / f"GHSR_overlap_top{top_k}.pdf", bbox_inches='tight')
    plt.close()

    print(f"   ✓ overlapvisualization: {overlap_file}")

def generate_comparison_pymol_script(predicted_df, true_pocket, output_dir, top_k=30):
    """generate PyMOL comparisonprediction vs. pocket

    important: convert dataset positions (0-indexed) PDB residue numbers
    Mapping: pdb_residue = dataset_position + 2
    """

    # Dataset position to PDB residue number offset
    OFFSET = 2  # pdb_residue = dataset_position + 2

    top_k_positions = set(predicted_df.iloc[:top_k]['Position'].astype(int).tolist())
    true_set = set(true_pocket)

    tp_positions = top_k_positions & true_set # 
    fp_positions = top_k_positions - true_set # 
    fn_positions = true_set - top_k_positions # 

    # convert PDB residue numbers for PyMOL
    tp_pdb = sorted([p + OFFSET for p in tp_positions])
    fp_pdb = sorted([p + OFFSET for p in fp_positions])
    fn_pdb = sorted([p + OFFSET for p in fn_positions])
    true_pdb = sorted([p + OFFSET for p in true_set])

    script_file = Path(output_dir) / "compare_prediction_vs_truth.pml"

    with open(script_file, 'w') as f:
        f.write("# PyMOL script to compare predicted vs. true binding pocket\n")
        f.write("# Compatible with both Linux and Windows\n")
        f.write("#\n")
        f.write("# Usage:\n")
        f.write("#   Linux:   pymol datasets/GPCR_resarch/validation_results/compare_prediction_vs_truth.pml\n")
        f.write("#   Windows: 1. cd C:\\path\\to\\DrugBAN_BiLSTM\n")
        f.write("#            2. pymol datasets/GPCR_resarch/validation_results/compare_prediction_vs_truth.pml\n")
        f.write("#\n")
        f.write("# Color coding:\n")
        f.write("#   Green:  True Positive (TP) - Correctly predicted pocket residues\n")
        f.write("#   Yellow: False Positive (FP) - Predicted but not in true pocket\n")
        f.write("#   White:  False Negative (FN) - True pocket but not predicted\n")
        f.write("#   Cyan:   Ligand (Anamorelin)\n")
        f.write("#\n")
        f.write(f"# Statistics (Top-{top_k}):\n")
        f.write(f"#   True Positives:  {len(tp_positions)}\n")
        f.write(f"#   False Positives: {len(fp_positions)}\n")
        f.write(f"#   False Negatives: {len(fn_positions)}\n")
        precision = len(tp_positions) / top_k if top_k > 0 else 0
        recall = len(tp_positions) / len(true_set) if len(true_set) > 0 else 0
        f.write(f"#   Precision: {precision:.2%}\n")
        f.write(f"#   Recall:    {recall:.2%}\n")
        f.write("#\n\n")

        f.write("# Set working directory (PyMOL will use paths relative to script location)\n")
        f.write("# For Windows: Modify this path to your actual installation directory\n")
        f.write("# cd C:/Users/YourName/800milion_GPR/predict_affi/New_Docking/network/DrugBAN_BiLSTM\n\n")

        f.write("# Load PDB structure (using forward slashes works on both Linux and Windows)\n")
        f.write("load datasets/GPCR_resarch/GSHR_PDB/8JSR_R.pdb, GHSR\n\n")

        f.write("# Basic setup\n")
        f.write("hide everything\n")
        f.write("show cartoon, GHSR\n")
        f.write("color grey80, GHSR\n")
        f.write("set cartoon_transparency, 0.3\n\n")

        # - use PDB residue numbers
        if len(tp_pdb) > 0:
            tp_list = '+'.join([str(p) for p in tp_pdb])
            f.write("# True Positive: Correctly predicted pocket residues (PDB numbering)\n")
            f.write(f"select tp, GHSR and resi {tp_list}\n")
            f.write("show sticks, tp\n")
            f.write("color green, tp\n")
            f.write("util.cnc tp\n\n")

        # - use PDB residue numbers
        if len(fp_pdb) > 0:
            fp_list = '+'.join([str(p) for p in fp_pdb])
            f.write("# False Positive: Predicted but not in true pocket (PDB numbering)\n")
            f.write(f"select fp, GHSR and resi {fp_list}\n")
            f.write("show sticks, fp\n")
            f.write("color yellow, fp\n")
            f.write("util.cnc fp\n\n")

        # /- use PDB residue numbers
        if len(fn_pdb) > 0:
            fn_list = '+'.join([str(p) for p in fn_pdb])
            f.write("# False Negative: True pocket but not predicted (PDB numbering)\n")
            f.write(f"select fn, GHSR and resi {fn_list}\n")
            f.write("show sticks, fn\n")
            f.write("color white, fn\n")
            f.write("util.cnc fn\n\n")

        # displayligand
        f.write("# Show ligand\n")
        f.write("show sticks, organic\n")
        f.write("color cyan, organic\n")
        f.write("util.cnc organic\n\n")

        # label
        if len(tp_positions) > 0:
            f.write("# Labels for True Positives\n")
            f.write("label tp and name CA, \"%s%s\" % (resn, resi)\n")
            f.write("set label_size, -0.5\n")
            f.write("set label_color, green\n\n")

        # - use PDB residue numbers
        true_list = '+'.join([str(p) for p in true_pdb])
        f.write(f"select true_pocket, GHSR and resi {true_list}\n")
        f.write("zoom true_pocket\n")
        f.write("orient\n\n")

        # statistics
        f.write("# Print statistics\n")
        f.write(f"print 'True Pocket: {len(true_set)} residues'\n")
        f.write(f"print 'Predicted (Top-{top_k}): {len(top_k_positions)} residues'\n")
        f.write(f"print 'True Positive (TP): {len(tp_positions)} residues'\n")
        f.write(f"print 'False Positive (FP): {len(fp_positions)} residues'\n")
        f.write(f"print 'False Negative (FN): {len(fn_positions)} residues'\n")
        precision = len(tp_positions) / len(top_k_positions) if len(top_k_positions) > 0 else 0
        recall = len(tp_positions) / len(true_set) if len(true_set) > 0 else 0
        f.write(f"print 'Precision: {precision:.2%}'\n")
        f.write(f"print 'Recall: {recall:.2%}'\n")

    print(f" ✓ PyMOL comparison: {script_file}")
    print(f" row: pymol {script_file}")

def save_validation_report(results_df, predicted_df, true_pocket, protein_len, output_dir):
    """generatevalidatereport"""

    report_file = Path(output_dir) / "GHSR_validation_report.txt"

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GHSR pocketpredictionvalidatereport\n")
        f.write("Growth Hormone Secretagogue Receptor Binding Pocket Validation Report\n")
        f.write("="*80 + "\n\n")

        f.write("1. data\n")
        f.write("-" * 40 + "\n")
        f.write(f"proteinlength: {protein_len} aa\n")
        f.write(f"pocketresidue: {len(true_pocket)}\n")
        f.write(f"predictionresidue: {len(predicted_df)}\n\n")

        f.write("pocketresidue:\n")
        f.write(f"  {true_pocket}\n\n")

        f.write("2. Top-K \n")
        f.write("-" * 40 + "\n")
        f.write(f"{'K':>5}  {'Precision':>10}  {'Recall':>10}  {'F1-Score':>10}  {'Enrichment':>12}  {'P-value':>10}\n")
        for _, row in results_df.iterrows():
            f.write(f"{int(row['K']):5d}  "
                    f"{row['precision']*100:9.2f}%  "
                    f"{row['recall']*100:9.2f}%  "
                    f"{row['f1_score']:10.4f}  "
                    f"{row['enrichment']:11.2f}x  "
                    f"{row['p_value']:10.2e}\n")
        f.write("\n")

        # analysis K value
        best_k_idx = results_df['f1_score'].idxmax()
        best_k_row = results_df.iloc[best_k_idx]

        f.write("3. (F1-Score)\n")
        f.write("-" * 40 + "\n")
        f.write(f"Top-K: {int(best_k_row['K'])}\n")
        f.write(f"Precision: {best_k_row['precision']*100:.2f}%\n")
        f.write(f"Recall: {best_k_row['recall']*100:.2f}%\n")
        f.write(f"F1-Score: {best_k_row['f1_score']:.4f}\n")
        f.write(f"Enrichment: {best_k_row['enrichment']:.2f}x\n")
        f.write(f"P-value: {best_k_row['p_value']:.2e}\n")
        f.write(f"Overlap: {best_k_row['n_overlap']}/{len(true_pocket)} true pocket residues\n\n")

        # comparison
        k = int(best_k_row['K'])
        random_expected = (k * len(true_pocket)) / protein_len

        f.write("4. comparison\n")
        f.write("-" * 40 + "\n")
        f.write(f" Top-{k} : {random_expected:.2f}\n")
        f.write(f"modelactual: {best_k_row['n_overlap']}\n")
        f.write(f": {best_k_row['enrichment']:.2f}x\n\n")

        # Top predictionresidue
        f.write("5. Top 20 predictionresidue\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Rank':>4}  {'Position':>8}  {'Frequency':>10}  {'In True Pocket':>15}\n")
        for i, row in predicted_df.head(20).iterrows():
            pos = int(row['Position'])
            in_pocket = "✓ YES" if pos in true_pocket else "  No"
            f.write(f"{i+1:4d}  {pos:8d}  {row['Frequency']*100:9.2f}%  {in_pocket:>15}\n")
        f.write("\n")

        # 
        f.write("6. result\n")
        f.write("-" * 40 + "\n")

        if best_k_row['precision'] > 0.5 and best_k_row['recall'] > 0.5:
            f.write("✅ modelsuccesspocket\n\n")
            f.write("modelattentiondrug-proteinkeyregion\n")
            f.write("validateattentionmodelpredictiondrug\n")
            f.write("binding site\n")
        elif best_k_row['precision'] > 0.3 and best_k_row['recall'] > 0.3:
            f.write("⚠️ intermediatemodelpartialpocket\n\n")
            f.write("modelpartialpocketregion:\n")
            f.write("1. drug\n")
            f.write("2. attentionfunctionregion\n")
            f.write("3. modelfunctionregion\n")
        else:
            f.write("❌ modelpocket\n\n")
            f.write(":\n")
            f.write("1. sequencealignPDB vs. datasequence\n")
            f.write("2. datadrug\n")
            f.write("3. attention\n")

        f.write("\n")
        f.write("="*80 + "\n")
        f.write("reportgenerate: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

    print(f"   ✓ validatereport: {report_file}")

def main():
    args = parse_args()

    print("="*80)
    print("🧬 GHSR pocketpredictionvalidatemodelprediction vs. structure")
    print("   Validation: Model Prediction vs. Crystal Structure")
    print("="*80)
    print()

    # checkfile
    if not os.path.exists(args.predicted):
        print(f"❌ predictionfile: {args.predicted}")
        sys.exit(1)

    if not os.path.exists(args.true_pocket):
        print(f"❌ pocketfile: {args.true_pocket}")
        sys.exit(1)

    # data
    true_pocket = load_true_pocket(args.true_pocket)
    predicted_df = load_predicted_residues(args.predicted)

    # Top-K 
    results_df = evaluate_top_k(predicted_df, true_pocket, args.protein_length, args.top_k)

    # createoutputdirectory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 savevalidateresult: {args.output_dir}")

    # saveresult
    results_file = output_path / "GHSR_validation_metrics.csv"
    results_df.to_csv(results_file, index=False)
    print(f" ✓ : {results_file}")

    # generatevisualization
    plot_performance_vs_k(results_df, output_path)
    visualize_overlap(predicted_df, true_pocket, args.protein_length, output_path, top_k=30)

    # generate PyMOL 
    generate_comparison_pymol_script(predicted_df, true_pocket, output_path, top_k=30)

    # generatevalidatereport
    save_validation_report(results_df, predicted_df, true_pocket, args.protein_length, output_path)

    # 
    print("\n" + "="*80)
    print("✅ validatedone！")
    print("="*80)
    print()

    # 
    best_k_idx = results_df['f1_score'].idxmax()
    best_k_row = results_df.iloc[best_k_idx]

    print(f"🏆 (Top-{int(best_k_row['K'])}):")
    print(f"   Precision:  {best_k_row['precision']*100:6.2f}%")
    print(f"   Recall:     {best_k_row['recall']*100:6.2f}%")
    print(f"   F1-Score:   {best_k_row['f1_score']:6.4f}")
    print(f"   Enrichment: {best_k_row['enrichment']:6.2f}x (vs. random)")
    print(f"   P-value:    {best_k_row['p_value']:.2e}")
    print(f"   Overlap:    {int(best_k_row['n_overlap'])}/{len(true_pocket)} true pocket residues")
    print()

    # 
    if best_k_row['precision'] > 0.5 and best_k_row['recall'] > 0.5:
        print("✅ : modelsuccess GHSR pocket")
        print(" attentiondrug-proteinkeyregion")
    elif best_k_row['precision'] > 0.3 and best_k_row['recall'] > 0.3:
        print("⚠️ : modelpartial GHSR pocket")
        print(" analysisresidue")
    else:
        print("❌ : model GHSR pocket")
        print(" checksequencealigndatamodeltraining")
    print()

if __name__ == "__main__":
    main()
