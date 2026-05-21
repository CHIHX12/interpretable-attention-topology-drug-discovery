#!/usr/bin/env python3
"""
attentionplotNature Machine Intelligence 
affinity vs affinity
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import argparse
from pathlib import Path
from matplotlib.gridspec import GridSpec


def select_best_cases(results, n_high=5, n_low=5):
    """
 

 
    - affinityY=1, Z>0.7, prediction
    - affinityY=0, Z<0.3, prediction
    """
    y_true = np.array(results['y_true'])
    y_pred = np.array(results['y_pred'])
    z_true = np.array(results['z_true'])
    z_pred = np.array(results['z_pred']) if results['z_pred'][0] is not None else z_true

    # affinityY=1, Z>0.7, prediction
    high_mask = (y_true == 1) & (z_true > 0.7) & (y_pred > 0.5)
    high_indices = np.where(high_mask)[0]

    if len(high_indices) > 0:
        # predictionsortY_pred 1 
        high_confidence = y_pred[high_indices]
        high_sorted = high_indices[np.argsort(-high_confidence)]
        high_selected = high_sorted[:n_high]
    else:
        # 
        high_mask = (y_true == 1) & (y_pred > 0.5)
        high_indices = np.where(high_mask)[0]
        high_selected = high_indices[:n_high] if len(high_indices) > 0 else []

        # affinityY=0, Z<0.3, prediction
    low_mask = (y_true == 0) & (z_true < 0.3) & (y_pred < 0.5)
    low_indices = np.where(low_mask)[0]

    if len(low_indices) > 0:
        # predictionsortY_pred 0 
        low_confidence = y_pred[low_indices]
        low_sorted = low_indices[np.argsort(low_confidence)]
        low_selected = low_sorted[:n_low]
    else:
        # 
        low_mask = (y_true == 0) & (y_pred < 0.5)
        low_indices = np.where(low_mask)[0]
        low_selected = low_indices[:n_low] if len(low_indices) > 0 else []

    print("\n📌 Selected Cases:")
    print(f"   High affinity: {len(high_selected)} cases")
    print(f"   Low affinity: {len(low_selected)} cases")

    return high_selected, low_selected


def plot_single_attention_case(attention, y_true, y_pred, z_true, z_pred,
                                case_idx, case_type, save_path, max_display=200):
    """
                                attentionplot

    Args:
        attention: [drug_tokens, protein_residues] numpy array
        y_true, y_pred: Classification labels and predictions
        z_true, z_pred: Regression targets and predictions
        case_idx: Case index
        case_type: 'High' or 'Low' affinity
        save_path: Output path
        max_display: maxresidueplot
    """
    # 
    if attention.shape[1] > max_display:
        # attention
        avg_att = attention.mean(axis=0)
        top_indices = np.argsort(-avg_att)[:max_display]
        top_indices = np.sort(top_indices) # 
        attention_display = attention[:, top_indices]
        residue_indices = top_indices
    else:
        attention_display = attention
        residue_indices = np.arange(attention.shape[1])

        # createplot
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 1, height_ratios=[4, 1, 1], hspace=0.3)

    # Panel A: attentionplot
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(attention_display, cmap='RdYlBu_r', aspect='auto',
                    interpolation='nearest', vmin=0, vmax=attention_display.max())

                    # color
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=13, fontweight='bold')

    ax1.set_xlabel('Protein Residue Position', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Drug SELFIES Tokens', fontsize=14, fontweight='bold')

    # Title with predictions
    color = 'green' if case_type == 'High' else 'blue'
    title = f'{case_type} Affinity Case #{case_idx}\n'
    title += f'Classification: Y={int(y_true)}, Pred={y_pred:.3f} | '
    title += f'Regression: Z={z_true:.3f}, Pred={z_pred:.3f}'

    ax1.set_title(title, fontsize=13, fontweight='bold', pad=15, color=color)

    # Xaxisactualresidue
    if len(residue_indices) <= 50:
        tick_step = 5
    elif len(residue_indices) <= 100:
        tick_step = 10
    else:
        tick_step = 20

    tick_positions = range(0, len(residue_indices), tick_step)
    tick_labels = [str(residue_indices[i]) for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=9)

    # Yaxis
    if attention_display.shape[0] <= 50:
        ax1.set_yticks(range(0, attention_display.shape[0], 5))
    else:
        ax1.set_yticks(range(0, attention_display.shape[0], 10))
    ax1.tick_params(axis='y', labelsize=9)

    # Panel B: averageattentionresidue
    ax2 = fig.add_subplot(gs[1])
    avg_attention = attention_display.mean(axis=0)
    x_pos = np.arange(len(avg_attention))

    bars = ax2.bar(x_pos, avg_attention, color='steelblue', alpha=0.7, edgecolor='none')

    # Top-5 residue
    top5_indices = np.argsort(-avg_attention)[:5]
    for idx in top5_indices:
        bars[idx].set_color('red')
        bars[idx].set_alpha(0.9)

    ax2.set_xlabel('Residue Position (within displayed region)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average\\nAttention', fontsize=11, fontweight='bold')
    ax2.set_xlim(-0.5, len(avg_attention) - 0.5)
    ax2.set_ylim(0, avg_attention.max() * 1.1)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Top-5 residue
    for idx in top5_indices:
        actual_residue = residue_indices[idx]
        ax2.text(idx, avg_attention[idx], f'{actual_residue}',
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='red')

                # Panel C: averageattentiondrugtoken
    ax3 = fig.add_subplot(gs[2])
    avg_attention_drug = attention_display.mean(axis=1)
    y_pos = np.arange(len(avg_attention_drug))

    bars_drug = ax3.barh(y_pos, avg_attention_drug, color='orange', alpha=0.7, edgecolor='none')

    # Top-5 drugtokens
    top5_drug = np.argsort(-avg_attention_drug)[:5]
    for idx in top5_drug:
        bars_drug[idx].set_color('darkred')
        bars_drug[idx].set_alpha(0.9)

    ax3.set_ylabel('Drug Token Index', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Average Attention', fontsize=12, fontweight='bold')
    ax3.set_ylim(-0.5, len(avg_attention_drug) - 0.5)
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3, linestyle='--')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"   ✅ {save_path.name}")
    print(f"      Top-5 Residues: {residue_indices[top5_indices].tolist()}")
    print(f"      Max attention: {avg_attention.max():.4f}")


def plot_comparison_grid(results, high_indices, low_indices, save_path, n_display=4):
    """
    affinitypairplot2x2 2x3

    Args:
    results: extractresultdict
    high_indices: affinityindex
        low_indices: affinityindex
        save_path: outputpath
        n_display: 
    """
    n_display = min(n_display, len(high_indices), len(low_indices))

    fig, axes = plt.subplots(2, n_display, figsize=(6*n_display, 10))

    if n_display == 1:
        axes = axes.reshape(2, 1)

    for col in range(n_display):
        # affinity
        high_idx = high_indices[col]
        att_high = results['attentions'][high_idx]

        # size
        if att_high.shape[1] > 150:
            avg_att = att_high.mean(axis=0)
            top_res = np.argsort(-avg_att)[:150]
            top_res = np.sort(top_res)
            att_high = att_high[:, top_res]

        im = axes[0, col].imshow(att_high, cmap='Reds', aspect='auto', interpolation='nearest')
        axes[0, col].set_title(f'High Affinity #{high_idx}\nY={results["y_true"][high_idx]}, '
                              f'Z={results["z_true"][high_idx]:.3f}',
                              fontsize=11, fontweight='bold', color='red')
        axes[0, col].set_ylabel('Drug Tokens', fontsize=10)
        axes[0, col].set_xlabel('Protein Residues', fontsize=10)

        if col == n_display - 1:
            plt.colorbar(im, ax=axes[0, col], fraction=0.046, pad=0.04)

            # affinity
        low_idx = low_indices[col]
        att_low = results['attentions'][low_idx]

        if att_low.shape[1] > 150:
            avg_att = att_low.mean(axis=0)
            top_res = np.argsort(-avg_att)[:150]
            top_res = np.sort(top_res)
            att_low = att_low[:, top_res]

        im = axes[1, col].imshow(att_low, cmap='Blues', aspect='auto', interpolation='nearest')
        axes[1, col].set_title(f'Low Affinity #{low_idx}\nY={results["y_true"][low_idx]}, '
                              f'Z={results["z_true"][low_idx]:.3f}',
                              fontsize=11, fontweight='bold', color='blue')
        axes[1, col].set_ylabel('Drug Tokens', fontsize=10)
        axes[1, col].set_xlabel('Protein Residues', fontsize=10)

        if col == n_display - 1:
            plt.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    plt.suptitle('Attention Pattern Comparison: High vs Low Affinity',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Comparison grid saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot attention heatmap cases')
    parser.add_argument('--input', type=str, required=True, help='Input pickle file from extraction')
    parser.add_argument('--output-dir', type=str, default='nature_mi_figures/attention_cases',
                       help='Output directory')
    parser.add_argument('--n-high', type=int, default=5, help='Number of high affinity cases')
    parser.add_argument('--n-low', type=int, default=5, help='Number of low affinity cases')
    parser.add_argument('--max-display', type=int, default=200, help='Max residues to display')

    args = parser.parse_args()

    # Load data
    print("\n" + "="*80)
    print("📊 Plotting Attention Heatmap Cases")
    print("="*80)
    print(f"Input: {args.input}")

    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    print(f"Loaded {len(results['attentions'])} samples")
    print(f"Multi-task: {results['metadata']['use_multitask']}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select best cases
    high_indices, low_indices = select_best_cases(results, args.n_high, args.n_low)

    if len(high_indices) == 0 or len(low_indices) == 0:
        print("⚠️  Not enough cases found. Try lowering selection criteria.")
        return

    # Plot individual cases
    print("\n📈 Plotting individual cases...")

    # High affinity cases
    for i, idx in enumerate(high_indices):
        save_path = output_dir / f'high_affinity_case_{i+1}_idx{idx}.png'
        plot_single_attention_case(
            attention=results['attentions'][idx],
            y_true=results['y_true'][idx],
            y_pred=results['y_pred'][idx],
            z_true=results['z_true'][idx],
            z_pred=results['z_pred'][idx] if results['z_pred'][idx] is not None else results['z_true'][idx],
            case_idx=idx,
            case_type='High',
            save_path=save_path,
            max_display=args.max_display
        )

    # Low affinity cases
    for i, idx in enumerate(low_indices):
        save_path = output_dir / f'low_affinity_case_{i+1}_idx{idx}.png'
        plot_single_attention_case(
            attention=results['attentions'][idx],
            y_true=results['y_true'][idx],
            y_pred=results['y_pred'][idx],
            z_true=results['z_true'][idx],
            z_pred=results['z_pred'][idx] if results['z_pred'][idx] is not None else results['z_true'][idx],
            case_idx=idx,
            case_type='Low',
            save_path=save_path,
            max_display=args.max_display
        )

    # Plot comparison grid
    print("\n📊 Plotting comparison grid...")
    comparison_path = output_dir / 'attention_comparison_grid.png'
    plot_comparison_grid(results, high_indices, low_indices, comparison_path, n_display=min(4, len(high_indices)))

    print("\n" + "="*80)
    print("✅ All plots complete!")
    print("="*80)
    print(f"Output directory: {output_dir}")
    print(f"Total files: {len(list(output_dir.glob('*.png')))}")


if __name__ == "__main__":
    main()
