#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regenerate the "exact five" figures (heatmap + pairing network) for the
user-specified 5 target residues x 5 constitutive pairs (25 pairs total).

Inputs  (per class):
    result/class_attention_analysis_pdb/exact_five_<class>_25pairs.csv
Outputs (per class, 600 dpi PNG + vector PDF):
    exact_five_<class>_heatmap.png / .pdf
    exact_five_<class>_network.png / .pdf

These generators were missing from the repo; reconstructed to match the
original figures and upgraded to journal-quality 600 dpi output.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# Unicode-safe fonts (the column header uses the Greek delta in 'ΔImp')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Journal-quality defaults: 600 dpi output + larger, less-crowded fonts
plt.rcParams.update({
    'savefig.dpi': 600,
    'figure.dpi': 150,
    'font.size': 13,
    'axes.titlesize': 17,
    'axes.labelsize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})

BASE_DIR = Path('result/class_attention_analysis_pdb')


def const_label(row):
    """'G' + 183 -> 'G183'."""
    return f"{row['Constitutive']}{row['Const_PDB']}"


def create_heatmap(df, class_name, out_stem):
    """5 targets (rows) x 5 pair-ranks (cols), colored by ΔImp.

    Each cell is annotated with the ΔImp value and the constitutive residue.
    """
    targets = df.sort_values('Target_Rank')['Target'].unique()
    pair_ranks = sorted(df['Pair_Rank'].unique())

    n_rows, n_cols = len(targets), len(pair_ranks)
    values = np.full((n_rows, n_cols), np.nan)
    labels = np.empty((n_rows, n_cols), dtype=object)

    t_to_idx = {t: i for i, t in enumerate(targets)}
    for _, row in df.iterrows():
        i = t_to_idx[row['Target']]
        j = pair_ranks.index(row['Pair_Rank'])
        values[i, j] = row['ΔImp']
        labels[i, j] = const_label(row)

    fig, ax = plt.subplots(figsize=(13, 9))
    im = ax.imshow(values, cmap='YlOrRd', aspect='auto')

    # cell annotations: ΔImp on top, constitutive residue below (blue, bold)
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    threshold = vmin + 0.6 * (vmax - vmin)
    for i in range(n_rows):
        for j in range(n_cols):
            if np.isnan(values[i, j]):
                continue
            val_color = 'white' if values[i, j] > threshold else 'black'
            lbl_color = 'white' if values[i, j] > threshold else 'navy'
            ax.text(j, i - 0.18, f"{values[i, j]:.4f}", ha='center', va='center',
                    fontsize=12, color=val_color)
            ax.text(j, i + 0.20, labels[i, j], ha='center', va='center',
                    fontsize=14, weight='bold', color=lbl_color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"Pair {r}" for r in pair_ranks], fontsize=13)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(targets, fontsize=14)
    ax.set_xlabel('Constitutive Pair Rank', fontsize=16, weight='bold')
    ax.set_ylabel(f'{class_name} Target Residue', fontsize=16, weight='bold')
    ax.set_title(
        f'User-Specified 5 {class_name} Residues × 5 Constitutive Pairs\n'
        'ΔImp Heatmap (25 Total Pairs)\n'
        'Values show ΔImp; Labels show Constitutive Residue (e.g., G183)',
        fontsize=17, weight='bold', pad=18)

    # grid lines between cells
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='gray', linewidth=1.0)
    ax.tick_params(which='minor', length=0)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('ΔImp (Importance Difference)', fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(BASE_DIR / f'{out_stem}.png', dpi=600, bbox_inches='tight')
    plt.savefig(BASE_DIR / f'{out_stem}.pdf', bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out_stem}.png / .pdf")


def create_network(df, class_name, out_stem):
    """One subplot per target: red star (target) linked to its top-5
    constitutive partners (blue circles), edges labeled with ΔImp."""
    targets = df.sort_values('Target_Rank')['Target'].unique()
    n = len(targets)

    # 2 subplots per row; large elements so each panel stays legible even when
    # the whole figure is scaled down to fit a Word column.
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 6.6 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, target in zip(axes, targets):
        sub = df[df['Target'] == target].sort_values('Pair_Rank')

        # blue partners on a ring around the central star
        k = len(sub)
        angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, k, endpoint=False)
        radius = 1.0

        for angle, (_, row) in zip(angles, sub.iterrows()):
            x, y = radius * np.cos(angle), radius * np.sin(angle)
            ax.plot([0, x], [0, y], color='steelblue', linewidth=2.0,
                    alpha=0.6, zorder=1)
            # edge label (ΔImp) offset to the side of the link so it does not
            # sit under the node or the central star
            ax.text(x * 0.46, y * 0.46, f"ΔImp={row['ΔImp']:.3f}",
                    ha='center', va='center', fontsize=11, color='dimgray',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                              edgecolor='none', alpha=0.7), zorder=2)
            ax.scatter([x], [y], s=2400, c='lightblue', edgecolors='steelblue',
                       linewidths=2.5, zorder=3)
            ax.text(x, y, const_label(row), ha='center', va='center',
                    fontsize=16, weight='bold', color='navy', zorder=4)

        # central target star
        ax.scatter([0], [0], s=4200, marker='*', c='red', edgecolors='darkred',
                   linewidths=2.5, zorder=5)
        ax.text(0, -0.27, target, ha='center', va='center', fontsize=18,
                weight='bold', color='darkred', zorder=6)

        rank = sub['Target_Rank'].iloc[0]
        ax.set_title(f'Target #{rank}: {target}', fontsize=18, weight='bold')
        ax.set_xlim(-1.42, 1.42)
        ax.set_ylim(-1.42, 1.42)
        ax.set_aspect('equal')
        ax.axis('off')

    # hide any unused panels (e.g. 6th cell when n=5)
    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(
        f'User-Specified {n} {class_name} Residues: Constitutive Pairing Network\n'
        '(Red star = Target, Blue circles = Top 5 Constitutive pairs)\n'
        'Labels show Residue+Number (e.g., G183)',
        fontsize=17, weight='bold', y=0.99)

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(BASE_DIR / f'{out_stem}.png', dpi=600, bbox_inches='tight')
    plt.savefig(BASE_DIR / f'{out_stem}.pdf', bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out_stem}.png / .pdf")


def main():
    for class_name in ('Active', 'Inactive'):
        key = class_name.lower()
        csv = BASE_DIR / f'exact_five_{key}_25pairs.csv'
        if not csv.exists():
            print(f"   ✗ missing input: {csv}")
            continue
        df = pd.read_csv(csv)
        print(f"\n=== {class_name}: {len(df)} pairs ===")
        create_heatmap(df, class_name, f'exact_five_{key}_heatmap')
        create_network(df, class_name, f'exact_five_{key}_network')


if __name__ == '__main__':
    main()
