#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
networkanalysis - 25 Constitutive residue
Network Overlap Analysis - Find shared Constitutive residues from 25 pairs

residue 3D structurefunctionregion
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict, Counter

# set
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

def _build_region_points(regions, extract_pdb, spread=14.0, seed=42):
    """Assign each residue an x (PDB position) and a single-RNG random y.

    Using one RNG for all regions (instead of reseeding per region) avoids the
    repeated y-pattern that put residues from different regions on top of each
    other.
    """
    rng = np.random.default_rng(seed)
    pts, region_meta = [], {}
    for ri, (residues, label, color, marker) in enumerate(regions):
        region_meta[ri] = (label, color, marker)
        ys = rng.normal(0, spread, size=len(residues))
        for r, y in zip(residues, ys):
            pts.append({'x': extract_pdb(r), 'y': float(y), 'res': r, 'ri': ri})
    return pts, region_meta


def _deoverlap_y(pts, x_frac=0.045, min_gap=5.5, iters=300):
    """Greedily separate markers that sit at nearby sequence positions by
    nudging their y-values apart, so labels no longer overlap."""
    if len(pts) < 2:
        return
    xs = [p['x'] for p in pts]
    xspan = (max(xs) - min(xs)) or 1.0
    for _ in range(iters):
        moved = False
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                if abs(pts[i]['x'] - pts[j]['x']) < xspan * x_frac:
                    dy = pts[i]['y'] - pts[j]['y']
                    if abs(dy) < min_gap:
                        push = (min_gap - abs(dy)) / 2 + 0.1
                        s = 1.0 if dy >= 0 else -1.0
                        pts[i]['y'] += s * push
                        pts[j]['y'] -= s * push
                        moved = True
        if not moved:
            break


def main():
    # read 25 pairs data
    base_dir = Path('result/class_attention_analysis_pdb')
    pairs_file = base_dir / 'exact_five_active_25pairs.csv'

    if not pairs_file.exists():
        print(f"error: file {pairs_file}")
        return

    df = pd.read_csv(pairs_file)

    print("="*80)
    print("networkanalysis - activeresidue - Constitutive residue")
    print("Network Overlap Analysis - Shared Constitutive Residues")
    print("="*80)
    print(f": {len(df)}")
    print(f"residue: {df['Target'].nunique()}")
    print()

    # analysiseach Constitutive residue Target use
    const_usage = defaultdict(list)

    for _, row in df.iterrows():
        const_name = f"{row['Constitutive']}{row['Const_PDB']}"
        target_name = row['Target']
        rank = row['Pair_Rank']
        delta_imp = row['ΔImp']

        const_usage[const_name].append({
            'Target': target_name,
            'Rank': rank,
            'ΔImp': delta_imp,
            'Target_PDB': row['Target_PDB']
        })

    # statisticsfrequency
    print("\n" + "="*80)
    print("Constitutive residuestatistics")
    print("="*80)

    sharing_stats = []
    for const_name, targets in const_usage.items():
        n_targets = len(targets)
        target_list = [t['Target'] for t in targets]
        avg_delta = np.mean([t['ΔImp'] for t in targets])

        sharing_stats.append({
            'Constitutive': const_name,
            'N_Targets': n_targets,
            'Targets': ', '.join(target_list),
            'Avg_ΔImp': avg_delta,
            'Details': targets
        })

    sharing_df = pd.DataFrame(sharing_stats)
    sharing_df = sharing_df.sort_values('N_Targets', ascending=False)

    print(f"\nfrequency:")
    print(f"{'count':<12}{'Constitutive ':<20}{'':<10}")
    print("-" * 45)
    for n in range(5, 0, -1):
        count = (sharing_df['N_Targets'] == n).sum()
        pct = count / len(sharing_df) * 100
        print(f"{n}x          {count:<20}{pct:>6.1f}%")

    # heightresidue2 target use
    highly_shared = sharing_df[sharing_df['N_Targets'] >= 2].copy()

    print("\n" + "="*80)
    print(f"height Constitutive residue ≥2 Target use: {len(highly_shared)}")
    print("="*80)
    print(f"{'Constitutive':<15}{'count':<10}{'averageΔImp':<12}{'use Targets':<40}")
    print("-" * 80)

    for _, row in highly_shared.iterrows():
        print(f"{row['Constitutive']:<15}{row['N_Targets']:<10}{row['Avg_ΔImp']:<12.4f}{row['Targets']:<40}")

    # functionregion
    print("\n" + "="*80)
    print("functionregion")
    print("="*80)

    # analysiseach constitutive target use
    core_region = []
    s125_region = []
    low_imp_region = []
    other_region = []

    for const_name, targets_list in const_usage.items():
        target_names = [t['Target'] for t in targets_list]

        if 'E124' in target_names and 'V122' in target_names:
            core_region.append(const_name)
        elif 'S125' in target_names and len(target_names) == 1:
            s125_region.append(const_name)
        elif 'C198' in target_names and 'P200' in target_names:
            low_imp_region.append(const_name)
        else:
            other_region.append(const_name)

    print(f"\nregion 1: region (E124 & V122 )")
    print(f" residue: {len(core_region)}")
    print(f"  residue: {', '.join(core_region)}")
    print(f" : important active-preferred ")

    print(f"\nregion 2: S125 region")
    print(f" residue: {len(s125_region)}")
    print(f"  residue: {', '.join(s125_region)}")
    print(f" : S125 agonist ")

    print(f"\nregion 3: importantregion (C198 & P200 )")
    print(f" residue: {len(low_imp_region)}")
    print(f"  residue: {', '.join(low_imp_region)}")
    print(f" : importantregion")

    # creatematrix
    print("\ncreate Target-Constitutive matrix...")
    create_sharing_matrix(df, base_dir)

    # createnetworkoverlap
    print("createnetworkoverlap...")
    create_overlap_network(sharing_df, base_dir)

    # create 3D functionregion
    print("createfunctionregion...")
    create_functional_regions_plot(core_region, s125_region, low_imp_region, base_dir)

    # saveresult
    output_txt = base_dir / 'network_overlap_analysis_active.txt'
    output_csv = base_dir / 'constitutive_sharing_stats_active.csv'

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("networkanalysis - Constitutive residue\n")
        f.write("Network Overlap Analysis - Shared Constitutive Residues\n")
        f.write("="*80 + "\n\n")

        f.write(f": {len(df)}\n")
        f.write(f" Constitutive residue: {len(const_usage)}\n\n")

        f.write("="*80 + "\n")
        f.write("region 1: region (E124 & V122 )\n")
        f.write("="*80 + "\n")
        f.write(f"residue: {len(core_region)}\n")
        f.write(f"residuecolumn:\n")
        for res in core_region:
            details = const_usage[res]
            f.write(f"\n  {res}:\n")
            for d in details:
                f.write(f"    {d['Target']} (Rank {d['Rank']}, ΔImp={d['ΔImp']:.4f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("region 2: S125 region\n")
        f.write("="*80 + "\n")
        f.write(f"residue: {len(s125_region)}\n")
        f.write(f"residuecolumn:\n")
        for res in s125_region:
            details = const_usage[res]
            f.write(f"\n  {res}:\n")
            for d in details:
                f.write(f"    {d['Target']} (Rank {d['Rank']}, ΔImp={d['ΔImp']:.4f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write("region 3: importantregion (C198 & P200 )\n")
        f.write("="*80 + "\n")
        f.write(f"residue: {len(low_imp_region)}\n")
        f.write(f"residuecolumn:\n")
        for res in low_imp_region:
            details = const_usage[res]
            f.write(f"\n  {res}:\n")
            for d in details:
                f.write(f"    {d['Target']} (Rank {d['Rank']}, ΔImp={d['ΔImp']:.4f})\n")

    sharing_df.to_csv(output_csv, index=False, encoding='utf-8')

    print("\n" + "="*80)
    print("analysisdone！")
    print("="*80)
    print(f"resultfile:")
    print(f"  - {output_txt}")
    print(f"  - {output_csv}")
    print(f"  - {base_dir / 'target_constitutive_sharing_matrix_active.png/pdf'}")
    print(f"  - {base_dir / 'network_overlap_diagram_active.png/pdf'}")
    print(f"  - {base_dir / 'functional_regions_3d_active.png/pdf'}")

def create_sharing_matrix(df, base_dir):
    """create Target-Constitutive matrix"""
    # target constitutive
    targets = df['Target'].unique()

    # create constitutive 
    df['Const_Full'] = df['Constitutive'] + df['Const_PDB'].astype(str)
    constitutives = df['Const_Full'].unique()

    # creatematrix
    matrix = np.zeros((len(targets), len(constitutives)))

    target_to_idx = {t: i for i, t in enumerate(targets)}
    const_to_idx = {c: i for i, c in enumerate(constitutives)}

    for _, row in df.iterrows():
        t_idx = target_to_idx[row['Target']]
        c_idx = const_to_idx[row['Const_Full']]
        matrix[t_idx, c_idx] = 1 # use

    # 
    fig, ax = plt.subplots(figsize=(17, 8))

    sns.heatmap(matrix, cmap='RdYlBu_r',
                cbar_kws={'label': 'Used (1) or Not (0)'},
                xticklabels=constitutives, yticklabels=targets,
                linewidths=0.5, linecolor='gray', ax=ax)
    ax.figure.axes[-1].yaxis.label.set_size(14)

    ax.set_title('Target-Constitutive Sharing Matrix\n'
                 '(Red = Used, Blue = Not Used)',
                 fontsize=18, weight='bold', pad=22)
    ax.set_xlabel('Constitutive Residue', fontsize=16, weight='bold')
    ax.set_ylabel('Active Target Residue', fontsize=16, weight='bold')

    plt.xticks(rotation=90, fontsize=18, weight='bold')
    plt.yticks(fontsize=18, weight='bold')
    plt.tight_layout()
    plt.savefig(base_dir / 'target_constitutive_sharing_matrix_active.png', dpi=600, bbox_inches='tight')
    plt.savefig(base_dir / 'target_constitutive_sharing_matrix_active.pdf', bbox_inches='tight')
    plt.close()

def create_overlap_network(sharing_df, base_dir):
    """createnetworkoverlap - display Constitutive Target """
    fig, ax = plt.subplots(figsize=(16, 12))

    # countgroup
    share_5 = sharing_df[sharing_df['N_Targets'] == 5]
    share_4 = sharing_df[sharing_df['N_Targets'] == 4]
    share_3 = sharing_df[sharing_df['N_Targets'] == 3]
    share_2 = sharing_df[sharing_df['N_Targets'] == 2]
    share_1 = sharing_df[sharing_df['N_Targets'] == 1]

    # - (use)
    circles = [
        (share_1, 5.0, 'dodgerblue', 'Only 1 Target'),
        (share_2, 4.0, 'mediumblue', 'Shared by 2 Targets'),
        (share_3, 3.0, 'darkorange', 'Shared by 3 Targets'),
        (share_4, 2.0, 'orangered', 'Shared by 4 Targets'),
        (share_5, 1.0, 'darkred', 'Shared by 5 Targets (Core)')
    ]

    for group_df, radius, color, label in circles:
        if len(group_df) == 0:
            continue

        # 
        circle = plt.Circle((0, 0), radius, fill=False, edgecolor=color, linewidth=3, label=label)
        ax.add_patch(circle)

        # residue
        n = len(group_df)
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)

        for i, (_, row) in enumerate(group_df.iterrows()):
            x = radius * np.cos(angles[i])
            y = radius * np.sin(angles[i])

            # 
            ax.scatter([x], [y], s=300, c=color, edgecolors='black', linewidths=2, zorder=3, alpha=0.7)

            # 
            # compute
            text_radius = radius * 1.15
            text_x = text_radius * np.cos(angles[i])
            text_y = text_radius * np.sin(angles[i])

            ax.text(text_x, text_y, row['Constitutive'], ha='center', va='center',
                   fontsize=15, weight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=color, alpha=0.9, linewidth=2))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax.set_title('Network Overlap Diagram\n'
                 'Constitutive Residues by Sharing Level\n'
                 '(Inner circles = More shared = Core functional regions)',
                 fontsize=18, weight='bold', pad=22)

    plt.tight_layout()
    plt.savefig(base_dir / 'network_overlap_diagram_active.png', dpi=600, bbox_inches='tight')
    plt.savefig(base_dir / 'network_overlap_diagram_active.pdf', bbox_inches='tight')
    plt.close()

def create_functional_regions_plot(core_region, s125_region, low_imp_region, base_dir):
    """createfunctionregion - 3D region"""
    fig, ax = plt.subplots(figsize=(16, 11))

    # extract PDB mapping
    def extract_pdb(res_name):
        """ 'G183' extract 183"""
        return int(''.join(filter(str.isdigit, res_name)))

    # computeeachregion"" PDB sequence
    regions = [
        (core_region, 'Core Region\n(E124 & V122)', 'red', 'o'),
        (s125_region, 'S125 Region\n(Independent)', 'green', 's'),
        (low_imp_region, 'Low Importance\n(C198 & P200)', 'blue', '^')
    ]

    pts, region_meta = _build_region_points(regions, extract_pdb)
    _deoverlap_y(pts)

    for ri, (label, color, marker) in region_meta.items():
        rpts = [p for p in pts if p['ri'] == ri]
        if not rpts:
            continue
        ax.scatter([p['x'] for p in rpts], [p['y'] for p in rpts],
                   s=340, c=color, marker=marker, edgecolors='black',
                   linewidths=2, alpha=0.75, label=label, zorder=3)
        for p in rpts:
            ax.annotate(p['res'], xy=(p['x'], p['y']), xytext=(0, 13),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=12, weight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                                  edgecolor=color, alpha=0.92, linewidth=1.5),
                        arrowprops=dict(arrowstyle='-', color=color,
                                        lw=1.0, alpha=0.55), zorder=4)

    all_y_positions = [p['y'] for p in pts]

    # give the scattered points headroom so labels stay inside the axes
    if all_y_positions:
        ymin, ymax = min(all_y_positions), max(all_y_positions)
        yr = (ymax - ymin) or 1.0
        ax.set_ylim(ymin - 0.18 * yr - 3, ymax + 0.42 * yr + 4)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('PDB Residue Number (Sequence Position)', fontsize=16, weight='bold')
    ax.set_ylabel('Spatial Distribution (Arbitrary Units)', fontsize=16, weight='bold')
    ax.set_title('Functional Regions in 3D Space\n'
                 'Constitutive Residues Grouped by Target Sharing\n'
                 '(X-axis = Sequence position, Y-axis = Spatial clustering)',
                 fontsize=18, weight='bold', pad=22)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(base_dir / 'functional_regions_3d_active.png', dpi=600, bbox_inches='tight')
    plt.savefig(base_dir / 'functional_regions_3d_active.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
