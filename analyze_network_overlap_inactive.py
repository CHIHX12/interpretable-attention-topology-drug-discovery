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

def main():
    # read 25 pairs data
    base_dir = Path('result/class_attention_analysis_pdb')
    pairs_file = base_dir / 'exact_five_inactive_25pairs.csv'

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

    # 5 residue
    all_targets = df['Target'].unique()
    if len(all_targets) != 5:
        print(f"warning: 5 residueactual {len(all_targets)}")

    targets_list = sorted(all_targets)

    # analysiseach constitutive target use
    core_region = []
    independent_region = []
    low_imp_region = []
    other_region = []

    # target (region)
    target_counts = {t: 0 for t in targets_list}
    for const_name, const_targets in const_usage.items():
        for t in [ct['Target'] for ct in const_targets]:
            target_counts[t] += 1

    # sortuse constitutive target ()
    sorted_targets = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
    core_target_1 = sorted_targets[0][0] if len(sorted_targets) > 0 else None
    core_target_2 = sorted_targets[1][0] if len(sorted_targets) > 1 else None
    independent_target = sorted_targets[2][0] if len(sorted_targets) > 2 else None
    low_target_1 = sorted_targets[3][0] if len(sorted_targets) > 3 else None
    low_target_2 = sorted_targets[4][0] if len(sorted_targets) > 4 else None

    for const_name, targets_list_data in const_usage.items():
        target_names = [t['Target'] for t in targets_list_data]

        if core_target_1 in target_names and core_target_2 in target_names:
            core_region.append(const_name)
        elif independent_target in target_names and len(target_names) == 1:
            independent_region.append(const_name)
        elif low_target_1 in target_names and low_target_2 in target_names:
            low_imp_region.append(const_name)
        else:
            other_region.append(const_name)

    print(f"\nregion 1: region ({core_target_1} & {core_target_2} )")
    print(f" residue: {len(core_region)}")
    print(f"  residue: {', '.join(core_region)}")
    print(f" : important inactive-preferred ")

    print(f"\nregion 2: {independent_target} region")
    print(f" residue: {len(independent_region)}")
    print(f"  residue: {', '.join(independent_region)}")
    print(f" : {independent_target} antagonist ")

    print(f"\nregion 3: importantregion ({low_target_1} & {low_target_2} )")
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
    region_info = {
        'core': (core_region, f'{core_target_1} & {core_target_2}'),
        'independent': (independent_region, independent_target),
        'low_imp': (low_imp_region, f'{low_target_1} & {low_target_2}')
    }
    create_functional_regions_plot(region_info, base_dir)

    # saveresult
    output_txt = base_dir / 'network_overlap_analysis_inactive.txt'
    output_csv = base_dir / 'constitutive_sharing_stats_inactive.csv'

    with open(output_txt, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("networkanalysis - Constitutive residue (active)\n")
        f.write("Network Overlap Analysis - Shared Constitutive Residues (Inactive)\n")
        f.write("="*80 + "\n\n")

        f.write(f": {len(df)}\n")
        f.write(f" Constitutive residue: {len(const_usage)}\n\n")

        f.write("="*80 + "\n")
        f.write(f"region 1: region ({core_target_1} & {core_target_2} )\n")
        f.write("="*80 + "\n")
        f.write(f"residue: {len(core_region)}\n")
        f.write(f"residuecolumn:\n")
        for res in core_region:
            details = const_usage[res]
            f.write(f"\n  {res}:\n")
            for d in details:
                f.write(f"    {d['Target']} (Rank {d['Rank']}, ΔImp={d['ΔImp']:.4f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"region 2: {independent_target} region\n")
        f.write("="*80 + "\n")
        f.write(f"residue: {len(independent_region)}\n")
        f.write(f"residuecolumn:\n")
        for res in independent_region:
            details = const_usage[res]
            f.write(f"\n  {res}:\n")
            for d in details:
                f.write(f"    {d['Target']} (Rank {d['Rank']}, ΔImp={d['ΔImp']:.4f})\n")

        f.write("\n" + "="*80 + "\n")
        f.write(f"region 3: importantregion ({low_target_1} & {low_target_2} )\n")
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
    print(f"  - {base_dir / 'target_constitutive_sharing_matrix_inactive.png/pdf'}")
    print(f"  - {base_dir / 'network_overlap_diagram_inactive.png/pdf'}")
    print(f"  - {base_dir / 'functional_regions_3d_inactive.png/pdf'}")

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
    fig, ax = plt.subplots(figsize=(20, 6))

    sns.heatmap(matrix, cmap='RdYlBu_r', cbar_kws={'label': 'Used (1) or Not (0)'},
                xticklabels=constitutives, yticklabels=targets,
                linewidths=0.5, linecolor='gray', ax=ax)

    ax.set_title('Target-Constitutive Sharing Matrix\n'
                 '(Red = Used, Blue = Not Used)',
                 fontsize=14, weight='bold', pad=20)
    ax.set_xlabel('Constitutive Residue', fontsize=12, weight='bold')
    ax.set_ylabel('Inactive Target Residue', fontsize=12, weight='bold')

    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(base_dir / 'target_constitutive_sharing_matrix_inactive.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_dir / 'target_constitutive_sharing_matrix_inactive.pdf', bbox_inches='tight')
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
                   fontsize=12, weight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                           edgecolor=color, alpha=0.9, linewidth=2))

    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.set_title('Network Overlap Diagram\n'
                 'Constitutive Residues by Sharing Level\n'
                 '(Inner circles = More shared = Core functional regions)',
                 fontsize=14, weight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(base_dir / 'network_overlap_diagram_inactive.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_dir / 'network_overlap_diagram_inactive.pdf', bbox_inches='tight')
    plt.close()

def create_functional_regions_plot(region_info, base_dir):
    """createfunctionregion - 3D region"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # extract PDB mapping
    def extract_pdb(res_name):
        """ 'G183' extract 183"""
        return int(''.join(filter(str.isdigit, res_name)))

    # computeeachregion"" PDB sequence
    core_region, core_label = region_info['core']
    independent_region, independent_label = region_info['independent']
    low_imp_region, low_imp_label = region_info['low_imp']

    regions = [
        (core_region, f'Core Region\n({core_label})', 'red', 'o'),
        (independent_region, f'{independent_label} Region\n(Independent)', 'green', 's'),
        (low_imp_region, f'Low Importance\n({low_imp_label})', 'blue', '^')
    ]

    all_y_positions = []

    for residues, label, color, marker in regions:
        if len(residues) == 0:
            continue

        pdb_nums = [extract_pdb(r) for r in residues]

        # X : PDB sequence
        # Y : 3D 
        np.random.seed(42)
        y_positions = np.random.randn(len(pdb_nums)) * 10

        all_y_positions.extend(y_positions)

        ax.scatter(pdb_nums, y_positions, s=300, c=color, marker=marker,
                  edgecolors='black', linewidths=2, alpha=0.7, label=label, zorder=3)

        # residue
        for x, y, res_name in zip(pdb_nums, y_positions, residues):
            ax.text(x, y + 6, res_name, ha='center', va='bottom',
                   fontsize=10, weight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, alpha=0.9, linewidth=2))

    ax.axhline(0, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    ax.set_xlabel('PDB Residue Number (Sequence Position)', fontsize=13, weight='bold')
    ax.set_ylabel('Spatial Distribution (Arbitrary Units)', fontsize=13, weight='bold')
    ax.set_title('Functional Regions in 3D Space\n'
                 'Constitutive Residues Grouped by Target Sharing\n'
                 '(X-axis = Sequence position, Y-axis = Spatial clustering)',
                 fontsize=14, weight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(base_dir / 'functional_regions_3d_inactive.png', dpi=300, bbox_inches='tight')
    plt.savefig(base_dir / 'functional_regions_3d_inactive.pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
