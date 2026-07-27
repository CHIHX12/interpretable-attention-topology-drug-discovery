#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Network overlap analysis for INACTIVE residues with 3-way sharing detection.

Unlike the active residues (which share constitutive partners in 2-way
patterns), inactive residues form 3-way sharing groups. This script
classifies each constitutive residue by how many target residues use it
(3-way / 2-way / unique) and draws the grouped network diagram.

Input : result/class_attention_analysis_pdb/exact_five_inactive_25pairs.csv
Output: network_overlap_diagram_inactive_3way.png / .pdf  (600 dpi)
        network_overlap_analysis_inactive_3way.txt

Reconstructed generator, upgraded to journal-quality 600 dpi output.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Journal-quality defaults: 600 dpi output + larger, less-crowded fonts
plt.rcParams.update({
    'savefig.dpi': 600,
    'figure.dpi': 150,
    'font.size': 13,
})

BASE_DIR = Path('result/class_attention_analysis_pdb')


def target_label(row):
    """'F286' + 286 -> 'F286286' (matches original diagram labels)."""
    return f"{row['Target']}{row['Target_PDB']}"


def const_label(row):
    """'S' + 123 -> 'S123'."""
    return f"{row['Constitutive']}{row['Const_PDB']}"


def classify(df):
    """Group constitutive residues by their number of sharing targets."""
    const_usage = defaultdict(list)
    for _, row in df.iterrows():
        const_usage[const_label(row)].append({
            'Target': row['Target'],
            'TargetFull': target_label(row),
            'Rank': row['Pair_Rank'],
            'ΔImp': row['ΔImp'],
        })

    sharing_3way, sharing_2way, sharing_1way = [], [], []
    for const_name, targets in const_usage.items():
        item = {'Constitutive': const_name, 'targets': targets,
                'n': len(targets)}
        if len(targets) >= 3:
            sharing_3way.append(item)
        elif len(targets) == 2:
            sharing_2way.append(item)
        else:
            sharing_1way.append(item)

    # group 3-way constitutives by their (sorted) target combination
    groups_3way = defaultdict(list)
    for item in sharing_3way:
        combo = tuple(sorted({t['TargetFull'] for t in item['targets']}))
        groups_3way[combo].append(item)

    return groups_3way, sharing_2way, sharing_1way


def draw_circle_row(ax, members, y, color, edgecolor):
    """Place member circles in a centered horizontal row at height y."""
    n = len(members)
    spacing = 2.4
    x0 = -(n - 1) * spacing / 2
    for i, name in enumerate(members):
        x = x0 + i * spacing
        ax.scatter([x], [y], s=2600, c=color, edgecolors=edgecolor,
                   linewidths=2.5, alpha=0.85, zorder=2)
        ax.text(x, y, name, ha='center', va='center', fontsize=12,
                weight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor=edgecolor, alpha=0.95, linewidth=2),
                zorder=3)


def draw_group_label(ax, text, y, edgecolor):
    ax.text(0, y, text, ha='center', va='center', fontsize=14, weight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor=edgecolor, alpha=0.95, linewidth=2.5),
            zorder=4)


def create_3way_diagram(groups_3way, sharing_2way, sharing_1way):
    fig, ax = plt.subplots(figsize=(12, 13))

    # red palette for the 3-way groups, then orange (2-way), light blue (1-way)
    three_way_colors = [('#FF6B6B', '#E03131'), ('#A05252', '#7A2E2E')]

    y = 0.0
    band = 2.6

    # 3-way groups (ordered by number of shared constitutives, desc keeps
    # the layout stable across runs)
    ordered_groups = sorted(groups_3way.items(),
                            key=lambda kv: (-len(kv[1]), kv[0]))
    for idx, (combo, items) in enumerate(ordered_groups):
        fill, edge = three_way_colors[idx % len(three_way_colors)]
        draw_group_label(ax, f"3-way: {', '.join(combo)}", y + 1.0, edge)
        members = [it['Constitutive'] for it in items]
        draw_circle_row(ax, members, y, fill, edge)
        y -= band

    # 2-way sharing
    if sharing_2way:
        draw_group_label(ax, '2-way Sharing', y + 1.0, '#F08C00')
        draw_circle_row(ax, [it['Constitutive'] for it in sharing_2way],
                        y, '#FFC078', '#F08C00')
        y -= band

    # unique (1-way)
    if sharing_1way:
        draw_group_label(ax, 'Unique (1-way)', y + 1.0, '#74C0FC')
        members = [it['Constitutive'] for it in sharing_1way]
        # wrap into rows of at most 3 to avoid overflow
        per_row = 3
        for r in range(0, len(members), per_row):
            draw_circle_row(ax, members[r:r + per_row], y, '#C5E4F3', '#74C0FC')
            y -= band * 0.85

    ax.set_title('Inactive Residue Network - 3-Way Sharing Analysis',
                 fontsize=18, weight='bold', pad=20)
    ax.set_xlim(-6, 6)
    ax.set_ylim(y - 0.5, 2.5)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()
    out = BASE_DIR / 'network_overlap_diagram_inactive_3way'
    plt.savefig(f'{out}.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{out}.pdf', bbox_inches='tight')
    plt.close()
    print(f"   ✓ {out.name}.png / .pdf")


def write_report(groups_3way, sharing_2way, sharing_1way, n_pairs, n_const):
    out = BASE_DIR / 'network_overlap_analysis_inactive_3way.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Network Overlap Analysis - 3-Way Sharing Detection (Inactive)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total pairs: {n_pairs}\n")
        f.write(f"Unique constitutive residues: {n_const}\n\n")

        f.write("=" * 80 + "\n3-WAY SHARING GROUPS\n" + "=" * 80 + "\n\n")
        ordered = sorted(groups_3way.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        for gi, (combo, items) in enumerate(ordered, 1):
            f.write(f"Group {gi}: {' & '.join(combo)}\n")
            f.write(f"Number of shared constitutives: {len(items)}\n")
            f.write("Constitutive residues:\n")
            for it in items:
                f.write(f"\n  {it['Constitutive']}:\n")
                avg = sum(t['ΔImp'] for t in it['targets']) / len(it['targets'])
                for t in it['targets']:
                    f.write(f"    {t['TargetFull']} (Rank {t['Rank']}, "
                            f"ΔImp={t['ΔImp']:.4f})\n")
                f.write(f"    Average ΔImp: {avg:.4f}\n")
            f.write("\n\n")

        f.write("=" * 80 + "\n2-WAY SHARING\n" + "=" * 80 + "\n")
        f.write(f"Number of constitutives: {len(sharing_2way)}\n")
        for it in sharing_2way:
            f.write(f"\n  {it['Constitutive']}:\n")
            for t in it['targets']:
                f.write(f"    {t['TargetFull']} (Rank {t['Rank']}, "
                        f"ΔImp={t['ΔImp']:.4f})\n")

        f.write("\n" + "=" * 80 + "\nUNIQUE (1-WAY)\n" + "=" * 80 + "\n")
        f.write(f"Number of constitutives: {len(sharing_1way)}\n\n")
        for it in sharing_1way:
            t = it['targets'][0]
            f.write(f"  {it['Constitutive']}: {t['TargetFull']} "
                    f"(Rank {t['Rank']}, ΔImp={t['ΔImp']:.4f})\n")
    print(f"   ✓ {out.name}")


def main():
    csv = BASE_DIR / 'exact_five_inactive_25pairs.csv'
    if not csv.exists():
        print(f"   ✗ missing input: {csv}")
        return
    df = pd.read_csv(csv)
    groups_3way, sharing_2way, sharing_1way = classify(df)

    n_const = (sum(len(v) for v in groups_3way.values())
               + len(sharing_2way) + len(sharing_1way))
    print(f"\n=== Inactive 3-way: {len(df)} pairs, {n_const} constitutives ===")
    print(f"   3-way groups: {len(groups_3way)} | 2-way: {len(sharing_2way)} "
          f"| unique: {len(sharing_1way)}")

    create_3way_diagram(groups_3way, sharing_2way, sharing_1way)
    write_report(groups_3way, sharing_2way, sharing_1way, len(df), n_const)


if __name__ == '__main__':
    main()
