#!/usr/bin/env python3
"""
GHSR 注意力共識分析
GHSR Attention Consensus Analysis

目的：
從所有 GHSR 藥物的注意力分數中找出高頻率被關注的殘基
這些殘基可能是重要的結合口袋位點

方法：
1. 對每個藥物，找出高注意力殘基（>Mean + 1σ）
2. 統計每個殘基被高注意力關注的頻率
3. 按頻率排序，找出共識殘基
4. 生成 PyMOL 腳本可視化共識殘基
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

def parse_args():
    parser = argparse.ArgumentParser(description="Consensus analysis of GHSR attention scores")
    parser.add_argument('--attention_file', type=str,
                        default='datasets/GPCR_resarch/attention_results/GHSR_attention_scores.npz',
                        help='Path to attention scores NPZ file')
    parser.add_argument('--output_dir', type=str,
                        default='datasets/GPCR_resarch/consensus_results',
                        help='Output directory for consensus analysis')
    parser.add_argument('--protein_length', type=int, default=523,
                        help='Protein sequence length')
    parser.add_argument('--threshold_method', type=str, default='mean_std',
                        choices=['mean_std', 'percentile', 'top_k'],
                        help='Method to define high attention (mean_std: Mean+1σ, percentile: top 10%, top_k: top 30 positions)')
    parser.add_argument('--percentile', type=float, default=90,
                        help='Percentile threshold if using percentile method')
    parser.add_argument('--top_k', type=int, default=30,
                        help='Number of top residues if using top_k method')
    parser.add_argument('--min_frequency', type=float, default=0.25,
                        help='Minimum frequency to be considered consensus residue (0-1)')
    return parser.parse_args()

def load_attention_scores(attention_file):
    """載入注意力分數"""
    print(f"📖 載入注意力分數: {attention_file}")

    data = np.load(attention_file)
    attention_matrix = data['attention_scores']  # [n_drugs, protein_len]
    drug_ids = data['drug_ids']

    print(f"   ✓ 藥物數量: {attention_matrix.shape[0]}")
    print(f"   ✓ 蛋白質長度: {attention_matrix.shape[1]}")

    return attention_matrix, drug_ids

def identify_high_attention_residues(attention_matrix, method='mean_std', percentile=90, top_k=30):
    """
    識別每個藥物的高注意力殘基

    Returns:
        high_attention_mask: [n_drugs, protein_len], boolean mask
        thresholds: [n_drugs], threshold for each drug
    """
    n_drugs, protein_len = attention_matrix.shape
    high_attention_mask = np.zeros_like(attention_matrix, dtype=bool)
    thresholds = np.zeros(n_drugs)

    print(f"\n🔍 識別高注意力殘基...")
    print(f"   方法: {method}")

    for i in range(n_drugs):
        att_i = attention_matrix[i]

        if method == 'mean_std':
            # Mean + 1 標準差
            mean_i = att_i.mean()
            std_i = att_i.std()
            threshold_i = mean_i + std_i
        elif method == 'percentile':
            # 百分位數
            threshold_i = np.percentile(att_i, percentile)
        elif method == 'top_k':
            # Top K 位置
            sorted_indices = np.argsort(att_i)[::-1]
            top_k_indices = sorted_indices[:top_k]
            high_attention_mask[i, top_k_indices] = True
            threshold_i = att_i[top_k_indices[-1]]
            thresholds[i] = threshold_i
            continue

        high_attention_mask[i] = att_i > threshold_i
        thresholds[i] = threshold_i

    # 統計
    n_high_per_drug = high_attention_mask.sum(axis=1)
    print(f"   ✓ 每個藥物的高注意力殘基數:")
    print(f"     Mean: {n_high_per_drug.mean():.1f}")
    print(f"     Median: {np.median(n_high_per_drug):.1f}")
    print(f"     Range: [{n_high_per_drug.min()}, {n_high_per_drug.max()}]")

    return high_attention_mask, thresholds

def compute_consensus_residues(high_attention_mask, min_frequency=0.25):
    """
    計算共識殘基

    Returns:
        consensus_freq: [protein_len], frequency of each residue
        consensus_residues: list of (position, frequency) tuples, sorted by frequency
    """
    n_drugs, protein_len = high_attention_mask.shape

    # 計算每個殘基被高注意力關注的頻率
    consensus_freq = high_attention_mask.sum(axis=0) / n_drugs  # [protein_len]

    print(f"\n📊 共識殘基統計:")
    print(f"   頻率統計:")
    print(f"     Mean: {consensus_freq.mean():.4f}")
    print(f"     Median: {np.median(consensus_freq):.4f}")
    print(f"     Max: {consensus_freq.max():.4f}")

    # 找出高頻率殘基
    consensus_mask = consensus_freq >= min_frequency
    n_consensus = consensus_mask.sum()

    print(f"\n   共識殘基 (頻率 >= {min_frequency:.2%}):")
    print(f"     數量: {n_consensus}")

    # 按頻率排序
    consensus_positions = np.where(consensus_mask)[0]
    consensus_residues = [(int(pos), float(consensus_freq[pos]))
                          for pos in consensus_positions]
    consensus_residues.sort(key=lambda x: x[1], reverse=True)

    # 頻率分層
    freq_categories = {
        'ultra_high': [],  # >75%
        'high': [],        # 50-75%
        'medium': [],      # 25-50%
    }

    for pos, freq in consensus_residues:
        if freq > 0.75:
            freq_categories['ultra_high'].append((pos, freq))
        elif freq > 0.50:
            freq_categories['high'].append((pos, freq))
        else:
            freq_categories['medium'].append((pos, freq))

    print(f"\n   頻率分層:")
    print(f"     🔴 超高頻率 (>75%): {len(freq_categories['ultra_high'])} 殘基")
    print(f"     🟠 高頻率 (50-75%): {len(freq_categories['high'])} 殘基")
    print(f"     🟡 中等頻率 (25-50%): {len(freq_categories['medium'])} 殘基")

    return consensus_freq, consensus_residues, freq_categories

def save_consensus_results(consensus_freq, consensus_residues, freq_categories,
                           attention_matrix, output_dir, protein_length):
    """保存共識分析結果"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 保存共識分析結果到: {output_dir}")

    # 1. 保存共識頻率向量
    freq_file = output_path / "GHSR_consensus_frequencies.txt"
    with open(freq_file, 'w') as f:
        f.write("# GHSR Consensus Residue Frequencies\n")
        f.write(f"# Total positions: {protein_length}\n")
        f.write(f"# Position\tFrequency\n")
        for pos in range(protein_length):
            f.write(f"{pos}\t{consensus_freq[pos]:.6f}\n")
    print(f"   ✓ 共識頻率: {freq_file}")

    # 2. 保存排序的共識殘基列表
    residues_file = output_path / "GHSR_consensus_residues.csv"
    consensus_df = pd.DataFrame(consensus_residues, columns=['Position', 'Frequency'])
    consensus_df['Position'] = consensus_df['Position'].astype(int)
    consensus_df['Frequency_Percent'] = (consensus_df['Frequency'] * 100).round(2)

    # 添加分類
    def categorize_freq(freq):
        if freq > 0.75:
            return 'Ultra High (>75%)'
        elif freq > 0.50:
            return 'High (50-75%)'
        else:
            return 'Medium (25-50%)'

    consensus_df['Category'] = consensus_df['Frequency'].apply(categorize_freq)

    consensus_df.to_csv(residues_file, index=False)
    print(f"   ✓ 共識殘基列表: {residues_file}")

    # 3. 保存分類結果
    for category, residues in freq_categories.items():
        if len(residues) > 0:
            cat_file = output_path / f"GHSR_consensus_{category}.txt"
            with open(cat_file, 'w') as f:
                f.write(f"# GHSR Consensus Residues - {category.upper().replace('_', ' ')}\n")
                f.write(f"# Count: {len(residues)}\n")
                f.write(f"# Position\tFrequency\n")
                for pos, freq in residues:
                    f.write(f"{pos}\t{freq:.4f} ({freq*100:.2f}%)\n")
            print(f"   ✓ {category.upper()}: {cat_file}")

    # 4. 保存統計信息
    stats = {
        'protein_length': protein_length,
        'n_drugs': attention_matrix.shape[0],
        'n_consensus_residues': len(consensus_residues),
        'frequency_categories': {
            'ultra_high': len(freq_categories['ultra_high']),
            'high': len(freq_categories['high']),
            'medium': len(freq_categories['medium'])
        },
        'top_10_residues': [
            {'position': int(pos), 'frequency': float(freq)}
            for pos, freq in consensus_residues[:10]
        ]
    }

    stats_file = output_path / "GHSR_consensus_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"   ✓ 統計信息: {stats_file}")

    # 5. 生成可視化
    plot_consensus_heatmap(consensus_freq, protein_length, output_path)
    plot_top_residues(consensus_residues[:30], output_path)

    # 6. 生成 PyMOL 腳本
    generate_pymol_script(consensus_residues, freq_categories, output_path)

    return stats

def plot_consensus_heatmap(consensus_freq, protein_length, output_dir):
    """繪製共識頻率熱圖"""

    fig, ax = plt.subplots(figsize=(16, 4))

    # 創建熱圖
    freq_matrix = consensus_freq.reshape(1, -1)
    im = ax.imshow(freq_matrix, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1, interpolation='nearest')

    ax.set_xlabel('Residue Position', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.set_title('GHSR Consensus Attention Frequency Across All Drugs',
                 fontsize=14, fontweight='bold', pad=15)

    # 添加 colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, aspect=40)
    cbar.set_label('Attention Frequency', fontsize=11)

    # 標記高頻率位置
    high_freq_positions = np.where(consensus_freq > 0.5)[0]
    if len(high_freq_positions) > 0:
        ax.scatter(high_freq_positions, np.zeros(len(high_freq_positions)),
                   marker='v', s=100, c='red', edgecolors='black',
                   linewidth=0.5, zorder=10,
                   label=f'High Frequency (>50%, n={len(high_freq_positions)})')
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    heatmap_file = output_dir / "GHSR_consensus_heatmap.png"
    plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "GHSR_consensus_heatmap.pdf", bbox_inches='tight')
    plt.close()

    print(f"   ✓ 共識熱圖: {heatmap_file}")

def plot_top_residues(top_residues, output_dir):
    """繪製 Top N 共識殘基柱狀圖"""

    if len(top_residues) == 0:
        return

    positions = [pos for pos, freq in top_residues]
    frequencies = [freq * 100 for pos, freq in top_residues]

    # 顏色編碼
    colors = []
    for freq in frequencies:
        if freq > 75:
            colors.append('#E74C3C')  # 紅色
        elif freq > 50:
            colors.append('#F39C12')  # 橘色
        else:
            colors.append('#F1C40F')  # 黃色

    fig, ax = plt.subplots(figsize=(14, 6))

    bars = ax.bar(range(len(positions)), frequencies, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Residue Rank', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attention Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {len(top_residues)} Consensus Residues (GHSR)',
                 fontsize=14, fontweight='bold', pad=15)

    # X 軸標籤顯示位置
    ax.set_xticks(range(len(positions)))
    ax.set_xticklabels([str(pos) for pos in positions], rotation=45, ha='right', fontsize=9)

    # 添加數值標籤
    for i, (bar, freq, pos) in enumerate(zip(bars, frequencies, positions)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{freq:.1f}%',
                ha='center', va='bottom', fontsize=7)

    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#E74C3C', edgecolor='black', label='Ultra High (>75%)'),
        Patch(facecolor='#F39C12', edgecolor='black', label='High (50-75%)'),
        Patch(facecolor='#F1C40F', edgecolor='black', label='Medium (25-50%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 110)

    plt.tight_layout()

    bar_file = output_dir / "GHSR_top_consensus_residues.png"
    plt.savefig(bar_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "GHSR_top_consensus_residues.pdf", bbox_inches='tight')
    plt.close()

    print(f"   ✓ Top 殘基柱狀圖: {bar_file}")

def generate_pymol_script(consensus_residues, freq_categories, output_dir):
    """生成 PyMOL 腳本來可視化共識殘基"""

    script_file = output_dir / "visualize_GHSR_consensus.pml"

    with open(script_file, 'w') as f:
        f.write("# PyMOL script to visualize GHSR consensus attention residues\n")
        f.write("# Compatible with both Linux and Windows\n")
        f.write("#\n")
        f.write("# Usage:\n")
        f.write("#   Linux:   pymol datasets/GPCR_resarch/consensus_results/visualize_GHSR_consensus.pml\n")
        f.write("#   Windows: 1. cd C:\\path\\to\\DrugBAN_BiLSTM\n")
        f.write("#            2. pymol datasets/GPCR_resarch/consensus_results/visualize_GHSR_consensus.pml\n")
        f.write("#\n")
        f.write("# Color coding:\n")
        f.write("#   Red (>75%):    Ultra high frequency residues - Core hotspots\n")
        f.write("#   Orange (50-75%): High frequency residues - Secondary sites\n")
        f.write("#   Yellow (25-50%): Medium frequency residues - Selective sites\n")
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

        # PDB residue number = dataset position + 2
        # 超高頻率殘基（紅色）
        if len(freq_categories['ultra_high']) > 0:
            ultra_high_pos = [str(pos + 2) for pos, freq in freq_categories['ultra_high']]
            f.write("# Ultra high frequency residues (>75%)\n")
            f.write(f"select ultra_high, GHSR and resi {'+'.join(ultra_high_pos)}\n")
            f.write("show sticks, ultra_high\n")
            f.write("color red, ultra_high\n")
            f.write("util.cnc ultra_high\n\n")

        # 高頻率殘基（橘色）
        if len(freq_categories['high']) > 0:
            high_pos = [str(pos + 2) for pos, freq in freq_categories['high']]
            f.write("# High frequency residues (50-75%)\n")
            f.write(f"select high, GHSR and resi {'+'.join(high_pos)}\n")
            f.write("show sticks, high\n")
            f.write("color orange, high\n")
            f.write("util.cnc high\n\n")

        # 中等頻率殘基（黃色）
        if len(freq_categories['medium']) > 0:
            medium_pos = [str(pos + 2) for pos, freq in freq_categories['medium']]
            f.write("# Medium frequency residues (25-50%)\n")
            f.write(f"select medium, GHSR and resi {'+'.join(medium_pos)}\n")
            f.write("show sticks, medium\n")
            f.write("color yellow, medium\n")
            f.write("util.cnc medium\n\n")

        # 顯示配體
        f.write("# Show ligand\n")
        f.write("show sticks, organic\n")
        f.write("color cyan, organic\n")
        f.write("util.cnc organic\n\n")

        # 標籤
        if len(freq_categories['ultra_high']) > 0:
            f.write("# Labels for ultra high frequency residues\n")
            f.write("label ultra_high and name CA, \"%s%s\" % (resn, resi)\n")
            f.write("set label_size, -0.5\n")
            f.write("set label_color, red\n\n")

        # 視圖
        if len(consensus_residues) > 0:
            all_consensus_pos = '+'.join([str(pos + 2) for pos, freq in consensus_residues])
            f.write(f"select all_consensus, GHSR and resi {all_consensus_pos}\n")
            f.write("zoom all_consensus\n")
        f.write("orient\n\n")

        # 打印信息
        f.write("# Print summary\n")
        f.write(f"print 'Total consensus residues: {len(consensus_residues)}'\n")
        f.write(f"print 'Ultra high (>75%%): {len(freq_categories['ultra_high'])}'\n")
        f.write(f"print 'High (50-75%%): {len(freq_categories['high'])}'\n")
        f.write(f"print 'Medium (25-50%%): {len(freq_categories['medium'])}'\n")

    print(f"   ✓ PyMOL 腳本: {script_file}")
    print(f"     執行: pymol {script_file}")

def main():
    args = parse_args()

    print("="*80)
    print("🧬 GHSR 注意力共識分析")
    print("   Growth Hormone Secretagogue Receptor - Attention Consensus Analysis")
    print("="*80)
    print()

    # 檢查文件
    if not os.path.exists(args.attention_file):
        print(f"❌ 注意力分數文件不存在: {args.attention_file}")
        sys.exit(1)

    # 載入注意力分數
    attention_matrix, drug_ids = load_attention_scores(args.attention_file)

    # 識別高注意力殘基
    high_attention_mask, thresholds = identify_high_attention_residues(
        attention_matrix,
        method=args.threshold_method,
        percentile=args.percentile,
        top_k=args.top_k
    )

    # 計算共識殘基
    consensus_freq, consensus_residues, freq_categories = compute_consensus_residues(
        high_attention_mask,
        min_frequency=args.min_frequency
    )

    # 保存結果
    stats = save_consensus_results(
        consensus_freq, consensus_residues, freq_categories,
        attention_matrix, args.output_dir, args.protein_length
    )

    print("\n" + "="*80)
    print("✅ 共識分析完成！")
    print("="*80)
    print()
    print(f"📁 輸出目錄: {args.output_dir}")
    print(f"🔬 共識殘基數: {len(consensus_residues)}")
    print(f"   🔴 超高頻率 (>75%): {stats['frequency_categories']['ultra_high']}")
    print(f"   🟠 高頻率 (50-75%): {stats['frequency_categories']['high']}")
    print(f"   🟡 中等頻率 (25-50%): {stats['frequency_categories']['medium']}")
    print()

    # 顯示 Top 10
    print("📊 Top 10 共識殘基:")
    for i, (pos, freq) in enumerate(consensus_residues[:10], 1):
        print(f"   {i:2d}. 位置 {pos:3d}  -  頻率: {freq*100:5.2f}%")
    print()

    print("📝 下一步：比較預測 vs. 真實口袋")
    print()
    print("   python compare_prediction_vs_truth_ghsr.py \\")
    print(f"       --predicted {args.output_dir}/GHSR_consensus_residues.csv \\")
    print("       --true_pocket datasets/GPCR_resarch/validation_results/GHSR_true_pocket.txt \\")
    print(f"       --output_dir {args.output_dir}")
    print()

if __name__ == "__main__":
    main()
