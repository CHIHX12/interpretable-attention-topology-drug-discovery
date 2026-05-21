#!/usr/bin/env python3
"""
validatesequence PDB structure
Verify residue numbering correspondence between dataset and PDB structure
"""

import pandas as pd
from pathlib import Path

def read_pdb_sequence(pdb_file, chain='R'):
    """ PDB filereadsequenceresidue"""
    residues = []
    res_numbers = []

    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }

    seen_residues = set()

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                # PDB format
                atom_chain = line[21:22].strip()
                res_name = line[17:20].strip()
                res_num = int(line[22:26].strip())

                if atom_chain == chain and res_name in three_to_one:
                    if res_num not in seen_residues:
                        residues.append(three_to_one[res_name])
                        res_numbers.append(res_num)
                        seen_residues.add(res_num)

    return ''.join(residues), res_numbers


def main():
    print("=" * 80)
    print("validateresidue")
    print("Verifying Residue Numbering Correspondence")
    print("=" * 80)
    print()

    # 1. readsequence
    print("1. readsequence...")
    df = pd.read_csv('datasets/GPCR_resarch/GHSR_training_data.csv')
    dataset_seq = df['Protein'].iloc[0]
    print(f" sequencelength: {len(dataset_seq)} aa")
    print(f" 50 amino acid: {dataset_seq[:50]}")
    print()

    # 2. read PDB sequence
    print("2. read PDB structuresequence (Chain R)...")
    pdb_file = 'datasets/GPCR_resarch/GSHR_PDB/8JSR_R.pdb'
    pdb_seq, pdb_res_numbers = read_pdb_sequence(pdb_file, chain='R')
    print(f"   PDB sequencelength: {len(pdb_seq)} aa")
    print(f" PDB residuerange: {pdb_res_numbers[0]} - {pdb_res_numbers[-1]}")
    print(f" 50 amino acid: {pdb_seq[:50]}")
    print()

    # 3. sequence
    print("3. sequence...")

    # check PDB sequencesequence
    pdb_start_in_dataset = dataset_seq.find(pdb_seq)

    if pdb_start_in_dataset != -1:
        print(f" ✓ PDB sequence")
        print(f" ✓ PDB sequence Position {pdb_start_in_dataset + 1}")
    else:
        # sequence
        print(f" ⚠️ PDB sequencesequencematch")
        print(f" 100 residue...")

        # 100 
        match_count = sum(1 for i in range(min(100, len(pdb_seq), len(dataset_seq)))
                         if pdb_seq[i] == dataset_seq[i])
        print(f" 100 residuematch: {match_count}/100")

    print()

    # 4. compute (offset)
    print("4. compute...")
    print()

    # PDB residue
    pdb_first_res = pdb_res_numbers[0]

    # 1-based
    if pdb_start_in_dataset != -1:
        dataset_first_pos = pdb_start_in_dataset + 1
    else:
        dataset_first_pos = 1 # start

    # compute
    offset = pdb_first_res - dataset_first_pos

    print(f" PDB residue: {pdb_first_res} ({pdb_seq[0]})")
    print(f" : {dataset_first_pos} ({dataset_seq[dataset_first_pos-1]})")
    print(f" ✓ (Offset): {offset}")
    print()
    print(f" convert:")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f" PDB residue = Position + {offset}")
    print(f" Position = PDB residue - {offset}")
    print(f"   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()

    # 5. validate
    print("5. validateconvert 10 residue:")
    print()
    print(f"   {'Dataset Pos':<15} {'Dataset AA':<12} {'PDB Res':<12} {'PDB AA':<12} {'Match':<10}")
    print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")

    for i in range(min(10, len(pdb_seq))):
        dataset_pos = i + 1
        pdb_res = pdb_res_numbers[i]
        dataset_aa = dataset_seq[i]
        pdb_aa = pdb_seq[i]
        match = "✓" if dataset_aa == pdb_aa else "✗"

        print(f"   {dataset_pos:<15} {dataset_aa:<12} {pdb_res:<12} {pdb_aa:<12} {match:<10}")

    print()

    # 6. testkeyresidue
    print("6. convertanalysisresultkeyresidue:")
    print()

    # result
    key_residues = [
        (398, 'Q'),
        (286, 'S'),
        (29, 'L'),
        (397, 'L'),
        (28, 'S'),
    ]

    print(f"   {'Dataset Pos':<15} {'Dataset AA':<12} {'PDB Res':<12} {'Verified':<12}")
    print(f"   {'-'*15} {'-'*12} {'-'*12} {'-'*12}")

    for pos, aa in key_residues:
        pdb_res = pos + offset

        # validate
        if pos <= len(dataset_seq):
            dataset_aa = dataset_seq[pos - 1]
            verified = "✓" if dataset_aa == aa else f"✗ ({dataset_aa})"
        else:
            verified = "Out of range"

        print(f"   {pos:<15} {aa:<12} {pdb_res:<12} {verified:<12}")

    print()

    # 7. saveconvert
    print("7. generateconvert...")

    conversion_data = []
    for i in range(len(dataset_seq)):
        dataset_pos = i + 1
        dataset_aa = dataset_seq[i]
        pdb_res = dataset_pos + offset

        # check PDB range
        in_pdb = (pdb_res >= pdb_res_numbers[0] and pdb_res <= pdb_res_numbers[-1])

        conversion_data.append({
            'Dataset_Position': dataset_pos,
            'Dataset_AA': dataset_aa,
            'PDB_Residue': pdb_res,
            'In_PDB_Structure': in_pdb
        })

    conversion_df = pd.DataFrame(conversion_data)
    output_file = 'result/residue_numbering_conversion.csv'
    conversion_df.to_csv(output_file, index=False)

    print(f" ✓ convertsave: {output_file}")
    print(f" ✓ {len(conversion_df)} residue")
    print(f" ✓ PDB structure: {conversion_df['In_PDB_Structure'].sum()} residue")
    print()

    # 8. summary
    print("=" * 80)
    print("summary")
    print("=" * 80)
    print()
    print(f"✓ sequencelength: {len(dataset_seq)} aa")
    print(f"✓ PDB structurelength: {len(pdb_seq)} aa (Residues {pdb_res_numbers[0]}-{pdb_res_numbers[-1]})")
    print(f"✓ : {offset}")
    print()
    print(f" PyMOL use:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"# Position 398 (Q398)")
    print(f"# PDB residue = 398 + {offset} = {398 + offset}")
    print(f"select res398, chain R and resi {398 + offset}")
    print(f"show spheres, res398")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()


if __name__ == "__main__":
    main()
