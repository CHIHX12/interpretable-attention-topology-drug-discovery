#!/usr/bin/env python3
"""
pocket_enrichment.py — is the residue ranking enriched for binding-pocket residues?

For each attention configuration we take the strongest agonist-side and
antagonist-side residues and ask whether they sit closer to the co-crystallised
ligand than residues drawn at random from the same 302 analysed positions.
Distances are minimum heavy-atom distances to the ligand in the agonist-bound
structure 8JSR (anamorelin, ligand UYI) and in the antagonist-bound structure
6KO5 (ligand 8QX). Significance is a two-sided permutation test (100,000 draws)
on the median distance of the selected set.

It also reports how much of the variance of Delta is explained by amino-acid
identity alone (one-way ANOVA R^2), which distinguishes "the model marks
specific positions" from "the model marks specific residue types".

Usage:
    python revision/pocket_enrichment.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIRST, LAST, OFFSET = 37, 338, 2
DATA = Path("datasets/GPCR_resarch/GHSR_training_data.csv")
STRUCTURES = [("/home/cycheng/GSHR/8jsr.pdb", "R", "UYI", "8JSR agonist"),
              ("/home/cycheng/GSHR/6ko5.pdb", "A", "8QX", "6KO5 antagonist")]
TOPK = 10
NPERM = 100_000
RNG = np.random.default_rng(0)


def parse_pdb(path, chain, ligand):
    prot, lig = {}, []
    for line in open(path):
        tag = line[:6]
        if tag not in ("ATOM  ", "HETATM") or line[16] not in " A":
            continue
        name = line[12:16].strip()
        if not name or name[0] == "H":
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if tag == "HETATM" and line[17:20].strip() == ligand:
            lig.append(xyz)
        elif tag == "ATOM  " and line[21] == chain:
            prot.setdefault(int(line[22:26]), []).append(xyz)
    return prot, np.array(lig)


def distances(path, chain, ligand, residues):
    prot, lig = parse_pdb(path, chain, ligand)
    out = []
    for r in residues:
        atoms = prot.get(int(r))
        out.append(np.min(np.linalg.norm(np.array(atoms)[:, None, :] - lig[None, :, :], axis=-1))
                   if atoms is not None and len(lig) else np.nan)
    return np.array(out)


def permutation_p(values, selected_idx, valid):
    obs = np.nanmedian(values[selected_idx])
    pool = np.flatnonzero(valid)
    draws = np.array([np.median(values[RNG.choice(pool, len(selected_idx), replace=False)])
                      for _ in range(NPERM // 100)])
    p = (np.sum(draws <= obs) + 1) / (len(draws) + 1)
    return float(obs), float(min(1.0, 2 * min(p, 1 - p + 1e-12)))


def anova_r2(delta, letters):
    grand = delta.mean()
    ss_tot = ((delta - grand) ** 2).sum()
    ss_between = sum(len(delta[letters == a]) * (delta[letters == a].mean() - grand) ** 2
                     for a in np.unique(letters))
    return float(ss_between / ss_tot)


def main():
    df = pd.read_csv(DATA)
    seq = df.Protein.iloc[0]
    residues = np.arange(FIRST, LAST + 1)
    letters = np.array([seq[r - OFFSET] for r in residues])
    dists = {label: distances(path, ch, lig, residues) for path, ch, lig, label in STRUCTURES}
    for label, d in dists.items():
        print(f"{label}: {np.sum(~np.isnan(d))}/{len(d)} residues resolved, "
              f"{np.sum(d <= 4.5)} within 4.5 A of the ligand")

    configs = {}
    for name, path, key in [
        # main model: trained and read without the physicochemical channels
        ("main_trained_and_read_without_descriptors",
         "result/rev22/attention/trainfeatoff_seed42.npz", "unmasked_max"),
        # previous configuration: trained with descriptors, read without them
        ("read_without_descriptors_model_trained_with_them",
         "result/rev22/attention/featoff_epoch44_dec.npz", "unmasked_max"),
        ("descriptors_enabled_max", "result/rev22/attention/dec2025_seed42_epoch44.npz", "unmasked_max"),
        ("descriptors_enabled_masked_mean", "result/rev22/attention/dec2025_seed42_epoch44.npz", "masked_mean"),
    ]:
        f = Path(path)
        if not f.exists():
            print(f"missing {f} — skipped")
            continue
        z = np.load(f, allow_pickle=True)
        A = z[key][:, residues - OFFSET]
        y = z["label"].astype(int)
        configs[name] = A[y == 1].mean(0) - A[y == 0].mean(0)

    report = {}
    for name, delta in configs.items():
        entry = {"anova_r2_by_amino_acid": anova_r2(delta, letters)}
        order = np.argsort(delta)
        picks = {"agonist_side": order[-TOPK:], "antagonist_side": order[:TOPK]}
        for side, idx in picks.items():
            entry[side] = {"residues": [f"{letters[i]}{residues[i]}" for i in idx]}
            for label, d in dists.items():
                valid = ~np.isnan(d)
                obs, p = permutation_p(d, idx, valid)
                entry[side][label] = {"median_distance_A": round(obs, 2),
                                      "background_median_A": round(float(np.nanmedian(d)), 2),
                                      "permutation_p": round(p, 4)}
        report[name] = entry
        print(f"\n=== {name} (ANOVA R^2 by amino-acid identity = {entry['anova_r2_by_amino_acid']:.3f})")
        for side in picks:
            print(f"  {side}: {' '.join(entry[side]['residues'])}")
            for label in dists:
                s = entry[side][label]
                print(f"     {label}: median {s['median_distance_A']} A "
                      f"(background {s['background_median_A']} A), permutation p = {s['permutation_p']}")

    out = Path("result/rev22/pocket_enrichment.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
