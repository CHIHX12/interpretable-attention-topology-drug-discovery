#!/usr/bin/env python3
"""
table1_rev22.py — Table 1 of the revision: class-differential residue attention.

Read-out: attention of the fine-tuned model with the four physicochemical
channels set to zero, averaged over heads and maximised over drug-graph rows
(the read-out used throughout this work). Delta(i) is the raw difference of
class means, as in the previous version, with Welch's t-test and
Benjamini-Hochberg q-values over the 302 analysed residues, plus the
across-seed stability of every entry and the distance of each residue to the
co-crystallised ligand.

Outputs (never overwritten):
    result/rev22/table1_residues.csv   all 302 residues
    result/rev22/table1_top.csv        the ten strongest residues per direction
    result/rev22/table1_summary.json   stability and control statistics

Usage:
    python revision/table1_rev22.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FIRST, LAST, OFFSET = 37, 338, 2
KEY = "unmasked_max"
PRETRAINED = "result/rev22/attention/featoff_pretrained.npz"
STRUCTURES = [("/home/cycheng/GSHR/8jsr.pdb", "R", "UYI", "dist_8JSR"),
              ("/home/cycheng/GSHR/6ko5.pdb", "A", "8QX", "dist_6KO5")]


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def ligand_distances(path, chain, ligand, residues):
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
    lig = np.array(lig)
    return np.array([np.min(np.linalg.norm(np.array(prot[r])[:, None] - lig[None], axis=-1))
                     if r in prot and len(lig) else np.nan for r in residues])


def delta_of(path, cols):
    z = np.load(path, allow_pickle=True)
    A = z[KEY][:, cols]
    y = z["label"].astype(int)
    return A[y == 1], A[y == 0]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="result/rev22/attention/featoff_epoch36.npz",
                    help="attention file of the model the table is computed from")
    ap.add_argument("--seed_glob", default="result/rev22/attention/featoff_seed*.npz",
                    help="attention files of the seed models used for the stability columns")
    ap.add_argument("--out_prefix", default="table1",
                    help="output files are result/rev22/<prefix>_residues.csv etc.")
    args = ap.parse_args()
    global REFERENCE, SEEDS
    REFERENCE = args.reference
    SEEDS = sorted(Path().glob(args.seed_glob))
    df = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv")
    seq = df.Protein.iloc[0]
    residues = np.arange(FIRST, LAST + 1)
    cols = residues - OFFSET
    letters = np.array([seq[c] for c in cols])

    act, inact = delta_of(REFERENCE, cols)
    delta = act.mean(0) - inact.mean(0)
    t, p = stats.ttest_ind(act, inact, equal_var=False)
    seed_deltas = np.vstack([np.subtract(*[a.mean(0) for a in delta_of(str(f), cols)]) for f in SEEDS])
    pre_act, pre_inact = delta_of(PRETRAINED, cols)
    pre_delta = pre_act.mean(0) - pre_inact.mean(0)

    table = pd.DataFrame({
        "residue": [f"{l}{r}" for l, r in zip(letters, residues)],
        "resnum": residues,
        "delta": delta,
        "p_welch": p,
        "q_bh": bh(np.nan_to_num(p, nan=1.0)),
        "rank_agonist_side": (-delta).argsort().argsort() + 1,
        "delta_mean_10seeds": seed_deltas.mean(0),
        "delta_sd_10seeds": seed_deltas.std(0, ddof=1),
        "delta_pretrained_only": pre_delta,
        "responder": np.load(REFERENCE, allow_pickle=True)[KEY][:, cols].std(0) > 1e-6,
    })
    for path, chain, lig, name in STRUCTURES:
        table[name] = ligand_distances(path, chain, lig, residues)
    within = table.groupby(table.residue.str[0]).delta.transform(lambda s: (s - s.mean()) / s.std(ddof=0))
    table["z_within_amino_acid_type"] = within

    out_dir = Path("result/rev22")
    table.sort_values("rank_agonist_side").to_csv(out_dir / f"{args.out_prefix}_residues.csv", index=False)
    top = pd.concat([table.nlargest(10, "delta"), table.nsmallest(10, "delta")])
    top.to_csv(out_dir / f"{args.out_prefix}_top.csv", index=False)

    n_seeds = len(SEEDS)
    summary = {
        "reference_model": REFERENCE, "n_seed_models": n_seeds,
        "spearman_reference_vs_pretrained": float(stats.spearmanr(delta, pre_delta).correlation),
        "median_pairwise_spearman_across_seeds": float(np.median([
            stats.spearmanr(seed_deltas[i], seed_deltas[j]).correlation
            for i in range(n_seeds) for j in range(i + 1, n_seeds)])),
        "n_responders": int(table.responder.sum()),
        "n_zero_delta": int((table.delta.abs() < 1e-3).sum()),
        "top10_agonist": table.nlargest(10, "delta").residue.tolist(),
        "top10_antagonist": table.nsmallest(10, "delta").residue.tolist(),
        "ten_v21_residues": {},
    }
    for r in (124, 125, 122, 198, 200, 287, 286, 278, 147, 116, 283):
        row = table[table.resnum == r].iloc[0]
        summary["ten_v21_residues"][row.residue] = {
            "delta": round(float(row.delta), 4), "q_bh": float(f"{row.q_bh:.3g}"),
            "rank": int(row.rank_agonist_side),
            "delta_mean_10seeds": round(float(row.delta_mean_10seeds), 4),
            "delta_sd_10seeds": round(float(row.delta_sd_10seeds), 4),
            "delta_pretrained_only": round(float(row.delta_pretrained_only), 4),
            "dist_8JSR": None if np.isnan(row.dist_8JSR) else round(float(row.dist_8JSR), 2),
            "dist_6KO5": None if np.isnan(row.dist_6KO5) else round(float(row.dist_6KO5), 2),
            "z_within_type": round(float(row.z_within_amino_acid_type), 2)}
    (out_dir / f"{args.out_prefix}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
