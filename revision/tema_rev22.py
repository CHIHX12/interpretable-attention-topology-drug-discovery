#!/usr/bin/env python3
"""
tema_rev22.py — TEMA constitutive pairing recomputed on padding-masked attention.

Implements the pairing score exactly as written in Methods 2.5:
    S(t, c) = I_t * I_c * corr(A_t, A_c)
where A is the per-ligand attention profile normalised by softmax across the
302 resolved residues (so importances are positive and comparable), I_r is the
mean of A_r over all ligands, and corr is Pearson's r across ligands.

Targets   : top-k agonist- and antagonist-preferred residues from the
            cross-seed stability table (differential_attention.py, z / all).
Constitutive pool: residues in the lowest quartile of |mean Delta| (i.e. least
            class-discriminative) and in the top quartile of importance. With
            1,539 ligands and sequence-smooth profiles almost every residue is
            FDR-significant in at least one model, so significance cannot be
            used as the constitutive criterion as it was in version 21.
For every target the top-5 partners by S are reported, together with
dImp = |I_t - I_c| / I_t, the usage count U_c and the sequence separation
|t - c| (a diagnostic for sequence-smoothing of BiLSTM representations).

Usage:
    python revision/tema_rev22.py --att result/rev22/attention/random_pretrained_seed*.npz \
        --stability result/rev22/differential/random_pretrained/masked_mean/STABILITY__z__all.csv \
        --out_dir result/rev22/tema/random_pretrained
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FIRST, LAST, OFFSET = 37, 338, 2
K_TARGETS, K_PARTNERS = 5, 5


def softmax_rows(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="+", required=True)
    ap.add_argument("--stability", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists — refusing to overwrite")
    out_dir.mkdir(parents=True)

    stab = pd.read_csv(args.stability)
    resnums = np.arange(FIRST, LAST + 1)
    cols = resnums - OFFSET
    label = dict(zip(stab.resnum, stab.residue))

    ago = stab.sort_values(["top10_agonist_count", "delta_mean"], ascending=[False, False]).head(K_TARGETS).resnum.tolist()
    ant = stab.sort_values(["top10_antagonist_count", "delta_mean"], ascending=[False, True]).head(K_TARGETS).resnum.tolist()
    low_delta = stab[stab.delta_mean.abs() <= stab.delta_mean.abs().quantile(0.25)].resnum.values

    # average A across models (profiles are per ligand; models share row order)
    mats = [softmax_rows(np.load(f, allow_pickle=True)["masked_mean"][:, cols]) for f in sorted(args.att)]
    A = np.mean(mats, axis=0)
    imp = A.mean(0)
    pos = {r: i for i, r in enumerate(resnums)}
    q75 = np.quantile(imp, 0.75)
    pool = [r for r in low_delta if imp[pos[r]] >= q75]
    corr = np.corrcoef(A.T)

    rows = []
    for direction, targets in (("agonist", ago), ("antagonist", ant)):
        for t in targets:
            cand = [c for c in pool if c != t]
            s = np.array([imp[pos[t]] * imp[pos[c]] * corr[pos[t], pos[c]] for c in cand])
            for rank, j in enumerate(np.argsort(-s)[:K_PARTNERS], 1):
                c = cand[j]
                rows.append({"direction": direction, "target": label[t], "target_resnum": t,
                             "rank": rank, "partner": label[c], "partner_resnum": c,
                             "S": s[j], "corr": corr[pos[t], pos[c]],
                             "I_target": imp[pos[t]], "I_partner": imp[pos[c]],
                             "dImp_rel": abs(imp[pos[t]] - imp[pos[c]]) / imp[pos[t]],
                             "seq_separation": abs(int(t) - int(c))})
    pairs = pd.DataFrame(rows)
    usage = pairs.groupby(["direction", "partner"]).target.nunique().rename("U_c").reset_index()
    pairs = pairs.merge(usage, on=["direction", "partner"])
    pairs.to_csv(out_dir / "tema_pairs.csv", index=False)

    summary = {
        "n_models": len(mats), "constitutive_pool_size": len(pool),
        "agonist_targets": [label[t] for t in ago], "antagonist_targets": [label[t] for t in ant],
        "by_direction": {}}
    for d, g in pairs.groupby("direction"):
        summary["by_direction"][d] = {
            "median_corr": float(g["corr"].median()),
            "median_dImp_rel": float(g.dImp_rel.median()),
            "median_seq_separation": float(g.seq_separation.median()),
            "frac_partners_within_10_residues": float((g.seq_separation <= 10).mean()),
            "n_unique_partners": int(g.partner.nunique()),
            "frac_pairs_with_shared_partner": float((g.U_c > 1).mean())}
    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(pairs.to_string(index=False))


if __name__ == "__main__":
    main()
