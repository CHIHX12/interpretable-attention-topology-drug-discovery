#!/usr/bin/env python3
"""
differential_attention.py — class-differential residue attention (v22 revision).

For each attention file produced by extract_attention_masked.py:
  1. keep GHSR residues 37-338 (the 302 residues resolved in PDB 8JSR, as in
     the original analysis); position index + 2 = GHSR residue number;
  2. normalise each ligand's attention profile (z-score across residues,
     primary; softmax across residues, secondary) so that ligand-level offsets
     and scale do not enter the residue comparison;
  3. compute Delta(i) = mean_agonist - mean_antagonist, Welch's t-test,
     Benjamini-Hochberg q-value and Hedges' g, for each ligand subset
     (all ligands / deposited test split / dual-endpoint ligands removed).

Across several models (e.g. ten seeds) it also reports rank stability:
pairwise Spearman correlation of the Delta profiles and how often each residue
appears among the top-k agonist- or antagonist-preferred residues.

Usage:
    python revision/differential_attention.py --att result/rev22/attention/random_pre_s*.npz \
        --tag random_pretrained --out_dir result/rev22/differential
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DATA = Path("datasets/GPCR_resarch/GHSR_training_data.csv")
TEST = Path("datasets/GPCR_resarch/random/test.csv")
MANUSCRIPT = Path("result/class_attention_analysis_pdb/attention_analysis_pdb_numbering_CORRECTED.csv")
FIRST, LAST, OFFSET = 37, 338, 2
TOPK = 10


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    d = (a.mean(0) - b.mean(0)) / sp
    return d * (1 - 3 / (4 * (na + nb) - 9))


def normalise(x, how):
    if how == "z":
        return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True)
    if how == "softmax":
        e = np.exp(x - x.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    raise ValueError(how)


def differential(x, y):
    a, b = x[y == 1], x[y == 0]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    return pd.DataFrame({
        "delta": a.mean(0) - b.mean(0), "t": t, "p": p, "q": bh(p), "hedges_g": hedges_g(a, b),
        "mean_agonist": a.mean(0), "sd_agonist": a.std(0, ddof=1),
        "mean_antagonist": b.mean(0), "sd_antagonist": b.std(0, ddof=1),
        "n_agonist": len(a), "n_antagonist": len(b)})


def subsets(df):
    key = list(zip(df.SMILES, df.Y))
    test = pd.read_csv(TEST)
    test_keys = set(zip(test.SMILES, test.Y))
    dual = df.groupby("SMILES").Y.nunique()
    dual = set(dual[dual > 1].index)
    return {
        "all": np.ones(len(df), bool),
        "random_test": np.array([k in test_keys for k in key]),
        "no_dual": ~df.SMILES.isin(dual).values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", default="result/rev22/differential")
    ap.add_argument("--agg", default="masked_mean",
                    choices=["masked_mean", "masked_max", "unmasked_mean", "virtual_mean"])
    args = ap.parse_args()

    out_dir = Path(args.out_dir) / args.tag / args.agg
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists — refusing to overwrite")
    out_dir.mkdir(parents=True)

    df = pd.read_csv(DATA)
    seq = df.Protein.iloc[0]
    residues = np.arange(FIRST, LAST + 1)
    cols = residues - OFFSET
    labels = [f"{seq[c]}{r}" for c, r in zip(cols, residues)]
    subs = subsets(df)
    summary = {"tag": args.tag, "agg": args.agg, "models": [], "subsets": {}}

    per_model = {}
    for f in sorted(args.att):
        z = np.load(f, allow_pickle=True)
        assert np.array_equal(z["label"].astype(int), df.Y.values), f"row order mismatch in {f}"
        x_raw = z[args.agg][:, cols]
        size_corr = float(np.corrcoef(z[args.agg][:, :365].mean(1), z["n_atoms"])[0, 1])
        name = Path(f).stem
        summary["models"].append({"file": f, "corr_ligand_mean_vs_n_atoms": size_corr})
        for how in ("z", "softmax"):
            x = normalise(x_raw, how)
            for sub, mask in subs.items():
                res = differential(x[mask], df.Y.values[mask])
                res.insert(0, "residue", labels)
                res.insert(1, "resnum", residues)
                res.to_csv(out_dir / f"{name}__{how}__{sub}.csv", index=False)
                per_model.setdefault((how, sub), {})[name] = res

    for (how, sub), models in per_model.items():
        names = sorted(models)
        deltas = np.vstack([models[n].delta.values for n in names])
        rho = stats.spearmanr(deltas, axis=1).correlation if len(names) > 1 else np.array([[1.0]])
        rho = np.atleast_2d(rho)
        off = rho[np.triu_indices(len(names), 1)] if len(names) > 1 else np.array([1.0])
        top_pos = pd.Series(0, index=labels)
        top_neg = pd.Series(0, index=labels)
        sig_pos = pd.Series(0, index=labels)
        sig_neg = pd.Series(0, index=labels)
        for n in names:
            r = models[n].set_index("residue")
            top_pos[r.delta.nlargest(TOPK).index] += 1
            top_neg[r.delta.nsmallest(TOPK).index] += 1
            sig_pos[r.index[(r.q < 0.05) & (r.delta > 0)]] += 1
            sig_neg[r.index[(r.q < 0.05) & (r.delta < 0)]] += 1
        stab = pd.DataFrame({
            "residue": labels, "resnum": residues,
            "delta_mean": deltas.mean(0), "delta_sd": deltas.std(0, ddof=1) if len(names) > 1 else 0.0,
            "top10_agonist_count": top_pos.values, "top10_antagonist_count": top_neg.values,
            "fdr_sig_agonist_count": sig_pos.values, "fdr_sig_antagonist_count": sig_neg.values,
            "n_models": len(names)})
        stab.to_csv(out_dir / f"STABILITY__{how}__{sub}.csv", index=False)
        first = models[names[0]]
        summary["subsets"][f"{how}__{sub}"] = {
            "n_agonist": int(first.n_agonist.iloc[0]), "n_antagonist": int(first.n_antagonist.iloc[0]),
            "spearman_between_models_median": float(np.median(off)),
            "spearman_between_models_min": float(np.min(off)),
            "fdr_sig_agonist_per_model": [int(((models[n].q < 0.05) & (models[n].delta > 0)).sum()) for n in names],
            "fdr_sig_antagonist_per_model": [int(((models[n].q < 0.05) & (models[n].delta < 0)).sum()) for n in names],
            "consensus_top_agonist": stab.sort_values(["top10_agonist_count", "delta_mean"], ascending=False)
            .head(TOPK)[["residue", "top10_agonist_count", "delta_mean"]].values.tolist(),
            "consensus_top_antagonist": stab.sort_values(["top10_antagonist_count", "delta_mean"], ascending=[False, True])
            .head(TOPK)[["residue", "top10_antagonist_count", "delta_mean"]].values.tolist(),
        }

    if MANUSCRIPT.exists():
        old = pd.read_csv(MANUSCRIPT).set_index("PDB_Residue")
        new = pd.read_csv(out_dir / "STABILITY__z__all.csv").set_index("resnum")
        common = old.index.intersection(new.index)
        summary["vs_manuscript_v21"] = {
            "spearman_delta": float(stats.spearmanr(old.loc[common, "Difference"], new.loc[common, "delta_mean"]).correlation),
            "manuscript_residues_new_rank": {
                str(r): {"new_delta_mean": float(new.loc[r, "delta_mean"]),
                         "new_rank_desc": int((new.delta_mean > new.loc[r, "delta_mean"]).sum() + 1),
                         "top10_agonist_count": int(new.loc[r, "top10_agonist_count"]),
                         "top10_antagonist_count": int(new.loc[r, "top10_antagonist_count"])}
                for r in (124, 125, 122, 198, 200, 287, 286, 278, 147, 116, 283)},
        }

    (out_dir / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
