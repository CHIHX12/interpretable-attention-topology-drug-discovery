#!/usr/bin/env python3
"""
make_figures.py — regenerate the data-driven main-text figures for the v22 revision.

Figures written to 文章草稿/圖片_v22/ as 1200-dpi PNG and vector PDF:

  Fig3_data_and_performance : dataset composition, per-partition performance of
                              the model and of the ligand-only reference models,
                              and chemotype novelty (nearest-neighbour Tanimoto
                              and single-class scaffold fraction). Replaces the
                              LogP panel, whose argument is withdrawn.
  Fig4_residue_difference   : class difference per residue for the ten strongest
                              positions in each direction, with the spread over
                              the ten fine-tuning seeds and the value obtained
                              before efficacy fine-tuning.
  Fig5_pairing_network      : constitutive pairing of the class-associated
                              residues, drawn as sharing rings, with the
                              self-pairing error of the previous version removed.

Fig. 1 and Fig. 2 carry no data and are drawn by revision/make_schematics.py.

Usage:
    python revision/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

OUT = Path("文章草稿/圖片_v22")
RES_FIRST, RES_LAST, OFFSET = 37, 338, 2
SEEDS = "result/rev22/attention/featoff_seed*.npz"
PRETRAINED = "result/rev22/attention/featoff_pretrained.npz"
plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 1200,
                     "figure.constrained_layout.use": True})
RED, BLUE, GREY = "#C0392B", "#2471A3", "#7F8C8D"


def panel_label(ax, text, x=-0.13, y=1.02):
    """Panel letter only. The descriptive title belongs in the figure legend."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="left", va="bottom")


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=1200)
    plt.close(fig)
    print(f"saved {OUT/name}.png and .pdf")


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1)
    return out


def seed_deltas():
    import glob
    from scipy import stats
    cols = np.arange(RES_FIRST, RES_LAST + 1) - OFFSET
    mats, qs = [], []
    for f in sorted(glob.glob(SEEDS)):
        z = np.load(f, allow_pickle=True)
        a = z["unmasked_max"][:, cols]
        y = z["label"].astype(int)
        mats.append(a[y == 1].mean(0) - a[y == 0].mean(0))
        _, pv = stats.ttest_ind(a[y == 1], a[y == 0], equal_var=False)
        qs.append(bh(np.nan_to_num(pv, nan=1.0)))
    z = np.load(PRETRAINED, allow_pickle=True)
    a = z["unmasked_max"][:, cols]
    y = z["label"].astype(int)
    return np.vstack(mats), a[y == 1].mean(0) - a[y == 0].mean(0), (np.vstack(qs) < 0.05).sum(0)


def figure3():
    df = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv")
    summary = pd.read_csv("result/rev22/summary_table.csv")
    splits = json.loads(Path("datasets/GPCR_resarch/rev22_split_summary.json").read_text())

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))

    # (a) composition
    ax = axes[0]
    n1, n0 = int((df.Y == 1).sum()), int((df.Y == 0).sum())
    ax.pie([n1, n0], labels=[f"EC50-derived\nn = {n1}", f"IC50-derived\nn = {n0}"],
           colors=[RED, BLUE], autopct="%1.1f%%", startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.2},
           textprops={"color": "black"})
    panel_label(ax, "(a)", x=-0.02)

    # (b) performance per partition family
    ax = axes[1]
    fams = ["random", "compound", "scaffold"]
    labels = ["record-level\n(deposited)", "compound-\ndisjoint", "scaffold-\ndisjoint"]
    models = [("DrugBAN-BiLSTM (pretrained, descriptors on)", "this model", "#1F618D"),
              ("DrugBAN-BiLSTM (scratch, descriptors on)", "no pre-training", "#85C1E9"),
              ("logreg_ecfp4", "ECFP4 logistic regression", "#D68910"),
              ("rf_ecfp4", "ECFP4 random forest", "#F5CBA7")]
    x = np.arange(len(fams))
    w = 0.2
    for i, (key, lab, col) in enumerate(models):
        m = [summary[(summary.family == f) & (summary.model == key)].auroc_mean.values for f in fams]
        e = [summary[(summary.family == f) & (summary.model == key)].auroc_sd.values for f in fams]
        m = [v[0] if len(v) else np.nan for v in m]
        e = [0 if not len(v) or np.isnan(v[0]) else v[0] for v in e]
        ax.bar(x + (i - 1.5) * w, m, w, yerr=e, capsize=2, label=lab, color=col,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.9, 1.0); ax.set_ylabel("test AUROC")
    ax.legend(fontsize=7, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, -0.17), ncol=2, columnspacing=1.4,
              handlelength=1.4, handletextpad=0.6)
    panel_label(ax, "(b)")

    # (c) chemotype novelty: per-ligand nearest-neighbour similarity to training
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    ax = axes[2]

    def fp(smiles):
        return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)

    curves = [("record-level (deposited)", "datasets/GPCR_resarch/random", "#AEB6BF"),
              ("compound-disjoint", "datasets/GPCR_resarch/rev22_compound_s0", "#5DADE2"),
              ("scaffold-disjoint", "datasets/GPCR_resarch/rev22_scaffold_s0", "#1F618D")]
    for label, folder, colour in curves:
        tr = pd.read_csv(f"{folder}/train.csv"); te = pd.read_csv(f"{folder}/test.csv")
        tr_fp = [fp(s) for s in tr.SMILES.unique()]
        sims = np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), tr_fp))
                         for s in te.SMILES.unique()])
        xs = np.sort(sims)
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=colour, lw=1.6, label=label)
    ax.set_xlabel("nearest-neighbour ECFP4 Tanimoto\nfrom each test ligand to the training set")
    ax.set_ylabel("cumulative fraction of test ligands")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    panel_label(ax, "(c)")
    save(fig, "Fig3_data_and_performance")


def figure4():
    d, pre, nsig = seed_deltas()
    df = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv")
    seq = df.Protein.iloc[0]
    resn = np.arange(RES_FIRST, RES_LAST + 1)
    mean, sd = d.mean(0), d.std(0, ddof=1)
    order = np.argsort(mean)
    pick = np.r_[order[-10:][::-1], order[:10]]
    labels = [f"{seq[r - OFFSET]}{r}" for r in resn[pick]]

    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ypos = np.arange(len(pick))[::-1]
    cols = [RED if mean[i] > 0 else BLUE for i in pick]
    ax.barh(ypos, mean[pick], xerr=sd[pick], color=cols, edgecolor="black",
            linewidth=0.4, error_kw={"lw": 0.8, "capsize": 2}, height=0.7)
    ax.plot(pre[pick], ypos, "o", color="black", ms=3.5, label="before efficacy fine-tuning")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos); ax.set_yticklabels(labels)
    ax.set_xlabel("Δ = mean attention, EC50-annotated − IC50-annotated ligands")
    xmax = np.abs(mean[pick]).max()
    for yv, i in zip(ypos, pick):
        off = 0.035 * xmax * (1 if mean[i] > 0 else -1)
        ax.text(mean[i] + sd[i] * np.sign(mean[i]) + off, yv, f"{nsig[i]}/10",
                va="center", ha="left" if mean[i] > 0 else "right", fontsize=6.5, color="#555555")
    ax.set_xlim(min(mean[pick]) * 1.25, max(mean[pick]) * 1.7)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.text(0.02, 0.98, "EC50-associated", transform=ax.transAxes, color=RED,
            va="top", fontsize=8)
    ax.text(0.02, 0.02, "IC50-associated", transform=ax.transAxes, color=BLUE,
            va="bottom", fontsize=8)
    save(fig, "Fig4_residue_difference")


def pairing(npz):
    cols = np.arange(RES_FIRST, RES_LAST + 1) - OFFSET
    resn = np.arange(RES_FIRST, RES_LAST + 1)
    pos = {r: i for i, r in enumerate(resn)}
    z = np.load(npz, allow_pickle=True)
    A = z["unmasked_max"][:, cols]
    y = z["label"].astype(int)
    m1, m0 = A[y == 1].mean(0), A[y == 0].mean(0)
    imp, delta = (m1 + m0) / 2, m1 - m0
    const = [r for r in resn if abs(delta[pos[r]]) < 0.1 and imp[pos[r]] > 1.5]
    out = {}
    for side, targets in (("EC50", [124, 125, 122, 198, 200]),
                          ("IC50", [287, 286, 278, 147, 116])):
        pairs = []
        for t in targets:
            cand = sorted([c for c in const if c != t],
                          key=lambda c: abs(imp[pos[t]] - imp[pos[c]]))[:5]
            pairs += [(t, c) for c in cand]
        out[side] = pairs
    return out


def figure5():
    df = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv")
    seq = df.Protein.iloc[0]
    pairs = pairing("result/rev22/attention/featoff_seed42.npz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (side, label) in zip(axes, [("EC50", "(a)"), ("IC50", "(b)")]):
        share = {}
        for t, c in pairs[side]:
            share.setdefault(c, set()).add(t)
        rings = {1: [], 2: [], 3: []}
        for c, ts in share.items():
            rings[min(3, len(ts))].append(c)
        for radius, level, lc in ((1.0, 1, "#AED6F1"), (0.68, 2, "#5499C7"), (0.36, 3, "#E67E22")):
            ax.add_patch(Circle((0, 0), radius, fill=False, color=lc, lw=1.4))
            nodes = sorted(rings[level])
            for k, c in enumerate(nodes):
                ang = 2 * np.pi * k / max(1, len(nodes)) + 0.3 * level
                x, y = radius * np.cos(ang), radius * np.sin(ang)
                ax.plot(x, y, "o", color=lc, ms=7, markeredgecolor="black", markeredgewidth=0.4)
                ax.text(x, y + 0.075, f"{seq[c - OFFSET]}{c}", ha="center", fontsize=7)
        ax.set_xlim(-1.25, 1.25); ax.set_ylim(-1.25, 1.25); ax.axis("off")
        panel_label(ax, label, x=0.0, y=0.98)
    save(fig, "Fig5_pairing_network")


if __name__ == "__main__":
    figure3()
    figure4()
    figure5()
