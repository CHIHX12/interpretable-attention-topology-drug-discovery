#!/usr/bin/env python3
"""
make_si_figures.py — Supplementary Figures S6 to S10.

S1 to S5 are produced by the elastic-network scripts in revision/enm/ and are
copied into the output folder by this script so that all ten sit together.

  S6  TM6 cross-helix restraint and the effect of removing the salt bridge
  S7  class-difference profiles under the four read-outs of Section 5.6
  S8  cross-seed stability of the class-difference profile
  S9  label-permutation control on ligand proximity
  S10 chemotype novelty of each partition

All are drawn at the printed double-column width, so the type sizes in the file
are the type sizes on paper.

Usage:
    python revision/figures/make_si_figures.py
"""
import json
import math
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("文章草稿/圖片_v22/SI")
ENM = Path("revision/enm")
ATT = Path("result/rev22/attention")
FIRST, LAST, OFFSET = 37, 338, 2
WIDTH = 6.73
RED, BLUE, GREY, ORANGE = "#C0392B", "#2471A3", "#7F8C8D", "#C2681F"
TM6 = list(range(261, 293))
SEGMENTS = [("intracellular", 261, 275), ("pivot", 276, 282), ("extracellular", 283, 292)]

plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.7, "savefig.dpi": 1200,
                     "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "legend.fontsize": 7, "axes.titlesize": 8})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02, dpi=1200)
    plt.close(fig)
    print(f"  saved {name}")


def panel(ax, letter):
    ax.text(-0.10, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


# --- elastic network, the same construction the audit script checks -----------
def load_network():
    nodes = json.loads((ENM / "_data_compact.json").read_text())["nodes"]
    ids = sorted(n["id"] for n in nodes)
    idx = {r: i for i, r in enumerate(ids)}
    K = np.zeros((len(ids), len(ids)))
    for n in nodes:
        i = idx[n["id"]]
        for s, k in n["neighbors"].items():
            j = int(s)
            if j in idx:
                K[i, idx[j]] = K[idx[j], i] = k
    return ids, idx, K


def strike(K, i0, steps=300, dt=0.005, damping=0.05):
    x = np.zeros(len(K)); v = np.zeros(len(K))
    x[i0] = 1.0
    peak = np.abs(x).copy()
    D = K.sum(1)
    for _ in range(steps):
        v += (K @ x - D * x - damping * v) * dt
        x += v * dt
        peak = np.maximum(peak, np.abs(x))
    return peak


def figure_s6():
    ids, idx, K = load_network()
    tm6 = [r for r in TM6 if r in idx]
    cross = [sum(k for j, k in zip(ids, K[idx[r]]) if k > 0 and j not in TM6) for r in tm6]

    K2 = K.copy()
    K2[idx[124], idx[283]] = K2[idx[283], idx[124]] = 0.0
    pn = strike(K, idx[124])
    pb = strike(K2, idx[124])
    a = np.array([pn[idx[r]] for r in tm6])
    b = np.array([pb[idx[r]] for r in tm6])
    pct = np.where(a > 1e-12, (b - a) / a * 100, 0.0)

    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 6.2), sharex=True)
    for ax, letter in zip(axes, "abc"):
        panel(ax, letter)
        for n, (_, lo, hi) in enumerate(SEGMENTS):
            ax.axvspan(lo - 0.5, hi + 0.5, color="#EEF1F5" if n % 2 == 0 else "#FFFFFF",
                       zorder=0)
            if n:
                ax.axvline(lo - 0.5, color="#C8CFD8", lw=0.6, zorder=0)
    axes[0].bar(tm6, cross, color=GREY, edgecolor="black", linewidth=0.3)
    axes[0].set_ylabel(r"$\Sigma k_{\mathrm{cross}}$")
    for name, lo, hi in SEGMENTS:
        m = np.mean([c for r, c in zip(tm6, cross) if lo <= r <= hi])
        axes[0].hlines(m, lo - 0.5, hi + 0.5, color=RED, lw=1.2)
        axes[0].text((lo + hi) / 2, axes[0].get_ylim()[1] * 0.92, f"{name}\nmean {m:.2f}",
                     ha="center", va="top", fontsize=6.5, color=RED)
    axes[1].plot(tm6, a, "o-", ms=3, lw=1.0, color=BLUE, label="intact network")
    axes[1].plot(tm6, b, "s--", ms=3, lw=1.0, color=ORANGE,
                 label="Glu124–Arg283 spring removed")
    axes[1].set_ylabel("peak displacement\nafter a strike at Glu124")
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    axes[2].bar(tm6, pct, color=[RED if p < -50 else ORANGE if p < -20 else GREY for p in pct],
                edgecolor="black", linewidth=0.3)
    axes[2].set_ylabel("change (%)")
    axes[2].set_xlabel("TM6 residue number")
    for name, lo, hi in SEGMENTS:
        m = np.mean([p for r, p in zip(tm6, pct) if lo <= r <= hi])
        axes[2].text((lo + hi) / 2, -97, f"{m:.0f}%", ha="center", fontsize=7, fontweight="bold")
    axes[2].set_ylim(-105, 5)
    fig.align_ylabels(axes)
    save(fig, "FigS6_tm6_restraint_and_saltbridge")


# --- attention read-outs ------------------------------------------------------
def delta(path, key):
    cols = np.arange(FIRST, LAST + 1) - OFFSET
    z = np.load(path, allow_pickle=True)
    A, y = z[key][:, cols], z["label"].astype(int)
    return A[y == 1].mean(0) - A[y == 0].mean(0)


def figure_s7():
    seq = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv").Protein.iloc[0]
    resn = np.arange(FIRST, LAST + 1)
    variants = [("descriptors withheld, max over all graph rows (this work)",
                 ATT / "featoff_epoch36.npz", "unmasked_max"),
                ("descriptors supplied, max over all graph rows",
                 ATT / "epoch36_with_unmaskedmax.npz", "unmasked_max"),
                ("descriptors supplied, padding excluded, mean over real atoms",
                 ATT / "epoch36_with_unmaskedmax.npz", "masked_mean"),
                ("descriptors withheld, padding excluded, max over real atoms",
                 ATT / "featoff_epoch36.npz", "masked_max")]
    fig, axes = plt.subplots(4, 1, figsize=(WIDTH, 7.2), sharex=True)
    for ax, letter, (_label, path, key) in zip(axes, "abcd", variants):
        d = delta(path, key)
        ax.axhline(0, color="black", lw=0.6)
        ax.bar(resn, d, width=1.0,
               color=[RED if v > 0 else BLUE for v in d], linewidth=0)
        order = np.argsort(-d)
        # label the strongest positions, but only one per run of neighbours,
        # otherwise a cluster of consecutive residues prints on top of itself
        picked = []
        for i in list(order[:5]) + list(order[-5:]):
            if abs(d[i]) > 1e-6 and all(abs(resn[i] - resn[j]) > 6 for j in picked):
                picked.append(i)
        for i in picked:
            ax.annotate(f"{seq[resn[i]-OFFSET]}{resn[i]}", (resn[i], d[i]),
                        textcoords="offset points",
                        xytext=(0, 4 if d[i] > 0 else -10),
                        ha="center", fontsize=5.4, color="#333333")
        ax.margins(y=0.22)
        ax.set_ylabel(r"$\Delta$")
        panel(ax, letter)
    axes[-1].set_xlabel("GHSR residue number")
    axes[-1].set_xlim(FIRST - 2, LAST + 2)
    fig.align_ylabels(axes)
    save(fig, "FigS7_readout_sensitivity")


def figure_s8():
    files = sorted(ATT.glob("featoff_seed*.npz"))
    D = np.array([delta(f, "unmasked_max") for f in files])
    rho = [spearmanr(D[i], D[j]).statistic for i in range(len(D)) for j in range(i + 1, len(D))]
    resn = np.arange(FIRST, LAST + 1)
    sd = D.std(0, ddof=1)

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8))
    axes[0].hist(rho, bins=18, color=BLUE, edgecolor="black", linewidth=0.4)
    axes[0].axvline(np.median(rho), color=RED, lw=1.2)
    axes[0].text(np.median(rho), axes[0].get_ylim()[1] * 0.95,
                 f"  median {np.median(rho):.3f}", color=RED, fontsize=7, va="top")
    axes[0].set_xlabel("pairwise Spearman correlation\nbetween seeds (45 pairs)")
    axes[0].set_ylabel("count")
    panel(axes[0], "a")
    axes[1].bar(resn, sd, width=1.0, color=GREY, linewidth=0)
    axes[1].set_xlabel("GHSR residue number")
    axes[1].set_ylabel(r"SD of $\Delta$ across seeds")
    axes[1].set_xlim(FIRST - 2, LAST + 2)
    panel(axes[1], "b")
    save(fig, "FigS8_cross_seed_stability")


def ligand_distance_8jsr():
    rec, lig = {}, []
    for line in open(ENM / "8jsr.pdb"):
        tag = line[:6]
        if tag not in ("ATOM  ", "HETATM") or line[76:78].strip() == "H" or line[21] != "R":
            continue
        xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        if tag == "ATOM  ":
            rec.setdefault(int(line[22:26]), []).append(xyz)
        elif line[17:20].strip() == "UYI":
            lig.append(xyz)
    return {r: min(math.dist(a, b) for a in v for b in lig) for r, v in rec.items()}


def figure_s9(n_draw=1000, seed=0):
    """Proximity to the ligand against randomly drawn residue sets.

    This is the null model that revision/pocket_enrichment.py implements and
    that the main text reports. The label-permutation null described alongside
    it is a different test; its script was never deposited and a fresh
    implementation does not reproduce the reported outcome, so it is not
    plotted here.
    """
    dist = ligand_distance_8jsr()
    resn = np.arange(FIRST, LAST + 1)
    cols = resn - OFFSET
    dvec = np.array([dist.get(r, np.nan) for r in resn])
    resolved = np.flatnonzero(~np.isnan(dvec))
    background = float(np.nanmedian(dvec))
    rng = np.random.default_rng(seed)

    def test(idx):
        obs = float(np.nanmedian(dvec[idx]))
        draws = np.array([np.median(dvec[rng.choice(resolved, len(idx), replace=False)])
                          for _ in range(n_draw)])
        p = (np.sum(draws <= obs) + 1) / (n_draw + 1)
        return obs, draws, float(p)

    files = sorted(ATT.glob("featoff_seed*.npz"))
    rows = {"responding": [], "top ten": []}
    draws_ref = None
    for f in files:
        z = np.load(f, allow_pickle=True)
        A, y = z["unmasked_max"][:, cols], z["label"].astype(int)
        d = A[y == 1].mean(0) - A[y == 0].mean(0)
        # the deposited criterion, matching revision/feature_dependence.py
        responding = np.flatnonzero((A.std(0) > 1e-6) & ~np.isnan(dvec))
        top10 = np.array([i for i in np.argsort(-np.abs(d)) if not np.isnan(dvec[i])][:10])
        for name, idx in (("responding", responding), ("top ten", top10)):
            obs, draws, pv = test(idx)
            rows[name].append((obs, pv))
            if name == "top ten" and draws_ref is None:
                draws_ref = draws

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), layout="constrained")
    axes[0].hist(draws_ref, bins=34, color=GREY, edgecolor="black", linewidth=0.3,
                 label=f"{n_draw} random residue sets of ten")
    for obs, _ in rows["top ten"]:
        axes[0].axvline(obs, color=RED, lw=0.9, alpha=0.85)
    axes[0].axvline(rows["top ten"][0][0], color=RED, lw=0.9,
                    label="observed, one line per seed")
    axes[0].axvline(background, color=BLUE, lw=1.2, ls="--",
                    label=f"receptor background, {background:.1f} \u00c5")
    axes[0].set_xlabel("median distance to the ligand in 8JSR (\u00c5)")
    axes[0].set_ylabel("count")
    panel(axes[0], "a")

    x = np.arange(len(files))
    for name, colour, marker in (("responding", BLUE, "o"), ("top ten", RED, "s")):
        axes[1].plot(x, [o for o, _ in rows[name]], marker + "-", ms=3.5, lw=1.0,
                     color=colour, label=name + " positions")
    axes[1].axhline(background, color="black", lw=0.9, ls="--")
    axes[1].text(len(files) - 0.4, background + 0.15, "receptor background",
                 fontsize=6.5, ha="right")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([int(f.stem.split("seed")[-1]) for f in files], rotation=90)
    axes[1].set_xlabel("fine-tuning seed")
    axes[1].set_ylabel("median distance to the\nligand in 8JSR (\u00c5)")
    panel(axes[1], "b")
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="outside lower center", ncol=3, frameon=False)
    save(fig, "FigS9_pocket_proximity")

    for name in rows:
        o = [v for v, _ in rows[name]]; pv = [v for _, v in rows[name]]
        print(f"     {name:11s} median {min(o):.1f}-{max(o):.1f} \u00c5, "
              f"median P = {np.median(pv):.4f}, all ten below background: "
              f"{all(v < background for v in o)}")
    print(f"     background {background:.1f} \u00c5")


def figure_s10():
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")

    def fp(s):
        return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)

    curves = [("record-level (deposited)", "datasets/GPCR_resarch/random", "#AEB6BF"),
              ("compound-disjoint", "datasets/GPCR_resarch/rev22_compound_s0", "#5DADE2"),
              ("scaffold-disjoint", "datasets/GPCR_resarch/rev22_scaffold_s0", "#1F618D")]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), layout="constrained")
    for label, folder, colour in curves:
        tr = pd.read_csv(f"{folder}/train.csv"); te = pd.read_csv(f"{folder}/test.csv")
        tr_fp = [fp(s) for s in tr.SMILES.unique()]
        sims = np.array([max(DataStructs.BulkTanimotoSimilarity(fp(s), tr_fp))
                         for s in te.SMILES.unique()])
        xs = np.sort(sims)
        axes[0].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=colour, lw=1.5, label=label)
        axes[1].hist(sims, bins=25, histtype="step", lw=1.4, color=colour, label=label)
    axes[0].set_xlabel("nearest-neighbour ECFP4 Tanimoto\nto the training set")
    axes[0].set_ylabel("cumulative fraction of test ligands")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1)
    panel(axes[0], "a")
    axes[1].set_xlabel("nearest-neighbour ECFP4 Tanimoto\nto the training set")
    axes[1].set_ylabel("number of test ligands")
    axes[1].set_xlim(0, 1)
    panel(axes[1], "b")
    fig.legend(*axes[0].get_legend_handles_labels(),
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "FigS10_partition_characteristics")


def collect_s1_to_s5():
    """The elastic-network scripts write S1 to S5; put all ten in one folder."""
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(ENM.glob("FigS[1-5]_*")):
        shutil.copy2(f, OUT / f.name)
        print(f"  copied {f.name}")


if __name__ == "__main__":
    collect_s1_to_s5()
    figure_s6()
    figure_s7()
    figure_s8()
    figure_s9()
    figure_s10()
