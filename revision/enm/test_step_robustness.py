#!/usr/bin/env python3
"""
test_step_robustness.py — does the choice of step count change the CONCLUSION?

R magnitudes drift with the number of integration steps (the system is weakly
damped; far hubs keep accumulating peak). This script tests whether the two
things the paper actually relies on are robust to step count:
  (1) the RANKING of all 293 residues by R  (Spearman vs the 300-step reference)
  (2) the DIRECTION of the 10 ML hubs       (Active stay R>1, Inactive stay R<1)

Outputs: FigS1_enm_step_robustness.{png,pdf}
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- submission figure convention -------------------------------------------
# Panels carry a letter only. All descriptive wording lives in the figure
# legend of the Supplementary Information. Figures are drawn at the printed
# double-column width so the type sizes here are the printed ones.
import matplotlib as _mpl
_mpl.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                      "xtick.labelsize": 6.5, "ytick.labelsize": 7,
                      "legend.fontsize": 7, "axes.linewidth": 0.7,
                      "lines.linewidth": 1.0, "savefig.dpi": 1200})
SI_WIDTH = 6.73  # sizes scaled to this width


def _panel(ax, letter):
    ax.text(-0.09, 1.03, f"({letter})", transform=ax.transAxes, fontsize=5.5,
            fontweight="bold", ha="left", va="bottom")
# ----------------------------------------------------------------------------


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVE = [124, 122, 125, 198, 200]
INACTIVE = [287, 286, 278, 147, 116]
CHECKPOINTS = [100, 200, 300, 500, 1000, 2000]
REF = 300

with open(os.path.join(SCRIPT_DIR, "_data_compact.json")) as f:
    data = json.load(f)
id2name = {n["id"]: n["name"] for n in data["nodes"]}
ids = sorted(id2name); N = len(ids); idx = {r: i for i, r in enumerate(ids)}
K = np.zeros((N, N))
for n in data["nodes"]:
    i = idx[n["id"]]
    for s, k in n["neighbors"].items():
        j = int(s)
        if j in idx:
            K[i][idx[j]] = k; K[idx[j]][i] = k

ai = [idx[h] for h in ACTIVE]; ii = [idx[h] for h in INACTIVE]


def rankdata(a):
    """Average ranks (ties shared), no scipy needed."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    inv = np.empty(len(a), int); inv[order] = np.arange(len(a))
    a_sorted = a[order]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    counts = np.r_[np.nonzero(obs)[0], len(a)]
    return 0.5 * (counts[dense] + counts[dense - 1] + 1)


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


# ── Strike ALL residues at once (columns = strikes), record peak at checkpoints ──
X = np.eye(N); V = np.zeros((N, N)); peak = np.abs(X).copy(); D = K.sum(1)
Rcp = {}
for step in range(1, max(CHECKPOINTS) + 1):
    F = K @ X - D[:, None] * X - 0.05 * V
    V += F * 0.005; X += V * 0.005
    np.maximum(peak, np.abs(X), out=peak)
    if step in CHECKPOINTS:
        asum = peak[ai, :].sum(0); isum = peak[ii, :].sum(0)
        Rcp[step] = asum / (isum + 1e-12)

print("Spearman rank correlation of all 293 residues vs the 300-step ranking:")
rho = {}
for s in CHECKPOINTS:
    rho[s] = spearman(Rcp[s], Rcp[REF])
    print(f"  {s:>5} steps:  rho = {rho[s]:.4f}")

print("\nThe 10 hubs' R across step counts (Active should stay >1, Inactive <1):")
for h in ACTIVE + INACTIVE:
    vals = "  ".join(f"{Rcp[s][idx[h]]:8.3f}" for s in CHECKPOINTS)
    print(f"  {id2name[h]:>8} ({'A' if h in ACTIVE else 'I'}): {vals}")

# ── Figure ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "mathtext.default": "regular",
})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(SI_WIDTH, 2.97), constrained_layout=True)

fig.set_layout_engine("constrained")
# Panel A: Spearman rho vs step count
xs = list(CHECKPOINTS)
axA.plot(xs, [rho[s] for s in xs], "o-", color="#1A9E8F", lw=0.95, ms=4.4,
         markeredgecolor="white", markeredgewidth=1.2)
for s in xs:
    axA.annotate(f"{rho[s]:.3f}", (s, rho[s]), textcoords="offset points",
                 xytext=(0, 12), ha="center", fontsize=5.5, fontweight="bold")
axA.axvline(REF, color="#888", ls="--", lw=0.55)
axA.text(REF, axA.get_ylim()[0], " reference (300)", color="#666", fontsize=5.5,
         va="bottom", ha="left")
axA.set_xscale("log")
axA.set_xticks(xs); axA.set_xticklabels([str(s) for s in xs])
axA.set_ylim(0.72, 1.01)
axA.set_xlabel("Integration steps")
axA.set_ylabel("Spearman ρ  (293-residue ranking vs 300-step)")
_panel(axA, "a")
axA.grid(alpha=0.3)
axA.spines[["top", "right"]].set_visible(False)

# Panel B: each hub's R across step counts (labels spread to avoid overlap)
def spread_labels(ax, pairs, x_end, x_lab, gap=0.42):
    """Place every label in one pass, so the two colour groups cannot collide."""
    pairs = sorted(pairs, key=lambda t: t[0])      # (log10 y, name, colour)
    placed = []
    for ly, _, _ in pairs:
        placed.append(ly if not placed else max(ly, placed[-1] + gap))
    for (ly, name, colour), yp in zip(pairs, placed):
        ax.annotate(name, xy=(x_end, 10 ** ly), xytext=(x_lab, 10 ** yp),
                    fontsize=5.5, color=colour, va="center", ha="left", fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=colour, lw=0.4, alpha=0.5))

for h in ACTIVE:
    axB.plot(xs, [Rcp[s][idx[h]] for s in xs], "o-", color="#E8731C", lw=0.79, ms=3.2, alpha=0.9)
for h in INACTIVE:
    axB.plot(xs, [max(Rcp[s][idx[h]], 1e-4) for s in xs], "s-", color="#1F6FBF", lw=0.79, ms=2.8, alpha=0.9)
x_end = xs[-1]; x_lab = x_end * 1.35
spread_labels(axB,
              [(np.log10(Rcp[x_end][idx[h]]), id2name[h], "#E8731C") for h in ACTIVE]
              + [(np.log10(max(Rcp[x_end][idx[h]], 1e-4)), id2name[h], "#1F6FBF")
                 for h in INACTIVE],
              x_end, x_lab)
axB.set_xlim(xs[0] * 0.85, x_end * 2.4)
axB.axhline(1.0, color="black", ls="--", lw=0.55)
axB.text(xs[0], 1.15, "R = 1 (crossover)", fontsize=5.5, va="bottom")
axB.set_xscale("log"); axB.set_yscale("log")
axB.set_xticks(xs); axB.set_xticklabels([str(s) for s in xs])
axB.set_xlabel("Integration steps")
axB.set_ylabel("R (log scale)")
_panel(axB, "b")
axB.grid(alpha=0.3, which="both")
axB.spines[["top", "right"]].set_visible(False)
axB.legend(handles=[
    plt.Line2D([0], [0], color="#E8731C", marker="o", lw=0.8, label="Active hub"),
    plt.Line2D([0], [0], color="#1F6FBF", marker="s", lw=0.8, label="Inactive hub"),
], loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
for ext in ("png", "pdf"):
    out = os.path.join(SCRIPT_DIR, f"FigS1_enm_step_robustness.{ext}")
    fig.savefig(out, dpi=1200, facecolor="white")
    print(f"saved {out}")
