#!/usr/bin/env python3
"""
pif_cwxp_compare.py — CWxP toggle + PIF connector across 3 GHSR states,
baseline = Antagonist (6KO5).  Agonist (8JSR) and Inverse (7F83) compared to it.

Two measurements, both from the crystals (reproducible):
  (a) PIF contact distances  : F272–P224 (F6.44–P5.50) and F272–V131 (F6.44–I3.40)
  (B) Toggle rotamers (χ)    : W276 (χ1,χ2), F272 (χ1,χ2)

Output: FigS3_pif_cwxp_compare.{png,pdf}
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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
STATES = [("6ko5.pdb", "A", "Antagonist\n(6KO5)", "#9E9E9E"),
          ("8jsr.pdb", "R", "Agonist\n(8JSR)", "#2E9E5B"),
          ("7f83.pdb", "A", "Inverse\n(7F83)", "#E8731C")]


def parse(pdb, chain):
    res = {}
    for l in open(os.path.join(SCRIPT_DIR, pdb)):
        if l[:6] in ("ATOM  ", "HETATM") and l[21] == chain and l[16] in " A":
            try:
                r = int(l[22:26])
            except ValueError:
                continue
            nm = l[12:16].strip()
            if nm and nm[0] != "H":
                res.setdefault(r, {})[nm] = np.array(
                    [float(l[30:38]), float(l[38:46]), float(l[46:54])])
    return res


def mindist(res, r1, r2):
    a, b = res.get(r1, {}).values(), res.get(r2, {}).values()
    a, b = list(a), list(b)
    return min(np.linalg.norm(x - y) for x in a for y in b) if a and b else np.nan


def dihedral(p):
    b0, b1, b2 = p[0] - p[1], p[2] - p[1], p[3] - p[2]
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def chi(res, r, names):
    try:
        return dihedral([res[r][n] for n in names])
    except KeyError:
        return np.nan


# ---- collect data ----
dist = {}      # state_label -> {contact: d}
rot = {}       # state_label -> {angle: deg}
labels = []
colors = []
for pdb, ch, lab, col in STATES:
    res = parse(pdb, ch)
    labels.append(lab); colors.append(col)
    dist[lab] = {
        "F272–P224\n(F6.44–P5.50)": mindist(res, 272, 224),
        "F272–V131\n(F6.44–I3.40)": mindist(res, 272, 131),
    }
    rot[lab] = {
        "W276 χ1": chi(res, 276, ["N", "CA", "CB", "CG"]),
        "W276 χ2": chi(res, 276, ["CA", "CB", "CG", "CD1"]),
        "F272 χ1": chi(res, 272, ["N", "CA", "CB", "CG"]),
        "F272 χ2": chi(res, 272, ["CA", "CB", "CG", "CD1"]),
    }

# ---- figure ----
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "mathtext.default": "regular",
})
fig, (axA, axB) = plt.subplots(1, 2, figsize=(SI_WIDTH, 2.97))
fig.set_layout_engine("constrained")
# Panel A: PIF distances
contacts = list(dist[labels[0]].keys())
x = np.arange(len(contacts)); w = 0.26
for i, lab in enumerate(labels):
    vals = [dist[lab][c] for c in contacts]
    bars = axA.bar(x + (i - 1) * w, vals, w, color=colors[i],
                   edgecolor="#333", lw=0.4, label=lab.replace("\n", " "))
    for b, v in zip(bars, vals):
        axA.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=5.5, fontweight="bold")
axA.axhline(4.0, color="red", ls="--", lw=0.51)
axA.text(len(contacts) - 0.5, 4.05, "~4 Å van der Waals contact", color="red",
         fontsize=5.5, ha="right", va="bottom")
axA.set_xticks(x); axA.set_xticklabels(contacts, fontsize=5.5)
axA.set_ylabel("min heavy-atom distance (Å)")
_panel(axA, "a")
axA.set_ylim(0, 7.2)
axA.spines[["top", "right"]].set_visible(False)

# Panel B: rotamers
angles = list(rot[labels[0]].keys())
xb = np.arange(len(angles))
for i, lab in enumerate(labels):
    vals = [rot[lab][a] for a in angles]
    bars = axB.bar(xb + (i - 1) * w, vals, w, color=colors[i],
                   edgecolor="#333", lw=0.4, label=lab.replace("\n", " "))
    for b, v in zip(bars, vals):
        axB.text(b.get_x() + b.get_width() / 2, v + (4 if v >= 0 else -4),
                 f"{v:.0f}", ha="center", va="bottom" if v >= 0 else "top",
                 fontsize=5.5, fontweight="bold")
axB.axhline(0, color="black", lw=0.4)
axB.set_xticks(xb); axB.set_xticklabels(angles)
axB.set_ylabel("side-chain dihedral (°)")
_panel(axB, "b")
axB.set_ylim(-200, 200)
axB.spines[["top", "right"]].set_visible(False)

# shared color key, one row, at the bottom
handles = [mpatches.Patch(facecolor=colors[i], edgecolor="#333",
                          label=labels[i].replace("\n", " ")) for i in range(len(labels))]
fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)

for ext in ("png", "pdf"):
    out = os.path.join(SCRIPT_DIR, f"FigS3_pif_cwxp_compare.{ext}")
    fig.savefig(out, dpi=1200, facecolor="white")
    print("saved", out)

# also print the table
print("\nPIF distances (Å):")
for c in contacts:
    print(f"  {c.splitlines()[0]:12}", "  ".join(f"{dist[l][c]:5.2f}" for l in labels))
print("Rotamers (deg):")
for a in angles:
    print(f"  {a:8}", "  ".join(f"{rot[l][a]:6.0f}" for l in labels))
