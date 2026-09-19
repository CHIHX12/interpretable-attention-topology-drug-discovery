#!/usr/bin/env python3
"""
prs_tm6.py — Perturbation Response Scanning (PRS): apply a FORCE, get DIRECTION.

This is the right tool for "who pushes TM6 out". Unlike striking (scalar), PRS
applies a real force vector at one residue and computes the directional
displacement everywhere from the ANM:   Δr = H^{-1} F .

Test of the user's hypothesis:
  apply a force on F272 (F6.44) pointed AT P224 (P5.50) — i.e. F272 engaging the
  conserved TM5 proline — and ask: does TM6's cytoplasmic (bottom) half move
  OUTWARD (the activation swing)?

Network: 6KO5 (antagonist) + the active-state F272–P224 spring (so the force can
transmit through the engaged contact).

HONEST: linear harmonic response to a small static force (no chemistry, no bond
breaking). A directional elastic prediction, not MD.

Output: FigS5_prs_tm6.{png,pdf}
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
TM6 = list(range(261, 293))
with open(os.path.join(SCRIPT_DIR, "_data_compact.json")) as f:
    data = json.load(f)
id2node = {n["id"]: n for n in data["nodes"]}
ids = sorted(id2node); N = len(ids); idx = {r: i for i, r in enumerate(ids)}
name = {r: id2node[r]["name"] for r in ids}
coords = np.array([[id2node[r]["x3d"], id2node[r]["y3d"], id2node[r]["z3d"]] for r in ids])


def build_hessian(extra=None):
    H = np.zeros((3 * N, 3 * N))
    edges = []
    for n in data["nodes"]:
        i = idx[n["id"]]
        for s, k in n["neighbors"].items():
            j = int(s)
            if j in idx and idx[j] > i:
                edges.append((i, idx[j], k))
    if extra:
        a, b, k = extra
        edges.append((idx[a], idx[b], k))
    for i, j, k in edges:
        d = coords[j] - coords[i]; L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        u = d / L; sub = k * np.outer(u, u)
        H[3*i:3*i+3, 3*j:3*j+3] -= sub
        H[3*j:3*j+3, 3*i:3*i+3] -= sub
        H[3*i:3*i+3, 3*i:3*i+3] += sub
        H[3*j:3*j+3, 3*j:3*j+3] += sub
    return H


H = build_hessian(extra=(272, 224, 1.0 / 3.7))   # active-like: F272–P224 engaged
w, V = np.linalg.eigh(H)
# pseudo-inverse (drop zero/rigid-body & disconnected modes)
inv = np.zeros_like(w)
nz = w > 1e-6
inv[nz] = 1.0 / w[nz]
Hinv = (V * inv) @ V.T

# bundle axis + outward(radial) direction
ctr = coords.mean(0); cc = coords - ctr
axis = np.linalg.svd(cc, full_matrices=False)[2][0]
def radial_out(r):
    v = coords[idx[r]] - ctr
    v = v - np.dot(v, axis) * axis
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 1e-6 else v

def prs(push_res, toward_res):
    F = np.zeros(3 * N)
    d = coords[idx[toward_res]] - coords[idx[push_res]]
    d = d / np.linalg.norm(d)
    F[3*idx[push_res]:3*idx[push_res]+3] = d
    dr = (Hinv @ F).reshape(N, 3)
    return dr

dr = prs(272, 224)        # push F272 toward P224

# radial response of TM6 zones
def zone_rad(dr, zone):
    return np.mean([np.dot(dr[idx[r]], radial_out(r)) for r in zone])

bottom = list(range(261, 271)); mid = list(range(272, 277)); top = list(range(283, 293))
print("PRS: push F272 toward P224 (F6.44 engaging P5.50).")
print("Radial response of TM6 (+ = outward / − = inward):")
print(f"  bottom 261-270 : {zone_rad(dr, bottom):+.3f}")
print(f"  middle 272-276 : {zone_rad(dr, mid):+.3f}")
print(f"  top    283-292 : {zone_rad(dr, top):+.3f}")

# normalize for display
radvals = np.array([np.dot(dr[idx[r]], radial_out(r)) for r in TM6])
scale = np.abs(radvals).max()
radvals = radvals / scale

# ---- figure ----
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
    "mathtext.default": "regular",
})
fig, ax = plt.subplots(figsize=(SI_WIDTH, 3.14))
fig.set_layout_engine("constrained")
xa = np.arange(len(TM6))
ax.bar(xa, radvals, color=["#C62828" if v > 0 else "#1F6FBF" for v in radvals],
       edgecolor="#333", lw=0.4)
ax.axhline(0, color="black", lw=0.45)
for rr, lbl in [(272, "F272 (push)"), (276, "W276"), (283, "R283")]:
    k = TM6.index(rr)
    ax.annotate(lbl, (k, radvals[k]), textcoords="offset points", xytext=(0, 8 if radvals[k] >= 0 else -14),
                ha="center", fontsize=5.5, fontweight="bold")
ax.set_xticks(xa); ax.set_xticklabels([str(r) for r in TM6], rotation=90)
ax.set_ylabel("TM6 radial response (normalized)\n+ outward / − inward")
ax.set_xlabel("TM6 residue number")
ax.spines[["top", "right"]].set_visible(False)
for ext in ("png", "pdf"):
    out = os.path.join(SCRIPT_DIR, f"FigS5_prs_tm6.{ext}")
    fig.savefig(out, dpi=1200, facecolor="white")
    print("saved", out)
