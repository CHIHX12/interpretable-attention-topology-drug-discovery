#!/usr/bin/env python3
"""
anm_tm6_direction.py — vector ANM to test how TM6 actually moves.

Three things:
  1. Build an Anisotropic Network Model (ANM) from the 6KO5 contact network
     (same edges as the scalar ENM, but 3D Hessian -> directional modes).
  2. Measure the REAL activation motion from the crystals (6KO5 -> 8JSR) and the
     inverse motion (6KO5 -> 7F83) by superposition.
  3. Ask: does a low-frequency ANM mode of the RESTING structure predict the
     activation direction? (mode/transition overlap). And is 272-276 the pivot?

Honest note: a single ANM mode is sign-ambiguous (it oscillates both ways), so
the ANM gives the AXIS of easy motion, not outward-vs-inward. The crystal
transition supplies the actual direction; the overlap tells us if the resting
structure's intrinsic flexibility already points along it.

Outputs printed diagnostics; figure built in a second step.
"""
import os, json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TM6 = list(range(261, 293))

# ---------- load reference network (6KO5) ----------
with open(os.path.join(SCRIPT_DIR, "_data_compact.json")) as f:
    data = json.load(f)
nodes = data["nodes"]
id2node = {n["id"]: n for n in nodes}
ids = sorted(id2node)
N = len(ids)
idx = {r: i for i, r in enumerate(ids)}
id2tm = {n["id"]: n.get("tm", "?") for n in nodes}
coords = np.array([[id2node[r]["x3d"], id2node[r]["y3d"], id2node[r]["z3d"]] for r in ids])

# ---------- build ANM Hessian (3N x 3N) ----------
H = np.zeros((3 * N, 3 * N))
for n in nodes:
    i = idx[n["id"]]
    for s, k in n["neighbors"].items():
        j = int(s)
        if j not in idx:
            continue
        jj = idx[j]
        if jj <= i:
            continue
        d = coords[jj] - coords[i]
        L = np.linalg.norm(d)
        if L < 1e-6:
            continue
        u = d / L
        sub = k * np.outer(u, u)          # 3x3
        H[3*i:3*i+3, 3*jj:3*jj+3] -= sub
        H[3*jj:3*jj+3, 3*i:3*i+3] -= sub
        H[3*i:3*i+3, 3*i:3*i+3] += sub
        H[3*jj:3*jj+3, 3*jj:3*jj+3] += sub

w, V = np.linalg.eigh(H)               # ascending eigenvalues
print(f"ANM built: {N} nodes, Hessian {H.shape}")
print(f"6 lowest eigenvalues (should be ~0, rigid body): {np.round(w[:6], 6)}")
print(f"first non-trivial eigenvalues (modes 7-12): {np.round(w[6:12], 4)}")

def mode_vec(m):
    """mode m (0-based eigenvector index) as N x 3 per-residue displacement."""
    return V[:, m].reshape(N, 3)

# ---------- parse other structures, match Cα by residue number ----------
def parse_ca(pdb):
    """return {chain: {resseq: (resname, np.array xyz)}} for CA atoms."""
    out = {}
    for line in open(os.path.join(SCRIPT_DIR, pdb)):
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[12:16].strip() != "CA":
            continue
        alt = line[16]
        if alt not in (" ", "A"):
            continue
        ch = line[21]
        try:
            rs = int(line[22:26])
            xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        except ValueError:
            continue
        out.setdefault(ch, {})[rs] = (line[17:20].strip(), xyz)
    return out

def best_chain(ca_by_chain, want_ids, id2resn):
    """Pick the chain that matches our residues by NUMBER *and* NAME (the real
    receptor), not just a chain that happens to share residue numbering."""
    best, bestscore = None, -1
    for ch, d in ca_by_chain.items():
        score = sum(1 for r in want_ids if r in d and d[r][0] == id2resn.get(r))
        if score > bestscore:
            best, bestscore = ch, score
    return best, bestscore

def kabsch(P, Q):
    """rotate+translate P onto Q (both N x 3). returns P aligned to Q frame."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    Hm = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return (Pc @ R.T) + Q.mean(0), R

id2resn = {n["id"]: n["resn"] for n in nodes}

def transition(pdb, label):
    ca = parse_ca(pdb)
    ch, score = best_chain(ca, ids, id2resn)
    # keep only residues whose NAME matches (true receptor residues)
    common = [r for r in ids if r in ca[ch] and ca[ch][r][0] == id2resn[r]]
    P = np.array([ca[ch][r][1] for r in common])            # mobile (other state)
    Q = coords[[idx[r] for r in common]]                    # reference (6KO5)
    Paln, R = kabsch(P, Q)
    rmsd = np.sqrt(((Paln - Q) ** 2).sum(1).mean())
    disp = {r: Paln[k] - Q[k] for k, r in enumerate(common)}   # per-residue 3D shift
    print(f"\n{label} ({pdb}): receptor chain '{ch}', matched {len(common)}/{N} residues, "
          f"superposed RMSD={rmsd:.2f} A")
    return disp, common

disp_act, common_act = transition("8jsr.pdb", "ACTIVATION 6KO5->8JSR (agonist)")
disp_inv, common_inv = transition("7f83.pdb", "INVERSE   6KO5->7F83 (inverse agonist)")

# ---------- bundle axis + outward(radial) direction ----------
ctr = coords.mean(0)
cc = coords - ctr
axis = np.linalg.svd(cc, full_matrices=False)[2][0]      # principal axis ~ membrane normal
def radial_out(r):
    v = coords[idx[r]] - ctr
    v = v - np.dot(v, axis) * axis                        # remove axial component
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v

# ---------- characterize the activation transition on TM6 ----------
def zone_radial(disp, zone):
    vals = []
    for r in zone:
        if r in disp:
            vals.append(np.dot(disp[r], radial_out(r)))    # + = outward, - = inward
    return np.mean(vals) if vals else float("nan")

tm6_bottom = list(range(261, 271)); tm6_mid = list(range(272, 277)); tm6_top = list(range(283, 293))
print("\n=== Crystal transition: radial motion of TM6 zones (+=outward, -=inward) ===")
for lbl, disp in [("ACTIVATION (agonist)", disp_act), ("INVERSE (inv-agonist)", disp_inv)]:
    print(f"  {lbl}: bottom={zone_radial(disp,tm6_bottom):+.2f}  "
          f"mid(272-276)={zone_radial(disp,tm6_mid):+.2f}  top={zone_radial(disp,tm6_top):+.2f} A")

# per-residue magnitude (to find pivot = min motion)
mag_act = {r: np.linalg.norm(disp_act[r]) for r in TM6 if r in disp_act}
piv = min(mag_act, key=mag_act.get)
print(f"\nTM6 least-moving residue in activation (candidate pivot): {id2node[piv]['name']} "
      f"({mag_act[piv]:.2f} A);  272={mag_act.get(272,float('nan')):.2f}, "
      f"276={mag_act.get(276,float('nan')):.2f}, bottom max="
      f"{max((mag_act[r] for r in tm6_bottom if r in mag_act)):.2f}")

# ---------- ANM overlap with the activation transition ----------
# build 3N transition vector on common residues; compare to ANM modes (restricted)
# first genuinely non-trivial mode (skip rigid-body + any disconnected zero modes)
first_mode = int(np.searchsorted(w, 1e-3))
print(f"\nFirst non-trivial mode index (eigenvalue > 1e-3): #{first_mode+1} "
      f"(skipped {first_mode} near-zero modes)")

def overlap_with_modes(disp, common, start, n_modes=20):
    sel = [idx[r] for r in common]
    rows = np.concatenate([[3*s, 3*s+1, 3*s+2] for s in sel])
    t = np.concatenate([disp[r] for r in common]); t = t / np.linalg.norm(t)
    ov = []
    for m in range(start, start + n_modes):
        vm = V[rows, m]; vm = vm / np.linalg.norm(vm)
        ov.append(abs(np.dot(t, vm)))
    return np.array(ov)

ov_act = overlap_with_modes(disp_act, common_act, first_mode)
cum = np.sqrt(np.cumsum(ov_act ** 2))
print("\n=== ANM modes vs ACTIVATION transition (overlap) ===")
print(f"  best single mode: #{first_mode+int(np.argmax(ov_act))+1} (overlap {ov_act.max():.2f})")
print(f"  cumulative overlap, first 5 modes:  {cum[4]:.2f}")
print(f"  cumulative overlap, first 10 modes: {cum[9]:.2f}")
print(f"  cumulative overlap, first 20 modes: {cum[19]:.2f}")
print("  (1.0 = the transition is fully captured by those soft modes)")

np.savez(os.path.join(SCRIPT_DIR, "_anm_cache.npz"),
         w=w, ids=np.array(ids), ov_act=ov_act,
         mag_act=np.array([mag_act.get(r, np.nan) for r in TM6]),
         tm6=np.array(TM6))
print("\nSaved intermediate results to _anm_cache.npz")

# ===================== FIGURE =====================
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
SI_WIDTH = 6.73


def _panel(ax, letter):
    ax.text(-0.09, 1.03, f"({letter})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")
# ----------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 12, "ytick.labelsize": 13, "legend.fontsize": 12,
    "mathtext.default": "regular",
})

present = [r for r in TM6 if r in disp_act]
xs = np.arange(len(present))
lbl = [str(r) for r in present]
mag = np.array([np.linalg.norm(disp_act[r]) for r in present])
rad = np.array([np.dot(disp_act[r], radial_out(r)) for r in present])

def zone_color(r):
    if r <= 275: return "#1F6FBF"      # bottom
    if r <= 282: return "#7B1FA2"      # middle (pivot region)
    return "#C62828"                   # top

fig, (axA, axB, axC) = plt.subplots(3, 1, figsize=(SI_WIDTH, 6.4))
fig.subplots_adjust(top=0.88, bottom=0.07, hspace=0.32, left=0.08, right=0.97)

# Panel A: how far each TM6 residue moves
axA.bar(xs, mag, color=[zone_color(r) for r in present], edgecolor="#333", lw=0.4)
for rr in (272, 276):
    if rr in present:
        k = present.index(rr)
        axA.annotate(f"{id2node[rr]['name']}", (k, mag[k]), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=11, fontweight="bold", color="#7B1FA2")
_panel(axA, "a")
axA.set_ylabel("Cα displacement |Δr| (Å)")
axA.set_xticks(xs); axA.set_xticklabels(lbl, rotation=90)
axA.spines[["top", "right"]].set_visible(False)
# Panel B: direction (radial, + outward / - inward)
axB.bar(xs, rad, color=["#C62828" if v > 0 else "#1F6FBF" for v in rad], edgecolor="#333", lw=0.4)
axB.axhline(0, color="black", lw=1.0)
_panel(axB, "b")
axB.set_ylabel("Radial motion (Å)")
axB.set_xlabel("TM6 residue number")
axB.set_xticks(xs); axB.set_xticklabels(lbl, rotation=90)
axB.spines[["top", "right"]].set_visible(False)
# Panel C: how much of the crystal transition the soft modes already span
axC.plot(range(1, len(cum) + 1), cum, "o-", ms=3, color="#1F6FBF")
for k, style in ((10, "--"), (20, ":")):
    if len(cum) >= k:
        axC.axvline(k, color="#999", lw=0.7, ls=style)
        axC.annotate(f"{cum[k-1]:.2f} at {k} modes", (k, cum[k-1]),
                     textcoords="offset points", xytext=(5, -9), fontsize=7)
axC.set_xlabel("number of low-frequency ANM modes")
axC.set_ylabel("cumulative overlap\nwith the activation transition")
axC.set_ylim(0, 1)
axC.set_xlim(0, min(40, len(cum)))
axC.spines[["top", "right"]].set_visible(False)
_panel(axC, "c")
fig.align_ylabels([axA, axB, axC])

for ext in ("png", "pdf"):
    out = os.path.join(SCRIPT_DIR, f"FigS2_anm_tm6_direction.{ext}")
    fig.savefig(out, dpi=600, facecolor="white")
    print(f"saved {out}")
