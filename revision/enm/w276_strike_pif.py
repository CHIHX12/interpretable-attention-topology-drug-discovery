#!/usr/bin/env python3
"""
w276_strike_pif.py — ENM "butterfly" test of the user's hypothesis.

Start from the ANTAGONIST 6KO5 network. "Strike" W276 (the CWxP toggle) and watch
where the vibrational wave goes. Then ask whether forming the F272-P224
(F6.44-P5.50) contact — which only exists in the ACTIVE state (6.5 A -> 3.7 A) —
opens a new propagation route from the toggle to the conserved TM5 proline.

Two networks:
  (1) antagonist   : 6KO5 contacts as-is (F272-P224 NOT connected, 6.5 A)
  (2) +F272-P224   : antagonist + one extra spring F272-P224 (k = 1/3.7), i.e.
                     the active-state engagement grafted onto the resting network

HONEST: the scalar ENM measures vibrational COUPLING, not real forces. Striking =
a connectivity/coupling probe, not a force-field simulation.

Output: FigS4_w276_strike_pif.{png,pdf}
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, "_data_compact.json")) as f:
    data = json.load(f)
id2node = {n["id"]: n for n in data["nodes"]}
ids = sorted(id2node); N = len(ids); idx = {r: i for i, r in enumerate(ids)}
name = {r: id2node[r]["name"] for r in ids}


def build_K(extra=None):
    K = np.zeros((N, N))
    for n in data["nodes"]:
        i = idx[n["id"]]
        for s, k in n["neighbors"].items():
            j = int(s)
            if j in idx:
                K[i][idx[j]] = k; K[idx[j]][i] = k
    if extra:
        a, b, k = extra
        K[idx[a]][idx[b]] = k; K[idx[b]][idx[a]] = k
    return K


def strike(K, sid, steps=300, dt=0.005, g=0.05):
    x = np.zeros(N); v = np.zeros(N); x[idx[sid]] = 1.0
    peak = np.abs(x).copy(); D = K.sum(1)
    for _ in range(steps):
        f = K @ x - D * x - g * v; v += f * dt; x += v * dt
        peak = np.maximum(peak, np.abs(x))
    return peak


# is F272-P224 connected in the antagonist network?
edge_272_224 = id2node[272]["neighbors"].get("224")
print(f"F272-P224 spring in antagonist 6KO5 network: {edge_272_224}  (None = not in contact)")

K_anta = build_K()
K_act = build_K(extra=(272, 224, 1.0 / 3.7))   # graft the active-state contact

peak_anta = strike(K_anta, 276)
peak_act = strike(K_act, 276)

# cascade residues to report: toggle -> connector -> TM5 proline -> TM3 -> down TM6
cascade = [276, 272, 224, 131, 278, 283, 262]
print(f"\nStrike W276 — peak displacement at key residues:")
print(f"{'residue':10}{'role':22}{'antag':>9}{'+F272-P224':>12}{'fold':>8}")
for r in cascade:
    role = {276: "W6.48 toggle(struck)", 272: "F6.44 PIF (relay)", 224: "P5.50 PIF (TM5)",
            131: "I3.40 PIF (TM3)", 278: "P6.50 CWxP", 283: "R283 (salt bridge)",
            262: "V262 TM6 bottom"}[r]
    a, b = peak_anta[idx[r]], peak_act[idx[r]]
    fold = b / a if a > 1e-9 else float("inf")
    print(f"{name[r]:10}{role:22}{a:>9.4f}{b:>12.4f}{fold:>8.1f}x")

# ---- figure ----
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 13, "legend.fontsize": 12,
})
labels = [f"{name[r]}\n" + {276: "W6.48*", 272: "F6.44", 224: "P5.50", 131: "I3.40",
          278: "P6.50", 283: "R283", 262: "V262"}[r] for r in cascade]
xa = np.arange(len(cascade)); w = 0.38
fig, ax = plt.subplots(figsize=(14, 7.5))
fig.subplots_adjust(top=0.90, bottom=0.20, left=0.09, right=0.97)
ax.bar(xa - w / 2, [peak_anta[idx[r]] for r in cascade], w, color="#9E9E9E",
       edgecolor="#333", lw=0.5, label="Antagonist network (no F272–P224)")
ax.bar(xa + w / 2, [peak_act[idx[r]] for r in cascade], w, color="#2E9E5B",
       edgecolor="#333", lw=0.5, label="+ F272–P224 contact (active-like)")
ax.set_yscale("log")
ax.set_xticks(xa); ax.set_xticklabels(labels)
ax.set_ylabel("peak displacement when W276 is struck (log)")
ax.set_title("(A)  Strike W276 (CWxP toggle): does forming F272–P224 open a route to P5.50?",
             loc="left", fontweight="bold")
ax.legend(loc="upper right", framealpha=0.95, edgecolor="#ccc")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.5, -0.16, "Grey = resting 6KO5 network. Green = same network + the active-state "
        "F272–P224 spring. Scalar ENM = mechanical coupling, not real force.",
        transform=ax.transAxes, ha="center", fontsize=11, style="italic", color="#666")

for ext in ("png", "pdf"):
    out = os.path.join(SCRIPT_DIR, f"FigS4_w276_strike_pif.{ext}")
    fig.savefig(out, dpi=600, facecolor="white")
    print("saved", out)
