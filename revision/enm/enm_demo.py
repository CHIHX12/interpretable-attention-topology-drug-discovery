#!/usr/bin/env python3
"""
enm_demo.py — Strike any single residue and watch how its energy
distributes to the 10 ML-identified hubs.

This is a teaching/inspection tool that reproduces, for ONE residue,
exactly what the full pipeline does for all 293 residues to build
`enm_ratio_map.json`.

How it works
------------
    node      = one residue, placed at its Cα coordinate (x3d/y3d/z3d)
    spring    = k_ij = 1 / (min heavy-atom distance), only if < 5 Å
    motion    = damped harmonic oscillator network
                  F_i = -Σ_j k_ij·(x_i - x_j) - γ·v_i
                integrated by symplectic Euler for 300 steps
    output    = R(r) = Σ peak(Active hubs) / Σ peak(Inactive hubs)

The 10 hubs are a FIXED set. We strike one residue, run the wave, and
only read the peak displacement that arrives at those 10 hubs.

Usage
-----
    python3 enm_demo.py            # default: strike E124
    python3 enm_demo.py 125        # strike by residue id
    python3 enm_demo.py SER125     # strike by residue name
    python3 enm_demo.py W276       # works with 1-letter or 3-letter prefix match
"""
import os
import sys
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fixed ML-identified hubs (same set used for every strike)
ACTIVE_HUBS = [124, 122, 125, 198, 200]
INACTIVE_HUBS = [287, 286, 278, 147, 116]

# Simulation parameters (match the main pipeline / ENM_WAVE_METHODS.md §2.3)
STEPS = 300
DT = 0.005
GAMMA = 0.05


def load_network() -> tuple[dict, list[int], dict, np.ndarray]:
    """Load the contact network and build the symmetric spring matrix K."""
    with open(os.path.join(SCRIPT_DIR, "_data_compact.json")) as f:
        data = json.load(f)

    id2node = {n["id"]: n for n in data["nodes"]}
    ids = sorted(id2node)
    idx = {rid: i for i, rid in enumerate(ids)}
    n = len(ids)

    k_matrix = np.zeros((n, n))
    for node in data["nodes"]:
        i = idx[node["id"]]
        for nid_str, k in node["neighbors"].items():
            nid = int(nid_str)
            if nid in idx:
                j = idx[nid]
                k_matrix[i][j] = k
                k_matrix[j][i] = k
    return id2node, ids, idx, k_matrix


def strike(k_matrix: np.ndarray, struck_idx: int,
           steps: int = STEPS, dt: float = DT, gamma: float = GAMMA) -> np.ndarray:
    """Strike one node (initial displacement 1.0) and return the peak
    absolute displacement reached by every node over `steps` time steps."""
    n = k_matrix.shape[0]
    x = np.zeros(n)
    v = np.zeros(n)
    x[struck_idx] = 1.0
    peak = np.abs(x).copy()
    d_row = k_matrix.sum(axis=1)  # d_row[i] = Σ_j k_ij
    for _ in range(steps):
        # Damped harmonic oscillator network:
        #   F_i = -Σ_j k_ij (x_i - x_j) - γ v_i
        #   K@x = Σ_j k_ij x_j ,  d_row*x = Σ_j k_ij x_i
        force = k_matrix @ x - d_row * x - gamma * v
        v += force * dt                       # symplectic Euler
        x += v * dt
        peak = np.maximum(peak, np.abs(x))    # biggest swing each node reaches
    return peak


def resolve_residue(arg: str, id2node: dict) -> int:
    """Resolve a CLI argument to a residue id.

    The residue number equals the node id here, so any form works:
    an id (276), a 3-letter name (TRP276), or a 1-letter code (W276).
    """
    digits = "".join(c for c in arg if c.isdigit())
    if not digits:
        sys.exit(f"No residue number found in '{arg}'. "
                 f"Try an id (124), a name (SER125), or a code (W276).")
    rid = int(digits)
    if rid not in id2node:
        sys.exit(f"Residue {rid} (from '{arg}') is not in the network.")
    return rid


def main() -> None:
    id2node, ids, idx, k_matrix = load_network()
    id2name = {rid: node["name"] for rid, node in id2node.items()}

    arg = sys.argv[1] if len(sys.argv) > 1 else "124"
    rid = resolve_residue(arg, id2node)

    peak = strike(k_matrix, idx[rid])

    print(f"STRIKE residue {id2name[rid]} (id={rid})")
    print(f"  -> initial displacement 1.0 at its Cα, run {STEPS} steps "
          f"(dt={DT}, γ={GAMMA})")
    print(f"  -> record peak |displacement| of every node, then read the 10 hubs\n")

    print(f"{'hub':>8} {'type':>9} {'peak displ.':>12}")
    print("-" * 33)

    active_sum = 0.0
    for h in ACTIVE_HUBS:
        p = peak[idx[h]]
        active_sum += p
        print(f"{id2name[h]:>8} {'Active':>9} {p:>12.5f}")

    inactive_sum = 0.0
    for h in INACTIVE_HUBS:
        p = peak[idx[h]]
        inactive_sum += p
        print(f"{id2name[h]:>8} {'Inactive':>9} {p:>12.5f}")

    print("-" * 33)
    print(f"Sum over 5 Active   hubs = {active_sum:.5f}")
    print(f"Sum over 5 Inactive hubs = {inactive_sum:.5f}")

    if inactive_sum > 0:
        ratio = active_sum / inactive_sum
        print(f"\nR({id2name[rid]}) = Sum(Active) / Sum(Inactive) "
              f"= {active_sum:.5f} / {inactive_sum:.5f} = {ratio:.4f}")
        print(f"  R > 1  -> energy goes preferentially to the Active-hub side")
        print(f"  R < 1  -> energy goes preferentially to the Inactive-hub side")
    else:
        print(f"\nR({id2name[rid]}) = ∞ (no displacement reached Inactive hubs)")


if __name__ == "__main__":
    main()
