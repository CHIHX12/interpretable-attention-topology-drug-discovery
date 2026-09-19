#!/usr/bin/env python3
"""
reverse.py — BLIND reverse-derivation test.

Question it answers
-------------------
The 10 hubs come from an external ML model (deltaImp analysis). A natural
verification is: can the Elastic Network Model, WITHOUT being told which
residues are hubs, independently point back to those same 10 residues?

This script ranks all 293 residues by four BLIND importance metrics
(none of which use the hub labels) and reports where the 10 ML hubs land.

The four metrics measure DIFFERENT notions of "important":
    WeightedDegree  — how strongly a residue is connected (Σk)
    SoftModePartic  — how much it moves in the soft collective motions
                      (low-frequency normal modes of the Kirchhoff matrix)
    Betweenness     — how many shortest paths run through it (allosteric
                      bottleneck / bridge)   [needs networkx; skipped if absent]
    Broadcast       — how much total displacement a strike on it spreads

Interpretation
--------------
If the 10 hubs rank at the TOP across metrics -> ENM independently recovers
the ML result (strong cross-validation). If they rank near or below random
(~50th percentile) -> ENM importance (structural centrality) is measuring a
DIFFERENT thing than ML importance (state discrimination), and ENM cannot
reverse-derive the hubs. The latter is itself an honest, useful finding:
it means the ML sees something pure geometry/ENM does not.

Requirements: numpy (required), networkx (optional, for Betweenness).
Usage: python3 reverse.py
"""
import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

ACTIVE_HUBS = [124, 122, 125, 198, 200]
INACTIVE_HUBS = [287, 286, 278, 147, 116]
HUBS = ACTIVE_HUBS + INACTIVE_HUBS

STEPS, DT, GAMMA = 300, 0.005, 0.05


def load() -> tuple[dict, list[int], dict, np.ndarray]:
    with open(os.path.join(SCRIPT_DIR, "_data_compact.json")) as f:
        data = json.load(f)
    id2node = {n["id"]: n for n in data["nodes"]}
    ids = sorted(id2node)
    idx = {r: i for i, r in enumerate(ids)}
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


def weighted_degree(k_matrix: np.ndarray) -> np.ndarray:
    return k_matrix.sum(axis=1)


def soft_mode_participation(k_matrix: np.ndarray, n_modes: int = 10) -> np.ndarray:
    """Sum of squared eigenvector components over the lowest non-trivial modes,
    weighted by 1/eigenvalue so softer (more functional) modes count more."""
    laplacian = np.diag(k_matrix.sum(axis=1)) - k_matrix
    w, v = np.linalg.eigh(laplacian)
    part = np.zeros(k_matrix.shape[0])
    for m in range(1, n_modes + 1):       # skip mode 0 (rigid-body)
        part += (v[:, m] ** 2) / w[m]
    return part


def betweenness(k_matrix: np.ndarray) -> np.ndarray | None:
    """Weighted betweenness centrality (strong spring -> short path).
    Returns None if networkx is unavailable."""
    try:
        import networkx as nx
    except ImportError:
        return None
    n = k_matrix.shape[0]
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if k_matrix[i][j] > 0:
                g.add_edge(i, j, weight=1.0 / k_matrix[i][j])
    bc = nx.betweenness_centrality(g, weight="weight")
    return np.array([bc.get(i, 0.0) for i in range(n)])


def broadcast(k_matrix: np.ndarray) -> np.ndarray:
    """For each residue: total peak displacement spread to the whole network
    when it is struck."""
    n = k_matrix.shape[0]
    d_row = k_matrix.sum(axis=1)
    out = np.zeros(n)
    for s in range(n):
        x = np.zeros(n)
        v = np.zeros(n)
        x[s] = 1.0
        peak = np.abs(x).copy()
        for _ in range(STEPS):
            force = k_matrix @ x - d_row * x - GAMMA * v
            v += force * DT
            x += v * DT
            peak = np.maximum(peak, np.abs(x))
        out[s] = peak.sum()
    return out


def pct_rank(arr: np.ndarray, i: int) -> float:
    """Percentile of element i (100 = most important)."""
    return float((arr < arr[i]).sum()) / len(arr) * 100.0


def main() -> None:
    id2node, ids, idx, k_matrix = load()
    id2name = {r: id2node[r]["name"] for r in ids}
    n = len(ids)

    metrics: dict[str, np.ndarray] = {
        "WeightedDegree": weighted_degree(k_matrix),
        "SoftModePartic": soft_mode_participation(k_matrix),
        "Broadcast": broadcast(k_matrix),
    }
    bc = betweenness(k_matrix)
    if bc is not None:
        metrics["Betweenness"] = bc
    else:
        print("[note] networkx not installed -> Betweenness metric skipped.\n")

    label = {**{h: "ACT" for h in ACTIVE_HUBS}, **{h: "INA" for h in INACTIVE_HUBS}}

    print("BLIND reverse-derivation: rank all 293 residues by ENM importance,")
    print("then see where the 10 ML hubs land (percentile, 100 = most important).\n")
    header = f"{'hub':>8} {'type':>4} " + " ".join(f"{m[:13]:>14}" for m in metrics)
    print(header)
    print("-" * len(header))
    for h in HUBS:
        i = idx[h]
        row = f"{id2name[h]:>8} {label[h]:>4} "
        row += " ".join(f"{pct_rank(arr, i):>12.0f}% " for arr in metrics.values())
        print(row)

    print("\nTop-10 most important residues by each BLIND metric "
          "(* = one of your 10 ML hubs):")
    for m, arr in metrics.items():
        top = sorted(range(n), key=lambda i: -arr[i])[:10]
        names = [id2name[ids[i]] + ("*" if ids[i] in HUBS else "") for i in top]
        print(f"  {m:>14}: " + ", ".join(names))

    print("\nAvg percentile of the 10 hubs "
          "(50 = random; >50 means hubs tend to be important):")
    for m, arr in metrics.items():
        avg = np.mean([pct_rank(arr, idx[h]) for h in HUBS])
        print(f"  {m:>14}: {avg:5.1f}%")


if __name__ == "__main__":
    main()
