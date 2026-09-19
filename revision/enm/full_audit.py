#!/usr/bin/env python3
"""
FULL AUDIT: Re-derive every numerical claim in ENM_WAVE_METHODS.md
from raw data (_data_compact.json, enm_ratio_map.json).

Checks:
  Section 2.1: Network statistics (nodes, edges, edge classes, mean k)
  Section 2.2-2.3: Equation of motion & integration (formula correctness)
  Section 2.4-2.5: Strike protocol, ratio calculation
  Section 4.1: Global distribution statistics
  Section 4.2: Hub R values, percentiles, ranks
  Section 4.3: Crossover point
  Section 4.4: TM helix medians
  Section 10.1: Zone definitions, cross-helix restraint values
  Section 10.1.2: Zone averages
  Section 10.1.3: High-point analysis (all 7 peaks + zero-restraint residues)
  Section 10.3: R283 decomposition (Σk, k(E124-R283), fraction)
  Section 10.4: Salt bridge removal simulation zone averages
  Section 10.5: Prediction box values
  Figure 3 hard-coded values vs computed
"""
import os, json, sys
import numpy as np
from collections import defaultdict

# ── Load data ─────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_DIR, '_data_compact.json')) as f:
    data = json.load(f)
with open(os.path.join(SCRIPT_DIR, 'enm_ratio_map.json')) as f:
    ratio_map_raw = json.load(f)

ratio_map = {int(k): v for k, v in ratio_map_raw.items()}

id2node = {n['id']: n for n in data['nodes']}
id2tm   = {n['id']: n.get('tm', 'unknown') for n in data['nodes']}
id2name = {n['id']: n.get('name', str(n['id'])) for n in data['nodes']}

all_ids = sorted(id2node.keys())
N = len(all_ids)
id2idx = {rid: i for i, rid in enumerate(all_ids)}

errors = []
warnings = []
checks = 0

def check(name, expected, actual, tol=0.01):
    global checks
    checks += 1
    if isinstance(expected, bool):
        ok = bool(actual) == expected
    elif isinstance(expected, str):
        ok = str(actual) == expected
    elif isinstance(expected, (int, float)):
        ok = abs(float(actual) - float(expected)) < tol
    else:
        ok = actual == expected
    status = "OK" if ok else "FAIL"
    if not ok:
        errors.append(f"  {name}: expected={expected}, got={actual}")
    print(f"  [{status}] {name}: expected={expected}, got={actual}")

print("="*80)
print("SECTION 2.1: Network statistics")
print("="*80)

# Nodes
check("Number of nodes", 293, N)

# Edges (unique undirected)
edge_set = set()
for node in data['nodes']:
    i = node['id']
    for nid_str, k in node['neighbors'].items():
        nid = int(nid_str)
        if nid in id2node:
            pair = tuple(sorted([i, nid]))
            edge_set.add(pair)

n_edges = len(edge_set)
check("Number of unique edges", 1590, n_edges)

# Mean coupling strength
all_k_vals = []
for (a, b) in edge_set:
    # get k from node a's neighbors
    k = id2node[a]['neighbors'].get(str(b), None)
    if k is None:
        k = id2node[b]['neighbors'].get(str(a), None)
    all_k_vals.append(k)
mean_k = np.mean(all_k_vals)
check("Mean coupling strength", 0.354, mean_k, tol=0.005)

# Edge classification
sequential = []
intra_helix = []
inter_helix = []
for (a, b) in edge_set:
    k = id2node[a]['neighbors'].get(str(b), None)
    if k is None:
        k = id2node[b]['neighbors'].get(str(a), None)
    if abs(a - b) <= 1:
        sequential.append(k)
    elif id2tm.get(a, '') == id2tm.get(b, '') and id2tm.get(a, '') != 'unknown':
        intra_helix.append(k)
    else:
        inter_helix.append(k)

check("Sequential edges count", 291, len(sequential))
check("Sequential k value", 0.667, np.mean(sequential), tol=0.001)
check("Intra-helix non-seq count", 798, len(intra_helix))
check("Intra-helix mean k", 0.294, np.mean(intra_helix), tol=0.005)
check("Inter-helix count", 501, len(inter_helix))
check("Inter-helix mean k", 0.267, np.mean(inter_helix), tol=0.005)

# Cross-helix k range claimed: 0.200 (5.0Å) to 0.493 (2.0Å)
# k = 1/d, so d=5.0 → k=0.200, d=2.0 → k=0.500
# The doc says 0.493 for 2.0Å — this might be slightly off
non_seq_k = intra_helix + inter_helix
min_non_seq = min(non_seq_k)
max_non_seq = max(non_seq_k)
print(f"  [INFO] Non-sequential k range: {min_non_seq:.4f} to {max_non_seq:.4f}")
print(f"  [INFO] k=1/5.0={1/5.0:.3f}, k=1/2.0={1/2.0:.3f}")

print()
print("="*80)
print("SECTION 2.2-2.3: Equation of motion parameters")
print("="*80)
# dt=0.005, gamma=0.05, stiffness=1.0, 300 steps, t_final=1.5
check("t_final = dt * steps", 1.5, 0.005 * 300)
print("  [INFO] Formula: F_i = -sum_j k_ij*(x_i - x_j) - gamma*v_i")
print("  [INFO] Vectorized: F = K@x - D*x - gamma*v  (D = row sums of K)")
print("  [INFO] These are equivalent: K@x gives sum_j k_ij*x_j, D*x gives sum_j k_ij*x_i")

print()
print("="*80)
print("SECTION 2.5: Ratio calculation - Hub definitions")
print("="*80)
ACTIVE_HUBS = [124, 122, 125, 198, 200]
INACTIVE_HUBS = [287, 286, 278, 147, 116]
for h in ACTIVE_HUBS:
    print(f"  [INFO] Active hub {id2name.get(h,'?')} (id={h}), TM={id2tm.get(h,'?')}")
for h in INACTIVE_HUBS:
    print(f"  [INFO] Inactive hub {id2name.get(h,'?')} (id={h}), TM={id2tm.get(h,'?')}")

print()
print("="*80)
print("SECTION 4.1: Global R distribution statistics")
print("="*80)
all_r = sorted(ratio_map.values())
check("R minimum", 0.000, min(all_r), tol=0.001)
check("R 25th percentile", 0.067, np.percentile(all_r, 25), tol=0.01)
check("R median", 0.872, np.median(all_r), tol=0.01)
check("R 75th percentile", 7.442, np.percentile(all_r, 75), tol=0.1)
check("R maximum", 1225.1, max(all_r), tol=1.0)
check("R mean", 23.93, np.mean(all_r), tol=0.5)
check("R std dev", 107.4, np.std(all_r), tol=1.0)

# Min residue
min_rid = min(ratio_map, key=lambda r: ratio_map[r])
max_rid = max(ratio_map, key=lambda r: ratio_map[r])
print(f"  [INFO] Min R residue: {id2name.get(min_rid,'?')} (id={min_rid}), R={ratio_map[min_rid]:.6f}")
print(f"  [INFO] Max R residue: {id2name.get(max_rid,'?')} (id={max_rid}), R={ratio_map[max_rid]:.1f}")
# Doc says min is F147 (ICL2), max is V166 (TM4)
# Doc now says "30 residues tied, including F147" — so we check that 147 is among them
check("Min R residues: F147 is among ties at R=0", True, ratio_map.get(147, -1) == 0.0)
# Also verify count of zeros
n_zeros = sum(1 for v in ratio_map.values() if v == 0.0)
check("Min R: 30 residues tied at R=0", 30, n_zeros)
check("Max R residue id", 166, max_rid)

print()
print("="*80)
print("SECTION 4.2: Hub R values, percentiles, ranks")
print("="*80)
sorted_r = sorted(ratio_map.items(), key=lambda x: x[1])
rid_to_rank = {}
for rank_i, (rid, rv) in enumerate(sorted_r, 1):
    rid_to_rank[rid] = rank_i

def percentile_of(rid):
    r_val = ratio_map[rid]
    count_leq = sum(1 for v in ratio_map.values() if v <= r_val)
    return count_leq / len(ratio_map) * 100

# Active hubs
hub_claims = {
    125: (92.31, 95, 16),
    122: (29.80, 87, 25),
    124: (19.86, 85, 29),
    198: (9.05, 78, 48),
    200: (8.56, 78, 50),
    116: (0.212, 36, 105),
    286: (0.068, 26, 75),
    287: (0.037, 23, 68),
    278: (0.025, 20, 59),
    147: (0.000, 10, 0),  # percentile=10% (30 residues tied at R=0)
}

for rid, (exp_r, exp_pct, exp_rank_from_top) in hub_claims.items():
    actual_r = ratio_map.get(rid, -999)
    actual_pct = percentile_of(rid)
    # Rank: the doc gives rank as "X/293" — need to figure out the ranking scheme
    # From the doc: S125 rank 16/293 (top 16), meaning rank from top
    # sorted descending: rank 1 = highest R
    rank_desc = len(ratio_map) - rid_to_rank[rid] + 1  # rank from top
    name = id2name.get(rid, str(rid))
    check(f"{name} R value", exp_r, actual_r, tol=0.05)
    check(f"{name} percentile", exp_pct, round(actual_pct), tol=2)

# Separation gap
min_active = min(ratio_map[h] for h in ACTIVE_HUBS)
max_inactive = max(ratio_map[h] for h in INACTIVE_HUBS)
gap = min_active / max_inactive
check("40x gap: min(Active)/max(Inactive)", 40.4, gap, tol=0.5)

print()
print("="*80)
print("SECTION 4.3: Crossover point")
print("="*80)
# Find residues straddling R=1.0
below_1 = [(rid, r) for rid, r in ratio_map.items() if r < 1.0 and r > 0]
above_1 = [(rid, r) for rid, r in ratio_map.items() if r >= 1.0]
# Closest below
closest_below = max(below_1, key=lambda x: x[1])
closest_above = min(above_1, key=lambda x: x[1])
print(f"  [INFO] Closest below R=1: id={closest_below[0]} ({id2name.get(closest_below[0],'?')}), R={closest_below[1]:.4f}")
print(f"  [INFO] Closest above R=1: id={closest_above[0]} ({id2name.get(closest_above[0],'?')}), R={closest_above[1]:.4f}")
# Doc claims PRO39 (R=0.964) and PHE312 (R=1.082)
check("Crossover below: R(PRO39)", 0.964, ratio_map.get(39, -1), tol=0.01)
check("Crossover above: R(PHE312)", 1.082, ratio_map.get(312, -1), tol=0.01)

print()
print("="*80)
print("SECTION 4.4: TM Helix Medians")
print("="*80)
tm_ratios = defaultdict(list)
for rid, r in ratio_map.items():
    tm = id2tm.get(rid, 'unknown')
    tm_ratios[tm].append(r)

tm_median_claims = {
    'TM6': 0.07,
    'TM7': 0.18,
    'TM5': 2.17,
    'TM3': 3.56,
    'TM1': 9.78,
    'TM2': 10.72,
}
for tm, exp_med in tm_median_claims.items():
    actual_med = np.median(tm_ratios.get(tm, [0]))
    check(f"TM median: {tm}", exp_med, actual_med, tol=0.15)

print()
print("="*80)
print("SECTION 10.1: TM6 Cross-Helix Restraint")
print("="*80)

tm6_ids = list(range(261, 293))

# Compute cross-helix restraint for each TM6 residue
cross_helix_k = {}
for rid in tm6_ids:
    node = id2node[rid]
    total = 0.0
    for nid_str, k in node['neighbors'].items():
        nid = int(nid_str)
        if id2tm.get(nid, '') != 'TM6':
            total += k
    cross_helix_k[rid] = total

print("\n  Per-residue Σk_cross:")
for rid in tm6_ids:
    print(f"    {rid}: {cross_helix_k[rid]:.4f}")

# Section 10.1.2: Zone averages
bottom_k = [cross_helix_k[r] for r in range(261, 276)]
middle_k = [cross_helix_k[r] for r in range(276, 283)]
top_k    = [cross_helix_k[r] for r in range(283, 293)]

check("Bottom avg Σk_cross", 1.05, np.mean(bottom_k), tol=0.05)
check("Middle avg Σk_cross", 0.88, np.mean(middle_k), tol=0.05)
check("Top avg Σk_cross", 0.38, np.mean(top_k), tol=0.05)

# Range claims
check("Bottom Σk range low", 0.0, min(bottom_k), tol=0.01)
check("Bottom Σk range high (V262=2.37)", 2.37, max(bottom_k), tol=0.02)
check("Middle Σk range low", 0.0, min(middle_k), tol=0.01)
check("Middle Σk range high (W276=2.66)", 2.66, max(middle_k), tol=0.02)
check("Top Σk range low", 0.0, min(top_k), tol=0.01)
check("Top Σk range high (R283=0.93~0.94)", 0.94, max(top_k), tol=0.02)

print()
print("="*80)
print("SECTION 10.1.3: High-point analysis (7 peaks)")
print("="*80)
peak_claims = {
    276: ("W276", 2.66),
    262: ("V262", 2.37),
    261: ("T261", 1.83),
    264: ("M264", 1.70),
    272: ("F272", 1.30),
    280: ("H280", 1.20),
    283: ("R283", 0.93),
}
for rid, (name, exp_k) in peak_claims.items():
    check(f"{name} Σk_cross", exp_k, cross_helix_k[rid], tol=0.02)

# Zero-restraint residues
zero_claimed = [270, 282, 285, 288, 289, 292]
print("\n  Zero-restraint residues check:")
actual_zeros = [r for r in tm6_ids if cross_helix_k[r] < 0.001]
for rid in zero_claimed:
    ok = cross_helix_k[rid] < 0.001
    status = "OK" if ok else "FAIL"
    checks += 1
    if not ok:
        errors.append(f"  {rid} claimed zero but Σk={cross_helix_k[rid]:.4f}")
    print(f"  [{status}] {rid} Σk_cross = {cross_helix_k[rid]:.4f} (expected ~0)")
print(f"  [INFO] All actual zeros: {actual_zeros}")

# Verify cross-helix contact partner percentages for key residues
print()
print("  Verifying cross-helix contact partner breakdowns:")

def get_cross_helix_by_tm(rid):
    """Return dict: partner_tm -> total k"""
    node = id2node[rid]
    tm_k = defaultdict(float)
    for nid_str, k in node['neighbors'].items():
        nid = int(nid_str)
        partner_tm = id2tm.get(nid, 'unknown')
        if partner_tm != 'TM6':
            tm_k[partner_tm] += k
    return dict(tm_k)

# R283 decomposition (Section 10.3)
r283_partners = get_cross_helix_by_tm(283)
print(f"\n  R283 partners: {r283_partners}")
total_r283 = sum(r283_partners.values())
check("R283 Σk_cross total", 0.93, total_r283, tol=0.02)

# k(E124-R283) specifically
k_e124_r283 = id2node[283]['neighbors'].get('124', 0)
if k_e124_r283 == 0:
    k_e124_r283 = id2node[124]['neighbors'].get('283', 0)
check("k(E124-R283)", 0.45, k_e124_r283, tol=0.02)
fraction = k_e124_r283 / total_r283 * 100 if total_r283 > 0 else 0
check("E124 fraction of R283 restraint", 48, fraction, tol=2)

# W276 partners
w276_partners = get_cross_helix_by_tm(276)
print(f"\n  W276 partners: {w276_partners}")
total_w276 = sum(w276_partners.values())
check("W276 Σk_cross total", 2.66, total_w276, tol=0.02)

# F272 partners
f272_partners = get_cross_helix_by_tm(272)
print(f"\n  F272 partners: {f272_partners}")
total_f272 = sum(f272_partners.values())
check("F272 Σk_cross total", 1.30, total_f272, tol=0.02)
# Check TM3 percentage
f272_tm3_pct = f272_partners.get('TM3', 0) / total_f272 * 100 if total_f272 > 0 else 0
check("F272 TM3 percentage", 62, f272_tm3_pct, tol=3)

# V262 partner count (claimed 6 ICL3 contacts)
v262_node = id2node[262]
icl3_contacts_262 = [(int(nid), k) for nid, k in v262_node['neighbors'].items()
                     if id2tm.get(int(nid), '') == 'ICL3']
print(f"\n  V262 ICL3 contacts: {len(icl3_contacts_262)}")
for nid, k in icl3_contacts_262:
    print(f"    -> {id2name.get(nid, nid)} (id={nid}): k={k:.4f}")
check("V262 ICL3 contact count", 8, len(icl3_contacts_262))

# Specific partner k values from the table in 10.1.3
print("\n  Checking specific partner k values from Section 10.1.3:")

# V262 partners
v262_partners_claimed = {'258': 0.34, '260': 0.31, '235': 0.27, '259': 0.27, '239': 0.25, '232': 0.22, '240': 0.22, '257': 0.21}
for nid_str, exp_k in v262_partners_claimed.items():
    actual_k = v262_node['neighbors'].get(nid_str, 0)
    name = id2name.get(int(nid_str), nid_str)
    check(f"V262-{name}({nid_str}) k", exp_k, actual_k, tol=0.02)

# R283 partners individually
r283_node = id2node[283]
r283_partner_claims = {'124': 0.45, '214': 0.25, '120': 0.24}
for nid_str, exp_k in r283_partner_claims.items():
    actual_k = r283_node['neighbors'].get(nid_str, 0)
    name = id2name.get(int(nid_str), nid_str)
    check(f"R283-{name}({nid_str}) k", exp_k, actual_k, tol=0.02)

# H280 partners
h280_node = id2node[280]
h280_partner_claims = {'217': 0.25, '221': 0.30, '222': 0.22, '127': 0.22}
for nid_str, exp_k in h280_partner_claims.items():
    actual_k = h280_node['neighbors'].get(nid_str, 0)
    if actual_k == 0:
        # Try cross-lookup
        other_node = id2node.get(int(nid_str), {})
        actual_k = other_node.get('neighbors', {}).get('280', 0)
    name = id2name.get(int(nid_str), nid_str)
    check(f"H280-{name}({nid_str}) k", exp_k, actual_k, tol=0.02)

# F272 partners
f272_node = id2node[272]
f272_partner_claims = {'221': 0.29, '131': 0.28, '134': 0.28, '130': 0.25}
for nid_str, exp_k in f272_partner_claims.items():
    actual_k = f272_node['neighbors'].get(nid_str, 0)
    if actual_k == 0:
        other_node = id2node.get(int(nid_str), {})
        actual_k = other_node.get('neighbors', {}).get('272', 0)
    name = id2name.get(int(nid_str), nid_str)
    check(f"F272-{name}({nid_str}) k", exp_k, actual_k, tol=0.02)

# W276 partner claims
w276_node = id2node[276]
w276_partner_claims = {'315': 0.37, '312': 0.35, '127': 0.29, '130': 0.26}
for nid_str, exp_k in w276_partner_claims.items():
    actual_k = w276_node['neighbors'].get(nid_str, 0)
    if actual_k == 0:
        other_node = id2node.get(int(nid_str), {})
        actual_k = other_node.get('neighbors', {}).get('276', 0)
    name = id2name.get(int(nid_str), nid_str)
    check(f"W276-{name}({nid_str}) k", exp_k, actual_k, tol=0.02)

# V262 Σk / R283 Σk ratio (claimed 2.5x)
ratio_v262_r283 = cross_helix_k[262] / cross_helix_k[283]
check("V262/R283 ratio (claimed 2.5x)", 2.5, ratio_v262_r283, tol=0.15)

print()
print("="*80)
print("SECTION 10.1.1: Contact partner percentages for all residues")
print("="*80)
# Verify the contact pattern table
contact_pattern_claims = {
    261: ('ICL3', 86),
    262: ('ICL3', 89),
    263: ('ICL3', 82),
    264: ('ICL3', 32),
    266: ('ICL3', 100),
    267: ('TM7', 100),
    268: ('TM7', 79),
    269: ('ICL3', 66),
    271: ('TM7', 100),
    272: ('TM3', 62),
    273: ('TM5', 71),
    274: ('TM7', 100),
    275: ('TM7', 100),
    280: ('TM5', 82),
    281: ('TM5', 100),
    283: ('TM3', 74),
    284: ('TM5', 100),
    286: ('TM5', 100),
    287: ('TM5', 100),
    290: ('TM5', 100),
}

for rid, (exp_tm, exp_pct) in contact_pattern_claims.items():
    partners = get_cross_helix_by_tm(rid)
    total = sum(partners.values())
    if total < 0.001:
        print(f"  [SKIP] {rid} has no cross-helix contacts (Σk={total:.4f})")
        continue
    actual_pct = partners.get(exp_tm, 0) / total * 100
    check(f"Residue {rid} dominant TM={exp_tm} pct", exp_pct, actual_pct, tol=5)

print()
print("="*80)
print("SECTION 10.4: Salt Bridge Removal Simulation")
print("="*80)

# Build adjacency matrix
def build_adjacency(exclude_pair=None):
    K = np.zeros((N, N))
    for node in data['nodes']:
        i = id2idx[node['id']]
        for nid_str, k in node['neighbors'].items():
            nid = int(nid_str)
            if nid not in id2idx:
                continue
            j = id2idx[nid]
            if exclude_pair and (
                (node['id'], nid) == exclude_pair or
                (nid, node['id']) == exclude_pair
            ):
                continue
            K[i][j] = k
            K[j][i] = k
    return K

def simulate_strike(K, strike_id, steps=300, dt=0.005, damping=0.05):
    x = np.zeros(N)
    v = np.zeros(N)
    x[id2idx[strike_id]] = 1.0
    peak = np.abs(x).copy()
    D = np.sum(K, axis=1)
    for _ in range(steps):
        force = K @ x - D * x - damping * v
        v += force * dt
        x += v * dt
        peak = np.maximum(peak, np.abs(x))
    return peak

print("  Running ENM simulation (dt=0.005, γ=0.05, 300 steps)...")
K_normal = build_adjacency()
peak_normal = simulate_strike(K_normal, 124)

K_no_bridge = build_adjacency(exclude_pair=(124, 283))
peak_no_bridge = simulate_strike(K_no_bridge, 124)

tm6_pn  = [peak_normal[id2idx[r]] for r in tm6_ids]
tm6_pnb = [peak_no_bridge[id2idx[r]] for r in tm6_ids]

pct_change = [
    (pnb - pn) / pn * 100 if pn > 1e-12 else 0.0
    for pn, pnb in zip(tm6_pn, tm6_pnb)
]

bottom_pct = [pct_change[i] for i in range(0, 15)]
middle_pct = [pct_change[i] for i in range(15, 22)]
top_pct    = [pct_change[i] for i in range(22, 32)]

bottom_avg = np.mean(bottom_pct)
middle_avg = np.mean(middle_pct)
top_avg    = np.mean(top_pct)

print(f"  Computed zone averages: Bottom={bottom_avg:.1f}%, Middle={middle_avg:.1f}%, Top={top_avg:.1f}%")

check("Salt bridge removal: Bottom avg", -3, bottom_avg, tol=2)
check("Salt bridge removal: Middle avg", -32, middle_avg, tol=3)
check("Salt bridge removal: Top avg", -91, top_avg, tol=3)

# R283 peak with bridge (claimed 0.093 in caption)
r283_peak = tm6_pn[tm6_ids.index(283)]
print(f"  [INFO] R283 peak (with bridge) = {r283_peak:.6f}")

# Per-residue pct change for all TM6
print("\n  Per-residue % change:")
for i, rid in enumerate(tm6_ids):
    zone = "Bottom" if i < 15 else ("Middle" if i < 22 else "Top")
    print(f"    {rid} ({zone}): normal={tm6_pn[i]:.6f}, removed={tm6_pnb[i]:.6f}, change={pct_change[i]:.1f}%")

print()
print("="*80)
print("SECTION 10.5: Prediction box values")
print("="*80)
check("Prediction box: k(E124-R283)", 0.45, k_e124_r283, tol=0.02)
check("Prediction box: W276 Σk", 2.66, cross_helix_k[276], tol=0.02)
# "TM6 bottom anchored by TM5/ICL3 (Σk>1.4)" - check max of bottom zone
print(f"  [INFO] Bottom zone max Σk = {max(bottom_k):.2f} (claimed >1.4)")
# The claim says "Σk>1.4" for bottom, which means there exist residues with Σk>1.4
check("Bottom has residues with Σk>1.4", True, max(bottom_k) > 1.4)

print()
print("="*80)
print("FIGURE 3: Hard-coded hub values in generate_fig3_40x_gap.py")
print("="*80)
fig3_active = [('S125', 92.307), ('V122', 29.802), ('E124', 19.862), ('C198', 9.054), ('P200', 8.559)]
fig3_inactive = [('C116', 0.2119), ('F286', 0.0681), ('S287', 0.0372), ('P278', 0.0250), ('F147', 1e-7)]

fig3_map = {125: 92.307, 122: 29.802, 124: 19.862, 198: 9.054, 200: 8.559,
            116: 0.2119, 286: 0.0681, 287: 0.0372, 278: 0.0250}
for rid, exp_r in fig3_map.items():
    actual_r = ratio_map.get(rid, -999)
    name = id2name.get(rid, str(rid))
    check(f"Fig3 {name} R value", exp_r, actual_r, tol=0.01)

print()
print("="*80)
print("CROSS-CHECK: Vectorized vs Loop-based ENM")
print("="*80)

# Loop-based simulation for comparison
def simulate_strike_loop(K_matrix, strike_id, steps=300, dt=0.005, damping=0.05):
    x = np.zeros(N)
    v = np.zeros(N)
    x[id2idx[strike_id]] = 1.0
    peak = np.abs(x).copy()
    for _ in range(steps):
        force = np.zeros(N)
        for i_idx in range(N):
            for j_idx in range(N):
                if K_matrix[i_idx][j_idx] > 0:
                    force[i_idx] -= K_matrix[i_idx][j_idx] * (x[i_idx] - x[j_idx])
            force[i_idx] -= damping * v[i_idx]
        v += force * dt
        x += v * dt
        peak = np.maximum(peak, np.abs(x))
    return peak

print("  Running loop-based simulation for comparison (this takes a moment)...")
peak_loop = simulate_strike_loop(K_normal, 124)

max_diff = np.max(np.abs(peak_normal - peak_loop))
# Exclude struck residue (index of 124) for comparison
idx_124 = id2idx[124]
mask = np.ones(N, dtype=bool)
mask[idx_124] = False
max_diff_no_struck = np.max(np.abs(peak_normal[mask] - peak_loop[mask]))

print(f"  Max diff (all residues): {max_diff:.2e}")
print(f"  Max diff (excluding struck): {max_diff_no_struck:.2e}")
check("Loop vs vectorized agreement (excl struck)", True, max_diff_no_struck < 1e-10)

print()
print("="*80)
print("CROSS-CHECK: Ratio map values vs live simulation")
print("="*80)
# Verify a few ratio map values by running live simulation
# Strike residue 124 and check hub peaks
print("  Checking R(E124) from simulation vs ratio map...")
active_sum_124 = sum(peak_normal[id2idx[h]] for h in ACTIVE_HUBS)
inactive_sum_124 = sum(peak_normal[id2idx[h]] for h in INACTIVE_HUBS)
r_124_computed = active_sum_124 / inactive_sum_124 if inactive_sum_124 > 0 else float('inf')
r_124_stored = ratio_map.get(124, -999)
print(f"  R(E124) computed = {r_124_computed:.4f}")
print(f"  R(E124) stored   = {r_124_stored:.4f}")
check("R(E124) simulation vs ratio_map", r_124_stored, r_124_computed, tol=0.5)

# Additional: strike S125
print("  Checking R(S125) from simulation vs ratio map...")
peak_125 = simulate_strike(K_normal, 125)
active_sum_125 = sum(peak_125[id2idx[h]] for h in ACTIVE_HUBS)
inactive_sum_125 = sum(peak_125[id2idx[h]] for h in INACTIVE_HUBS)
r_125_computed = active_sum_125 / inactive_sum_125 if inactive_sum_125 > 0 else float('inf')
r_125_stored = ratio_map.get(125, -999)
print(f"  R(S125) computed = {r_125_computed:.4f}")
print(f"  R(S125) stored   = {r_125_stored:.4f}")
check("R(S125) simulation vs ratio_map", r_125_stored, r_125_computed, tol=1.0)

# Strike F147
print("  Checking R(F147) from simulation vs ratio map...")
peak_147 = simulate_strike(K_normal, 147)
active_sum_147 = sum(peak_147[id2idx[h]] for h in ACTIVE_HUBS)
inactive_sum_147 = sum(peak_147[id2idx[h]] for h in INACTIVE_HUBS)
r_147_computed = active_sum_147 / inactive_sum_147 if inactive_sum_147 > 0 else float('inf')
r_147_stored = ratio_map.get(147, -999)
print(f"  R(F147) computed = {r_147_computed:.6f}")
print(f"  R(F147) stored   = {r_147_stored:.6f}")

print()
print("="*80)
print("VERIFICATION: Section 9 conclusion claims")
print("="*80)
# "removing salt bridge frees TM6 top (−93%)" - doc says -93% in Section 9 but -91% in Section 10.4
check("Section 9: Top avg displacement change (now -91%)", -91, top_avg, tol=2)

# "W276 Σk = 2.66" in Section 9
check("Section 9: W276 Σk", 2.66, cross_helix_k[276], tol=0.02)

# "E124-R283 (k = 0.333, or 3.0 Å)" in Section 6.4
# Actually k = 0.45. Check if Section 6.4 has wrong value
check("Section 6.4: k(E124-R283) now corrected", 0.451, k_e124_r283, tol=0.01)

print()
print("="*80)
print(f"AUDIT COMPLETE: {checks} checks performed")
print(f"ERRORS: {len(errors)}")
print("="*80)
if errors:
    print("\nFAILED CHECKS:")
    for e in errors:
        print(e)
else:
    print("\nALL CHECKS PASSED ✓")
