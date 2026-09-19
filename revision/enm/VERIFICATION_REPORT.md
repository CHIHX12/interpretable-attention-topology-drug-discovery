# Verification Report — ENM claims vs. live computation

**Purpose:** authenticate every substantive claim in the GHSR/ENM documents
against actual computation, and flag anything not true. Generated to support
the goal: *"only write results that are real."*

**Date:** 2026-06-21
**Scripts used:** `full_audit.py`, `enm_demo.py`, `reverse.py` (all in this folder,
all reproducible). Data: `_data_compact.json` (293 residues, Cα + contact network).

---

## TL;DR

| Question | Verdict |
|----------|---------|
| Does the ENM math/network reproduce the stated numbers? | ✅ YES — `full_audit.py` passes 143/143 checks |
| Is the R-ratio "40× separation" an *independent* validation of the ML hubs? | ❌ NO — it is a **self-inclusion artifact** (random 10 points reproduce it; §2.4) |
| Can ENM, run blind, **reverse-derive** the 10 ML hubs? | ❌ NO — no blind metric ranks them at the top |
| Are the TM6 mechanism / salt-bridge / W276 findings real? | ✅ YES — they are hub-independent geometry |
| Are the "deltaImp" numbers in `ENM_WAVE_METHODS.md` §3.1 real? | ❌ NO — they are **Ballesteros-Weinstein numbers, mislabeled** |

**Bottom line:** the *mechanistic* use of ENM (explain how, given the hubs) is
sound. The *validation* use of ENM (claim it independently confirms or
rediscovers the hubs) is **not supported** and must be corrected.

---

## 1. What checks out (KEEP)

### 1.1 Network & math — VERIFIED
- 293 nodes, 1590 edges, mean k = 0.354, edge classes — all confirmed.
- Node = **Cα coordinate**; spring `k_ij = 1 / (min heavy-atom distance)`, edge if < 5 Å.
  (Confirmed: backbone pair 39–40 has k=0.667 ⇒ 1.50 Å peptide bond, **not** the
  3.81 Å Cα–Cα distance. So springs use heavy atoms; nodes sit on Cα.)
- Equation of motion (damped harmonic oscillator network) and symplectic-Euler
  integration: vectorized vs explicit double-loop agree to **2.78e-17**.
- `full_audit.py`: **143 checks, 0 errors.**

### 1.2 Hub-independent mechanism findings — VERIFIED & CREDIBLE
These do **not** depend on the 10 hub labels and stand on their own:
- **TM6 cross-helix restraint gradient** Σk_cross: Bottom 1.05 / Middle 0.88 / Top 0.38.
- **W276 = highest cross-helix restraint on TM6 (Σk = 2.66)**, matching the
  conserved CWxP 6.48 toggle switch. *(Notably, W276 is one of the few residues
  ENM flags blind — it is in the top-10 of WeightedDegree and Broadcast — so this
  is a genuine, independent structural result.)*
- **E124–R283 salt bridge** k = 0.451 = 48% of R283's cross-helix restraint.
- **Salt-bridge removal** (strike E124, delete the spring): TM6 Top −91%,
  Middle −32%, Bottom −3% — a real, asymmetric result.
- **Fig 5** three-state distances are crystal-structure facts (6KO5/8JSR/7F83).

> These support **Fig 4** and **Fig 5**.

---

## 2. What is NOT true / overclaimed (CORRECT)

### 2.1 "deltaImp values" in `ENM_WAVE_METHODS.md` §3.1 — FALSE LABEL
The doc lists: `E124 (3.33), V122 (3.32), S125 (3.34), C198 (5.24), P200 (5.26)`
… as **"ML deltaImp analysis."**

These are **Ballesteros–Weinstein generic GPCR position numbers**, not importance
scores. Proof:
- `GHSR_WAVE_METHODOLOGY.md` §7 lists a **"BW#" column** with the identical values
  (E124=3.33, S125=3.34, C198=5.24, …, Q120=3.29).
- `GHSR_WAVE_METHODOLOGY.md` §5 explicitly says "Ballesteros-Weinstein numbering
  (e.g., **3.33 for E124**)."
- The doc's own definition says inactive hubs have **negative** deltaImp, yet the
  §3.1 inactive values are positive (6.59, 6.58, 6.50) — i.e. they are TM6 BW
  positions (6.59 = 6.59), not signed importances.

**The real deltaImp numerical values are NOT in this folder** — only hub *ranks*
(1–5) appear (`GHSR_WAVE_METHODOLOGY.md` §7). The ML model itself is external.

### 2.2 "ENM reproduces the ML functional clustering" / "separation arises from 3D
geometry alone" (`ENM_WAVE_METHODS.md` §1, §9) — OVERCLAIMED
This **contradicts** `GHSR_WAVE_METHODOLOGY.md` §4 ("Would the 10 points be
visible without your analysis? **Short answer: NO**"). My tests support §4:

- **R-ratio is hub-dependent.** Striking E124 and only changing the observation
  set: original hubs R=19.86; swap Active/Inactive R=0.05 (=1/R); random 5+5 R=158.
  The "story" is entirely a function of which residues you pick.
- **Hub-free spectral partition (Fiedler vector)** does **not** separate
  Active from Inactive: 9 of 10 hubs fall on the **same** side; only F147 splits.
- **The hubs are not even two clean spatial clusters.** Active hubs are tight
  (mean Cα–Cα 11.8 Å) but **Inactive hubs are not** (mean 25.5 Å); and C198/P200
  (Active) have **C116 (Inactive)** as their nearest hub neighbor (~6 Å).

So "40× separation, zero overlap" is a real *number* but **not** an independent
geometric phenomenon — it is produced by the hub-weighted R metric.

### 2.3 Blind reverse-derivation — ENM does NOT recover the hubs
`reverse.py` ranks all 293 residues by four blind metrics (no hub labels used).
**None of the 10 ML hubs appear in any metric's top-10.** Hub average percentile:

| Metric | Avg hub percentile (50 = random) |
|--------|----------------------------------|
| WeightedDegree | 63% |
| Betweenness | 62% |
| Broadcast | 64% |
| SoftModePartic | **44% (below random)** |

The residues ENM flags as most important are **Q120, K130, Y128, W276** — i.e.
the constitutive **scaffold/bridges**, not the discriminative switches. This is
consistent with the docs' own note that **Q120 is the bridge but not a hub
(low deltaImp because it does not change between states).**

**Interpretation (the honest, and more interesting, story):**
ENM importance = *structural centrality*; ML deltaImp = *state discrimination*.
They measure different things. ENM **cannot** reverse-derive the hubs — which
means the ML genuinely sees something pure geometry does not. That strengthens
the ML result; it just means ENM's correct role is **explanation, not validation.**

### 2.4 The "40× separation" is a SELF-INCLUSION artifact (most important)

The headline "40× gap, zero overlap" of the 10 hubs (Fig 3, §4.2) is **not**
evidence that the ML hubs are special. It is a trivial consequence of each hub
being **its own observation station**: R(r) = Σpeak(Active)/Σpeak(Inactive), and
when the struck residue r is itself a hub, its own peak = 1.0 lands in *its own
group's* sum — automatically inflating Active hubs and deflating Inactive ones.

Verified two ways:

| Test (300 steps) | ML 10 hubs | Random 10 points (3000 trials) |
|------------------|-----------|--------------------------------|
| **WITH self** (the metric as used) | 40× gap, zero overlap | **100% zero-overlap; median 120×; 78.6% reach ≥40×** |
| **WITHOUT self** (exclude struck residue) | **gap = 0; overlaps** (min Active R=0.59 < max Inactive R=127) | ~0.7% zero-overlap |

So: (a) **random** 10 points separate *even more cleanly* than the ML hubs
(120× vs 40×), and (b) once the trivial self-term is removed, the ML hubs **do
not separate at all**. The 40× gap therefore carries no information about ML
correctness or geometry.

**Scope:** this affects only the **10 hubs' own R values** (Fig 3, §4.2, and the
hub markers in Fig 1). The Fig 1 distribution over the ~283 **non-hub** residues
is unaffected (they are in neither group, so no self-term). Reproduce with the
self/no-self test embedded in the analysis (peak-matrix indexing; see
`test_step_robustness.py` for the peak-matrix construction).

### 2.5 Step-count dependence (R magnitude not converged)

300 steps is a fixed window, not a converged value (weak damping, γ=0.05). R
**magnitudes drift down** with more steps; but **direction** (Active R>1, Inactive
R<1) holds across 100–2000 steps with no crossover, and the 293-residue **ranking**
is stable near the window (Spearman ρ>0.98 for 100–500 vs 300; ρ≈0.77 at 2000).
See `FigS1_enm_step_robustness.png` / `test_step_robustness.py`.

---

## 3. Recommended corrections to the documents

| File | Location | Change |
|------|----------|--------|
| `ENM_WAVE_METHODS.md` | §3.1 | Relabel "deltaImp" values as **BW numbers**; state real deltaImp is external |
| `ENM_WAVE_METHODS.md` | §1, §9 | Drop "ENM reproduces/validates the ML clustering" and "geometry alone separates"; reframe as *mechanistic explanation given the ML hubs* |
| `ENM_WAVE_METHODS.md` | new section | Add the blind reverse-derivation result (ENM ≠ recovers hubs) and point to `reverse.py` |
| `ENM_WAVE_METHODS.md` | §4.2 | Keep the R numbers, but label the 40× as a **hub-weighted metric**, not independent proof |
| `README.md` | §6 results table | Add caveat to the "40.4× gap" headline |
| `GHSR_WAVE_METHODOLOGY.md` | §4 | Largely correct already; cross-reference this report |

---

## 4. How to re-verify (anytime)

```bash
python3 full_audit.py      # 143 numeric checks of the network/sim claims
python3 reverse.py         # blind test: does ENM recover the 10 hubs? (no)
python3 enm_demo.py 124    # strike any residue, see energy reach the 10 hubs
```
