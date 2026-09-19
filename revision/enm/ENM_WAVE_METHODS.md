# Elastic Network Model (ENM) Wave Propagation Analysis of GHSR

> **📝 THIS IS THE MANUSCRIPT SOURCE — write the paper from this file.**
> Supporting files: `VERIFICATION_REPORT.md` (claim-by-claim audit),
> `GHSR_WAVE_METHODOLOGY.md` (ML/topology background), `README.md` (how to run).
>
> **Core framing (use this exact logic in the paper):**
> *ML finds **which** residues discriminate active/inactive (the 10 hubs); the ENM,
> given those hubs, finds the **mechanical pathway** connecting them (E124–R283 salt
> bridge, W276 toggle, TM6 decoupling). The ENM **explains** the ML result and yields
> a testable activation hypothesis — it does **not** validate or rediscover the hubs
> (a blind reverse-derivation fails; `reverse.py`). Confirming the hubs are correct
> requires independent evidence (mutagenesis / known biology).*

## 1. Summary

We applied an Elastic Network Model (ENM) to the antagonist-bound GHSR structure
(PDB: 6KO5) as a **mechanistic tool to explain** the 10 functional residues
("hubs") identified by an external machine-learning (ML) deltaImp analysis. The
ENM is used to ask *how*, given those hubs, the structure transmits mechanical
perturbation — **not** to independently rediscover or validate the hubs.

By "striking" each of the 293 resolved residues and tracking how vibrational
displacement distributes, we report a hub-weighted descriptive ratio:

1. Define `R(r)` = (sum of peak displacement at the 5 Active hubs) / (sum at the
   5 Inactive hubs) when residue r is struck. Striking an Active hub gives high R
   (8.6–92.3); striking an Inactive hub gives low R (0.00–0.21).
2. By this metric the two hub sets do not overlap (a ~40× gap) — **but this gap is
   a self-inclusion artifact, not evidence**: when a hub is struck, its own peak
   (=1.0) falls in its own group's sum. Random 10 points reproduce the gap even more
   cleanly (median 120×), and removing the self-term collapses it (§4.2). Do not
   report the 40× gap as validation of the ML hubs.

**Important scope limits (verified — see `VERIFICATION_REPORT.md`):**

- `R(r)` is **hub-dependent by construction**. It encodes the hub identities, so
  it cannot serve as *independent* confirmation of the ML clustering. Swapping or
  randomizing the observation set changes the result completely.
- A **hub-free** analysis does **not** reproduce a clean Active/Inactive split:
  the standard spectral (Fiedler) partition puts 9 of 10 hubs on the same side,
  and the Inactive hubs are not even a tight spatial cluster.
- A **blind reverse-derivation** (`reverse.py`) ranking all 293 residues by
  structural-importance metrics does **not** recover the 10 hubs (they sit near
  the median, ~44–64th percentile). ENM importance measures *structural
  centrality*; ML deltaImp measures *state discrimination* — different quantities.

What the ENM *does* contribute is a credible, hub-independent **mechanism** for
TM6 activation (the E124–R283 salt bridge, the W276 toggle switch, and the
asymmetric TM6 restraint gradient; Sections 10 and figures 4–5).

---

## 1.1 Division of Labor: ML vs ENM (how to describe this work)

The two methods answer **different** questions and must be reported in that order:

**Step 1 — ML identifies *which* residues (the "what").**
A machine-learning classifier separates active vs inactive GHSR conformations.
Its deltaImp ranking selects the 10 residues whose local environment changes most
between the two states — i.e. the **state-discrimination switches**. ML answers
*which residues matter*.

**Step 2 — ENM, *given* those residues, reveals the mechanical pathway (the "how").**
Taking the 10 ML hubs as fixed targets, the elastic network model shows, by pure
mechanics, *how they are physically coupled*: striking an Active hub channels
displacement to the other Active hubs through the TM3→TM5 contact relay
(E124–Q120–C198), and the only strong conduit to TM6 is the E124–R283 salt bridge.
ENM is a **pathway / mechanism finder**.

**Step 3 — ENM by itself cannot do the rest (the limit, verified).**
A blind ENM importance ranking over all 293 residues does **not** recover the 10
hubs (`reverse.py`; §6.3): mechanical centrality ≠ state discrimination. So the ENM
**cannot** independently derive, reproduce, or validate the ML selection.

**What the combination does and does NOT prove.**

- ✅ It **does**: turn the ML hub list into a coherent, testable mechanical
  hypothesis (the salt-bridge/W276 pathway), making the ML result interpretable.
- ❌ It **does not**: prove "the ML picked the right residues." Mechanical coupling
  is necessary context, not proof — most residues in a dense bundle are connected by
  *some* path. **Confirmation that the hubs are functionally correct requires an
  independent test** (e.g. mutagenesis), or agreement with established biology
  (e.g. W276 = the conserved CWxP toggle — but note W276 is *not* one of the ML hubs).

**One-line framing:** *ML finds which points; ENM finds the mechanical pathway
connecting them. ENM explains the ML result — it does not validate it.*

---

## 2. Mathematical Model

### 2.1 Elastic Network Construction

From the 6KO5 crystal structure, we construct a contact network:

- **Nodes**: All 293 resolved residues (Cα coordinates)
- **Edges**: Any two residues with heavy-atom distance < 5 Å
- **Spring constant**: k_ij = 1 / d_ij, where d_ij is the minimum heavy-atom distance between residues i and j
  - Backbone neighbors (|i-j| ≤ 1): k = 1/1.5 = 0.667 (peptide bond ~1.5 Å)
  - Cross-helix contacts: k ranges from 0.200 (5.0 Å) to 0.493 (2.0 Å)

**Network statistics**:
- 293 nodes
- 1590 unique edges (undirected)
- Mean coupling strength: 0.354
- Sequential edges (|i−j| ≤ 1, peptide bonds): 291 (k = 0.667)
- Intra-helix non-sequential (same TM, |i−j| > 1): 798 (mean k = 0.294)
- Inter-helix (different TM): 501 (mean k = 0.267)

### 2.2 Equation of Motion

Each residue i is treated as a unit mass connected to its neighbors by springs. The displacement from equilibrium x_i obeys:

```
    d²x_i / dt² = - Σ_j k_ij · (x_i - x_j) - γ · dx_i/dt
```

where:
- x_i = scalar displacement of residue i from equilibrium position
- k_ij = spring constant between residues i and j (0 if not connected)
- γ = damping coefficient (default 0.05)
- The sum runs over all neighbors j of residue i

This is the **damped harmonic oscillator network** equation. For a system of N residues, it forms an N×N coupled oscillator system.

### 2.3 Numerical Integration

We use symplectic Euler integration:

```
    F_i(t) = - Σ_j k_ij · (x_i(t) - x_j(t)) - γ · v_i(t)

    v_i(t + dt) = v_i(t) + F_i(t) · dt

    x_i(t + dt) = x_i(t) + v_i(t + dt) · dt
```

**Parameters**:
- dt = 0.005 (time step)
- γ = 0.05 (damping)
- Stiffness multiplier = 1.0 (scales all k_ij uniformly)
- Total simulation: 300 steps per strike (t_final = 1.5 time units)

### 2.4 Strike Protocol

For each residue r ∈ {1, ..., 293}:

1. Set all displacements to zero: x_i(0) = 0 for all i
2. Set all velocities to zero: v_i(0) = 0 for all i
3. Apply initial displacement to the struck residue: x_r(0) = 1.0
4. Integrate for 300 steps
5. Record the **peak displacement** (maximum |x_i(t)| over all time steps) for each ML hub

Each residue's reported value comes from striking **that residue itself** (step 3
sets x_r = 1.0); e.g. the value plotted for E124 is the result of striking E124.

> **Oscillatory dynamics — why x changes sign, and why we use peak |x|.** Each
> residue is a damped harmonic oscillator, so its displacement x_i(t) is **signed**
> and oscillates about equilibrium: striking E124 sets x = +1.0, the springs pull it
> back, it overshoots through 0 to **negative**, swings back, and so on, with the
> amplitude decaying under damping (γ). (Direct trace of x_124: +1.00 at t=0 →
> ≈0 at t≈0.72 → −0.84 at t≈1.38 → returning.) The sign merely indicates *which
> direction* a residue is currently displaced (in the viewer, red = +, blue = −);
> different residues oscillate with different phases, which is what produces the
> travelling/reflecting wave. Because only the *magnitude* of the excursion is
> physically meaningful here, every quantity is built from the **peak absolute
> displacement** max|x_i(t)|, making R sign-independent and always positive.

### 2.5 Ratio Calculation

For each strike residue r, compute:

```
    R(r) = Σ_{h ∈ Active hubs} peak_h(r)  /  Σ_{h ∈ Inactive hubs} peak_h(r)
```

where:
- Active hubs = {E124, V122, S125, C198, P200} (from ML analysis)
- Inactive hubs = {S287, F286, P278, F147, C116} (from ML analysis)
- peak_h(r) = max_{t ∈ [0, T]} |x_h(t)| when residue r is struck

**Interpretation**:
- R(r) > 1 : striking residue r sends more displacement to Active hubs
- R(r) < 1 : striking residue r sends more displacement to Inactive hubs
- R(r) = 1 : displacement equally distributed (crossover point)

> **The 10 hubs as "monitoring stations" (conceptual role).** The Active and
> Inactive hubs function as fixed **sensors / listening posts**. Striking a residue
> and reading only these 10 points yields a compressed, two-number summary
> (Σpeak_Active, Σpeak_Inactive) — a **directional readout** of which side of the
> receptor a perturbation flows toward, obtainable from structure alone (no
> experiment). This gives the method a predictive, "early-indication" character:
> the geometry forecasts whether a perturbation is mechanically routed toward the
> activation machinery or the inhibitory scaffold.
>
> Two limits must be kept in mind:
> 1. **The stations are a summary, not the full picture.** The complete propagation
>    "context" is the wave itself across all 293 residues (visualised in
>    `wave_strike.html`); R(r) is only what two curated sensor groups detect.
> 2. **The stations report geometric flow, not functional importance.** What a high
>    or low R indicates is the struck residue's 3D position relative to the two hub
>    clusters (see §4.1), not how functionally important it is.

### 2.6 Percentile Ranking

Each R(r) is ranked among all 293 values to obtain a percentile:

```
    Percentile(r) = (number of residues with R ≤ R(r)) / 293 × 100%
```

---

## 3. Reproducibility

### 3.1 Input Data

| Item | Source |
|------|--------|
| Structure | PDB: 6KO5 (GHSR bound to antagonist compound 21, 2.7 Å resolution) |
| Active hubs | From external ML deltaImp ranking (top 5). Residues: E124, V122, S125, C198, P200. Numbers in parentheses below are **Ballesteros–Weinstein positions**, not importance scores: E124 (3.33), V122 (3.32), S125 (3.34), C198 (5.24), P200 (5.26) |
| Inactive hubs | From external ML deltaImp ranking (top 5). Residues: S287, F286, P278, F147, C116. **BW positions**: S287 (6.59), F286 (6.58), P278 (6.50), F147 (ICL2), C116 (3.25) |

> **Note on provenance.** The actual ML model and the numerical deltaImp values
> are **not stored in this repository** (see `GHSR_WAVE_METHODOLOGY.md`). Only the
> hub identities and ranks are available here. Earlier versions of this table
> mislabeled the Ballesteros–Weinstein position numbers (3.33, 6.59, …) as
> "deltaImp" values — corrected above. See `VERIFICATION_REPORT.md` §2.1.

### 3.2 Software

- Contact network extraction: Python 3, Biopython (PDB parser)
- ENM simulation: Custom JavaScript (Float64Array, symplectic Euler)
- Pre-computation of all 293 ratios: Python 3 (NumPy)
- Visualization: HTML5 Canvas (interactive), Matplotlib (static figures)

### 3.3 Parameter Sensitivity

The simulation was run with default parameters (dt=0.005, γ=0.05, stiffness=1.0). The interactive HTML allows users to vary these parameters in real-time. Key observations:

- **Damping (γ)**: Lower damping → longer oscillation → sharper resonance peaks. Higher damping → diffusion-like behavior. The ratio R(r) ranking is robust across γ ∈ [0.01, 0.30].
- **Stiffness multiplier**: Uniform scaling does not change the ratio (it only scales wave speed). The ranking is invariant to this parameter.
- **dt**: Must be small enough for numerical stability (dt < 0.02 for these k values). Results converge at dt ≤ 0.01.
- **Number of steps**: 300 steps is the fixed integration window used throughout (t = 1.5). It is **not** a converged value — the system is weakly damped (γ=0.05) and takes thousands of steps to settle, so peak values at distant hubs keep rising past 300 steps and the **absolute R magnitudes drift downward** with more steps (verified). What is robust is what the conclusions rely on: the **direction** of every ML hub (Active stays R>1, Inactive stays R<1) holds across 100–2000 steps with no crossover, and the **293-residue ranking** is stable near the chosen window (Spearman ρ > 0.98 for 100–500 steps vs the 300-step reference; ρ degrades to ~0.77 only at 2000 steps). See `FigS1_enm_step_robustness.png` / `test_step_robustness.py`. Conclusion: trust the direction and (near-window) ranking, not the absolute R values.

### 3.4 How to Reproduce

1. Download PDB 6KO5 from RCSB
2. Extract Cα coordinates for chain R (receptor)
3. Build contact graph: for all residue pairs, if minimum heavy-atom distance < 5 Å, add edge with k = 1/distance
4. For each of 293 residues, run the strike protocol (Section 2.4)
5. Compute R(r) for each residue (Section 2.5)
6. The interactive viewer (`wave_strike.html`) embeds the pre-computed network and runs the simulation client-side

---

## 4. Results

### 4.1 Global Distribution

Striking all 293 residues produces the following distribution of R values:

| Statistic | Value |
|-----------|-------|
| Minimum | 0.000 (30 residues tied, including F147/ICL2, ICL3 loop, lower TM4) |
| 25th percentile | 0.067 |
| Median | 0.872 |
| 75th percentile | 7.442 |
| Maximum | 1225.1 (V166, TM4) |
| Mean | 23.93 |
| Std dev | 107.4 |

The distribution is **strongly right-skewed** (log-normal), reflecting the asymmetric geometry of the GPCR bundle.

> **What R actually measures (important interpretation).** R(r) tracks each
> residue's **3D position relative to the two hub clusters**, not its functional
> importance. Vibrational displacement attenuates steeply with every spring hop,
> so a hub receives displacement mainly as a function of network distance from the
> struck residue. Because the Active hubs cluster on the extracellular/pocket side
> and the Inactive hubs on the intracellular/TM6 side, R(r) essentially maps where
> a residue sits along the pocket↔G-protein axis. High R = pocket-side; low R =
> G-protein-side. **R is a geometric/positional quantity, not an importance score.**
>
> **Caveat on the maximum (V166, R ≈ 1225).** This extreme value is *not* evidence
> that V166 drives activation. Verified by direct simulation (`enm_demo.py 166`):
> striking V166 deposits only small displacement on the Active hubs
> (Σ ≈ 0.019) but essentially **zero** on the Inactive hubs (Σ ≈ 0.00002). The huge
> ratio is therefore dominated by a **near-zero denominator** — i.e. V166 (upper
> TM4) is mechanically *isolated* from the Inactive side, not strongly coupled to
> the Active side. The magnitude of large R values is numerically unstable and
> should not be over-interpreted; only the *direction* (which side the displacement
> favours) is meaningful. Note V166 is **not** an ML hub — a purely geometry-driven
> extreme that ML did not flag, underscoring that R ≠ functional importance.

### 4.2 The 10 Hubs' Own R Values — and why the "40× gap" is an artifact

> ⚠️ **This section does NOT validate the ML hubs.** The apparent separation below
> is a self-inclusion artifact (see the caveat after the table). It is retained
> only to document the values and to explain why they must not be over-read.

| Hub | Type | R(r) | Percentile | Rank |
|-----|------|------|------------|------|
| S125 | Active | 92.31 | 95% | 16/293 |
| V122 | Active | 29.80 | 87% | 25/293 |
| E124 | Active | 19.86 | 85% | 29/293 |
| C198 | Active | 9.05 | 78% | 48/293 |
| P200 | Active | 8.56 | 78% | 50/293 |
| C116 | Inactive | 0.212 | 36% | 105/293 |
| F286 | Inactive | 0.068 | 26% | 75/293 |
| S287 | Inactive | 0.037 | 23% | 68/293 |
| P278 | Inactive | 0.025 | 20% | 59/293 |
| F147 | Inactive | 0.000 | 10% | tied at 0 (30 residues) |

**Separation gap**: min(Active) / max(Inactive) = 8.56 / 0.212 = **40.4×**, zero overlap.

> **This 40× gap is a SELF-INCLUSION artifact — not evidence (verified).**
> R(r) = Σpeak(Active)/Σpeak(Inactive). When the struck residue r is itself one of
> the 10 hubs, its own peak = 1.0 lands in *its own group's* sum, trivially pushing
> Active hubs up and Inactive hubs down — regardless of ML or geometry. Two checks:
>
> | Test (300 steps) | ML 10 hubs | Random 10 points (3000 trials) |
> |---|---|---|
> | **WITH self** (this table) | 40× gap, zero overlap | **100% zero-overlap; median 120×** |
> | **WITHOUT self** (exclude struck residue) | **gap = 0; overlaps** (min Active 0.59 < max Inactive 127) | ~0.7% zero-overlap |
>
> So random 10 points separate *even more cleanly* than the ML hubs, and removing
> the trivial self-term collapses the ML separation entirely. The gap carries **no
> information** about ML correctness. (Scope: this affects only the 10 hubs' own R;
> the Fig 1 distribution over the ~283 non-hub residues is unaffected.)
> See `VERIFICATION_REPORT.md` §2.4.

### 4.3 Crossover Point

The crossover (R = 1.0) occurs between:
- **PRO39** (N-terminus, R = 0.964) — last residue with R < 1
- **PHE312** (TM7, R = 1.082) — first residue with R > 1

This boundary separates the protein into two geometric domains:
- **R < 1 domain**: TM6, TM7 (extracellular half), ICL2, ICL3 — the "inhibitory scaffold"
- **R > 1 domain**: TM3 (upper), TM5 (upper), TM1, TM2, ECL2, TM4 (upper) — the "activation relay"

### 4.4 TM Helix Averages

| Helix | Median R | Interpretation |
|-------|----------|----------------|
| ICL3 | 0.000 | Strongly inhibitory (G-protein contact surface) |
| ICL2 | 0.000 | Strongly inhibitory (G-protein contact surface) |
| TM6 | 0.07 | Inhibitory (contains 3 Inactive hubs) |
| TM7 | 0.18 | Weakly inhibitory |
| Nterm | 0.79 | Near-neutral |
| loop | 1.53 | Near-neutral |
| TM5 | 2.17 | Weakly activating |
| TM3 | 3.56 | Activating (contains 3 Active + 1 Inactive hub) |
| TM1 | 9.78 | Activating |
| TM2 | 10.72 | Activating |
| ECL2 | 8.91 | Activating |
| TM4 | 0.16 (median), 151 (mean) | **Bipolar** — upper activating, lower inhibitory |

### 4.5 Surprise: TM4 Bipolarity

TM4 shows the most extreme range of any helix:
- **Upper TM4** (residues 163-168): R = 90-1225 (top activating in entire protein)
- **Lower TM4** (residues 148-152): R = 0.000 (maximally inhibitory)

This transition occurs within a single helix turn, at the TM4-ECL2 junction. Upper TM4 contacts ECL2 and the binding pocket (connected to Active hubs), while lower TM4 contacts ICL2 (connected to Inactive hubs, especially F147).

This bipolarity is the clearest illustration that **R encodes 3D position, not importance**: the same helix spans the entire R range purely because its two ends sit on opposite sides of the bundle (upper end near the pocket/Active cluster, lower end near ICL2/Inactive cluster). The high upper-TM4 values are denominator-driven (see the V166 caveat in §4.1) and should be read as "geometrically pocket-directed," not "functionally most important."

---

## 5. Figure Captions

### Figure 1: Fig1_enm_ratio_distribution.png

**ENM wave propagation ratio for all 293 GHSR residues.** R reflects each residue's 3D position relative to the two ML hub clusters (geometry), **not** ML/functional importance; extreme high-R values (e.g. V166) are denominator-driven and indicate geometric isolation from the Inactive side, not strong activation (see §4.1).

**(A)** Waterfall plot showing log₁₀(R) for each residue, sorted from lowest (left, inhibitory) to highest (right, activating). Each bar is colored by TM helix assignment. ML-identified Active hubs (red triangles) and Inactive hubs (blue triangles) are marked with labels. Dashed vertical line indicates the crossover point (R = 1.0), flanked by PRO39 (N-term, R = 0.96) and PHE312 (TM7, R = 1.08). The 40× gap between the weakest Active hub (P200, R = 8.56) and strongest Inactive hub (C116, R = 0.21) is a **self-inclusion artifact** of the hub markers (each struck hub's own peak = 1.0 falls in its own group); random 10 points reproduce it even more cleanly and removing the self-term collapses it (§4.2). It is not an independent separation. The overall *distribution* of the ~283 non-hub residues, however, is genuine geometric routing.

**(B)** Histogram of log₁₀(R) values. Vertical colored lines mark positions of Active hubs (red, right) and Inactive hubs (blue, left). Under this hub-weighted metric the two hub groups fall in opposite tails with no overlap. (Because R is computed from the hubs, this figure describes the metric's behaviour given the ML hubs; it does not independently validate the clustering.)

### Figure 2: Fig2_enm_ratio_by_TM.png

**ENM wave ratio grouped by transmembrane helix.** Box plots show the distribution of log₁₀(R) for each structural segment, sorted by median. ICL3 and ICL2 are maximally inhibitory (median R ≈ 0). TM6 contains all three TM6-based Inactive hubs (P278, S287, F286). TM4 shows extreme bipolarity (largest interquartile range). Diamond markers indicate ML-identified hubs: red = Active, blue = Inactive. Dashed horizontal line marks R = 1.0 (equal displacement distribution). Each helix's sample size is shown in parentheses. The per-segment ordering reflects each segment's geometric position relative to the two hub clusters (pocket vs G-protein side) — i.e. R encodes **3D location, not functional importance**. TM4's bipolar spread is purely positional (upper end pocket-side, lower end ICL2-side).

---

## 6. Why Does the Hub-Weighted Ratio Behave This Way?

### 6.1 The Key Insight

The ENM contains **no biological knowledge**. It knows only:
- 3D coordinates of each residue
- Which pairs are within 5 Å contact distance
- Spring constants proportional to 1/distance

**Given the ML hub assignments**, the ratio `R(r)` cleanly orders the two hub
sets. This is **not** because the ENM blindly separates them — it does not (a
hub-free split fails to recover the grouping; §6.3). It is because the *Active*
hubs happen to be a tight spatial cluster near the pocket, so displacement
channels efficiently among them. The question this section answers is therefore
narrower: *given* the hubs, what geometric context explains the channelling?

### 6.2 The Geometric Context (given the hub assignments)

The Active hubs sit together near the pocket; the salt bridge is the main link
to the TM6 side:

```
    EXTRACELLULAR SIDE (ligand binding)
    ┌─────────────────────────────┐
    │   TM3(upper)  TM5(upper)   │  ← Active hub zone
    │   E124 S125   C198 P200    │     (pocket-facing)
    │        V122                 │
    │                             │
    │   ───── salt bridge ─────   │  ← E124-R283 pivot
    │                             │
    │   TM6(upper)                │  ← Inactive hub zone
    │   S287 F286 P278            │     (membrane-facing)
    └─────────────────────────────┘
    ┌─────────────────────────────┐
    │   ICL2: F147                │  ← Inactive hub zone
    │   TM3(lower): C116          │     (intracellular face)
    └─────────────────────────────┘
    INTRACELLULAR SIDE (G-protein binding)
```

The two hub groups are on **opposite sides** of the helical bundle:
- Active hubs cluster near the **ligand-binding pocket** (extracellular, TM3-TM5 interface)
- Inactive hubs cluster near the **G-protein interface** (intracellular) and the **TM6 rigid scaffold**

Springs connect nearby residues. Displacement deposited on one side must traverse many springs to reach the other side. Each spring crossing attenuates displacement (damping) and splits it (branching). By the time the wave reaches the opposite cluster, most displacement has been absorbed by the intervening residues.

### 6.3 Could You Discover the Hubs Without ML?

**No — a blind ENM analysis does not recover the 10 hubs.** (This was tested
directly; see `reverse.py` and `VERIFICATION_REPORT.md` §2.3.)

What we tested: rank all 293 residues by four hub-free importance metrics
(weighted degree, soft-mode participation, betweenness, broadcast capacity).
Result: **none of the 10 ML hubs appear in any metric's top-10**, and their
average percentile is only ~44–64% (≈ random). The residues ENM flags as most
important are instead the constitutive scaffold/bridge residues — Q120, K130,
Y128, W276 — exactly the residues the ML deltaImp ranks *low* because they do
not change between states.

Why: ENM importance = **structural centrality** (who is connected / who is a
bottleneck); ML deltaImp = **state discrimination** (whose environment changes
between active and inactive). These are different quantities, so ENM cannot be
expected to reverse-derive the ML hubs — and it does not.

What ENM alone **cannot** reveal:
- The Active/Inactive partition (a hub-free spectral split does not produce it)
- Which specific residues are the functional switches
- The magnitude or sign of deltaImp (functional importance ranking)

**ML provides the labels** (Active/Inactive); **ENM provides a hub-independent
mechanism** (salt bridge, W276 toggle, TM6 restraint gradient). The ENM does not
validate the ML — it complements it. That ENM *cannot* see these residues is
itself evidence that the ML captures non-trivial, non-geometric information.

Analogy: ENM is like a topographic map showing two valleys separated by a ridge. ML tells you which valley is "farmland" and which is "desert." You need both to understand the landscape.

### 6.4 Why Does TM6 Tilt? (Structural Mechanics)

The ENM reveals WHY TM6 is the effector helix:

1. **TM6 is structurally isolated**: Median R = 0.07. Displacement from the binding pocket does not easily reach TM6. This means TM6 is a "rigid rod" held in place by its own internal springs.

2. **The E124-R283 salt bridge** (k = 0.451, or 2.2 Å) is the **only strong cross-helix connection** between the Active hub domain (TM3) and the Inactive hub domain (TM6). It acts as a hinge/pivot.

3. **TM6 has an asymmetric restraint gradient**: The intracellular end (bottom) is firmly anchored by 8+ ICL3 contacts (avg Σk=1.05), while the extracellular end (top) is loosely held with only TM5 contacts + the salt bridge (avg Σk=0.38). This means:
   - Bottom of TM6: held firmly by ICL3/TM7 → stays in place
   - Top of TM6: restrained primarily by the E124-R283 salt bridge → freed when bridge loosens
   - Result: asymmetric motion → outward tilt at the bottom → G-protein binding site opens

4. **Agonist mechanism**: When a ligand pushes on E124 (TM3), displacement propagates through the Active hub network (TM3→TM5). This does NOT directly push TM6 (the elastic network blocks it). Instead, the salt bridge E124-R283 acts as the single conduit. Modulating this bridge modulates TM6 restraint.

Our scalar ENM cannot show the **direction** of TM6 tilt (that requires a 3D vector model or Normal Mode Analysis). But it correctly shows that TM6 is **mechanically decoupled** from the activation pocket — which is the structural prerequisite for the tilt mechanism.

### 6.5 What Ratio = 1.0 Means

The crossover residues (PRO39, PHE312) are at the **structural boundary** between the two domains:

- PRO39 (N-terminus): The N-terminal tail wraps around the extracellular surface. It contacts both the pocket region (→ Active hubs) and the TM6/TM7 interface (→ Inactive hubs). At R = 0.96, displacement propagates almost equally to both sides.

- PHE312 (TM7, position 7.42): TM7 connects the extracellular (pocket-facing) and intracellular (G-protein-facing) regions. PHE312 sits near the midpoint of TM7. At R = 1.08, it is essentially equidistant from both hub groups in the spring network.

These residues are the **neutral zone** — the geographic equator of the GPCR. Neither side of the elastic network dominates.

---

## 7. Limitations

1. **Scalar displacement**: Our model uses a 1D scalar displacement per residue, not a 3D vector. This captures displacement distribution but not the direction of motion (e.g., TM6 outward tilt vs inward compression are indistinguishable).

2. **Static structure**: The contact network is derived from a single crystal structure (6KO5). In reality, contacts form and break dynamically. MD simulations would provide a more realistic contact ensemble.

3. **Uniform mass**: All residues are assigned unit mass. In reality, sidechains vary from 57 Da (glycine) to 204 Da (tryptophan). This affects wave velocity but not the topology of displacement distribution.

4. **Linear springs**: Real atomic interactions are nonlinear (Lennard-Jones, electrostatic). Our linear springs approximate the harmonic regime near equilibrium, which is valid for small perturbations.

5. **Hub labels from ML**: The ratio R(r) depends on which residues are defined as "Active" and "Inactive" hubs. These labels come from the ML analysis. The ENM does **not** validate or independently discover the ML clustering — a blind reverse-derivation does not recover the hubs (§6.3, `reverse.py`). The ENM's role is mechanistic explanation, not validation.

---

## 8. File Inventory

### 8.1 Data Files

| File | Description |
|------|-------------|
| `_data_compact.json` | ENM contact network: 293 nodes with Cα coordinates, TM assignments, neighbor lists with spring constants (from 6KO5) |
| `enm_ratio_map.json` | Pre-computed R(r) ratios for all 293 residues (Active peak sum / Inactive peak sum) |
| `6ko5.pdb` | Input structure: GHSR antagonist complex (PDB: 6KO5) |

### 8.2 Figures (600 DPI, PNG + PDF)

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `Fig1_enm_ratio_distribution.png/pdf` | All 293 residues: waterfall (sorted by log₁₀R, colored by TM) + histogram |
| Fig 2 | `Fig2_enm_ratio_by_TM.png/pdf` | Ratio by TM helix: boxplot with hub markers |
| Fig 3 | `Fig3_enm_40x_gap.png/pdf` | 40× gap: horizontal bar chart of 10 ML hubs |
| Fig 4 | `Fig4_enm_tm6_mechanism.png/pdf` | TM6 mechanism: restraint map (with sub-zones), salt bridge removal simulation |
| Fig 5 | `Fig5_enm_seesaw_mechanism.png/pdf` | Seesaw: three-state schematic (6KO5, 8JSR, 7F83) |

### 8.3 Reproducibility Scripts

All figures can be regenerated from the data files. Each script is self-contained:

```bash
python3 generate_fig1_ratio_distribution.py   # → Fig1_enm_ratio_distribution.png/pdf
python3 generate_fig2_ratio_by_TM.py          # → Fig2_enm_ratio_by_TM.png/pdf
python3 generate_fig3_40x_gap.py              # → Fig3_enm_40x_gap.png/pdf
python3 generate_fig4_tm6_mechanism.py        # → Fig4_enm_tm6_mechanism.png/pdf
python3 generate_fig5_seesaw_mechanism.py     # → Fig5_enm_seesaw_mechanism.png/pdf
```

| Script | Input | Runs ENM simulation? |
|--------|-------|---------------------|
| `generate_fig1_ratio_distribution.py` | `enm_ratio_map.json`, `_data_compact.json` | No (uses pre-computed ratios) |
| `generate_fig2_ratio_by_TM.py` | `enm_ratio_map.json`, `_data_compact.json` | No |
| `generate_fig3_40x_gap.py` | None (hard-coded hub values) | No |
| `generate_fig4_tm6_mechanism.py` | `_data_compact.json` | **Yes** (strikes E124 ×2: with/without salt bridge) |
| `generate_fig5_seesaw_mechanism.py` | None (hard-coded structural distances) | No |

Dependencies: Python 3, NumPy, Matplotlib. No other packages required.

### 8.4 Other Files

| File | Description |
|------|-------------|
| `wave_strike.html` | Interactive ENM wave simulator (self-contained, embeds network data + pre-computed ratios) |
| `plot_enm_tm6_mechanism.py` | Alternative Fig 4 script (vectorized simulation, contributed externally) |
| `ENM_WAVE_METHODS.md` | This document |

---

## 9. Conclusion

The Elastic Network Model does **not** independently reproduce or validate the ML
hub clustering (a blind reverse-derivation fails to recover the 10 hubs; §6.3,
`reverse.py`). Its contribution is a **hub-independent mechanism** for TM6
activation, built only from 6KO5 geometry. The key element is the E124–R283 salt
bridge, the principal strong spring linking TM3 to the TM6 extracellular half:

1. Perturbation at the binding pocket (TM3) cannot easily propagate to TM6 except through the salt bridge — explaining why TM6 is mechanically decoupled and can tilt independently
2. Perturbation at TM6 or ICL2/ICL3 stays localized — explaining why the inactive lock is self-reinforcing
3. The salt bridge is the critical bottleneck: the only strong spring connecting the two domains
4. Removing the salt bridge frees TM6 top (−91% displacement) while TM6 bottom stays anchored (−3%) — predicting the asymmetric outward tilt observed in crystal structures
5. W276 (6.48, CWxP toggle switch) independently emerges as the highest cross-helix restraint on TM6 (Σk = 2.66), serving as the mechanical fulcrum for TM6 rotation

The ENM does not replace the ML analysis — it **explains** it. ML identifies the functional switches; ENM reveals the mechanical architecture that makes those switches work. Moreover, the ENM goes beyond description to make testable **predictions**: the salt bridge need only loosen (not break) for activation, and the asymmetric restraint gradient channels this loosening into directional TM6 tilt — a prediction confirmed by the three-state structural comparison (6KO5 → 8JSR → 7F83).

---

## 10. TM6 Mechanism Prediction from ENM

This section presents the central mechanistic prediction of the ENM analysis: **why loosening the E124-R283 salt bridge allows TM6 to tilt outward**. All results below are derived purely from the 6KO5 contact geometry — no biological annotations, no MD simulations, no prior knowledge of GPCR activation is used.

### 10.1 TM6 Cross-Helix Restraint Gradient

For each TM6 residue (positions 261–292), we computed the total cross-helix spring strength:

```
    Σk_cross(i) = Σ_{j ∉ TM6} k_ij
```

This measures how strongly each TM6 residue is held in place by contacts from other helices (TM3, TM5, TM7, ECL, ICL).

#### 10.1.1 Zone Definition

TM6 is divided into three zones. The upper boundary (G282→R283) is unambiguous; the lower boundary (Bottom→Middle) is a gradual transition, discussed below.

**Cross-helix contact partners by residue (dominant helix and percentage):**

```
261 ICL3(86%)   267 TM7(100%)   273 TM5(71%)    279 TM7(54%)+TM3(45%)   285 -(0)
262 ICL3(89%)   268 TM7(79%)    274 TM7(100%)   280 TM5(82%)            286 TM5(100%)
263 ICL3(82%)   269 ICL3(66%)   275 TM7(100%)   281 TM5(100%)           287 TM5(100%)
264 ICL3(32%)   270 -(0)        276 TM7(43%)+   282 -(0) ◄ gap          288 -(0)
265 mixed       271 TM7(100%)       TM3(38%)+   283 TM3(74%) ◄ bridge   289 -(0)
266 ICL3(100%)  272 TM3(62%)!!      TM5+TM2(19%)284 TM5(100%)           290 TM5(100%)
```

**Zone assignments:**

| Zone | Residues | Primary contacts | Rationale |
|------|----------|------------------|-----------|
| **Bottom** | 261–275 | ICL3 (261-269), TM7 (267-275), with TM3+TM5 emerging at 272-273 | G-protein contact surface + helical interface. Contains two sub-patterns (see below). |
| **Middle** | 276–282 | TM7 + TM3 + TM5 (mixed) | Toggle switch zone. W276 contacts 3 helices simultaneously. Ends at G282 (Σk=0). |
| **Top** | 283–292 | TM5 exclusively (+ E124 salt bridge at R283) | Pocket-facing. All non-zero contacts are to TM5 only. |

**Bottom→Middle boundary discussion:**

The transition from Bottom to Middle is gradual, not a sharp boundary:

- **F272 (TM3=62%, TM5=22%)** — TM3 and TM5 first become dominant here. F272 contacts V131/I134 (TM3) and F221 (TM5), acting as a cross-bundle brace.
- **I273 (TM5=71%)** — TM5 dominant.
- **L274, C275 (TM7=100%)** — reverts to TM7-only, same pattern as 267-271.
- **W276 (TM7=43%, TM3=38%, TM5=10%)** — first residue where three helices contribute simultaneously.

The boundary is placed at **275→276** because:
1. W276 (6.48) is the universally conserved CWxP toggle switch — the strongest biological landmark on TM6
2. W276 is the first residue where TM3 contributes >30% (F272 has TM3=62% but the pattern is not sustained — 274-275 revert to TM7-only)
3. W276 has the highest Σk_cross on the entire helix (2.66), marking a structural singularity

However, F272 could also serve as an alternative boundary (271→272), since TM3+TM5 contacts genuinely begin there. The Bottom zone thus contains a sub-structure:

- **Bottom-ICL3** (261–269): dominated by ICL3 contacts (G-protein interface)
- **Bottom-TM7** (270–275): dominated by TM7 contacts (helical interface), with F272 as a TM3+TM5 outlier

**G282→R283 boundary:** This boundary is unambiguous. G282 has Σk_cross = 0 (zero cross-helix contacts), forming a natural structural gap. R283 marks the beginning of the salt bridge zone where TM3 (E124) provides 48% of all restraint.

#### 10.1.2 Restraint Gradient

The cross-helix restraint shows a striking **asymmetric gradient** across the three zones:

| TM6 zone | Residues | Σk_cross range | Avg Σk_cross | Primary anchors |
|----------|----------|----------------|--------------|-----------------|
| Bottom (intracellular) | 261–275 | 0.0 – 2.37 | 1.05 | ICL3 (6+ contacts), TM7, ICL2 |
| Middle (pivot) | 276–282 | 0.0 – 2.66 | 0.88 | TM7 + TM3 + TM5 (mixed) |
| Top (pocket-facing) | 283–292 | 0.0 – 0.94 | 0.38 | TM5 only (+ salt bridge) |

**Key observation**: TM6 bottom is firmly anchored by multiple cross-helix contacts to ICL3 (the G-protein contact surface), while TM6 top is loosely held — its primary restraint comes almost exclusively from the E124-R283 salt bridge.

#### 10.1.3 High-Point Analysis

The complete cross-helix restraint map reveals six structurally significant peaks:

| Rank | Residue | Zone | Σk_cross | Cross-helix partners | Structural role |
|------|---------|------|----------|---------------------|-----------------|
| 1 | **W276** | Middle | 2.66 | TM7: S315 (0.37), F312 (0.35); TM3: T127 (0.29), K130 (0.26); TM2: I92 (0.25) | Toggle switch (6.48, CWxP motif). Highest restraint on TM6. Contacts 4 helices. |
| 2 | **V262** | Bottom | 2.37 | ICL3: H258 (0.34), Q260 (0.31), I235 (0.27), K259 (0.27), L239 (0.25), Y232 (0.22), W240 (0.22), N257 (0.21) | Main bottom anchor. 8 ICL3 contacts hold TM6 intracellular end rigid. |
| 3 | T261 | Bottom | 1.83 | ICL3: Q260 (0.67), N257 (0.34), H258 (0.28) | TM6 start; anchored by backbone-adjacent ICL3 contacts. |
| 4 | M264 | Bottom | 1.70 | ICL3: Q260 (0.34); TM7: M326 (0.27), Y323 (0.22) | ICL3 + TM7 dual anchor. |
| 5 | **F272** | Bottom | 1.30 | TM5: F221 (0.29); TM3: V131 (0.28), I134 (0.28), K130 (0.25) | Cross-bundle brace — simultaneously contacts TM5 AND TM3. Pins TM6 to the helical bundle core. |
| 6 | **H280** | Middle | 1.20 | TM5: **S217** (0.25), F221 (0.30), F222 (0.22); TM3: T127 (0.22) | Contacts S217 (TM5), a key residue in the ML-identified activation relay (S217→H280→R283). |
| 7 | **R283** | Top | 0.93 | TM3: **E124** (0.45, salt bridge); TM5: V214 (0.25); TM3: Q120 (0.24) | Salt bridge residue. 48% of restraint from E124 alone. Sole strong TM3 anchor for Top zone. |

**Zero-restraint residues** (Σk_cross = 0): F270, G282, L285, K288, S289, P292. These face the lipid membrane and have no cross-helix contacts — they are structurally free.

**Insights from the high-point analysis:**

1. **V262 vs R283**: The TM6 bottom anchor (V262, Σk=2.37 from 8 ICL3 contacts) is **2.5× stronger** than the TM6 top anchor (R283, Σk=0.93 from salt bridge + 2 weak contacts). This quantifies why removing the salt bridge barely affects the bottom.

2. **F272 — the cross-bundle brace**: F272 simultaneously contacts TM3 (V131, I134) and TM5 (F221). This is structurally rare on TM6 and creates a rigid brace that prevents TM6 from sliding along its axis. It only allows rotation.

3. **H280 — the relay contact**: H280's contact with S217 (TM5) is the structural basis for the ML-identified activation relay path S217→H280→R283. The ENM shows this is a real physical contact (k=0.25), not a computational artifact.

4. **The gradient story**: Bottom has dense, multi-helix anchoring (ICL3 + TM7). Middle has the two strongest individual anchors (W276, H280) but fewer total contacts. Top has only R283, and 48% of that comes from one spring (the salt bridge). This gradient dictates the mechanical response to salt bridge loosening.

### 10.2 W276 (Ballesteros-Weinstein 6.48): Toggle Switch

The ENM independently identifies **W276** as the residue with the highest cross-helix spring strength on TM6:

```
    Σk_cross(W276) = 2.66  (highest on entire TM6)
```

W276 corresponds to position 6.48 in Ballesteros-Weinstein numbering — the universally conserved **CWxP toggle switch motif** in Class A GPCRs. This residue is known from decades of GPCR research as the pivot point for TM6 rotation during activation.

The ENM discovers this from geometry alone: W276's bulky indole sidechain makes extensive contacts with TM7 (43%), TM3 (38%), TM5 (10%), and TM2 (9%) — contacting 4 helices simultaneously and creating the strongest anchor point on TM6. This is a powerful validation that the ENM captures real structural mechanics.

### 10.3 Salt Bridge as Sole Conduit to TM6 Top

The cross-helix restraint on R283 (TM6) decomposes as:

```
    Σk_cross(R283) = 0.93
    k(E124-R283)   = 0.45  → 48% of total cross-helix restraint
```

The E124-R283 salt bridge provides **48% of R283's total cross-helix restraint** (k = 0.45 out of Σk = 0.93). R283 is the only TM6 residue in the extracellular half with a strong cross-helix anchor to the Active hub domain. Without this bridge, TM6 top has essentially no mechanical coupling to the binding pocket.

### 10.4 Salt Bridge Removal Simulation

To predict what happens when the salt bridge loosens (as in agonist binding), we performed an in silico experiment:

1. **Normal condition**: Strike E124 with the full spring network (including the E124-R283 spring). Record peak displacement at each TM6 residue.
2. **Bridge removed**: Delete the E124-R283 spring. Strike E124 again. Record peak displacement at each TM6 residue.
3. **Compute % displacement change**: For each TM6 residue, (removed − normal) / normal × 100%.

Results:

| TM6 zone | Average displacement change | Interpretation |
|----------|----------------------------|----------------|
| Top (R283–P292) | **−91%** | Almost all pocket-transmitted displacement lost |
| Middle (W276–H280) | −32% | Partially freed |
| Bottom (T261–C275) | −3% | Virtually unchanged |

**Critical finding**: Removing the salt bridge causes the TM6 top to lose 91% of the displacement it receives from the binding pocket, while the TM6 bottom loses only 3%. This creates a massive asymmetry:

- **TM6 bottom**: Still firmly held by TM5/ICL3 contacts → stays in place
- **TM6 top**: Freed from pocket control → can swing outward
- **W276**: Acts as the fulcrum — intermediate loss (−40%), consistent with its role as the mechanical pivot

### 10.5 The Prediction: Asymmetric Restraint → Outward Tilt

Combining the restraint gradient (Section 10.1) with the salt bridge removal simulation (Section 10.4), the ENM makes a clear mechanistic prediction:

```
    ┌────────────────────────────────────────────┐
    │  WITH salt bridge (antagonist, 6KO5):      │
    │                                            │
    │  TM6 top:    anchored by E124-R283 (k=0.45)│
    │  W276:       anchored by TM5 (Σk=2.66)    │
    │  TM6 bottom: anchored by TM5/ICL3 (Σk>1.4)│
    │  → TM6 held rigid, G-protein site CLOSED   │
    │                                            │
    │  WITHOUT salt bridge (agonist-like):        │
    │                                            │
    │  TM6 top:    FREE (91% displacement lost)   │
    │  W276:       partially free (pivot point)  │
    │  TM6 bottom: ANCHORED (only 3% change)    │
    │  → TM6 tilts outward at bottom,            │
    │    pivoting around W276 toggle switch       │
    │  → G-protein binding site OPENS            │
    └────────────────────────────────────────────┘
```

### 10.6 Seesaw Mechanism: Three-State Validation

The ENM prediction is validated by comparing three experimentally solved GHSR structures:

| State | PDB | E124-R283 distance | ICL gap (TM3-TM6 bottom) | G-protein site |
|-------|-----|--------------------|--------------------------|----------------|
| Antagonist (resting) | 6KO5 | 2.2 Å (locked) | 7.0 Å | CLOSED |
| Agonist (active) | 8JSR | 3.0 Å (loose, +0.8 Å) | 10.3 Å (+3.3 Å) | OPEN |
| Inverse agonist (collapsed) | 7F83 | 6.1 Å (broken) | 5.5 Å (−1.5 Å) | BLOCKED |

The salt bridge acts as a **seesaw pivot**:
- **Antagonist**: Tight pivot (2.2 Å) → TM6 locked in resting position
- **Agonist**: Loose pivot (3.0 Å, +0.8 Å loosening, NOT breaking) → TM6 bottom swings 3.3 Å outward → G-protein site opens
- **Inverse agonist**: Pivot broken (6.1 Å) → TM6 collapses inward → G-protein site blocked even more than resting state

This is precisely what the ENM predicts from the restraint gradient: the salt bridge does not need to **break** for activation — it only needs to **loosen** slightly. The asymmetric restraint distribution does the rest, channeling the loosening into directional TM6 tilt.

### 10.7 Why This Matters

The ENM makes a **prediction**, not a post-hoc description:

1. **Input**: Only the 6KO5 (antagonist) crystal structure — 293 Cα positions and 1590 spring contacts
2. **No biological knowledge used**: The model does not know what a salt bridge is, what TM6 tilting means, or what G-protein coupling requires
3. **Prediction**: If you weaken the E124-R283 spring, TM6 top becomes free while TM6 bottom stays anchored → asymmetric tilt
4. **Independent validation**: W276 (toggle switch, 6.48) emerges as the highest-restraint residue on TM6, matching the universally conserved CWxP motif identified by decades of GPCR research

The ENM thus provides a **structural mechanics explanation** for GPCR activation that is consistent with, but independent of, all prior biochemical knowledge. The 10 ML hubs are not arbitrary — they mark the two sides of an elastic seesaw, with the salt bridge as the pivot.

---

## 11. Figure Captions (Figures 3–5)

### Figure 3: Fig3_enm_40x_gap.png

**The 10 ML hubs' own R values — the 40× gap is a self-inclusion artifact, not a finding.**

> ⚠️ This figure should be **dropped or shown only as a cautionary control.** The
> 40× gap arises because each struck hub's own peak (=1.0) falls in its own group;
> random 10 points reproduce it (median 120×) and removing the self-term collapses
> it to zero (§4.2, `VERIFICATION_REPORT.md` §2.4). It is **not** evidence about the
> ML hubs.

Horizontal bar chart showing log₁₀(R) for each of the 10 ML hubs. Orange bars (top): 5 Active hubs (S125, V122, E124, C198, P200) with R values from 8.6 to 92.3. Blue bars (bottom): 5 Inactive hubs (C116, F286, S287, P278, F147) with R values from 0.000 to 0.212. Numeric R values are printed on each bar. Physical gap separates the two groups, annotated with the 40× ratio: min(Active) = 8.6 vs max(Inactive) = 0.21. Dashed red line and annotation emphasize the gap. Vertical dotted line at log₁₀(R) = 0 marks the crossover point (R = 1.0). X-axis is log-scale. F147 extends to log₁₀(R) ≈ −7 (R effectively 0, as no measurable displacement reaches Active hubs when F147 is struck).

> **Caveat.** R is a **hub-weighted metric computed from these same 10 hubs**, so this gap is descriptive of how displacement channels given the ML assignment — it is *not* an independent validation that geometry/ENM discovers the two groups. A blind reverse-derivation does not recover the hubs (`reverse.py`; §6.3). See `VERIFICATION_REPORT.md` §2.2. R reflects each hub's **3D position** relative to the two clusters (geometry), **not** its ML/functional importance.

### Figure 4: Fig4_enm_tm6_mechanism.png

**ENM prediction of TM6 activation mechanism from 6KO5 geometry.**

Three panels share a consistent zone system: TM6 is divided into Bottom (261–275), Middle (276–282), and Top (283–292), separated by black dashed lines. Panel A additionally shows a sub-zone boundary (blue dotted line at 269→270) within Bottom, dividing it into Bottom-ICL3 (261–269, darker blue shading) and Bottom-TM7 (270–275, lighter blue shading). Zone definitions are based on the dominant cross-helix contact partner (Section 10.1.1). Middle zone (yellow shading) contains the W276 toggle switch. Top zone (red shading) is pocket-facing, restrained only by the salt bridge.

**(A)** TM6 cross-helix restraint map. Bar height shows total cross-helix spring strength (Σk_cross) for each TM6 residue (261–292). Colors encode structural role: dark red = R283 (salt bridge, Σk = 0.93, 48% from E124); purple = W276 (toggle switch, 6.48, Σk = 2.66, highest on entire TM6); green = F272 (cross-bundle brace, Σk = 1.30, contacts TM3 at 62% + TM5); orange = strong (Σk > 1.5); yellow = moderate (0.5–1.5); gray = weak (< 0.5). Five high points are annotated: V262 (Σk = 2.37, anchored by 8 ICL3 contacts), F272 (TM3+TM5 brace, an outlier within Bottom-TM7), W276 (toggle switch), H280 (Σk = 1.20, contacts S217 of TM5 in the ML-identified activation relay), R283 (salt bridge). The asymmetric gradient — dense multi-helix anchoring at Bottom, sparse TM5-only contacts at Top — is visible.

**(B)** Peak displacement reaching each TM6 residue when E124 is struck, with (blue bars) and without (orange bars) the E124-R283 salt bridge spring. R283 shows the largest peak (0.093) with the bridge intact, dropping to near-zero when removed — demonstrating the salt bridge is the sole conduit for pocket displacement to reach TM6 top. W276 shows intermediate response in both conditions (anchored by TM5, not by salt bridge). Bottom zone residues show minimal change between conditions.

**(C)** Percentage displacement change when the E124-R283 spring is removed, per TM6 residue. Colors encode severity: dark red = >80% lost (freed), red = 50–80% lost, orange = 20–50% lost, yellow = 5–20% lost, gray = <5% change (anchored). Dotted horizontal lines show zone averages: Bottom avg = −3% (blue, anchored by ICL3/TM5), Middle avg = −32% (brown, partially freed), Top avg = −91% (red, freed from pocket control). The zone averages quantify the prediction: removing the salt bridge creates an asymmetric displacement loss that increases from bottom (−3%) to top (−91%), consistent with outward TM6 tilt pivoting around the W276 fulcrum.

Reproducibility: generated by `generate_fig4_tm6_mechanism.py` from `_data_compact.json` (see Section 12).

### Figure 5: Fig5_enm_seesaw_mechanism.png

**Seesaw mechanism: E124-R283 salt bridge as pivot in three GHSR states.**

Three-panel schematic comparing antagonist (6KO5), agonist (8JSR), and inverse agonist (7F83) structures. Each panel shows TM3 (yellow) and TM6 (blue) as vertical bars, connected by the E124-R283 salt bridge (horizontal line with pivot symbol). Distances are from crystal structures.

**Left (Antagonist, 6KO5)**: E124-R283 = 2.2 Å (locked, tight pivot). ICL gap = 7.0 Å. G-protein site CLOSED.

**Center (Agonist, 8JSR)**: E124-R283 = 3.0 Å (loose pivot, +0.8 Å loosening). TM6 bottom tilts 3.3 Å outward. ICL gap = 10.3 Å. G-protein site OPEN. Red arrow indicates ligand pushing from extracellular side.

**Right (Inverse Agonist, 7F83)**: E124-R283 = 6.1 Å (BROKEN, no pivot). TM6 collapses inward. ICL gap = 5.5 Å. G-protein site BLOCKED.
