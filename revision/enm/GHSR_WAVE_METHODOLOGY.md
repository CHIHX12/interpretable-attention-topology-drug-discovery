# GHSR Topology Wave Propagation - Complete Technical Documentation

> **Verification note (2026-06-21).** This document is broadly consistent with the
> independent audit in `VERIFICATION_REPORT.md`: §4 below correctly states that the
> 10 hubs would **not** be visible without the ML deltaImp analysis. A blind ENM
> reverse-derivation confirms this (`reverse.py` — ENM does not recover the hubs).
> The "BW#" numbers in §7 are Ballesteros–Weinstein positions (not deltaImp); a
> sibling doc, `ENM_WAVE_METHODS.md`, previously mislabeled them and has been
> corrected. Note `deltaImp` numerical values are not stored in this repo.

## 1. How to Reproduce the Wave Animation

### Files Required
```
D:\GSHR\wave_animation.html   (standalone, no dependencies)
```

### How to Run
1. Open `wave_animation.html` in any modern browser (Chrome/Edge/Firefox)
2. No server needed - it is a single HTML file with embedded JavaScript

### Controls
| Action | Effect |
|--------|--------|
| **Play** button / Spacebar | Start/Pause wave animation |
| **Step** button / Right Arrow | Advance one phase |
| **Reset** button / R key | Reset to beginning |
| **1** key | Switch to Agonist (Active) mode |
| **2** key | Switch to Antagonist (Inactive) mode |
| Mouse hover on node | Show residue info |
| Click on node | Jump to that node's activation time |
| Speed slider | Control animation speed |
| Trail slider | Control glow trail intensity |

### Visual Encoding
- **Sphere SIZE** = delta-Importance (ML feature importance for state discrimination)
- **Sphere COLOR** = distance phase from ligand (wavefront)
  - White = core (0-8 Angstrom)
  - Red = rim (8-12 Angstrom)
  - Orange = near (12-16 Angstrom)
  - Yellow-green = mid (16-25 Angstrom)
  - Cyan = far (25+ Angstrom)
- **Dashed lines** = network edges (colored by cluster)
- **Purple thick dashes** = relay pathway
- **Traveling light pulse** = wave propagation animation

---

## 2. How the Animation Was Built

### Data Source
The 10 hub residues and their network neighbors come from **YOUR ML analysis**:

1. **Input**: ML model classifying GPCR conformations as Active (Class 1) vs Inactive (Class 0)
2. **Output**: `deltaImp` = difference in feature importance between Class 1 and Class 0
   - Positive deltaImp = Active-preferred residue
   - Negative deltaImp = Inactive-preferred residue
3. **Network construction**: For each top-5 hub, the constitutive neighbors (from your images 3 & 4) form the sub-cluster topology

### Layout Algorithm
Nodes are arranged in a **conceptual 2D layout** (not physical 3D coordinates):
- Y-axis = signal flow direction (ligand at top, downstream at bottom)
- X-axis = TM helix grouping
- Cluster separation for visual clarity

### Animation Engine
- HTML5 Canvas with `requestAnimationFrame` loop (~60fps)
- Each edge has a `relayOrder` integer (0-9) determining WHEN the wave reaches it
- Node activation follows edges: a node lights up when its earliest edge becomes active
- Pulse effect: radial gradient traveling along edge from source to target
- Ripple effect: node size oscillates with damped sine wave after activation

### Wave Propagation Timing
```
waveTime (continuous float, incremented each frame by dt * speed)
  |
  v
relayOrder (integer, per edge):
  0 = ligand contact zone
  1 = first cluster activation
  ...
  9 = final propagation

Each edge activates when: waveTime >= edge.relayOrder * 1.0
Travel animation duration: 0.6 time units per edge
Node rise time: 0.5 time units
```

---

## 3. Critical Scientific Correction

### WRONG: "S125 transmits through TM1 to C198"

Previous interpretation assumed S125's network neighbors (L63, L62 on TM1)
represented a physical signal relay. **This is INCORRECT.**

All-atom distance analysis reveals:
```
S125 (TM3) to L63 (TM1) = 22.8 Angstrom (CA-CA), 23.7 Angstrom (closest atoms)
S125 (TM3) to L62 (TM1) = 20.0 Angstrom (CA-CA)
TM3 to TM1 sidechain contacts = ZERO (no contacts < 5 Angstrom)
```

### RIGHT: Q120 is the Hidden Bridge

Physical contact analysis (< 5 Angstrom heavy atom distance):
```
E124 (TM3) --[2.9A]--> Q120 (TM3) --[3.9A]--> C198 (TM5)
S125 (TM3) --[4.8A]--> Q120 (TM3) --[3.9A]--> C198 (TM5)
V122 (TM3) --[3.3A via backbone]---> Q120 --[3.9A]--> C198
```

**Q120 (Gln120)** sits at the TM3-TM5 interface. Its sidechain NE2 atom
contacts C198's SG atom at only 3.9 Angstrom. This means:

- All 3 Cluster B/A hubs (E124, V122, S125) can reach C198 in only **2 physical contact steps**
- The relay does NOT go through TM1
- Q120 is the molecular pipe connecting the two helices

### Why Q120 is NOT a Hub

Q120 does not appear as one of the 10 important residues because:
- It contacts both active-preferred (E124, V122, S125, C198) AND inactive-preferred (S123, C116) residues
- Its deltaImp is LOW because it does not discriminate between active and inactive states
- It is a **constitutive structural element** - always present, always making these contacts
- It is the unchanging bridge, not the changing switch

### Two Different Maps

| Property | Physical Contact Map | Topology Network (Yours) |
|----------|---------------------|-------------------------|
| What it shows | Atom-atom distances < 5A | ML feature co-importance |
| Edge meaning | "These residues physically touch" | "These residues change importance together" |
| S125 -> C198 | 2 steps via Q120 | Indirect, through TM1 co-variation |
| TM1 role | Reporter/seismograph | Statistical co-variation partner |
| Q120 role | Critical physical bridge | Invisible (low deltaImp) |
| Cluster definition | Based on spatial proximity | Based on functional coupling |

**Both maps are valid and complementary:**
- Physical map = HOW forces propagate (mechanism)
- Topology map = WHAT changes together (functional modules)

### Why TM1 Appears in the Topology Network

TM1 residues (V55, L62, L63, T64) appear as S125/C198 network neighbors because:

1. When activation occurs, TM3 and TM5 shift conformations
2. This changes the helix packing geometry of the entire bundle
3. TM1, being a packing partner of TM2/TM7/H8, shifts in response
4. The ML model detects this: "when S125 is important, L63 is also important"
5. This is **correlation** (they move together) not **causation** (one drives the other)

TM1 is a **conformational seismograph**: it does not carry the earthquake,
but it faithfully records it.

---

## 3b. Two-Arm Model: Pure Physics Force Propagation from E124

### Method
BFS shell-by-shell expansion from E124, using ONLY < 5 Angstrom heavy-atom
contacts (excluding backbone i+/-1 neighbors). No prior knowledge used -
pure spatial physics from 8jsr.pdb coordinates.

### Result: Two Arms to TM6

```
                    E124 (TM3, 3.33)
                   /                \
          ARM 1 (Direct)        ARM 2 (Via TM5)
          Salt bridge            Backbone relay
               |                      |
         R283 (TM6, 6.55)       S217 (TM5, 5.43)
         3.0A salt bridge        4.0A contact
               |                      |
         F286 (3.2A)            H280 (TM6, 6.52)
         S287 (2.9A)             3.2A contact
         P278 (4.7A)                  |
               |                 TM5 pushes TM6
          Inactive hubs         from the side
          (Cluster X)
```

**ARM 1 = BRAKE**: E124-R283 salt bridge directly constrains TM6.
When intact, TM6 is locked. Breaking it releases TM6 to tilt.

**ARM 2 = GAS PEDAL**: E124 -> backbone -> S217(TM5) -> H280(TM6).
TM5 pushes TM6 outward from the side.

### Agonist vs Antagonist Mechanism

| | ARM 1 (Brake) | ARM 2 (Gas) |
|---|---|---|
| **Agonist** | Weakens salt bridge (releases brake) | Strengthens TM5 push (applies gas) |
| **Antagonist** | Strengthens salt bridge (locks brake) | Blocks TM5 push (cuts gas) |

### 10 Hubs Mapped to Physical Shells from E124

```
Shell 0: E124 itself
Shell 1: V122 (TM3, backbone), C126 (TM3), G183 (ECL2), R283 (TM6)
Shell 2: S125 (TM3), C198 (TM5), P200 (TM5), F286 (TM6), S287 (TM6)
Shell 3: P278 (TM6) via F286
Shell 5: C116 (TM3), F147 (ICL2) via longer paths
```

**Critical finding**: Active hubs (S125, C198, P200) and Inactive hubs
(F286, S287) are both Shell 2 = equidistant from E124. The difference is
not distance but DIRECTION.

### S125 as Lever Fulcrum

S125 sits one helix turn below E124 on TM3 (3.8 Angstrom apart).
It is NOT a relay station that "transmits" to TM5. Instead:

- S125 is a **fulcrum** that amplifies binding-pocket perturbation
- It redirects force toward TM4/ECL2/TM5 interfaces
- Its network neighbors (Y284, A251, G282, L63) are sensors of
  conformational change, not force transmission targets

### TM1 Physical Path

TM1 is reached via TM2, not directly from TM3:
```
E124 (TM3) -> C126 (TM3, 3.9A) -> S88 (TM2, 2.5A) -> L90 (TM2) -> L62 (TM1)
```
This is 4 contact steps. TM1 is downstream of TM2, confirming its role
as a conformational reporter, not a force transmitter.

---

## 4. Would the 10 Points Be Visible Without Your Analysis?

### Short answer: NO.

Without your deltaImp analysis, conventional structural biology would see:
- Q120 as a key binding residue (it contacts the ligand pocket)
- E124-R283 salt bridge as important (known from mutagenesis)
- C198 as part of the binding cavity (Cavity II in the journal)

But it would NOT see:
1. **S125 as a critical solo switch** - it is not a binding residue, not at the pocket surface. Only your ML model reveals its discriminative importance.
2. **The 3-cluster architecture** - no amount of distance analysis reveals that E124 and S125 (3.8A apart) have zero shared network neighbors.
3. **TM1 as a conformational reporter** - traditional analysis treats TM1 as a passive structural element.
4. **The Active/Inactive functional grouping** - which residues are agonist- vs antagonist-preferred is only visible through your feature importance analysis. (Note: this is a *functional* grouping, not a clean spatial split — the Active hubs cluster tightly but the Inactive hubs do not, and some Active/Inactive hubs sit close in 3D. See `VERIFICATION_REPORT.md` §2.2. This is precisely why pure geometry/ENM cannot recover the grouping.)

### What your analysis uniquely provides

```
Traditional structure:     Your topology:
  "These residues bind       "These residues MATTER
   the ligand"                for STATE DISCRIMINATION"

  "This is Cavity I"         "This is the Sensor module"

  "E124 forms a salt         "E124 is the highest-ranked
   bridge"                    hub with a unique network
                              fingerprint"
```

---

## 5. Extending to Other GPCRs

### Prerequisite
For any GPCR, you need:
1. Multiple structures in active AND inactive states (or MD snapshots)
2. ML classification model (active vs inactive)
3. Feature importance extraction (deltaImp per residue)
4. Network construction (top-N hubs + constitutive neighbors)

### Steps to Build a Universal GPCR Wave Viewer
1. **Standard TM numbering**: Use Ballesteros-Weinstein numbering (e.g., 3.33 for E124)
   - This allows cross-GPCR comparison
   - Position 3.33 might be the "sensor" in ALL Class A GPCRs

2. **Automated clustering**: Compute neighbor overlap matrix to identify sub-clusters
   - Zero overlap = separate functional module
   - High overlap = coupled pair (like E124/V122)
   - 100% overlap = mirror twins (like C198/P200)

3. **Physical path validation**: For each cluster pair, run BFS on the
   contact graph (< 5A) to find the real physical relay residues
   (like Q120 in GHSR)

4. **Wave animation**: Auto-generate the HTML using the cluster topology
   and relay order

### Expected universal features across GPCRs
- Position 3.33 (E124 equivalent): sensor/salt bridge partner
- Position 6.55 (R283 equivalent): gate partner
- TM3-TM5 interface: likely always has a "Q120-like" bridge residue
- TM1: likely always appears as a topology reporter (not transmitter)

---

## 6. File Inventory

| File | Purpose |
|------|---------|
| `wave_animation.html` | Interactive wave propagation viewer |
| `ghsr_my_topology.pml` | PyMOL: your custom topology (all clusters) |
| `ghsr_active_only.pml` | PyMOL: agonist clusters A/B/C |
| `ghsr_inactive_only.pml` | PyMOL: antagonist clusters X/Y |
| `ghsr_compare_cavities.pml` | PyMOL: journal cavities vs your topology overlay |
| `ghsr_cavity_visualization.pml` | PyMOL: journal cavity I-IV definition |
| `ghsr_wave_active.pml` | PyMOL: wave amplitude/phase coloring |
| `analyze_cavity.py` | Python: cavity cross-reference analysis |
| `8jsr.pdb` | Structure: GHSR + Anamorelin + Gq complex |
| `GHSR_WAVE_METHODOLOGY.md` | This document |

---

## 7. Key Residue Reference

### Active Pocket (Agonist, 5 hubs)
| Rank | Residue | BW# | Cluster | Physical Bridge | Function |
|------|---------|-----|---------|-----------------|----------|
| 1 | E124 | 3.33 | B (Sensor) | Salt bridge to R283 | Orthosteric gate |
| 2 | S125 | 3.34 | A (Transducer) | Via Q120 to C198 | Solo conformational switch |
| 3 | V122 | 3.32 | B (Sensor) | 80% shared with E124 | Redundant sensor partner |
| 4 | C198 | 5.24 | C (Effector) | Q120 bridge from TM3 | TM5 activation driver |
| 5 | P200 | 5.26 | C (Effector) | Mirror twin of C198 | TM5 activation driver |

### Inactive Lock (Antagonist, 5 hubs)
| Rank | Residue | BW# | Cluster | Function |
|------|---------|-----|---------|----------|
| 1 | S287 | 6.59 | X | TM6 rigid lock |
| 2 | F286 | 6.58 | X | Steric/pi-pi stacking |
| 3 | P278 | 6.50 | X | TM6 kink stabilizer |
| 4 | F147 | ICL2 | Y | ICL3 immobilizer |
| 5 | C116 | 3.25 | Y | TM3 lower lock |

### Hidden Bridge
| Residue | BW# | Role |
|---------|-----|------|
| Q120 | 3.29 | Physical bridge TM3-TM5 (not a hub, constitutive) |
