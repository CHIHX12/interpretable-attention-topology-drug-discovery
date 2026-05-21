# PyMOL Visualization: Class 0 (Inactive) Preferred Residues
# These residues receive higher attention from INACTIVE compounds
# Color gradient: Red (highest) → Orange → Yellow (moderate)

# Load structure
load datasets/GPCR_resarch/GSHR_PDB/8JSR_R.pdb
hide everything
show cartoon, chain R
color gray80, chain R

# ===== Tier 1: Strongest Inactive Preference (RED) =====
select class0_tier1, chain R and resi 287+286+278+147+116+302+41+38+39+292
color red, class0_tier1
show spheres, class0_tier1
set sphere_scale, 0.8, class0_tier1

# Top residues in Tier 1:
# S286 (PDB 287): Diff=-0.773, p=1.60e-19 
# F285 (PDB 286): Diff=-0.685, p=3.74e-23 ***IN POCKET***
# P277 (PDB 278): Diff=-0.640, p=1.51e-15 ***IN POCKET***
# F146 (PDB 147): Diff=-0.640, p=3.47e-36 
# C115 (PDB 116): Diff=-0.623, p=5.82e-29 
# Q301 (PDB 302): Diff=-0.594, p=2.02e-12 
# P40 (PDB 41): Diff=-0.550, p=1.52e-22 
# F37 (PDB 38): Diff=-0.547, p=2.67e-23 
# P38 (PDB 39): Diff=-0.531, p=1.35e-24 
# P291 (PDB 292): Diff=-0.523, p=9.81e-15 

# ===== Tier 2: Moderate Inactive Preference (ORANGE) =====
select class0_tier2, chain R and resi 226+37+285+274+117+322+303+288+225+320
color orange, class0_tier2
show spheres, class0_tier2
set sphere_scale, 0.6, class0_tier2

# ===== Tier 3: Lower Inactive Preference (YELLOW) =====
select class0_tier3, chain R and resi 224+275+118+291+120+277+305+119+103+105
color yellow, class0_tier3
show spheres, class0_tier3
set sphere_scale, 0.4, class0_tier3

# ===== Experimental Binding Pocket (REFERENCE) =====
select exp_pocket, chain R and resi 37+40+99+102+103+106+110+123+126+127+278+282+285+286+289
color cyan, exp_pocket
show sticks, exp_pocket
set stick_radius, 0.15, exp_pocket

# ===== Overlap with Experimental Pocket =====
select overlap_class0, (class0_tier1 or class0_tier2 or class0_tier3) and exp_pocket
color magenta, overlap_class0
set sphere_scale, 1.0, overlap_class0

# ===== View Settings =====
bg_color white
set sphere_transparency, 0.2
set stick_transparency, 0.3
zoom chain R
center class0_tier1

# ===== Labels for Top 5 Residues =====
label chain R and resi 287 and name CA, "S287"
label chain R and resi 286 and name CA, "F286"
label chain R and resi 278 and name CA, "P278"
label chain R and resi 147 and name CA, "F147"
label chain R and resi 116 and name CA, "C116"
set label_size, 20
set label_color, black
