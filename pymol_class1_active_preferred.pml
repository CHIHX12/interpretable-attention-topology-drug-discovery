# PyMOL Visualization: Class 1 (Active) Preferred Residues
# These residues receive higher attention from ACTIVE compounds
# Color gradient: Red (highest) → Orange → Yellow (moderate)

# Load structure
load datasets/GPCR_resarch/GSHR_PDB/8JSR_R.pdb
hide everything
show cartoon, chain R
color gray80, chain R

# ===== Tier 1: Strongest Active Preference (RED) =====
select class1_tier1, chain R and resi 124+125+122+198+200+299+232+126+230+160
color red, class1_tier1
show spheres, class1_tier1
set sphere_scale, 0.8, class1_tier1

# Top residues in Tier 1:
# E123 (PDB 124): Diff=0.244, p=4.74e-08 
# S124 (PDB 125): Diff=0.216, p=1.09e-09 
# V121 (PDB 122): Diff=0.080, p=1.59e-03 
# C197 (PDB 198): Diff=0.065, p=1.63e-15 
# P199 (PDB 200): Diff=0.049, p=1.10e-02 
# Q298 (PDB 299): Diff=0.043, p=2.49e-09 
# Y231 (PDB 232): Diff=0.028, p=1.40e-14 
# C125 (PDB 126): Diff=0.028, p=1.02e-21 ***IN POCKET***
# V229 (PDB 230): Diff=0.024, p=4.82e-25 
# V159 (PDB 160): Diff=0.000, p=1.16e-60 

# ===== Tier 2: Moderate Active Preference (ORANGE) =====
select class1_tier2, chain R and resi 205+44+206+63+61+189+188+209+142+139
color orange, class1_tier2
show spheres, class1_tier2
set sphere_scale, 0.6, class1_tier2

# ===== Tier 3: Lower Active Preference (YELLOW) =====
select class1_tier3, chain R and resi 165+130+334+132+254+296+195+203+52+143
color yellow, class1_tier3
show spheres, class1_tier3
set sphere_scale, 0.4, class1_tier3

# ===== Experimental Binding Pocket (REFERENCE) =====
select exp_pocket, chain R and resi 37+40+99+102+103+106+110+123+126+127+278+282+285+286+289
color cyan, exp_pocket
show sticks, exp_pocket
set stick_radius, 0.15, exp_pocket

# ===== Overlap with Experimental Pocket =====
select overlap_class1, (class1_tier1 or class1_tier2 or class1_tier3) and exp_pocket
color magenta, overlap_class1
set sphere_scale, 1.0, overlap_class1

# ===== View Settings =====
bg_color white
set sphere_transparency, 0.2
set stick_transparency, 0.3
zoom chain R
center class1_tier1

# ===== Labels for Top 5 Residues =====
label chain R and resi 124 and name CA, "E124"
label chain R and resi 125 and name CA, "S125"
label chain R and resi 122 and name CA, "V122"
label chain R and resi 198 and name CA, "C198"
label chain R and resi 200 and name CA, "P200"
set label_size, 20
set label_color, black
