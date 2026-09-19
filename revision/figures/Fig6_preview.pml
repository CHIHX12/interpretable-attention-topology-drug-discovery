# Fig. 6 preview (small, ray traced)
# self-contained: loads and styles the scene first
@Fig6_setup.pml

# --- Fig6a_agonist_both ---
disable 6ko5
enable 8jsr
show sticks, (8jsr and chain R and resi 124+125+122+200+198+287+286+278+116+147) and sidechain
show spheres, (8jsr and chain R and resi 124+125+122+200+198+287+286+278+116+147) and name CA
disable 8jsr_EC50_d*
disable 8jsr_IC50_d*
label 8jsr and chain R and resi 124 and name CA, ""
label 8jsr and chain R and resi 125 and name CA, ""
label 8jsr and chain R and resi 122 and name CA, ""
label 8jsr and chain R and resi 200 and name CA, ""
label 8jsr and chain R and resi 198 and name CA, ""
label 8jsr and chain R and resi 287 and name CA, ""
label 8jsr and chain R and resi 286 and name CA, ""
label 8jsr and chain R and resi 278 and name CA, ""
label 8jsr and chain R and resi 116 and name CA, ""
label 8jsr and chain R and resi 147 and name CA, ""
show labels, 8jsr_hel_*
show sticks, 8jsr and resn UYI
set_view (-0.2635,-0.5030,0.8232,-0.9588,0.0430,-0.2807,0.1058,-0.8632,-0.4936,0.0000,0.0000,-284.4907,132.8057,130.9379,113.5213,230.5993,338.3821,-20.0000)
png Fig6a_agonist_both_preview.png, width=1500, height=1500, dpi=300, ray=1

# --- Fig6b_agonist_EC50 ---
disable 6ko5
enable 8jsr
hide sticks, 8jsr and chain R and resi 287+286+278+116+147
hide spheres, 8jsr and chain R and resi 287+286+278+116+147
disable 8jsr_IC50_d*
label 8jsr and chain R and resi 287 and name CA, ""
label 8jsr and chain R and resi 286 and name CA, ""
label 8jsr and chain R and resi 278 and name CA, ""
label 8jsr and chain R and resi 116 and name CA, ""
label 8jsr and chain R and resi 147 and name CA, ""
label 8jsr and chain R and resi 124 and name CA, "E124  3.4 A"
label 8jsr and chain R and resi 125 and name CA, "S125  6.9 A"
label 8jsr and chain R and resi 122 and name CA, "V122  6.9 A"
label 8jsr and chain R and resi 200 and name CA, "P200  4.7 A"
label 8jsr and chain R and resi 198 and name CA, "C198  5.6 A"
show sticks, (8jsr and chain R and resi 124+125+122+200+198) and sidechain
show spheres, (8jsr and chain R and resi 124+125+122+200+198) and name CA
show labels, (8jsr and chain R and resi 124+125+122+200+198) and name CA
enable 8jsr_EC50_d*
show labels, 8jsr_hel_*
show sticks, 8jsr and resn UYI
set_view (-0.2635,-0.5030,0.8232,-0.9588,0.0430,-0.2807,0.1058,-0.8632,-0.4936,0.0000,0.0000,-284.4907,132.8057,130.9379,113.5213,230.5993,338.3821,-20.0000)
png Fig6b_agonist_EC50_preview.png, width=1500, height=1500, dpi=300, ray=1

# --- Fig6c_agonist_IC50 ---
disable 6ko5
enable 8jsr
hide sticks, 8jsr and chain R and resi 124+125+122+200+198
hide spheres, 8jsr and chain R and resi 124+125+122+200+198
disable 8jsr_EC50_d*
label 8jsr and chain R and resi 124 and name CA, ""
label 8jsr and chain R and resi 125 and name CA, ""
label 8jsr and chain R and resi 122 and name CA, ""
label 8jsr and chain R and resi 200 and name CA, ""
label 8jsr and chain R and resi 198 and name CA, ""
label 8jsr and chain R and resi 287 and name CA, "S287  5.5 A"
label 8jsr and chain R and resi 286 and name CA, "F286  3.5 A"
label 8jsr and chain R and resi 278 and name CA, "P278  8.8 A"
label 8jsr and chain R and resi 116 and name CA, "C116  7.8 A"
label 8jsr and chain R and resi 147 and name CA, "F147  35.0 A"
show sticks, (8jsr and chain R and resi 287+286+278+116+147) and sidechain
show spheres, (8jsr and chain R and resi 287+286+278+116+147) and name CA
show labels, (8jsr and chain R and resi 287+286+278+116+147) and name CA
enable 8jsr_IC50_d*
show labels, 8jsr_hel_*
show sticks, 8jsr and resn UYI
set_view (-0.2635,-0.5030,0.8232,-0.9588,0.0430,-0.2807,0.1058,-0.8632,-0.4936,0.0000,0.0000,-284.4907,132.8057,130.9379,113.5213,230.5993,338.3821,-20.0000)
png Fig6c_agonist_IC50_preview.png, width=1500, height=1500, dpi=300, ray=1

# --- Fig6d_antagonist_both ---
disable 8jsr
enable 6ko5
show sticks, (6ko5 and chain A and resi 124+125+122+200+198+287+286+278+116+147) and sidechain
show spheres, (6ko5 and chain A and resi 124+125+122+200+198+287+286+278+116+147) and name CA
disable 6ko5_EC50_d*
disable 6ko5_IC50_d*
label 6ko5 and chain A and resi 124 and name CA, ""
label 6ko5 and chain A and resi 125 and name CA, ""
label 6ko5 and chain A and resi 122 and name CA, ""
label 6ko5 and chain A and resi 200 and name CA, ""
label 6ko5 and chain A and resi 198 and name CA, ""
label 6ko5 and chain A and resi 287 and name CA, ""
label 6ko5 and chain A and resi 286 and name CA, ""
label 6ko5 and chain A and resi 278 and name CA, ""
label 6ko5 and chain A and resi 116 and name CA, ""
label 6ko5 and chain A and resi 147 and name CA, ""
show labels, 6ko5_hel_*
show sticks, 6ko5 and resn 8QX
set_view (0.1606,0.9416,0.2961,-0.9870,0.1518,0.0527,0.0047,-0.3007,0.9537,0.0000,0.0000,-301.9714,-4.2867,-21.5618,16.0401,231.9364,372.0063,-20.0000)
png Fig6d_antagonist_both_preview.png, width=1500, height=1500, dpi=300, ray=1

print 'Fig. 6 preview (small, ray traced) done'
