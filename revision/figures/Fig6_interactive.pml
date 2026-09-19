# Fig. 6 interactive session
#   1. open it:           pymol Fig6_interactive.pml
#   2. turn and zoom with the mouse
#   3. save the view:     snap              (1200 dpi PNG of what you see)
#                         snap my_name      (choose the file name)
#   switch residue set:   ec50 | ic50 | both
#   switch structure:     ago (8JSR) | ant (6KO5)
@Fig6_setup.pml

# labels beside their residues: the column layout of the automatic panels
# is tied to one fixed camera and would drift once you rotate
set label_position, (2.5, 2.5, 3.0)
set orthoscopic, 0

run Fig6_commands.py

ago
both
orient 8jsr and chain R and resi 100-320
turn x, -20
