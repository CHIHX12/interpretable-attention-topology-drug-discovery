#!/usr/bin/env python3
"""
make_fig6_pymol.py — PyMOL scripts for the structural mapping figure (Fig. 6).

Writes three ASCII scripts into the figure folder, and a copy under revision/
whose path contains no non-ASCII characters, for PyMOL on Windows:

    Fig6_setup.pml         loads and styles the scene, no rendering
    Fig6_preview.pml       setup + four small ray-traced panels
    Fig6_render_final.pml  setup + four 1200-dpi panels

Design points, each of which follows from what the figure has to show:

  * the camera is not left to `orient`. The transmembrane axis is computed here
    from the Ca coordinates and written into an explicit set_view matrix, so the
    receptor stands upright with the pocket facing the viewer, which is the
    conventional class A GPCR view.
  * each panel carries one class direction only, so ten labels never compete for
    the same space.
  * a label gives the residue and its minimum heavy-atom distance to the
    co-crystallised ligand, and the dashed distance objects are drawn without
    their own numeric labels, so each number appears once.
  * distance objects are drawn only within 10 A of the ligand. Phe147 (35 A,
    ICL2) is shown and labelled but not connected, because the line would cross
    the whole panel.

Distances are computed here from the coordinates, so the figure and Table 1
cannot drift apart. Headless PyMOL has no OpenGL context, so every png command
ray-traces; ray=0 would write a blank image.

Usage:
    python revision/make_fig6_pymol.py
"""
from pathlib import Path

import numpy as np

OUT_DIR = Path("文章草稿/圖片_v22")
ASCII_DIR = Path("revision")
STRUCTURES = [("8jsr", "/home/cycheng/GSHR/8jsr.pdb", "R", "UYI", "agonist-bound (anamorelin)"),
              ("6ko5", "/home/cycheng/GSHR/6ko5.pdb", "A", "8QX", "antagonist-bound")]
EC50 = [124, 125, 122, 200, 198]
# one representative residue per segment, used only to place a helix label
HELIX_LABELS = {"TM3": 130, "TM5": 210, "TM6": 281, "TM7": 310, "ICL2": 145}
IC50 = [287, 286, 278, 116, 147]
DISTANCE_CUTOFF = 10.0
AA3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
          "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
          "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
          "TYR": "Y", "VAL": "V"}


def parse(path, chain, ligand):
    prot, lig, resname, ca = {}, [], {}, {}
    for line in open(path):
        tag = line[:6]
        if tag not in ("ATOM  ", "HETATM") or line[16] not in " A":
            continue
        name = line[12:16].strip()
        if not name or name[0] == "H":
            continue
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        if tag == "HETATM" and line[17:20].strip() == ligand:
            lig.append((name, xyz))
        elif tag == "ATOM  " and line[21] == chain:
            r = int(line[22:26])
            prot.setdefault(r, []).append((name, xyz))
            resname[r] = line[17:20].strip()
            if name == "CA":
                ca[r] = xyz
    return prot, lig, resname, ca


def closest_pair(prot, lig, resnum):
    atoms = prot.get(resnum)
    if not atoms or not lig:
        return None
    return min(((float(np.linalg.norm(a[1] - b[1])), a[0], b[0]) for a in atoms for b in lig),
               key=lambda t: t[0])


def upright_view(ca, lig, first=37, last=338):
    """Camera matrix with the transmembrane axis vertical and the pocket in front."""
    pts = np.array([v for r, v in ca.items() if first <= r <= last])
    centre = pts.mean(0)
    axis = np.linalg.eigh(np.cov((pts - centre).T))[1][:, -1]     # bundle axis
    ligand_centre = np.mean([x for _, x in lig], axis=0) if lig else centre + axis
    if np.dot(ligand_centre - centre, axis) < 0:                   # ligand side is up
        axis = -axis
    up = axis / np.linalg.norm(axis)
    to_pocket = ligand_centre - centre
    forward = to_pocket - np.dot(to_pocket, up) * up               # flatten into the plane
    if np.linalg.norm(forward) < 1e-6:
        forward = np.cross(up, np.array([1.0, 0.0, 0.0]))
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    rot = np.vstack([right, up, forward])                          # camera x, y, z axes
    radius = float(np.linalg.norm(pts - centre, axis=1).max())
    proj = np.array([rot @ (v - centre) for v in pts])
    half_w, half_h = float(np.abs(proj[:, 0]).max()), float(np.abs(proj[:, 1]).max())
    column_x = half_w + 5.0                       # label column just outside the receptor
    label_room = 26.0                             # space for the widest label text
    # PyMOL's default field of view is 20 degrees, so the visible half-size at
    # distance d is d*tan(10 deg) = 0.176 d; leave room for the columns as well.
    need = max(half_h * 1.05, column_x + label_room)
    dist = need / 0.176
    unit = LABEL_UNIT_PER_HALFWIDTH * (dist * 0.176)      # Angstrom per label unit
    return rot, centre, dist, radius, column_x, unit


def projected_extent(ca, rot, centre, first=37, last=338):
    """Half-width and half-height of the receptor as it appears in the camera frame."""
    pts = np.array([rot @ (v - centre) for r, v in ca.items() if first <= r <= last])
    return float(np.abs(pts[:, 0]).max()), float(np.abs(pts[:, 1]).max())


# PyMOL's label_position is not in Angstroms: it is a screen-space offset whose
# unit is a fixed fraction of the viewport. Calibrated by rendering one label at
# two offsets in this scene: one unit = 0.0114 of the scene half-width.
LABEL_UNIT_PER_HALFWIDTH = 0.0114


def layout_labels(anchors, rot, centre, column_x, side, unit_in_angstrom, min_gap=7.0):
    """Place labels in a column beside the receptor and return model-space offsets.

    PyMOL applies label_position along the screen axes, so the layout is done in
    camera coordinates and the offsets are used as they are. Labels are stacked
    with a minimum vertical separation and pushed to the left or right of the
    bundle, with connector lines drawn from each atom to its label.
    """
    proj = {k: rot @ (v - centre) for k, v in anchors.items()}          # camera frame
    order = sorted(proj, key=lambda k: -proj[k][1])                      # top to bottom
    column_x = (-1 if side == "left" else 1) * column_x
    placed, offsets = [], {}
    for key in order:
        x, y, _ = proj[key]
        while any(abs(y - py) < min_gap for py in placed):
            y -= min_gap * 0.55
        placed.append(y)
        offsets[key] = np.array([column_x - x, y - proj[key][1], 0.0]) / unit_in_angstrom
    return offsets


def view_block(rot, centre, dist, radius):
    # PyMOL's set_view takes the matrix column-major, i.e. the transpose of the
    # camera-axes-as-rows matrix built above (checked by rendering both).
    vals = list(rot.T.flatten()) + [0.0, 0.0, -dist] + list(centre) + \
           [dist - radius * 1.6, dist + radius * 1.6, -20.0]
    return "set_view (" + ",".join(f"{v:.4f}" for v in vals) + ")"


def build():
    setup = [
        "# Fig. 6 - class-associated residues on the GHSR structures",
        "# generated by revision/make_fig6_pymol.py - distances match Table 1",
        "reinitialize",
        "bg_color white",
        "set ray_opaque_background, 1",
        "set orthoscopic, 1",          # label offsets are then independent of depth
        "set cartoon_fancy_helices, 1",
        "set cartoon_transparency, 0.45",
        "set stick_radius, 0.26",
        "set sphere_scale, 0.30",
        "set label_size, 16",
        "set float_labels, 1",          # keep labels in front of the cartoon
        "set label_connector, 1",       # leader line from atom to displaced label
        "set label_connector_color, grey50",
        "set label_connector_width, 1.2",
        "set label_connector_mode, 1",
        "set label_color, black",
        # no label background: every label sits in the margin, and hidden labels
        # would otherwise leave an empty white box behind
        "set label_outline_color, white",
        "set label_position, (0, 2.2, 0)",
        "set dash_width, 2.2",
        "set dash_gap, 0.35",
        "set dash_color, grey40",
        "set depth_cue, 0",
        "set ray_shadows, 0",
        "set antialias, 2",
        "",
    ]
    views, chains, label_text = {}, {}, {}
    for tag, path, chain, ligand, title in STRUCTURES:
        prot, lig, resname, ca = parse(path, chain, ligand)
        rot, centre, dist, radius, column_x, unit = upright_view(ca, lig)
        views[tag] = view_block(rot, centre, dist, radius)
        chains[tag] = chain
        setup += [f"# ---------------- {tag}: {title} ----------------",
                  f"# expects {Path(path).name} beside this script, otherwise use:",
                  f"#   fetch {tag.upper()}, {tag}, async=0",
                  f"load {Path(path).name}, {tag}",
                  f"remove {tag} and not (chain {chain} or resn {ligand})",
                  f"# chain {chain} of 6KO5 also carries a BRIL fusion (residues >1000)",
                  f"remove {tag} and polymer and chain {chain} and resi 1000-2000",
                  f"remove {tag} and (solvent or hydro)",
                  f"hide everything, {tag}",
                  f"show cartoon, {tag} and chain {chain}",
                  f"color grey85, {tag} and chain {chain}",
                  f"set cartoon_color, grey85, {tag}",
                  f"show sticks, {tag} and resn {ligand}",
                  f"color yellow, {tag} and resn {ligand} and elem C",
                  ""]
        # segment labels on the right, residue labels on the left, both laid out
        # in camera coordinates so that they cannot overlap each other
        hel_anchors = {h: ca[r] for h, r in HELIX_LABELS.items() if r in ca}
        hel_off = layout_labels(hel_anchors, rot, centre, column_x, "right", unit)
        for hel, rep in HELIX_LABELS.items():
            if rep not in ca:
                continue
            o = hel_off[hel]
            setup += [f"select {tag}_hel_{hel}, {tag} and chain {chain} and resi {rep} and name CA",
                      f'label {tag}_hel_{hel}, "{hel}"',
                      f"set label_color, grey30, {tag}_hel_{hel}",
                      f"set label_size, 20, {tag}_hel_{hel}",
                      f"set label_position, ({o[0]:.2f}, {o[1]:.2f}, {o[2]:.2f}), {tag}_hel_{hel}"]
        setup.append("")
        res_off = {}
        for side_name, resl in (("EC50", EC50), ("IC50", IC50)):
            anchors = {r: ca[r] for r in resl if r in ca}
            res_off[side_name] = layout_labels(anchors, rot, centre, column_x, "left", unit)
        for side, resl, colour in (("EC50", EC50, "firebrick"), ("IC50", IC50, "marine")):
            members = []
            for r in resl:
                pair = closest_pair(prot, lig, r)
                sel = f"{tag}_{side}_{r}"
                members.append(sel)
                label = f"{AA3TO1.get(resname.get(r, ''), 'X')}{r}"
                setup += [f"select {sel}, {tag} and chain {chain} and resi {r}",
                          f"show sticks, {sel} and sidechain",
                          f"color {colour}, {sel} and elem C",
                          f"show spheres, {sel} and name CA",
                          f"color {colour}, {sel} and name CA"]
                if r in res_off[side]:
                    o = res_off[side][r]
                    setup.append(f"set label_position, ({o[0]:.2f}, {o[1]:.2f}, {o[2]:.2f}), {sel}")
                if pair:
                    d, pname, lname = pair
                    label_text[(tag, r)] = f"{label}  {d:.1f} A"
                    setup.append(f'label {sel} and name CA, "{label}  {d:.1f} A"')
                    if d <= DISTANCE_CUTOFF:
                        setup += [f"# {label}: minimum heavy-atom distance {d:.2f} A",
                                  f"distance {tag}_{side}_d{r}, {sel} and name {pname}, "
                                  f"{tag} and resn {ligand} and name {lname}",
                                  f"hide labels, {tag}_{side}_d{r}"]
                else:
                    label_text[(tag, r)] = label
                    setup.append(f'label {sel} and name CA, "{label}"')
                setup.append("")
            setup += [f"group {tag}_{side}, " + " ".join(members) + f" {tag}_{side}_d*", ""]
    setup += ["deselect",
              "print 'Fig. 6 scene ready: run Fig6_preview.pml or Fig6_render_final.pml'"]

    panels = [("Fig6a_agonist_both", "8jsr", "BOTH", None),
              ("Fig6b_agonist_EC50", "8jsr", "EC50", "IC50"),
              ("Fig6c_agonist_IC50", "8jsr", "IC50", "EC50"),
              ("Fig6d_antagonist_both", "6ko5", "BOTH", None)]

    def render(header, png_fmt, extra=()):
        out = ["# " + header, "# self-contained: loads and styles the scene first",
               "@Fig6_setup.pml", ""] + list(extra)
        for name, tag, show_side, hide_side in panels:
            other = "6ko5" if tag == "8jsr" else "8jsr"
            ch = chains[tag]
            lig = "UYI" if tag == "8jsr" else "8QX"
            both = "+".join(str(r) for r in EC50 + IC50)
            out += [f"# --- {name} ---", f"disable {other}", f"enable {tag}"]
            if show_side == "BOTH":
                # overview: both sets in one view, so the two faces of the pocket
                # can be compared; distances are read from the detail panels
                sel_all = f"{tag} and chain {ch} and resi {both}"
                out += [f"show sticks, ({sel_all}) and sidechain",
                        f"show spheres, ({sel_all}) and name CA",
                        f"disable {tag}_EC50_d*", f"disable {tag}_IC50_d*"]
                out += [f'label {tag} and chain {ch} and resi {r} and name CA, ""'
                        for r in EC50 + IC50]
            else:
                shown = "+".join(str(r) for r in (EC50 if show_side == "EC50" else IC50))
                hidden = "+".join(str(r) for r in (EC50 if hide_side == "EC50" else IC50))
                sel_show = f"{tag} and chain {ch} and resi {shown}"
                sel_hide = f"{tag} and chain {ch} and resi {hidden}"
                # groups collect named selections, which selection algebra does not
                # expand, so the panels address the residues explicitly
                out += [f"hide sticks, {sel_hide}", f"hide spheres, {sel_hide}",
                        f"disable {tag}_{hide_side}_d*"]
                out += [f'label {tag} and chain {ch} and resi {r} and name CA, ""'
                        for r in (EC50 if hide_side == "EC50" else IC50)]
                out += [f'label {tag} and chain {ch} and resi {r} and name CA, "{label_text[(tag, r)]}"'
                        for r in (EC50 if show_side == "EC50" else IC50) if (tag, r) in label_text]
                out += [
                        f"show sticks, ({sel_show}) and sidechain",
                        f"show spheres, ({sel_show}) and name CA",
                        f"show labels, ({sel_show}) and name CA",
                        f"enable {tag}_{show_side}_d*"]
            out += [f"show labels, {tag}_hel_*",
                    f"show sticks, {tag} and resn {lig}",
                    views[tag], png_fmt.format(name=name), ""]
        out.append(f"print '{header} done'")
        return "\n".join(out) + "\n"

    preview = render("Fig. 6 preview (small, ray traced)",
                     "png {name}_preview.png, width=1500, height=1500, dpi=300, ray=1")
    final = render("Fig. 6 final rendering at 1200 dpi",
                   "png {name}.png, width=4200, height=4200, dpi=1200, ray=1",
                   extra=["# if ray tracing is slow, uncomment the next line",
                          "# set cartoon_transparency, 0", ""])
    # an interactive entry point: labels sit beside their residues, the user
    # turns the molecule and saves the current view at publication resolution
    commands_py = f"""\
# Fig6_commands.py - helper commands for the interactive session
# loaded by Fig6_interactive.pml; defines: ec50, ic50, both, ago, ant, snap
from pymol import cmd

EC50 = "{'+'.join(str(r) for r in EC50)}"
IC50 = "{'+'.join(str(r) for r in IC50)}"


def _current():
    enabled = cmd.get_names("objects", enabled_only=1)
    return ("8jsr", "R") if "8jsr" in enabled else ("6ko5", "A")


def _show(resis):
    obj, chain = _current()
    cmd.hide("sticks", f"{{obj}} and polymer")
    cmd.hide("spheres", f"{{obj}} and polymer")
    sel = f"{{obj}} and chain {{chain}} and resi {{resis}}"
    cmd.show("sticks", f"({{sel}}) and sidechain")
    cmd.show("spheres", f"({{sel}}) and name CA")


def ec50():
    _show(EC50)


def ic50():
    _show(IC50)


def both():
    _show(EC50 + "+" + IC50)


def ago():
    cmd.disable("6ko5")
    cmd.enable("8jsr")


def ant():
    cmd.disable("8jsr")
    cmd.enable("6ko5")


def snap(name="Fig6_custom"):
    \"\"\"Save the current view at 1200 dpi.\"\"\"
    cmd.set("ray_trace_mode", 0)
    cmd.png(name, width=4200, height=4200, dpi=1200, ray=1)
    print(f"wrote {{name}}.png at 1200 dpi")


for _f in (ec50, ic50, both, ago, ant, snap):
    cmd.extend(_f.__name__, _f)
print("Commands: ec50 | ic50 | both | ago | ant | snap [name]")
"""

    interactive = [
        "# Fig. 6 interactive session",
        "#   1. open it:           pymol Fig6_interactive.pml",
        "#   2. turn and zoom with the mouse",
        "#   3. save the view:     snap              (1200 dpi PNG of what you see)",
        "#                         snap my_name      (choose the file name)",
        "#   switch residue set:   ec50 | ic50 | both",
        "#   switch structure:     ago (8JSR) | ant (6KO5)",
        "@Fig6_setup.pml",
        "",
        "# labels beside their residues: the column layout of the automatic panels",
        "# is tied to one fixed camera and would drift once you rotate",
        "set label_position, (2.5, 2.5, 3.0)",
        "set orthoscopic, 0",
        "",
        "run Fig6_commands.py",
        "",
        "ago",
        "both",
        "orient 8jsr and chain R and resi 100-320",
        "turn x, -20",
    ]
    for folder in (OUT_DIR, ASCII_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "Fig6_commands.py").write_text(commands_py, encoding="ascii")
        (folder / "Fig6_interactive.pml").write_text("\n".join(interactive) + "\n", encoding="ascii")

    setup_text = "\n".join(setup) + "\n"
    for folder in (OUT_DIR, ASCII_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        for name, text in (("Fig6_setup.pml", setup_text),
                           ("Fig6_preview.pml", preview),
                           ("Fig6_render_final.pml", final)):
            assert all(ord(c) < 128 for c in text), "script must be pure ASCII"
            # PyMOL splits a line on ';', so a comment containing one would be
            # parsed as a command and raise a SyntaxError
            assert not any(line.lstrip().startswith("#") and ";" in line
                           for line in text.splitlines()), "comment contains a semicolon"
            (folder / name).write_text(text, encoding="ascii")
    print(f"saved the three scripts into {OUT_DIR} and {ASCII_DIR}")
    for tag, path, chain, ligand, _ in STRUCTURES:
        prot, lig, _rn, _ca = parse(path, chain, ligand)
        pairs = {r: closest_pair(prot, lig, r) for r in EC50 + IC50}
        print(f"{tag}: " + ", ".join(f"{r} {v[0]:.1f}A" if v else f"{r} n/a"
                                     for r, v in pairs.items()))


if __name__ == "__main__":
    build()
