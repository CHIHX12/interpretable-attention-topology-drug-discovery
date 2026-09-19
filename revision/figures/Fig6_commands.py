# Fig6_commands.py - helper commands for the interactive session
# loaded by Fig6_interactive.pml; defines: ec50, ic50, both, ago, ant, snap
from pymol import cmd

EC50 = "124+125+122+200+198"
IC50 = "287+286+278+116+147"


def _current():
    enabled = cmd.get_names("objects", enabled_only=1)
    return ("8jsr", "R") if "8jsr" in enabled else ("6ko5", "A")


def _show(resis):
    obj, chain = _current()
    cmd.hide("sticks", f"{obj} and polymer")
    cmd.hide("spheres", f"{obj} and polymer")
    sel = f"{obj} and chain {chain} and resi {resis}"
    cmd.show("sticks", f"({sel}) and sidechain")
    cmd.show("spheres", f"({sel}) and name CA")


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
    """Save the current view at 1200 dpi."""
    cmd.set("ray_trace_mode", 0)
    cmd.png(name, width=4200, height=4200, dpi=1200, ray=1)
    print(f"wrote {name}.png at 1200 dpi")


for _f in (ec50, ic50, both, ago, ant, snap):
    cmd.extend(_f.__name__, _f)
print("Commands: ec50 | ic50 | both | ago | ant | snap [name]")
