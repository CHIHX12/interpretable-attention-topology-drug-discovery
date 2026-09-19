#!/usr/bin/env python3
"""
make_schematics.py — the two schematic main-text figures for the v22 revision.

  Fig1_architecture_and_readout : the two encoders, the bilinear attention
                                  module, the per-residue read-out, and the two
                                  training stages.
  Fig2_pairing_workflow         : the pairing analysis (TEMA) as implemented.

These two carry no data, so they live apart from make_figures.py, which reads
result/rev22/. Both are laid out on an explicit row grid with one card height,
so every connector is a straight line or a single right angle and every box
lines up with the ones beside it. Card text is measured against the card width
and wrapped, and a card that still does not fit raises rather than overflowing
silently. Neither figure carries an in-figure note: everything of that kind is
in the figure legend in the main text.

Usage:
    python revision/make_schematics.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

OUT = Path("文章草稿/圖片_v22")

# --- design tokens -----------------------------------------------------------
INK, MUTED, HAIR = "#111827", "#6B7280", "#B9C0CA"
DATA_F, DATA_E = "#F1F3F6", "#9AA4B2"      # data objects: inputs, tensors
MODEL_F, MODEL_E = "#DBE7F3", "#3B6FA0"    # learned components
FOCUS_F, FOCUS_E = "#FBE1CD", "#C2681F"    # the quantity analysed in this work
STAGE_F, STAGE_E = "#E6E1F0", "#6A5B94"    # downstream analysis

LEAD_PT, BODY_PT, TITLE_PT = 7.2, 6.5, 8.4
LEADING = 1.32          # line height as a multiple of the font size
PAD_V = 0.012           # vertical padding inside a card, in axis fraction
PAD_H = 0.012           # horizontal inset for card text


def line_h(pt, fig_h):
    """Height of one text line in axis fraction, for a figure fig_h inches tall."""
    return pt * LEADING / (72.0 * fig_h)


def text_w(fig, s, pt, weight="normal"):
    """Rendered width of a string, as a fraction of the figure width."""
    handle = fig.text(0, 0, s, fontsize=pt, weight=weight)
    width = handle.get_window_extent(renderer=fig.canvas.get_renderer()).width
    handle.remove()
    return width / fig.bbox.width


def wrap(fig, s, max_w, pt, weight="normal"):
    """Greedy wrap of s to lines no wider than max_w (figure fraction)."""
    lines, current = [], ""
    for word in s.split():
        trial = f"{current} {word}".strip()
        if current and text_w(fig, trial, pt, weight) > max_w:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def canvas(w, h):
    fig = plt.figure(figsize=(w, h), constrained_layout=False)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def card_lines(fig, w, lead, body):
    """Wrap a card's text to its width and return (lead lines, body lines)."""
    inner = w - 2 * PAD_H
    lead_lines = wrap(fig, lead, inner, LEAD_PT, "bold")
    body_lines = [ln for part in ([body] if isinstance(body, str) else body)
                  for ln in wrap(fig, part, inner, BODY_PT)]
    return lead_lines, body_lines


def card_height(fig, w, lead, body, fig_h, min_h=0.0):
    """Height a card needs for its wrapped text, in axis fraction."""
    lead_lines, body_lines = card_lines(fig, w, lead, body)
    block = (line_h(LEAD_PT, fig_h) * len(lead_lines)
             + line_h(BODY_PT, fig_h) * len(body_lines))
    return max(min_h, block + 2 * PAD_V)


def card(fig, ax, x, y_top, w, card_h, lead, body, *, fill, edge, fig_h,
         lead_colour=None, body_colour=None):
    """A box drawn at a height the caller chose, with the text centred in it.

    y_top is the top edge, so a row of cards shares one baseline whatever their
    contents, which is what keeps the horizontal connectors straight.
    """
    lead_lines, body_lines = card_lines(fig, w, lead, body)
    lh_lead, lh_body = line_h(LEAD_PT, fig_h), line_h(BODY_PT, fig_h)
    block = lh_lead * len(lead_lines) + lh_body * len(body_lines)

    ax.add_patch(Rectangle((x, y_top - card_h), w, card_h, facecolor=fill,
                           edgecolor=edge, linewidth=0.7, zorder=2))
    cursor = y_top - (card_h - block) / 2 - lh_lead * 0.76
    for text in lead_lines:
        ax.text(x + PAD_H, cursor, text, fontsize=LEAD_PT, weight="bold",
                color=lead_colour or INK, ha="left", va="baseline", zorder=3)
        cursor -= lh_lead
    cursor += lh_lead - lh_body
    for text in body_lines:
        ax.text(x + PAD_H, cursor, text, fontsize=BODY_PT,
                color=body_colour or INK, ha="left", va="baseline", zorder=3)
        cursor -= lh_body


def path_arrow(ax, pts, colour=INK, lw=0.85):
    """Polyline with an arrowhead on the final segment. Right angles only."""
    if len(pts) > 2:
        ax.plot([p[0] for p in pts[:-1]], [p[1] for p in pts[:-1]],
                color=colour, lw=lw, zorder=1, solid_capstyle="butt",
                solid_joinstyle="miter")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops={"arrowstyle": "-|>", "color": colour, "linewidth": lw,
                            "shrinkA": 0, "shrinkB": 0, "mutation_scale": 6.5},
                zorder=1)


def panel_title(fig, ax, x, y, w, text):
    if x + text_w(fig, text, TITLE_PT, "bold") > 1.0:
        raise ValueError(f"panel title runs off the canvas: {text!r}")
    ax.text(x, y, text, fontsize=TITLE_PT, weight="bold", color=INK,
            ha="left", va="baseline")
    ax.plot([x, x + w], [y - 0.022, y - 0.022], color=HAIR, lw=0.8, zorder=1)


def save(fig, name, y_min, fig_w, fig_h, pad=0.06):
    """Save cropped to the region actually used, so no dead band is shipped."""
    OUT.mkdir(parents=True, exist_ok=True)
    box = Bbox([[0, max(0.0, y_min * fig_h - pad)], [fig_w, fig_h]])
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches=box, dpi=1200)
    plt.close(fig)
    print(f"saved {OUT/name}.png and .pdf")


# --- Figure 1 ----------------------------------------------------------------
def figure1():
    FW, FH = 6.9, 3.6
    fig, ax = canvas(FW, FH)

    xa, xb, xc, xd = 0.010, 0.282, 0.530, 0.778
    wa, wb, wc, wd = 0.236, 0.212, 0.212, 0.212
    band_top, gap, min_h = 0.880, 0.028, 0.085

    panel_title(fig, ax, xa, 0.945, wa, "(a)  Encoders")
    panel_title(fig, ax, xb, 0.945, xc + wc - xb,
                "(b)  Bilinear attention and read-out")
    panel_title(fig, ax, xd, 0.945, wd, "(c)  Training and Δ")

    def item(r, x, w, lead, detail, fill, edge, **kw):
        return dict(r=r, x=x, w=w, lead=lead, body=detail, fill=fill,
                    edge=edge, kw=kw)

    items = [
        # (a) one lane per encoder, receptor above, ligand below
        item(0, xa, wa, "Receptor sequence",
             ["523 positions: GHSR 2–366 plus a fusion tail",
              "padded to 1,200 for the encoder"], DATA_F, DATA_E),
        item(1, xa, wa, "BiLSTM · 2 × 256",
             ["embedding 128 ⊕ 4 channels†",
              "replaces DrugBAN's 1D CNN"], MODEL_F, MODEL_E),
        item(2, xa, wa, "Ligand graph",
             "290 nodes; the padding nodes are ligand-independent",
             DATA_F, DATA_E),
        item(3, xa, wa, "GCN · 3 × 128", "unchanged from DrugBAN",
             MODEL_F, MODEL_E),
        # (b) the module, the branch this work does not read, and the read-out
        item(0, xb, xc + wc - xb, "Decoder → class probability",
             "not the quantity analysed in this work; Section 2.5 reports what "
             "the probing read-out costs it", MODEL_F, MODEL_E,
             lead_colour=MUTED, body_colour=MUTED),
        item(1, xb, wb, "Bilinear attention",
             "2 heads · joint dim 768 · pooled dim 256", MODEL_F, MODEL_E),
        item(2, xb, wb, "Attention tensor",
             "2 heads × 290 graph rows × 1,200 positions", DATA_F, DATA_E),
        item(2, xc, wc, "Read-out",
             ["mean over heads, then maximum over graph rows",
              "keeping the 523 real positions"],
             MODEL_F, MODEL_E),
        item(3, xc, wc, "One score per residue",
             "per ligand — the quantity analysed in this work",
             FOCUS_F, FOCUS_E),
        # (c) the two training stages, and where the read-out goes next
        item(0, xd, wd, "Stage 1 · BindingDB",
             "49,199 pairs over 2,623 targets; learns occupancy",
             MODEL_F, MODEL_E),
        item(1, xd, wd, "Stage 2 · GHSR",
             "1,539 records, EC50 vs IC50; fine-tunes all layers",
             MODEL_F, MODEL_E),
        item(3, xd, wd, "Δ per residue",
             ["mean(EC50) − mean(IC50)",
              "feeds Fig. 2 and Section 3.6"], STAGE_F, STAGE_E),
    ]

    # a row is as tall as its tallest card, so no box carries dead space and
    # cards in the same row still share a top and a mid-line
    height = {}
    for it in items:
        height[it["r"]] = max(height.get(it["r"], 0.0),
                              card_height(fig, it["w"], it["lead"], it["body"],
                                          FH, min_h))
    top, y = {}, band_top
    for r in sorted(height):
        top[r] = y
        y -= height[r] + gap

    def bot(r):
        return top[r] - height[r]

    def mid(r):
        return top[r] - height[r] / 2

    for it in items:
        card(fig, ax, it["x"], top[it["r"]], it["w"], height[it["r"]],
             it["lead"], it["body"], fill=it["fill"], edge=it["edge"],
             fig_h=FH, **it["kw"])

    # connectors: three hand-offs across the figure, everything else vertical
    path_arrow(ax, [(xa + wa / 2, bot(0)), (xa + wa / 2, top[1])])
    path_arrow(ax, [(xa + wa / 2, bot(2)), (xa + wa / 2, top[3])])
    path_arrow(ax, [(xd + wd / 2, bot(0)), (xd + wd / 2, top[1])])
    path_arrow(ax, [(xa + wa, mid(1)), (xb, mid(1))])
    leg = (xa + wa + xb) / 2
    path_arrow(ax, [(xa + wa, mid(3)), (leg, mid(3)),
                    (leg, mid(1) - 0.032), (xb, mid(1) - 0.032)])
    path_arrow(ax, [(xb + wb / 2, top[1]), (xb + wb / 2, bot(0))])
    path_arrow(ax, [(xb + wb / 2, bot(1)), (xb + wb / 2, top[2])])
    path_arrow(ax, [(xb + wb, mid(2)), (xc, mid(2))])
    path_arrow(ax, [(xc + wc / 2, bot(2)), (xc + wc / 2, top[3])])
    path_arrow(ax, [(xc + wc, mid(3)), (xd, mid(3))])

    save(fig, "Fig1_architecture_and_readout", bot(3), FW, FH)


# --- Figure 2 ----------------------------------------------------------------
def figure2():
    FW, FH = 5.0, 4.3
    fig, ax = canvas(FW, FH)

    x0, x1 = 0.02, 0.98
    rail_x, step_x = 0.068, 0.128
    steps = [("Rank the partners",
              "$\Delta$Imp$(t,c)=|I_t-I_c|$; keep the five smallest per target",
              DATA_F, DATA_E),
             ("Count the usage",
              "$U_c$, the number of targets that select the same partner",
              DATA_F, DATA_E),
             ("Summarise the sharing per direction",
              "the comparison reported in Table 2", DATA_F, DATA_E),
             ("Map the selected residues",
              "onto the structure (Fig. 6) and the elastic network (Section 3.6)",
              STAGE_F, STAGE_E)]

    wf = (x1 - x0 - 0.05) / 2
    head = ("Per-residue attention scores $A$",
            "one profile per ligand over the 302 analysed positions")
    fork = [("Targets $t$", "five per direction, the residues of Table 1"),
            ("Constitutive pool $c$",
             "$|\Delta|<0.1$ and importance $I>1.5$")]
    # one height for every card, so the rail and the step pitch stay regular
    card_h = max([card_height(fig, x1 - x0, *head, FH)]
                 + [card_height(fig, wf, lead, detail, FH) for lead, detail in fork]
                 + [card_height(fig, x1 - step_x, lead, detail, FH)
                    for lead, detail, _, _ in steps])

    def put(x, y_top, w, lead, detail, fill, edge):
        card(fig, ax, x, y_top, w, card_h, lead, detail, fill=fill, edge=edge,
             fig_h=FH)

    head_top = 0.975
    put(x0, head_top, x1 - x0, *head, DATA_F, DATA_E)
    head_bot = head_top - card_h

    # fork into the two residue sets, drawn as an H so no connector is diagonal
    fork_top = head_bot - 0.050
    put(x0, fork_top, wf, *fork[0], MODEL_F, MODEL_E)
    put(x1 - wf, fork_top, wf, *fork[1], MODEL_F, MODEL_E)
    lx, rx = x0 + wf / 2, x1 - wf / 2
    bus = head_bot - 0.027
    ax.plot([0.5, 0.5], [head_bot, bus], color=INK, lw=0.85)
    ax.plot([lx, rx], [bus, bus], color=INK, lw=0.85)
    for x in (lx, rx):
        path_arrow(ax, [(x, bus), (x, fork_top)])

    fork_bot = fork_top - card_h
    bus2 = fork_bot - 0.027
    for x in (lx, rx):
        ax.plot([x, x], [fork_bot, bus2], color=INK, lw=0.85)
    ax.plot([lx, rx], [bus2, bus2], color=INK, lw=0.85)

    step_top = bus2 - 0.033
    pitch = card_h + 0.046
    tops = [step_top - i * pitch for i in range(len(steps))]
    path_arrow(ax, [(0.5, bus2), (0.5, tops[0])])
    ax.plot([rail_x, rail_x], [tops[0] - card_h / 2, tops[-1] - card_h / 2],
            color=HAIR, lw=0.8, zorder=1)
    for i, ((lead, detail, fill, edge), y_top) in enumerate(zip(steps, tops), 1):
        put(step_x, y_top, x1 - step_x, lead, detail, fill, edge)
        ax.plot([rail_x], [y_top - card_h / 2], "o", ms=9.0, color=STAGE_E,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.text(rail_x, y_top - card_h / 2, str(i), fontsize=6.0, weight="bold",
                color="white", ha="center", va="center", zorder=4)
        if i < len(steps):
            centre = step_x + (x1 - step_x) / 2
            path_arrow(ax, [(centre, y_top - card_h), (centre, tops[i])])

    save(fig, "Fig2_pairing_workflow", tops[-1] - card_h, FW, FH)


if __name__ == "__main__":
    figure1()
    figure2()
