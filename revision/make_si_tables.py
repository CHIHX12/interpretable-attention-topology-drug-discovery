#!/usr/bin/env python3
"""
make_si_tables.py — generate Supplementary Tables 1 to 8.

The Supplementary Information described eight tables but contained none of them.
This script builds all eight from the deposited results, writes each as a CSV
under result/rev22/si_tables/, and writes one tab-separated text file,
si_tables.txt, in the same layout as the tables in the main text so that it can
be pasted straight into the Supplementary Information.

Ballesteros-Weinstein numbering is derived from the anchors that appear in the
deposited elastic-network scripts: Val131 = 3.40, Pro224 = 5.50 and Pro278 =
6.50, the last cross-checked against Trp276 = 6.48 and Phe272 = 6.44 in the same
scripts. The derivation puts Glu124 at 3.33, which is the value reported for
GHSR in the literature. Residues in loops carry no number.

Usage:
    python revision/make_si_tables.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("result/rev22/si_tables")
RES = Path("result/rev22")
FIRST, LAST, OFFSET = 37, 338, 2
ATT = RES / "attention"

# helix anchors, taken from revision/enm/{prs_tm6,w276_strike_pif,pif_cwxp_compare}.py
BW_ANCHOR = {"TM3": (131, 3, 40), "TM5": (224, 5, 50), "TM6": (278, 6, 50)}

TABLE1_RESIDUES = [124, 125, 122, 200, 198, 287, 286, 278, 147, 116]
ALSO_DISCUSSED = [123, 300, 231, 38, 302, 37, 285, 283, 276, 272, 224, 131]


def tm_map():
    nodes = json.loads(Path("revision/enm/_data_compact.json").read_text())["nodes"]
    return {n["id"]: n.get("tm", "") for n in nodes}


def bw(resnum, tm):
    if tm not in BW_ANCHOR:
        return ""
    anchor, helix, pos = BW_ANCHOR[tm]
    return f"{helix}.{pos + resnum - anchor:02d}"


def ligand_distances():
    """Minimum heavy-atom distance from each residue to the co-crystallised ligand."""
    out = {}
    for pdb, chain, het in (("8jsr", "R", "UYI"), ("6ko5", "A", None)):
        rec, lig = {}, []
        for line in open(f"revision/enm/{pdb}.pdb"):
            tag = line[:6]
            if tag not in ("ATOM  ", "HETATM") or line[76:78].strip() == "H":
                continue
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            if tag == "ATOM  " and line[21] == chain:
                rec.setdefault(int(line[22:26]), []).append(xyz)
            elif tag == "HETATM" and line[21] == chain:
                name = line[17:20].strip()
                if het and name == het:
                    lig.append(xyz)
                elif not het and name not in ("HOH", "NA", "CL", "SO4", "EDO",
                                              "GOL", "PEG", "OLC", "OLA", "CLR", "PLM"):
                    lig.append((name, xyz))
        if not het:                      # 6KO5: keep the largest hetero group
            groups = {}
            for name, xyz in lig:
                groups.setdefault(name, []).append(xyz)
            lig = max(groups.values(), key=len)
        out[pdb] = {r: min(math.dist(a, b) for a in v for b in lig)
                    for r, v in rec.items()}
    return out["8jsr"], out["6ko5"]


def seed_delta(files, key="unmasked_max"):
    cols = np.arange(FIRST, LAST + 1) - OFFSET
    rows = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        if key not in z.files:
            continue
        A, y = z[key][:, cols], z["label"].astype(int)
        rows.append(A[y == 1].mean(0) - A[y == 0].mean(0))
    return np.array(rows)


def table1():
    """Cost of the probing read-out, per fine-tuning seed."""
    runs = pd.read_csv(RES / "summary_runs.csv")
    on = runs[(runs.family == "random") & (runs.init == "pretrained")
              & (runs.descriptors == "on")][["seed", "auroc"]]
    probe = pd.read_csv(RES / "attention_probe.csv")
    off = probe[probe.file.str.startswith("featoff_seed")].copy()
    off["seed"] = off.file.str.extract(r"seed(\d+)").astype(int)
    t = (on.merge(off[["seed", "decoder_auroc_full_set", "probe_auroc_test"]],
                  on="seed").sort_values("seed"))
    t.columns = ["Seed", "AUROC, descriptors supplied",
                 "AUROC, descriptors withheld", "AUROC, linear probe on the map"]
    return t.round(4)


def table2(tms, d8, d6):
    """The ten residues with the largest class difference in each direction.

    Ranking the 302 residues by |delta| alone returns twenty IC50-side
    positions, because that side carries much larger magnitudes, so the table
    takes ten from each direction as Fig. 4 does.
    """
    t = pd.read_csv(RES / "table1_residues.csv").copy()
    t = t.sort_values("delta_mean_10seeds", ascending=False)
    t = pd.concat([t.head(10), t.tail(10)])
    out = pd.DataFrame({
        "Residue": t.residue, "Domain": [tms.get(r, "") for r in t.resnum],
        "BW": [bw(r, tms.get(r, "")) for r in t.resnum],
        "delta (mean over seeds)": t.delta_mean_10seeds.round(3),
        "SD over seeds": t.delta_sd_10seeds.round(3),
        "P (Welch)": t.p_welch.map(lambda v: f"{v:.1e}"),
        "q (BH)": t.q_bh.map(lambda v: f"{v:.1e}"),
        "delta, pre-trained": t.delta_pretrained_only.round(3),
        "z within amino-acid type": t.z_within_amino_acid_type.round(2),
        "d 8JSR (A)": [round(d8.get(r, float("nan")), 1) for r in t.resnum],
        "d 6KO5 (A)": [round(d6.get(r, float("nan")), 1) for r in t.resnum]})
    return out


def table3():
    """The fifty selected pairs of the reference model, seed 42."""
    t = pd.read_csv(RES / "table2_pairs_seed42.csv")
    out = t[["direction", "target", "partner", "rank", "I_target", "I_partner",
             "dImp", "U_c"]].copy()
    out.columns = ["Direction", "Target", "Partner", "Rank by gap", "I (target)",
                   "I (partner)", "Importance gap", "Usage count U_c"]
    for c in ("I (target)", "I (partner)", "Importance gap"):
        out[c] = out[c].round(4)
    return out


def table4(tms, d8, d6):
    """Every residue discussed in the main text, with its distances."""
    seq = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv").Protein.iloc[0]
    rows = []
    for r in TABLE1_RESIDUES + ALSO_DISCUSSED:
        tm = tms.get(r, "")
        def dist(table):
            v = table.get(r)
            return "n.d." if v is None else f"{v:.1f}"
        rows.append({"Residue": f"{seq[r - OFFSET]}{r}", "Domain": tm or "n.d.",
                     "BW": bw(r, tm) or "-",
                     "d 8JSR (A)": dist(d8), "d 6KO5 (A)": dist(d6),
                     "In Table 1": "yes" if r in TABLE1_RESIDUES else "no"})
    return pd.DataFrame(rows)


def table5():
    """Ten strongest residues per direction under four read-outs of one model."""
    seq = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv").Protein.iloc[0]
    resn = np.arange(FIRST, LAST + 1)
    variants = [("descriptors withheld, max over all graph rows (this work)",
                 ATT / "featoff_epoch36.npz", "unmasked_max"),
                ("descriptors supplied, max over all graph rows",
                 ATT / "epoch36_with_unmaskedmax.npz", "unmasked_max"),
                ("descriptors supplied, padding excluded, mean over real atoms",
                 ATT / "epoch36_with_unmaskedmax.npz", "masked_mean"),
                ("descriptors withheld, padding excluded, max over real atoms",
                 ATT / "featoff_epoch36.npz", "masked_max")]
    cols = {}
    for label, path, key in variants:
        d = seed_delta([path], key)[0]
        order = np.argsort(-d)

        def cell(i):
            if abs(d[i]) < 1e-6:
                return "no response"
            return f"{seq[resn[i] - OFFSET]}{resn[i]} ({d[i]:+.4f})"

        cols[label] = ([cell(i) for i in order[:10]]
                       + [cell(i) for i in order[-10:][::-1]]
                       + [f"{int((np.abs(d) > 1e-6).sum())} of 302",
                          f"{d.max():+.4f}", f"{d.min():+.4f}"])
    out = pd.DataFrame(cols)
    out.insert(0, "Rank", [f"EC50 side {i}" for i in range(1, 11)]
               + [f"IC50 side {i}" for i in range(1, 11)]
               + ["Residues responding", "Largest positive delta", "Largest negative delta"])
    return out


def table6():
    """Class counts, overlap and chemotype novelty for every partition."""
    s = json.loads(Path("datasets/GPCR_resarch/rev22_split_summary.json").read_text())["splits"]
    rows = []
    for name, v in s.items():
        nn = v.get("test_to_train_nn_tanimoto", {})
        rows.append({
            "Partition": "record-level (deposited)" if name == "random" else name.replace("rev22_", ""),
            "Train": v["n_train"], "Val": v["n_val"], "Test": v["n_test"],
            "Test class 1 / 0": f'{v["class_counts_test"].get("1", 0)} / {v["class_counts_test"].get("0", 0)}',
            "SMILES shared train-test": v["canonical_smiles_overlap_train_test"],
            "Test scaffolds seen in train": f'{v["n_test_scaffolds_seen_in_train"]} of {v["n_test_scaffolds"]}',
            "NN Tanimoto median": round(nn.get("median", float("nan")), 3),
            "Fraction with NN < 0.4": round(nn.get("frac_lt_0.4", float("nan")), 3)})
    return pd.DataFrame(rows)


def table7():
    """Test AUROC and AUPRC for every model on every partition family."""
    s = pd.read_csv(RES / "summary_table.csv")
    s = s[s.family != "nodual"].copy()
    s["AUROC"] = [f"{m:.3f} ± {0 if pd.isna(d) else d:.3f}" for m, d in zip(s.auroc_mean, s.auroc_sd)]
    s["AUPRC"] = [f"{m:.3f} ± {0 if pd.isna(d) else d:.3f}" for m, d in zip(s.auprc_mean, s.auprc_sd)]
    out = s[["family", "model", "AUROC", "AUPRC", "n"]].copy()
    out.columns = ["Partition family", "Model", "Test AUROC", "Test AUPRC", "Replicates"]
    return out.sort_values(["Partition family", "Model"])


def table8():
    """The dual-endpoint sensitivity analysis."""
    s = pd.read_csv(RES / "summary_table.csv")
    keep = s[s.family.isin(["compound", "nodual"])].copy()
    piv = keep.pivot_table(index="model", columns="family", values="auroc_mean")
    piv.columns = ["compound-disjoint, all ligands", "compound-disjoint, dual endpoints removed"]
    piv["change"] = piv.iloc[:, 1] - piv.iloc[:, 0]
    return piv.round(4).reset_index().rename(columns={"model": "Model"})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tms = tm_map()
    d8, d6 = ligand_distances()
    tables = [
        (1, "Cost of the probing read-out, per fine-tuning seed", table1()),
        (2, "The ten residues with the largest class difference in each direction", table2(tms, d8, d6)),
        (3, "The fifty selected pairs, reference model, seed 42", table3()),
        (4, "Every residue discussed in the main text, with ligand distances", table4(tms, d8, d6)),
        (5, "Ten strongest residues per direction under four read-outs", table5()),
        (6, "Partition characteristics and leakage audit", table6()),
        (7, "Test performance of every model on every partition family", table7()),
        (8, "Dual-endpoint sensitivity analysis", table8()),
    ]
    blocks = []
    for n, title, df in tables:
        df.to_csv(OUT / f"supplementary_table_{n}.csv", index=False)
        blocks.append(f"Supplementary Table {n} | {title}.\n"
                      + df.to_csv(sep="\t", index=False).rstrip("\n"))
        print(f"  Supplementary Table {n}: {len(df)} rows x {len(df.columns)} columns")
    (OUT / "si_tables.txt").write_text("\n\n\n".join(blocks) + "\n", encoding="utf-8")
    print(f"\nsaved {len(tables)} CSVs and {OUT/'si_tables.txt'}")


if __name__ == "__main__":
    main()
