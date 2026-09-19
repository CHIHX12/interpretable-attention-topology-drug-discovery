#!/usr/bin/env python3
"""
feature_dependence.py — does the physicochemical channel drive the residue read-out?

The attention read-out used in this work feeds the receptor sequence with the
four physicochemical channels (hydrophobicity, volume, charge, polarity) set to
zero. The stated reason is that the read-out should reflect the amino acid at a
given position rather than the physicochemical descriptors attached to it.
This script tests that claim on the per-residue class difference Delta:

  R2_phys  : linear regression of Delta on the four descriptors
  R2_type  : one-way ANOVA of Delta on amino-acid identity (20 levels)
  var split: between-type variance versus within-type (positional) variance
  responders: how many residues respond to the ligand at all in each read-out

A read-out "driven by physicochemical features" should show a high R2_phys and
a small within-type share; a read-out that resolves individual positions should
show the opposite.

Usage:
    python revision/feature_dependence.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import AA_FEATURES  # noqa: E402

FIRST, LAST, OFFSET = 37, 338, 2
CONFIGS = [
    ("features OFF, max over all atom rows (this work)",
     "result/rev22/attention/featoff_epoch44_dec.npz", "unmasked_max"),
    ("features OFF, max over real atoms",
     "result/rev22/attention/featoff_epoch44_dec.npz", "masked_max"),
    ("features ON, max over all atom rows",
     "result/rev22/attention/dec2025_seed42_epoch44.npz", "unmasked_max"),
    ("features ON, mean over real atoms",
     "result/rev22/attention/dec2025_seed42_epoch44.npz", "masked_mean"),
]


def r2_linear(x, y):
    x = np.c_[np.ones(len(y)), x]
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    return float(1 - resid.var() / y.var())


def variance_split(delta, letters):
    grand = delta.mean()
    between = sum(np.sum(letters == a) * (delta[letters == a].mean() - grand) ** 2
                  for a in np.unique(letters))
    total = ((delta - grand) ** 2).sum()
    return float(between / total), float(1 - between / total)


def main():
    df = pd.read_csv("datasets/GPCR_resarch/GHSR_training_data.csv")
    seq = df.Protein.iloc[0]
    residues = np.arange(FIRST, LAST + 1)
    cols = residues - OFFSET
    letters = np.array([seq[c] for c in cols])
    phys = np.array([AA_FEATURES[a] for a in letters])

    report = {}
    for name, path, key in CONFIGS:
        f = Path(path)
        if not f.exists():
            print(f"missing {f} — skipped")
            continue
        z = np.load(f, allow_pickle=True)
        A = z[key][:, cols]
        y = z["label"].astype(int)
        delta = A[y == 1].mean(0) - A[y == 0].mean(0)
        between, within = variance_split(delta, letters)
        entry = {
            "R2_physicochemical_descriptors": round(r2_linear(phys, delta), 3),
            "R2_amino_acid_identity": round(r2_linear(pd.get_dummies(letters).values.astype(float), delta), 3),
            "between_type_share": round(between, 3),
            "within_type_positional_share": round(within, 3),
            "n_responder_residues": int(np.sum(A.std(0) > 1e-6)),
            "n_residues_with_zero_delta": int(np.sum(np.abs(delta) < 1e-3)),
        }
        report[name] = entry
        print(f"\n{name}")
        for k, v in entry.items():
            print(f"    {k:34s} {v}")

    out = Path("result/rev22/feature_dependence.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
