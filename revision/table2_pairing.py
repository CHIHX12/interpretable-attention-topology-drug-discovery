#!/usr/bin/env python3
"""
table2_pairing.py — regenerate Table 2 of the main text.

Table 2 reports the pairing structure of the class-associated residues under the
rule of Section 2.6: the constitutive pool is the set of analysed residues whose
class difference is small (|delta| < 0.1) and whose overall importance I, the
mean of the two class means, is high (I > 1.5); the five partners of a target
are those minimising the importance gap dImp(t, c) = |I_t - I_c|; and a target is
excluded from its own pool. Five targets are used per class direction, namely the
residues listed in Table 1.

Every value is a mean +/- SD over the ten fine-tuning seeds, except the pool
size, which is reported as a range.

Output: result/rev22/table2_pairing.csv, and the per-seed values alongside it.

Usage:
    python revision/table2_pairing.py
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

FIRST, LAST, OFFSET = 37, 338, 2
DELTA_MAX, IMP_MIN, N_PARTNERS = 0.1, 1.5, 5
TARGETS = {"EC50": [124, 125, 122, 198, 200],
           "IC50": [287, 286, 278, 147, 116]}


def partners(imp, pool, target, pos):
    """The N_PARTNERS pool members closest to the target in importance."""
    return sorted((c for c in pool if c != target),
                  key=lambda c: abs(imp[pos[target]] - imp[pos[c]]))[:N_PARTNERS]


def seed_metrics(path):
    resn = np.arange(FIRST, LAST + 1)
    pos = {r: i for i, r in enumerate(resn)}
    z = np.load(path, allow_pickle=True)
    A = z["unmasked_max"][:, resn - OFFSET]
    y = z["label"].astype(int)
    mean_ec50, mean_ic50 = A[y == 1].mean(0), A[y == 0].mean(0)
    imp, delta = (mean_ec50 + mean_ic50) / 2, mean_ec50 - mean_ic50
    pool = [r for r in resn
            if abs(delta[pos[r]]) < DELTA_MAX and imp[pos[r]] > IMP_MIN]

    row = {"file": Path(path).name, "pool": len(pool)}
    for side, targets in TARGETS.items():
        chosen = {t: partners(imp, pool, t, pos) for t in targets}
        gaps = [abs(imp[pos[t]] - imp[pos[c]]) for t, cs in chosen.items() for c in cs]
        used = {}
        for t, cs in chosen.items():
            for c in cs:
                used.setdefault(c, set()).add(t)
        shared = sum(1 for cs in chosen.values() for c in cs if len(used[c]) > 1)
        row |= {f"gap_mean_{side}": float(np.mean(gaps)),
                f"gap_median_{side}": float(np.median(gaps)),
                f"max_sharing_{side}": max(len(v) for v in used.values()),
                f"distinct_partners_{side}": len(used),
                f"shared_fraction_{side}": shared / len(gaps)}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="+",
                    default=sorted(glob.glob("result/rev22/attention/featoff_seed*.npz")))
    ap.add_argument("--out", default="result/rev22/table2_pairing.csv")
    args = ap.parse_args()

    per_seed = pd.DataFrame([seed_metrics(f) for f in args.att])
    metrics = [c for c in per_seed.columns if c not in ("file", "pool")]
    summary = pd.DataFrame({
        "metric": metrics,
        "mean": [per_seed[c].mean() for c in metrics],
        "sd": [per_seed[c].std(ddof=1) for c in metrics],
        "min": [per_seed[c].min() for c in metrics],
        "max": [per_seed[c].max() for c in metrics],
    })
    smaller = int((per_seed.gap_mean_EC50 < per_seed.gap_mean_IC50).sum())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    per_seed.to_csv(out.with_name(out.stem + "_per_seed.csv"), index=False)

    print(summary.round(4).to_string(index=False))
    print(f"\nconstitutive pool: {per_seed['pool'].min()}-{per_seed['pool'].max()} residues")
    print(f"seeds in which the EC50 side has the smaller gap: {smaller} of {len(per_seed)}")
    print(f"saved {out} and {out.with_name(out.stem + '_per_seed.csv')}")


if __name__ == "__main__":
    main()
