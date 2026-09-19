#!/usr/bin/env python3
"""
attention_probe.py — does the per-residue attention map carry the class distinction?

The residue-level analysis of this work reads the interaction map with the four
physicochemical channels of the receptor encoder withheld. In that
configuration the decoder is out of distribution and no longer separates the
classes, so the natural question is whether the map itself still encodes them.

This script answers it with a linear probe: an L2 logistic regression is fitted
on the 302-dimensional per-ligand attention profile of the training partition
and evaluated on the held-out test partition. It reports, for each attention
file, the probe AUROC alongside the AUROC of the model's own decoder in the
same forward pass, so that the two can be compared directly.

Output: result/rev22/attention_probe.csv

Usage:
    python revision/attention_probe.py
    python revision/attention_probe.py --att result/rev22/attention/featoff_seed4*.npz
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIRST, LAST, OFFSET = 37, 338, 2
DATA = "datasets/GPCR_resarch/GHSR_training_data.csv"
SPLIT = "datasets/GPCR_resarch/random"


def masks():
    df = pd.read_csv(DATA)
    key = list(zip(df.SMILES, df.Y))
    out = {}
    for part in ("train", "test"):
        sub = pd.read_csv(f"{SPLIT}/{part}.csv")
        keys = set(zip(sub.SMILES, sub.Y))
        out[part] = np.array([k in keys for k in key])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--att", nargs="+", default=sorted(
        glob.glob("result/rev22/attention/featoff_seed*.npz")
        + glob.glob("result/rev22/attention/trainfeatoff_seed*.npz")))
    ap.add_argument("--key", default="unmasked_max")
    ap.add_argument("--out", default="result/rev22/attention_probe.csv")
    args = ap.parse_args()

    m = masks()
    cols = np.arange(FIRST, LAST + 1) - OFFSET
    rows = []
    for f in args.att:
        z = np.load(f, allow_pickle=True)
        if args.key not in z.files:
            print(f"{f}: no {args.key}, skipped")
            continue
        A = z[args.key][:, cols]
        y = z["label"].astype(int)
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        clf.fit(A[m["train"]], y[m["train"]])
        probe = roc_auc_score(y[m["test"]], clf.predict_proba(A[m["test"]])[:, 1])
        rows.append({"file": Path(f).name, "probe_auroc_test": probe,
                     "decoder_auroc_full_set": roc_auc_score(y, z["pred"])})
        print(f"{Path(f).name:34s} probe {probe:.4f}  decoder {rows[-1]['decoder_auroc_full_set']:.4f}")
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df[["probe_auroc_test", "decoder_auroc_full_set"]].agg(["mean", "std"]).round(4).to_string())
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
