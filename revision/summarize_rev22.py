#!/usr/bin/env python3
"""
summarize_rev22.py — collect v22 training runs and ligand-only baselines.

Reads result/rev22/<split>/<init>_seed<seed>/rev22_test_metrics.json and
result/rev22/ligand_baselines.csv, and writes
    result/rev22/summary_runs.csv        one row per run
    result/rev22/summary_table.csv       mean, SD, n per split family and model
    result/rev22/summary_paired.json     paired pretrained-vs-scratch differences

Usage:
    python revision/summarize_rev22.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("result/rev22")


def family(split):
    return re.sub(r"_s\d+$", "", split).replace("rev22_", "")


def main():
    rows = []
    for p in ROOT.glob("*/*/rev22_test_metrics.json"):
        rec = json.loads(p.read_text())
        # the run directory encodes whether the physicochemical channels were used
        rec["descriptors"] = "off" if "_featoff_" in p.parent.name else "on"
        rows.append(rec)
    runs = pd.DataFrame(rows)
    runs["family"] = runs.split.map(family)
    runs["model"] = ("DrugBAN-BiLSTM (" + runs["init"] + ", descriptors "
                     + runs["descriptors"] + ")")
    runs.sort_values(["family", "split", "init", "seed"]).to_csv(ROOT / "summary_runs.csv", index=False)

    frames = [runs[["family", "split", "model", "auroc", "auprc"]]]
    base_file = ROOT / "ligand_baselines.csv"
    if base_file.exists():
        base = pd.read_csv(base_file)
        base["family"] = base.split.map(family)
        frames.append(base[["family", "split", "model", "auroc", "auprc"]])
    allr = pd.concat(frames)
    table = allr.groupby(["family", "model"]).agg(
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        auprc_mean=("auprc", "mean"), auprc_sd=("auprc", "std"), n=("auroc", "size")).round(4)
    table.to_csv(ROOT / "summary_table.csv")
    print(table.to_string())

    paired = {}
    for (fam, desc), g in runs.groupby(["family", "descriptors"]):
        piv = g.pivot_table(index=["split", "seed", "descriptors"], columns="init", values="auroc")
        if {"pretrained", "scratch"} <= set(piv.columns):
            piv = piv.dropna()
            if len(piv):
                diff = piv.pretrained - piv.scratch
                test = stats.ttest_rel(piv.pretrained, piv.scratch) if len(piv) > 1 else None
                paired[f"{fam}_descriptors_{desc}"] = {"n_pairs": int(len(piv)), "mean_diff_auroc": float(diff.mean()),
                               "sd_diff": float(diff.std(ddof=1)) if len(piv) > 1 else None,
                               "paired_t_p": float(test.pvalue) if test is not None else None}
        if fam == "random":
            pre = g[g.init == "pretrained"].auroc
            scr = g[g.init == "scratch"].auroc
            if len(pre) > 1 and len(scr) > 1:
                paired[f"random_unpaired_descriptors_{desc}"] = {
                    "pretrained_mean": float(pre.mean()), "pretrained_sd": float(pre.std(ddof=1)),
                    "scratch_mean": float(scr.mean()), "scratch_sd": float(scr.std(ddof=1)),
                    "welch_p": float(stats.ttest_ind(pre, scr, equal_var=False).pvalue),
                    "pretrained_cv_percent": float(100 * pre.std(ddof=1) / pre.mean())}
    (ROOT / "summary_paired.json").write_text(json.dumps(paired, indent=2))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
