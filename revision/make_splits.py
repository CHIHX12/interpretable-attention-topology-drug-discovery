#!/usr/bin/env python3
"""
make_splits.py — leakage-controlled GHSR partitions for the v22 revision.

Creates NEW split folders next to the deposited random split (which is left
untouched so that results can be compared):

  datasets/GPCR_resarch/rev22_compound_s{k}/  compound-disjoint (grouped by
                                               canonical SMILES) 70/10/20
  datasets/GPCR_resarch/rev22_scaffold_s{k}/  Bemis-Murcko scaffold split 70/10/20
  datasets/GPCR_resarch/rev22_nodual_s{k}/    compound-disjoint split after
                                               removing every ligand that carries
                                               both an EC50 and an IC50 record

For every partition (including the deposited `random/` split) the script writes
a JSON report with class counts, exact/canonical SMILES overlap between subsets
and the nearest-neighbour ECFP4 Tanimoto similarity of each test ligand to the
training set.

Usage:
    python revision/make_splits.py            # k = 0..4
    python revision/make_splits.py --n_splits 5 --out_prefix rev22
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DATA_DIR = Path("datasets/GPCR_resarch")
SOURCE = DATA_DIR / "GHSR_training_data.csv"
FRACTIONS = (0.7, 0.1, 0.2)


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def fingerprint(smiles: str):
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=2048)


def assign_groups_random(groups: dict, n_total: int, rng: np.random.Generator):
    """Shuffle whole groups and fill train/val/test in order (group-disjoint)."""
    keys = list(groups)
    rng.shuffle(keys)
    return _fill(keys, groups, n_total)


def assign_groups_scaffold(groups: dict, n_total: int, rng: np.random.Generator):
    """Balanced scaffold split (chemprop-style): scaffold sets larger than half
    the test size go to training first; the remaining sets are shuffled."""
    big, small = [], []
    for key, idx in groups.items():
        (big if len(idx) > FRACTIONS[2] * n_total / 2 else small).append(key)
    rng.shuffle(big)
    rng.shuffle(small)
    return _fill(big + small, groups, n_total)


def _fill(ordered_keys, groups, n_total):
    cap_train = FRACTIONS[0] * n_total
    cap_val = FRACTIONS[1] * n_total
    train, val, test = [], [], []
    for key in ordered_keys:
        idx = groups[key]
        if len(train) + len(idx) <= cap_train:
            train += idx
        elif len(val) + len(idx) <= cap_val:
            val += idx
        else:
            test += idx
    return train, val, test


def overlap_report(parts: dict, fps: dict) -> dict:
    report = {}
    for name, df in parts.items():
        report[f"n_{name}"] = int(len(df))
        report[f"class_counts_{name}"] = {int(k): int(v) for k, v in df.Y.value_counts().items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        exact = set(parts[a].SMILES) & set(parts[b].SMILES)
        canon = set(parts[a].canon) & set(parts[b].canon)
        report[f"exact_smiles_overlap_{a}_{b}"] = len(exact)
        report[f"canonical_smiles_overlap_{a}_{b}"] = len(canon)
    train_fps = [fps[s] for s in parts["train"].canon.unique()]
    nn_sim = [max(DataStructs.BulkTanimotoSimilarity(fps[s], train_fps))
              for s in parts["test"].canon.unique()]
    nn_sim = np.array(nn_sim)
    report["test_to_train_nn_tanimoto"] = {
        "mean": float(nn_sim.mean()), "median": float(np.median(nn_sim)),
        "frac_ge_0.8": float((nn_sim >= 0.8).mean()), "frac_lt_0.4": float((nn_sim < 0.4).mean()),
        "n_test_ligands": int(len(nn_sim)),
    }
    report["n_test_scaffolds_seen_in_train"] = int(
        len(set(parts["test"].scaffold) & set(parts["train"].scaffold)))
    report["n_test_scaffolds"] = int(parts["test"].scaffold.nunique())
    return report


def write_split(name: str, df: pd.DataFrame, idx_sets, fps, columns):
    out = DATA_DIR / name
    if out.exists():
        raise FileExistsError(f"{out} already exists — refusing to overwrite")
    out.mkdir(parents=True)
    parts = {}
    for part, idx in zip(("train", "val", "test"), idx_sets):
        sub = df.loc[sorted(idx)].reset_index(drop=True)
        sub[columns].to_csv(out / f"{part}.csv", index=False)
        parts[part] = sub
    report = overlap_report(parts, fps)
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--out_prefix", default="rev22")
    args = ap.parse_args()

    df = pd.read_csv(SOURCE)
    columns = list(df.columns)
    df["canon"] = df.SMILES.map(canonical)
    df["scaffold"] = df.SMILES.map(scaffold)
    fps = {s: fingerprint(s) for s in df.canon.unique()}

    dual = df.groupby("canon").Y.nunique()
    dual = set(dual[dual > 1].index)
    summary = {
        "n_rows": int(len(df)),
        "n_unique_exact_smiles": int(df.SMILES.nunique()),
        "n_unique_canonical_smiles": int(df.canon.nunique()),
        "n_dual_endpoint_ligands": len(dual),
        "n_scaffolds": int(df.scaffold.nunique()),
        "splits": {},
    }

    # Report for the deposited split (not modified).
    dep = {}
    for part in ("train", "val", "test"):
        sub = pd.read_csv(DATA_DIR / "random" / f"{part}.csv")
        sub["canon"] = sub.SMILES.map(canonical)
        sub["scaffold"] = sub.SMILES.map(scaffold)
        dep[part] = sub
    summary["splits"]["random"] = overlap_report(dep, fps)

    by_canon = defaultdict(list)
    by_scaffold = defaultdict(list)
    for i, row in df.iterrows():
        by_canon[row.canon].append(i)
        by_scaffold[row.scaffold].append(i)

    nodual_df = df[~df.canon.isin(dual)]
    by_canon_nodual = defaultdict(list)
    for i, row in nodual_df.iterrows():
        by_canon_nodual[row.canon].append(i)

    for k in range(args.n_splits):
        rng = np.random.default_rng(k)
        name = f"{args.out_prefix}_compound_s{k}"
        summary["splits"][name] = write_split(
            name, df, assign_groups_random(by_canon, len(df), rng), fps, columns)

        rng = np.random.default_rng(k)
        name = f"{args.out_prefix}_scaffold_s{k}"
        summary["splits"][name] = write_split(
            name, df, assign_groups_scaffold(by_scaffold, len(df), rng), fps, columns)

        rng = np.random.default_rng(k)
        name = f"{args.out_prefix}_nodual_s{k}"
        summary["splits"][name] = write_split(
            name, df, assign_groups_random(by_canon_nodual, len(nodual_df), rng), fps, columns)

    out = DATA_DIR / f"{args.out_prefix}_split_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "splits"}, indent=2))
    for name, rep in summary["splits"].items():
        print(f"{name:22s} n={rep['n_train']}/{rep['n_val']}/{rep['n_test']} "
              f"exact_tr_te={rep['exact_smiles_overlap_train_test']} "
              f"canon_tr_te={rep['canonical_smiles_overlap_train_test']} "
              f"NN-sim median={rep['test_to_train_nn_tanimoto']['median']:.2f} "
              f"test scaffolds seen={rep['n_test_scaffolds_seen_in_train']}/{rep['n_test_scaffolds']}")


if __name__ == "__main__":
    main()
