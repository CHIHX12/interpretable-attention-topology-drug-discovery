#!/usr/bin/env python3
"""
ligand_baselines.py — ligand-only reference models for the GHSR task (v22 revision).

Models (ECFP4, 2048 bits, radius 2):
  * majority class
  * 1-nearest-neighbour (Tanimoto)
  * logistic regression (L2, C=1)
  * random forest (500 trees)

Every model sees only the ligand; the receptor is identical for all rows, so
these baselines measure how much of the agonist/antagonist signal is
recoverable from ligand chemistry alone. Models are fitted on train.csv and
evaluated on test.csv of each split folder.

Output: result/rev22/ligand_baselines.csv (never overwritten; a new file with
a timestamp suffix is written if it already exists).

Usage:
    python revision/ligand_baselines.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

RDLogger.DisableLog("rdApp.*")
DATA_DIR = Path("datasets/GPCR_resarch")
OUT = Path("result/rev22/ligand_baselines.csv")
SPLITS = ["random"] + [f"rev22_{kind}_s{k}" for kind in ("compound", "scaffold", "nodual") for k in range(5)]


def ecfp(smiles_list):
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048) for s in smiles_list]
    arr = np.zeros((len(fps), 2048), dtype=np.uint8)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return fps, arr


def nn_scores(train_fps, train_y, test_fps):
    """P(agonist) = label of the most similar training ligand (ties averaged)."""
    scores = []
    for fp in test_fps:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        best = sims == sims.max()
        scores.append(train_y[best].mean())
    return np.array(scores)


def evaluate(split):
    tr = pd.read_csv(DATA_DIR / split / "train.csv")
    te = pd.read_csv(DATA_DIR / split / "test.csv")
    tr_fps, Xtr = ecfp(tr.SMILES)
    te_fps, Xte = ecfp(te.SMILES)
    ytr, yte = tr.Y.values, te.Y.values
    preds = {
        "majority": np.full(len(yte), ytr.mean()),
        "1NN_tanimoto": nn_scores(tr_fps, ytr, te_fps),
        "logreg_ecfp4": LogisticRegression(C=1.0, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1],
        "rf_ecfp4": RandomForestClassifier(n_estimators=500, random_state=0, n_jobs=1)
        .fit(Xtr, ytr).predict_proba(Xte)[:, 1],
    }
    rows = []
    for name, p in preds.items():
        auroc = roc_auc_score(yte, p) if len(np.unique(p)) > 1 else 0.5
        rows.append({"split": split, "model": name, "n_train": len(tr), "n_test": len(te),
                     "auroc": auroc, "auprc": average_precision_score(yte, p)})
    return rows


def main():
    rows = []
    for split in SPLITS:
        t0 = time.time()
        rows += evaluate(split)
        print(f"{split}: done in {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = OUT if not OUT.exists() else OUT.with_name(f"ligand_baselines_{int(time.time())}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    df["family"] = df.split.str.replace(r"_s\d$", "", regex=True)
    print(df.groupby(["family", "model"])[["auroc", "auprc"]].agg(["mean", "std"]).round(3).to_string())
    print(f"saved {out}")


if __name__ == "__main__":
    main()
