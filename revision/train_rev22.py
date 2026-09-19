#!/usr/bin/env python3
"""
train_rev22.py — GHSR fine-tuning / from-scratch training for the v22 revision.

Identical to the non-domain-adaptation path of main.py (same DTIDataset,
DataLoader settings, DrugBAN model, Adam optimiser, Trainer and pretrained
weight loading), with two differences that do not change the numerics:
  * featurised molecular graphs are cached after the first epoch (the
    featurisation is deterministic; the host has a single CPU core);
  * the split folder, seed, output directory and the use of BindingDB
    pre-training are command-line arguments, so no existing config or result
    folder is modified.

Usage:
    python revision/train_rev22.py --split rev22_scaffold_s0 --seed 42 --init pretrained
    python revision/train_rev22.py --split random --seed 42 --init scratch
"""
import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import get_cfg_defaults  # noqa: E402
from dataloader import DTIDataset  # noqa: E402
from models import DrugBAN  # noqa: E402
from trainer import Trainer  # noqa: E402
from utils import graph_collate_func, mkdir, set_seed  # noqa: E402

BASE_CFG = "configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml"
PRETRAINED = "models/pretrained/DrugBAN_BiLSTM_BindingDB_epoch94.pth"


class CachedDataset(Dataset):
    """Memoise DTIDataset items; graphs are re-batched (copied) by dgl.batch."""

    def __init__(self, base):
        self.base = base
        self.cache = {}

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        if i not in self.cache:
            self.cache[i] = self.base[i]
        return self.cache[i]


def load_pretrained(model, path, device):
    checkpoint = torch.load(path, map_location=device)
    state = model.state_dict()
    filtered = {k: v for k, v in checkpoint.items() if k not in state or v.shape == state[k].shape}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(f"Loaded pretrained weights from {path} (missing={len(missing)}, unexpected={len(unexpected)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init", choices=["pretrained", "scratch"], required=True)
    ap.add_argument("--out_root", default="result/rev22")
    ap.add_argument("--no_protein_features", action="store_true",
                    help="train with the four physicochemical channels set to zero")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(BASE_CFG)
    suffix = "_featoff" if args.no_protein_features else ""
    out_dir = os.path.join(args.out_root, args.split, f"{args.init}{suffix}_seed{args.seed}")
    if os.path.exists(os.path.join(out_dir, "test_markdowntable.txt")):
        print(f"{out_dir} already finished — skipping (no overwrite)")
        return
    cfg.SOLVER.SEED = args.seed
    cfg.SOLVER.PRETRAINED_MODEL = PRETRAINED if args.init == "pretrained" else ""
    cfg.RESULT.OUTPUT_DIR = out_dir
    set_seed(cfg.SOLVER.SEED)
    mkdir(out_dir)

    folder = os.path.join("datasets", "GPCR_resarch", args.split)
    datasets = {}
    for part in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(folder, f"{part}.csv"))
        base = DTIDataset(df.index.values, df,
                          max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
                          use_features=(cfg.PROTEIN.get("USE_BILSTM", False)
                                        and not args.no_protein_features),
                          use_selfies=False, selfies_vocab=None,
                          max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
                          use_drug_features=cfg.DRUG.get("USE_FEATURES", False))
        datasets[part] = CachedDataset(base)

    params = {"batch_size": cfg.SOLVER.BATCH_SIZE, "shuffle": True,
              "num_workers": cfg.SOLVER.NUM_WORKERS, "drop_last": True,
              "collate_fn": graph_collate_func}
    train_gen = DataLoader(datasets["train"], **params)
    params.update(shuffle=False, drop_last=False)
    val_gen = DataLoader(datasets["val"], **params)
    test_gen = DataLoader(datasets["test"], **params)

    model = DrugBAN(**cfg).to(device)
    if args.init == "pretrained":
        load_pretrained(model, PRETRAINED, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, train_gen, val_gen, test_gen,
                      opt_da=None, discriminator=None, experiment=None, **cfg)
    result = trainer.train()
    record = {"split": args.split, "seed": args.seed, "init": args.init,
              **{k: (float(v) if isinstance(v, (int, float)) else str(v))
                 for k, v in trainer.test_metrics.items()}}
    Path(out_dir, "rev22_test_metrics.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))
    return result


if __name__ == "__main__":
    main()
