#!/usr/bin/env python3
"""
extract_attention_masked.py — per-residue BAN attention with the padding atoms removed.

DrugBAN pads every molecular graph to DRUG.MAX_NODES (290) with virtual nodes
whose input features are identical for all ligands, so any aggregation that
includes them mixes a ligand-independent term into the per-residue score
(and, for the mean, a term proportional to the number of real atoms).
This script aggregates over real atoms only, and also stores the
virtual-node-only aggregate so that the size of the artefact can be quantified.

For a batch, raw attention has shape [B, heads, 290, L_protein]; heads are
averaged first, then real atoms are aggregated by mean (primary) and by max
(secondary). Saved arrays cover the full 523-residue input sequence, whose
first 365 positions are GHSR residues 2-366 (position index + 2 = GHSR number).

Output (.npz):
    masked_mean, masked_max, virtual_mean, unmasked_mean  [N, L]
    pred, label, n_atoms, row_index                       [N]

Usage:
    python revision/extract_attention_masked.py \
        --model models/finetuned/DrugBAN_BiLSTM_GHSR_epoch36.pth \
        --out result/rev22/attention/epoch36_full.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs import get_cfg_defaults  # noqa: E402
from dataloader import DTIDataset  # noqa: E402
from models import DrugBAN  # noqa: E402
from utils import graph_collate_func  # noqa: E402

BASE_CFG = "configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml"
DEFAULT_DATA = "datasets/GPCR_resarch/GHSR_training_data.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default=BASE_CFG)
    ap.add_argument("--data_file", default=DEFAULT_DATA)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_protein_features", action="store_true",
                    help="feed zeros in the four physicochemical channels, reproducing the "
                         "December 2025 extraction, where create_dataset did not pass "
                         "use_features to DTIDataset")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"{out} exists — refusing to overwrite")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg.freeze()

    model = DrugBAN(**cfg).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    df = pd.read_csv(args.data_file)
    protein_len = len(df.Protein.iloc[0])
    dataset = DTIDataset(df.index.values, df,
                         max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
                         use_features=(cfg.PROTEIN.get("USE_BILSTM", False)
                                       and not args.no_protein_features),
                         use_selfies=False, selfies_vocab=None,
                         max_drug_length=cfg.DRUG.get("MAX_DRUG_LENGTH", 200),
                         use_drug_features=cfg.DRUG.get("USE_FEATURES", False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, drop_last=False, collate_fn=graph_collate_func)

    masked_mean, masked_max, virtual_mean, unmasked_mean, unmasked_max = [], [], [], [], []
    preds, labels, n_atoms = [], [], []

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            v_d, v_p, label = batch[0], batch[1], batch[2]
            v_d = v_d.to(device)
            v_p = tuple(x.to(device) for x in v_p) if isinstance(v_p, tuple) else v_p.to(device)

            # virtual-node flag is the last input feature; read it before the
            # model pops 'h' from the batched graph
            flag = v_d.ndata["h"][:, -1].view(-1, cfg.DRUG.MAX_NODES)  # [B, 290]
            real = (flag == 0)

            _, _, score, att = model(v_d, v_p, mode="eval")
            att = att.mean(dim=1)                       # [B, 290, L] (heads averaged)
            att = att[:, :, :protein_len]
            m = real.unsqueeze(-1).float()
            n_real = m.sum(dim=1)                       # [B, 1]

            masked_mean.append(((att * m).sum(1) / n_real).cpu().numpy())
            masked_max.append(att.masked_fill(~real.unsqueeze(-1), float("-inf")).max(1)[0].cpu().numpy())
            n_virtual = (1 - m).sum(dim=1)
            vm = (att * (1 - m)).sum(1) / torch.clamp(n_virtual, min=1)
            virtual_mean.append(vm.cpu().numpy())
            unmasked_mean.append(att.mean(1).cpu().numpy())
            # aggregation used before commit 6917a50: max over all 290 rows,
            # i.e. real atoms and padding together
            unmasked_max.append(att.max(1)[0].cpu().numpy())

            preds.append(torch.sigmoid(score).view(-1).cpu().numpy()
                         if score.size(1) == 1 else
                         torch.softmax(score, dim=1)[:, 1].cpu().numpy())
            labels.append(label.view(-1).numpy())
            n_atoms.append(n_real.view(-1).cpu().numpy())
            if bi % 10 == 0:
                print(f"batch {bi}/{len(loader)}", flush=True)

    np.savez_compressed(
        out,
        masked_mean=np.concatenate(masked_mean), masked_max=np.concatenate(masked_max),
        virtual_mean=np.concatenate(virtual_mean), unmasked_mean=np.concatenate(unmasked_mean),
        unmasked_max=np.concatenate(unmasked_max),
        pred=np.concatenate(preds), label=np.concatenate(labels),
        n_atoms=np.concatenate(n_atoms), row_index=df.index.values,
        model=str(args.model), data_file=str(args.data_file), protein_len=protein_len)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
