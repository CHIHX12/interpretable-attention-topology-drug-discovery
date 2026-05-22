"""
Train GHSR model and save a checkpoint at EVERY epoch.

Purpose: checkpoint-level inspection — allows comparing attention patterns
and AUROC across epochs to identify the best model without relying solely
on the final epoch. Complements the main trainer which saves only the
best-val checkpoint and the last epoch.

Usage:
  python train_save_all_epochs.py --max_epoch 50 --output_dir result/DrugBAN_BiLSTM_GHSR_EarlyCkpt
"""
import torch
import torch.nn as nn
import argparse
import os
import copy
import numpy as np
import pandas as pd
from functools import partial
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models import DrugBAN
from configs import get_cfg_defaults
from dataloader import DTIDataset
from utils import set_seed, mkdir, graph_collate_func

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml')
    p.add_argument('--max_epoch', type=int, default=15,
                   help='How many epochs to train (saves checkpoint each epoch)')
    p.add_argument('--output_dir', default='result/DrugBAN_BiLSTM_GHSR_EarlyCkpt')
    return p.parse_args()


def load_data(cfg):
    data_dir = './datasets/GPCR_resarch/random'
    use_features = cfg.PROTEIN.get('USE_BILSTM', False)
    use_drug_bilstm = cfg.DRUG.get('USE_BILSTM', False)

    def make_loader(split, shuffle):
        df = pd.read_csv(os.path.join(data_dir, f'{split}.csv'))
        ids = list(range(len(df)))
        ds = DTIDataset(ids, df,
                        max_drug_nodes=cfg.DRUG.MAX_NODES,
                        max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
                        use_features=use_features,
                        use_selfies=use_drug_bilstm)
        return DataLoader(ds, batch_size=cfg.SOLVER.BATCH_SIZE,
                          shuffle=shuffle, collate_fn=graph_collate_func,
                          num_workers=0, drop_last=False)

    return make_loader('train', True), make_loader('val', False)


def to_device(v_p):
    if isinstance(v_p, tuple):
        return tuple(x.to(device) for x in v_p)
    return v_p.to(device)


def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss()
    for v_d, v_p, labels, _ in loader:
        v_d = v_d.to(device)
        v_p = to_device(v_p)
        labels = labels.to(device)
        optimizer.zero_grad()
        _, _, _, score, _ = model(v_d, v_p)  # train mode: v_d, v_p_enc, f, score, score_da
        loss = criterion(score.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def eval_auroc(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for v_d, v_p, labels, _ in loader:
            v_d = v_d.to(device)
            v_p = to_device(v_p)
            _, _, score, _ = model(v_d, v_p, mode='eval')  # eval mode: v_d, v_p_enc, score, att
            p = torch.sigmoid(score).cpu().squeeze()
            preds.extend(p.tolist() if p.dim() > 0 else [p.item()])
            trues.extend(labels.cpu().squeeze().tolist())
    return roc_auc_score(trues, preds)


def main():
    args = parse_args()
    mkdir(args.output_dir)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    set_seed(cfg.SOLVER.SEED)

    print(f"Device: {device}")
    print(f"Config: {args.cfg}")
    print(f"Output: {args.output_dir}")
    print(f"Max epochs: {args.max_epoch}")
    print(f"Pretrained: {cfg.SOLVER.PRETRAINED_MODEL}")

    # Build model
    model = DrugBAN(**cfg).to(device)
    if cfg.SOLVER.PRETRAINED_MODEL:
        ckpt = torch.load(cfg.SOLVER.PRETRAINED_MODEL, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Loaded pretrained: {cfg.SOLVER.PRETRAINED_MODEL}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR)

    train_loader, val_loader = load_data(cfg)
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val AUROC':>12}")
    print("-" * 35)

    for epoch in range(1, args.max_epoch + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_auroc = eval_auroc(model, val_loader)

        # Save checkpoint at every epoch
        ckpt_path = os.path.join(args.output_dir, f'model_epoch_{epoch:02d}.pth')
        torch.save(model.state_dict(), ckpt_path)

        print(f"{epoch:>6} {train_loss:>12.4f} {val_auroc:>12.4f}  -> saved {ckpt_path}")

    print(f"\nDone. All {args.max_epoch} checkpoints saved to {args.output_dir}/")
    print("Next: run batch_predict on each checkpoint to find AUROC≈0.817")


if __name__ == '__main__':
    main()
