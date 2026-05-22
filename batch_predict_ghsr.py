#!/usr/bin/env python3
"""
Batch Prediction for GHSR Drugs with Attention Extraction.

Pipeline:
  1. Load a trained DrugBAN_BiLSTM model
  2. Run inference on all 1,539 GHSR drug–protein pairs
  3. Extract BAN attention maps and aggregate to per-residue scores
  4. Save attention matrix + predictions for downstream analysis

Attention aggregation
---------------------
Raw BAN attention shape: [batch, heads, N_drug_atoms, L_protein]

  Step 1 – average over attention heads  → [batch, N_drug_atoms, L_protein]
  Step 2 – mean over drug atoms          → [batch, L_protein]

Using *mean* (not max) over drug atoms is required to preserve
class-differential signal at residues such as Glu124 and Ser125.
Max-pooling collapses per-sample variance and eliminates these signals.

Outputs:
  <output_dir>/GHSR_attention_scores.npz   – attention matrix [N, L_protein]
  <output_dir>/GHSR_predictions.csv        – prediction scores and labels
  <output_dir>/GHSR_prediction_stats.json  – AUROC / accuracy summary
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

# Import project modules
from configs import get_cfg_defaults
from dataloader import DTIDataset, collate_selfies_fn
from models import DrugBAN
from utils import set_seed, graph_collate_func, build_selfies_vocab

def parse_args():
    parser = argparse.ArgumentParser(description="Batch predict GHSR drugs and extract attention scores")
    parser.add_argument('--config', type=str, default='configs/DrugBAN_BiLSTM.yaml',
                        help='Path to config file')
    parser.add_argument('--data_file', type=str,
                        default='datasets/GPCR_resarch/GHSR_training_data.csv',
                        help='Path to GHSR data CSV')
    parser.add_argument('--model_path', type=str,
                        default='result/DrugBAN_BiLSTM/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str,
                        default='datasets/GPCR_resarch/attention_results',
                        help='Output directory for attention scores')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for prediction')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()

def load_model(config_path, model_path, device):
    """trainingmodel"""
    print(f"📦 file: {config_path}")
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.freeze()

    print(f"🏗️ model...")
    model = DrugBAN(**cfg).to(device)

    print(f"⚙️ modelweight: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    print(f"✅ modelsuccess")
    return model, cfg

def create_dataset(data_file, cfg):
    """createdata"""
    print(f"📊 readdata: {data_file}")
    df = pd.read_csv(data_file)

    print(f" drug: {len(df)}")
    print(f" active (Y=1): {len(df[df['Y']==1])}")
    print(f" active (Y=0): {len(df[df['Y']==0])}")

    # proteinsequencelength
    protein_seq = df['Protein'].iloc[0]
    protein_len = len(protein_seq)
    print(f"   proteinlength: {protein_len} aa")

    # SELFIESBiLSTM
    use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
    use_drug_features = cfg.DRUG.get("USE_FEATURES", False)
    max_drug_length = cfg.DRUG.get("MAX_DRUG_LENGTH", 200)
    use_protein_features = cfg.PROTEIN.get("USE_BILSTM", False)  # BiLSTM uses physicochemical features

    selfies_vocab = None
    if use_drug_bilstm:
        print(f" 🔧 SELFIES vocabulary...")
        smiles_list = df['SMILES'].tolist()
        selfies_vocab = build_selfies_vocab(smiles_list, max_vocab_size=cfg.DRUG.get("VOCAB_SIZE", 100))
        print(f"   ✓ SELFIES vocabulary size: {len(selfies_vocab)} tokens")

    print(f"   Protein features (physicochemical): {'ON' if use_protein_features else 'OFF'}")

    dataset = DTIDataset(
        df.index.values,
        df,
        use_features=use_protein_features,
        use_selfies=use_drug_bilstm,
        selfies_vocab=selfies_vocab,
        max_drug_length=max_drug_length,
        use_drug_features=use_drug_features
    )

    return dataset, df, protein_len

def extract_attention_scores(model, dataloader, device, protein_len, use_drug_bilstm=False):
    """
    extractattention

    Returns:
        attention_scores: list of numpy arrays, each [protein_len]
        predictions: list of prediction scores
        labels: list of ground truth labels
        drug_ids: list of drug identifiers
    """
    attention_scores = []
    predictions = []
    labels = []
    drug_ids = []

    print(f"\n🔬 startprediction...")

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Unpack batch data — dataloader returns (v_d, v_p, label, z) when
            # physicochemical features are enabled (z = pvalue, ignored here)
            if len(batch_data) == 4:
                v_d, v_p, label, _ = batch_data
            else:
                v_d, v_p, label = batch_data

            # Move to device
            v_d = v_d.to(device) if not isinstance(v_d, tuple) else tuple(x.to(device) for x in v_d)
            v_p = v_p.to(device) if not isinstance(v_p, tuple) else tuple(x.to(device) for x in v_p)
            label = label.to(device)

            # Forward pass in eval mode to get attention
            # Returns: v_d, v_p_encoded, score, att
            _, _, score, att = model(v_d, v_p, mode="eval")

            # Process outputs
            batch_size = score.size(0)

            # Aggregate attention: [batch, heads, drug_len, protein_len] -> [batch, protein_len]
            # att shape: [batch, heads, drug_len, protein_len]
            # Strategy: average over heads, then mean over drug atoms
            # (mean preserves E124/S125 active-preferred signal; max collapses it)
            if len(att.shape) == 4:
                att_avg_heads = att.mean(dim=1)       # [batch, drug_len, protein_len]
                att_protein = att_avg_heads.mean(dim=1)  # [batch, protein_len]
            elif len(att.shape) == 3:
                att_protein = att.mean(dim=1)
            else:
                att_protein = att

            for i in range(batch_size):
                # Get attention for this sample
                # CRITICAL: Only take actual protein length, discard padding
                att_i = att_protein[i, :protein_len].cpu().numpy()  # [protein_len] - only actual protein, not padding

                # Get prediction score
                if score.size(1) == 2:
                    # Binary classification: take positive class probability
                    pred_i = torch.softmax(score[i], dim=0)[1].item()
                else:
                    # Single output
                    pred_i = torch.sigmoid(score[i]).item()

                # Get label
                label_i = label[i].item()

                # Store results
                attention_scores.append(att_i)
                predictions.append(pred_i)
                labels.append(label_i)
                drug_ids.append(batch_idx * dataloader.batch_size + i)

                print(f"✅ doneprocessing {len(attention_scores)} drug")

    return attention_scores, predictions, labels, drug_ids

def save_results(attention_scores, predictions, labels, drug_ids, df, protein_len, output_dir):
    """savepredictionresultattention"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n💾 saveresult: {output_dir}")

    # 1. saveattentionmatrix
    attention_file = output_path / "GHSR_attention_scores.npz"
    attention_matrix = np.array(attention_scores)  # [n_drugs, protein_len]

    # Validate attention matrix shape
    expected_shape = (len(drug_ids), protein_len)
    if attention_matrix.shape != expected_shape:
        raise ValueError(
            f"Attention matrix shape mismatch!\n"
            f"  Expected: {expected_shape} [n_drugs={len(drug_ids)}, protein_len={protein_len}]\n"
            f"  Got: {attention_matrix.shape}\n"
            f"  This usually means padding was not removed correctly."
        )

    np.savez_compressed(
        attention_file,
        attention_scores=attention_matrix,
        drug_ids=np.array(drug_ids)
    )
    print(f" ✓ attentionmatrix: {attention_file}")
    print(f"     shape: {attention_matrix.shape} [n_drugs={len(drug_ids)}, protein_len={protein_len}]")
    print(f" ✓ : paddingactualproteinlength")

    # 2. savepredictionresult
    predictions_df = pd.DataFrame({
        'Drug_ID': drug_ids,
        'SMILES': [df.iloc[i]['SMILES'] if i < len(df) else '' for i in drug_ids],
        'True_Label': labels,
        'Predicted_Score': predictions,
        'Predicted_Class': [1 if p >= 0.5 else 0 for p in predictions]
    })

    predictions_file = output_path / "GHSR_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"   ✓ predictionresult: {predictions_file}")

    # 3. computeprediction
    from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score

    y_true = np.array(labels)
    y_pred = np.array([1 if p >= 0.5 else 0 for p in predictions])
    y_score = np.array(predictions)

    accuracy = accuracy_score(y_true, y_pred)
    try:
        auroc = roc_auc_score(y_true, y_score)
    except:
        auroc = 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    print(f"\n📊 prediction:")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   AUROC:     {auroc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")

    # 4. savestatistics
    stats = {
        'n_drugs': len(drug_ids),
        'protein_length': protein_len,
        'n_active': int(sum(labels)),
        'n_inactive': int(len(labels) - sum(labels)),
        'performance': {
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'precision': float(precision),
            'recall': float(recall)
        },
        'attention_stats': {
            'mean': float(attention_matrix.mean()),
            'std': float(attention_matrix.std()),
            'min': float(attention_matrix.min()),
            'max': float(attention_matrix.max())
        }
    }

    stats_file = output_path / "GHSR_prediction_stats.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
        print(f" ✓ statistics: {stats_file}")

    # 5. drugsaveattentionfilevisualization
    individual_dir = output_path / "individual_attentions"
    individual_dir.mkdir(exist_ok=True)

    print(f"\n💾 saveattentionfile: {individual_dir}/")
    for i, (att, drug_id, pred, label) in enumerate(zip(attention_scores, drug_ids, predictions, labels)):
            if i < 10 or i % 100 == 0:  # save sample files
                drug_file = individual_dir / f"drug_{drug_id:04d}_att.txt"
                with open(drug_file, 'w') as f:
                    f.write(f"# Drug ID: {drug_id}\n")
                    f.write(f"# True Label: {label}\n")
                    f.write(f"# Predicted Score: {pred:.4f}\n")
                    f.write(f"# Attention scores (length: {len(att)})\n")
                    for pos, score in enumerate(att):
                        f.write(f"{pos}\t{score:.6f}\n")

    print(f" Saved individual attention files (first 10 + every 100th)")

    return stats

def main():
    args = parse_args()

    print("="*80)
    print("🧬 GHSR predictionattentionextract")
    print("   Growth Hormone Secretagogue Receptor - Batch Attention Extraction")
    print("="*80)
    print()

    # file
    if not os.path.exists(args.config):
        print(f"❌ file: {args.config}")
        sys.exit(1)

    if not os.path.exists(args.data_file):
        print(f"❌ datafile: {args.data_file}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"❌ modelfile: {args.model_path}")
        sys.exit(1)

        # device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"🖥️  device: {device}")
    print()

    # model
    model, cfg = load_model(args.config, args.model_path, device)
    print()

    # createdata
    dataset, df, protein_len = create_dataset(args.data_file, cfg)
    print()

    # type
    use_drug_bilstm = cfg.DRUG.get("USE_BILSTM", False)
    print(f"🔧 :")
    print(f"   Drug Encoder: {'BiLSTM (SELFIES)' if use_drug_bilstm else 'GCN (Graph)'}")
    print(f"   Protein Encoder: {'BiLSTM' if cfg.PROTEIN.get('USE_BILSTM', False) else 'CNN'}")
    print(f"   Batch Size: {args.batch_size}")
    print()

    # create DataLoader
    if use_drug_bilstm:
        collate_fn = collate_selfies_fn
    else:
        collate_fn = graph_collate_func

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0 # 
    )

    # extractattention
    attention_scores, predictions, labels, drug_ids = extract_attention_scores(
        model, dataloader, device, protein_len, use_drug_bilstm
    )

    # saveresult
    stats = save_results(
        attention_scores, predictions, labels, drug_ids,
        df, protein_len, args.output_dir
    )

    print("\n" + "="*80)
    print("✅ predictiondone")
    print("="*80)
    print()
    print(f"📁 outputdirectory: {args.output_dir}")
    print(f"🔬 processingdrug: {stats['n_drugs']}")
    print(f"💊 proteinlength: {stats['protein_length']} aa")
    print()
    print("📝 rowconsensusanalysis")
    print()
    print("   python consensus_analysis_ghsr.py \\")
    print(f"       --attention_file {args.output_dir}/GHSR_attention_scores.npz \\")
    print(f"       --output_dir {args.output_dir}/consensus \\")
    print(f"       --protein_length {protein_len}")
    print()

if __name__ == "__main__":
    main()
