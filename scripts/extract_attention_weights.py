#!/usr/bin/env python3
"""
extractattentionweight Nature Machine Intelligence visualization
multi-taskmodel 12 receptor
"""

import torch
import numpy as np
import pandas as pd
import pickle
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from configs import get_cfg_defaults
from models import DrugBAN
from dataloader import DTIDataset, collate_selfies_fn
from torch.utils.data import DataLoader
from utils import set_seed, graph_collate_func


def extract_attention_weights(model, dataloader, device, config, max_samples=None):
    """
    extractattentionweightpredictionvaluelabel

    Args:
    model: training DrugBAN model
        dataloader: dataload
        device: device (cuda/cpu)
        config: pair
    max_samples: maxsampleNone = 

    Returns:
    results: dict extract
    """
    model.eval()
    use_multitask = config.get("MULTITASK", {}).get("ENABLED", False)

    results = {
        'attentions': [],           # List of attention matrices
        'y_true': [],               # Classification labels
        'y_pred': [],               # Classification predictions
        'z_true': [],               # Regression targets (CNNscore)
        'z_pred': [],               # Regression predictions
        'smiles': [],               # SMILES strings
        'proteins': [],             # Protein sequences
        'drug_lengths': [],         # Actual drug token lengths
        'protein_lengths': []       # Actual protein sequence lengths
    }

    print("\n" + "="*80)
    print("🔍 Extracting Attention Weights")
    print("="*80)
    print(f"Multi-task Learning: {use_multitask}")
    print(f"Device: {device}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")

    total_samples = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Extracting")):
            if max_samples and total_samples >= max_samples:
                break

            # Unpack batch
            v_d, v_p, y, z = batch_data

            # Move to device
            if isinstance(v_d, tuple):
                # SELFIES mode: v_d = (drug_idx, drug_len) or (drug_idx, drug_feat, drug_len)
                v_d = tuple(item.to(device) for item in v_d)
            elif hasattr(v_d, 'to'):
                # Graph mode (DGLGraph) or Tensor
                v_d = v_d.to(device)
            else:
                # Already on device or doesn't need moving
                pass

            if isinstance(v_p, tuple):
                # BiLSTM with features: v_p = (p_idx, p_feat, p_len)
                v_p = tuple(item.to(device) for item in v_p)
            else:
                v_p = v_p.to(device)

            y = y.to(device)
            z = z.to(device)

            # Forward pass in eval mode to get attention
            try:
                forward_output = model(v_d, v_p, mode="eval")

                if use_multitask:
                    v_d_out, v_p_encoded, score, att, reg_output = forward_output
                else:
                    v_d_out, v_p_encoded, score, att = forward_output
                    reg_output = None
            except Exception as e:
                print(f"\n⚠️  Error in batch {batch_idx}: {e}")
                continue

            # Process each sample in batch
            batch_size = att.shape[0]

            for i in range(batch_size):
                if max_samples and total_samples >= max_samples:
                    break

                # Extract attention matrix [drug_tokens, protein_residues]
                att_matrix = att[i].cpu().numpy()

                # Get predictions
                y_pred_prob = torch.sigmoid(score[i]).item()
                y_true_label = y[i].item()
                z_true_val = z[i].item()
                z_pred_val = reg_output[i].item() if reg_output is not None else None

                # Store results
                results['attentions'].append(att_matrix)
                results['y_true'].append(y_true_label)
                results['y_pred'].append(y_pred_prob)
                results['z_true'].append(z_true_val)
                results['z_pred'].append(z_pred_val)

                # Get actual lengths (if available)
                if isinstance(v_d, tuple):
                    drug_len = v_d[-1][i].item()  # Last element is length
                    results['drug_lengths'].append(drug_len)
                elif hasattr(v_d, 'batch_num_nodes'):
                    # DGLGraph mode - get number of nodes for this graph
                    drug_len = v_d.batch_num_nodes()[i].item()
                    results['drug_lengths'].append(drug_len)
                else:
                    results['drug_lengths'].append(att_matrix.shape[0])

                if isinstance(v_p, tuple):
                    protein_len = v_p[-1][i].item()  # Last element is length
                    results['protein_lengths'].append(protein_len)
                else:
                    results['protein_lengths'].append(att_matrix.shape[1])

                # Placeholder for SMILES and protein sequences
                    # (dataindex)
                results['smiles'].append(f"sample_{total_samples}")
                results['proteins'].append(f"protein_{total_samples}")

                total_samples += 1

    print("\n✅ Extraction complete!")
    print(f"   Total samples: {total_samples}")
    print(f"   Attention matrices: {len(results['attentions'])}")

    # statistics
    print("\n📊 Statistics:")
    print(f"   Classification accuracy: {np.mean(np.abs(np.array(results['y_pred']) > 0.5) == np.array(results['y_true'])):.4f}")
    if use_multitask:
        from sklearn.metrics import r2_score
        r2 = r2_score(results['z_true'], results['z_pred'])
        print(f"   Regression R²: {r2:.4f}")

        # attentionmatrixshape
    att_shapes = [att.shape for att in results['attentions'][:10]]
    print(f"   Sample attention shapes: {att_shapes}")

    return results


def save_results(results, output_path):
    """saveextractresult"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n💾 Saved results to: {output_path}")
    print(f"   File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")


def load_model(config_path, model_path, device):
    """loadtrainingmodel"""
    # Load config
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)

    # Initialize model
    model = DrugBAN(**cfg).to(device)

    # Load weights
    print(f"\n📦 Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    print("✅ Model loaded successfully")

    return model, cfg


def create_dataloader(data_path, cfg, batch_size=32, num_workers=0):
    """createdataload"""
    print(f"\n📂 Loading data from: {data_path}")

    # Read data
    df = pd.read_csv(data_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Columns: {df.columns.tolist()}")

    # Check required columns
    required_cols = ['SMILES', 'Protein', 'Y']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create dataset
    use_selfies = cfg.DRUG.get("USE_BILSTM", False)
    use_protein_features = cfg.PROTEIN.get("USE_BILSTM", False)

    print(f"   Drug representation: {'SELFIES (BiLSTM)' if use_selfies else 'Graph (GCN)'}")
    print(f"   Protein representation: {'BiLSTM' if use_protein_features else 'CNN'}")

    # SELFIES vocabulary
    selfies_vocab = None
    if use_selfies:
        # dataload vocabulary
        vocab_path = Path("datasets/selfies_vocab.pkl")
        if vocab_path.exists():
            with open(vocab_path, 'rb') as f:
                selfies_vocab = pickle.load(f)
            print(f"   SELFIES vocab size: {len(selfies_vocab)}")
        else:
            print("   ⚠️  SELFIES vocab not found, will be created from data")

    # Create dataset
    list_IDs = list(range(len(df)))
    dataset = DTIDataset(
        list_IDs=list_IDs,
        df=df,
        max_drug_nodes=cfg.DRUG.MAX_NODES,
        max_protein_length=cfg.PROTEIN.MAX_PROTEIN_LENGTH,
        use_features=use_protein_features,
        use_selfies=use_selfies,
        selfies_vocab=selfies_vocab,
        max_drug_length=cfg.DRUG.MAX_DRUG_LENGTH,
        use_drug_features=cfg.DRUG.get("USE_FEATURES", False)
    )

    # Create dataloader
    # collate function
    if use_selfies:
        collate_fn = collate_selfies_fn
    else:
        # Graph graph_collate_func
        collate_fn = graph_collate_func

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False, # 
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    print(f"   Batches: {len(dataloader)}")

    return dataloader, df


def main():
    parser = argparse.ArgumentParser(description='Extract attention weights for visualization')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV')
    parser.add_argument('--output', type=str, required=True, help='Output pickle file')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to extract')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        args.device = 'cpu'

    device = torch.device(args.device)

    # Load model
    model, cfg = load_model(args.config, args.model, device)

    # Create dataloader
    dataloader, df = create_dataloader(args.data, cfg, args.batch_size, args.num_workers)

    # Extract attention weights
    results = extract_attention_weights(model, dataloader, device, cfg, args.max_samples)

    # Add metadata
    results['metadata'] = {
        'config_path': args.config,
        'model_path': args.model,
        'data_path': args.data,
        'total_samples': len(results['attentions']),
        'use_multitask': cfg.get("MULTITASK", {}).get("ENABLED", False),
        'use_selfies': cfg.DRUG.get("USE_BILSTM", False),
        'drug_representation': 'SELFIES' if cfg.DRUG.get("USE_BILSTM", False) else 'Graph',
        'protein_representation': 'BiLSTM' if cfg.PROTEIN.get("USE_BILSTM", False) else 'CNN'
    }

    # Save results
    save_results(results, args.output)

    print("\n" + "="*80)
    print("🎉 All done!")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Use the attention data for visualization:")
    print(f"   python scripts/plot_attention_heatmap_cases.py --input {args.output}")
    print(f"2. Generate all figures:")
    print(f"   python generate_all_nature_mi_figures.py")


if __name__ == "__main__":
    main()
