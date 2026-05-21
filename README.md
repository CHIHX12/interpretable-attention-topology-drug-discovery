# An Interpretable Attention-Topology Framework Decouples Affinity and Efficacy in Structure-Free Drug Discovery

**Chih-Yang Cheng<sup>1\*</sup>, Yi-Huan Wu<sup>2\*</sup>, and Feng-Yin Li<sup>1\*</sup>**

<sup>1</sup>Department of Chemistry, National Chung Hsing University, Taichung 402, Taiwan.  
<sup>2</sup>Department of Chemistry, R.O.C. Military Academy, Kaohsiung, Taiwan.

> **Built upon**: [DrugBAN](https://doi.org/10.1038/s42256-022-00605-1) (Bai et al., *Nature Machine Intelligence* 2023)

---

## Abstract

While deep learning predicts drug binding affinities, most models remain black boxes unable to deduce dynamic allosteric mechanisms from sequence data. We present an interpretable deep transfer learning framework that decouples binding affinity from signaling efficacy using a bilinear attention network. Validating this structure-free paradigm on the growth hormone secretagogue receptor (GHSR) with 1,539 ligands, our model achieves robust predictive stability (AUROC = 0.96) and autonomously infers structural dynamics without crystallographic priors. Network topology analysis reveals a fundamental divergence in ligand recognition: agonists utilize a precise, modular architecture driven by Ser125, whereas antagonists form a synergistic web around hydrophobic hubs Phe286 and Pro278. Structural mapping confirms distinct physical mechanisms: agonists trigger remote activation via a Glu124-Arg283 salt bridge outside the primary pocket, while antagonists physically jam the deep hydrophobic crevasse. This spatial decoupling of binding contacts and functional triggering sites demonstrates that artificial intelligence can transcend simple pattern matching to predict allosteric machinery, providing a computational paradigm for navigating orphan receptor efficacy and designing functionally selective therapeutics.

---

## Overview

This repository provides the implementation of the **Interpretable Attention-Topology Framework** described in the paper above. It extends the DrugBAN framework with **BiLSTM protein/drug encoders** and applies **transfer learning** to identify amino acid residues critical for GPCR **activation** vs. **inhibition** — using only sequence information, without 3D structures.

### Key Contributions over Original DrugBAN

| Feature | Original DrugBAN | This Work |
|---------|-----------------|-----------|
| Protein encoder | CNN | **BiLSTM** (bidirectional, physicochemical features) |
| Drug encoder | GCN (graph) | GCN or **BiLSTM** (SELFIES sequences) |
| Prediction task | Binary binding | **3-class** (Active / Intermediate / Inactive) |
| Training strategy | From scratch | **Transfer learning** (BindingDB → GPCR) |
| Multi-task | No | **Yes** (binding class + docking score regression) |
| Interpretability | Drug-protein attention | **Class-differential attention** (activation vs inhibition residues) |
| Validation | Random split | **LORO** (Leave-One-Receptor-Out) |

---

## Framework Architecture

```
Drug (SMILES/SELFIES)                Protein Sequence
        │                                    │
  GCN or BiLSTM                      BiLSTM + Physicochemical Features
  [B, N_atoms, 128]                   [B, L_res, 512]
        │                                    │
        └──────── BAN (Bilinear Attention) ──┘
                          │
                   [B, 256] interaction features
                          │
              ┌───────────┴───────────┐
         Classifier                Regressor (optional)
    3-class (A/I/N)          CNNScore prediction (multi-task)
              │
    Class-differential
      Attention Analysis
              │
    Key residues for          Key residues for
      ACTIVATION               INHIBITION
```

---

## Scientific Application: GPCR Residue Identification

The model identifies which amino acid residues in GPCR proteins are selectively important for:
- **Activation** (agonist binding): class 1 preferred residues
- **Inhibition** (antagonist/inverse agonist binding): class 0 preferred residues
- **Constitutive** (always important): residues important for both

This is achieved by comparing attention weights across drug-protein pairs in different activity classes — no 3D structure required.

---

## Installation

```bash
# Create conda environment
conda create -n drugban-bilstm python=3.8
conda activate drugban-bilstm

# Install PyTorch (adjust CUDA version as needed)
conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install DGL (match CUDA version)
pip install dgl -f https://data.dgl.ai/wheels/cu117/repo.html

# Install remaining dependencies
pip install -r requirements.txt
```

> **GPU Requirements**: CUDA-capable GPU with ≥ 8 GB VRAM recommended for full dataset training. CPU training is supported but slow.

---

## Data Preparation

### Pre-training Data (BindingDB)

Download from the [original DrugBAN repository](https://github.com/peizhenbai/DrugBAN):

```bash
# Place downloaded data in:
datasets/bindingdb/random/   # train.csv, val.csv, test.csv
datasets/bindingdb/cluster/  # for cross-domain experiments
```

### GPCR Fine-tuning Data (GHSR)

Prepare your GPCR dataset with the following CSV format:

```
SMILES,Protein,Y,cnnscore
CC(C)CC1=CC=CC=C1NC(=O)...,MGENSSPALLPETGEDKFPAMPL...,2,0.823
```

- `SMILES`: Drug SMILES string
- `Protein`: Protein amino acid sequence
- `Y`: Activity class (0=Inactive, 1=Intermediate, 2=Active)
- `cnnscore`: Docking score (for multi-task learning; set to 0 if unavailable)

Then split the data:

```bash
python split_ghsr_data.py --input datasets/GPCR_resarch/GHSR_training_data.csv \
                          --output datasets/GPCR_resarch/random \
                          --stratify
```

---

## Reproducing Results

### Step 1: Pre-train on BindingDB (binary binding)

```bash
python main.py \
    --cfg configs/DrugBAN_BiLSTM.yaml \
    --data bindingdb \
    --split random
```

This saves the best checkpoint to `result/DrugBAN_BiLSTM/best_model_epoch_XX.pth`.

### Step 2: Transfer Learning to GPCR (3-class, multi-task)

Edit `configs/DrugBAN_BiLSTM_CNNScore_Multitask_3Class.yaml` to set the pretrained model path:

```yaml
SOLVER:
  PRETRAINED_MODEL: "./result/DrugBAN_BiLSTM/best_model_epoch_94.pth"
```

Then run fine-tuning:

```bash
python main.py \
    --cfg configs/DrugBAN_BiLSTM_CNNScore_Multitask_3Class.yaml \
    --data GPCR_resarch \
    --split random
```

Or use the provided script:

```bash
bash run_ghsr_transfer_learning.sh
```

### Step 3: Leave-One-Receptor-Out (LORO) Validation

```bash
# Create LORO folds (one fold per receptor)
python scripts/create_loro_splits.py

# Train all folds
bash run_loro_transfer_learning.sh

# Evaluate LORO results
python scripts/evaluate_loro_results.py
```

### Step 4: Extract Attention Weights

```bash
python scripts/extract_attention_weights.py \
    --cfg configs/DrugBAN_BiLSTM_CNNScore_Multitask_3Class.yaml \
    --model_path result/LORO_3Class/best_model.pth \
    --data datasets/GPCR_resarch/random/test.csv \
    --output_dir datasets/GPCR_resarch/attention_results
```

### Step 5: Identify Key Residues (Class-Differential Analysis)

```bash
# Identify residues preferentially activated vs inhibited
python analyze_class_attention_difference_en.py \
    --attention_dir datasets/GPCR_resarch/attention_results \
    --output_dir datasets/GPCR_resarch/consensus_results

# Generate PyMOL visualization scripts
python scripts/plot_class_differential_attention.py \
    --consensus_dir datasets/GPCR_resarch/consensus_results \
    --output_dir pymol_output
```

---

## Configuration Guide

Key configuration files in `configs/`:

| Config File | Purpose |
|-------------|---------|
| `DrugBAN.yaml` | Baseline DrugBAN (CNN encoder, binary) |
| `DrugBAN_BiLSTM.yaml` | BiLSTM protein encoder, binary |
| `DrugBAN_BiLSTM_CNNScore_Multitask_3Class.yaml` | **Main GPCR config** (3-class + docking score) |
| `DrugBAN_BiLSTM_CNNScore_Multitask_Transfer.yaml` | Transfer learning from BindingDB |
| `DrugBAN_BiLSTM_CNNScore_Multitask_LORO.yaml` | LORO cross-validation |
| `DrugBAN_DA.yaml` | Domain adaptation (CDAN) |

### Key Parameters

```yaml
PROTEIN:
  USE_BILSTM: True          # Enable BiLSTM encoder (vs CNN)
  LSTM_HIDDEN_DIM: 256      # BiLSTM hidden size (output = 512 bidirectional)
  LSTM_NUM_LAYERS: 2
  FEATURE_DIM: 4            # Physicochemical features: hydrophobicity, volume, charge, polarity
  MAX_PROTEIN_LENGTH: 1200

DECODER:
  BINARY: 3                 # 1=binary, 2=binary logits, 3=3-class

MULTITASK:
  ENABLED: True             # Joint classification + regression
  ALPHA: 0.5                # Classification loss weight
  BETA: 0.5                 # Regression (docking score) loss weight

SOLVER:
  PRETRAINED_MODEL: "./result/DrugBAN_BiLSTM/best_model_epoch_94.pth"
```

---

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `scripts/extract_attention_weights.py` | Extract BAN attention for all drug-protein pairs |
| `analyze_class_attention_difference_en.py` | Class-differential attention: active vs inactive |
| `analyze_constitutive_variable_residues.py` | Identify constitutive vs variable residues |
| `aggregate_attention_analysis.py` | Aggregate attention across multiple drugs per protein |
| `scripts/plot_class_differential_attention.py` | Plot attention heatmaps |
| `scripts/statistical_tests.py` | Statistical significance testing |
| `scripts/evaluate_loro_results.py` | LORO cross-validation performance report |

---

## PyMOL Visualization

After running the analysis pipeline, PyMOL scripts are generated to visualize key residues on the protein 3D structure:

```bash
# Open in PyMOL:
pymol pymol_class0_inactive_preferred.pml  # Residues important for inhibition
pymol pymol_class1_active_preferred.pml    # Residues important for activation
```

Color scheme:
- **Red**: Class 1 (Active) preferred residues
- **Blue**: Class 0 (Inactive) preferred residues
- **Purple**: Constitutive residues (important for both)
- **Cyan**: Ligand structure

See `PYMOL_VISUALIZATION_GUIDE.md` for detailed instructions.

---

## Repository Structure

```
DrugBAN-BiLSTM/
├── main.py                    # Training entry point
├── models.py                  # Model definitions (DrugBAN, BiLSTM encoders)
├── ban.py                     # Bilinear Attention Network layer
├── trainer.py                 # Training loop
├── dataloader.py              # Dataset and data loading
├── configs.py                 # Config system (yacs)
├── utils.py                   # Utilities
├── domain_adaptator.py        # Domain adaptation discriminator
│
├── configs/                   # YAML configuration files
│   ├── DrugBAN.yaml           # Baseline
│   ├── DrugBAN_BiLSTM.yaml    # BiLSTM encoder
│   └── DrugBAN_BiLSTM_CNNScore_Multitask_3Class.yaml  # Main GPCR config
│
├── scripts/                   # Analysis and evaluation scripts
│   ├── extract_attention_weights.py
│   ├── plot_class_differential_attention.py
│   ├── evaluate_loro_results.py
│   └── statistical_tests.py
│
├── datasets/                  # Data directory (not tracked by git)
│   ├── bindingdb/             # Pre-training data
│   └── GPCR_resarch/          # GPCR fine-tuning data
│       └── GHSR_training_data.csv
│
├── analyze_class_attention_difference_en.py  # Key residue analysis
├── analyze_constitutive_variable_residues.py
├── aggregate_attention_analysis.py
│
├── run_ghsr_transfer_learning.sh   # Transfer learning pipeline
├── run_loro_transfer_learning.sh   # LORO validation pipeline
│
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Biological Interpretation

The attention mechanism in BAN provides direct biological interpretability:

1. **Training**: The model learns which protein residues are attended when drugs bind with high vs. low affinity
2. **Aggregation**: Attention weights are averaged across all drug-protein pairs in each activity class
3. **Differential analysis**: Residues showing high attention in active pairs but low in inactive pairs (or vice versa) are candidates for class-specific roles
4. **Constitutive residues**: Residues with uniformly high attention across all classes form the core binding scaffold

This approach has recovered known binding site residues in GPCR structures with up to 33% precision using only sequence information.

---

## Citation

If you use this work, please cite both the original DrugBAN paper and this repository:

**Original DrugBAN**:
```bibtex
@article{bai2023drugban,
  title   = {Interpretable bilinear attention network with domain adaptation improves drug-target prediction},
  author  = {Peizhen Bai and Filip Miljkovi{\'c} and Bino John and Haiping Lu},
  journal = {Nature Machine Intelligence},
  year    = {2023},
  doi     = {10.1038/s42256-022-00605-1}
}
```

**This Work**:
```bibtex
@article{cheng2026attention_topology,
  title   = {An Interpretable Attention-Topology Framework Decouples Affinity and Efficacy
             in Structure-Free Drug Discovery},
  author  = {Chih-Yang Cheng and Yi-Huan Wu and Feng-Yin Li},
  year    = {2026},
  url     = {https://github.com/CHIHX12/interpretable-attention-topology-drug-discovery}
}
```

---

## References

1. Bai et al. (2023). Interpretable bilinear attention network with domain adaptation improves drug-target prediction. *Nature Machine Intelligence*.
2. Liu et al. (2007). BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities. *Nucleic Acids Research*.
3. Huang et al. (2021). MolTrans: Molecular Interaction Transformer for drug-target interaction prediction. *Bioinformatics*.
4. Kim et al. (2018). Bilinear attention networks. *NeurIPS*.
