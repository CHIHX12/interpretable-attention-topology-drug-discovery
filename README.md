# TEMA-ENM: An Interpretable Attention-Topology Framework Decouples Affinity and Efficacy from Sequence Alone

**Chih-Yang Cheng<sup>1\*</sup>, Yi-Huan Wu<sup>2\*</sup>, and Feng-Yin Li<sup>1\*</sup>**

<sup>1</sup>Department of Chemistry, National Chung Hsing University, Taichung 402, Taiwan.  
<sup>2</sup>Department of Chemistry, R.O.C. Military Academy, Kaohsiung, Taiwan.

> **Base model**: [BiLSTM-Powered Bilinear Attention for Protein–Ligand Prediction](https://doi.org/10.64898/2026.05.10.724184)
> (Cheng, Chen, Li & Re, *bioRxiv* 2026) — TEMA-ENM performs transfer learning from this
> pre-trained BiLSTM-BAN model. Code: [BiLSTM-BAN-ProteinLigand](https://github.com/CHIHX12/BiLSTM-BAN-ProteinLigand)

---

## Abstract

While deep learning predicts drug binding affinities, most models remain black boxes unable to deduce dynamic allosteric mechanisms from sequence data. We present an interpretable deep transfer learning framework that decouples binding affinity from signaling efficacy using a bilinear attention network. Validating this structure-free paradigm on the growth hormone secretagogue receptor (GHSR) with 1,539 ligands, our model achieves robust predictive stability (AUROC = 0.96) and autonomously infers structural dynamics without crystallographic priors. Network topology analysis reveals a fundamental divergence in ligand recognition: agonists utilize a precise, modular architecture driven by Ser125, whereas antagonists form a synergistic web around hydrophobic hubs Phe286 and Pro278. Structural mapping confirms distinct physical mechanisms: agonists trigger remote activation via a Glu124-Arg283 salt bridge outside the primary pocket, while antagonists physically jam the deep hydrophobic crevasse. This spatial decoupling of binding contacts and functional triggering sites demonstrates that artificial intelligence can transcend simple pattern matching to predict allosteric machinery, providing a computational paradigm for navigating orphan receptor efficacy and designing functionally selective therapeutics.

---

## Overview

This repository provides the implementation of **TEMA-ENM**, the interpretable attention-topology
framework described in the paper above. It takes our previously published **BiLSTM-BAN** protein–ligand
model ([Cheng et al., *bioRxiv* 2026](https://doi.org/10.64898/2026.05.10.724184)) as the pre-trained
backbone and applies **transfer learning** to identify amino acid residues critical for GPCR
**activation** vs. **inhibition** — using only sequence information, without 3D structures.

### Key Contributions over the BiLSTM-BAN Base Model

| Feature | BiLSTM-BAN (base model, bioRxiv 2026) | TEMA-ENM (this work) |
|---------|---------------------------------------|----------------------|
| Protein encoder | BiLSTM (bidirectional, physicochemical features) | Same, fine-tuned on GPCR |
| Drug encoder | GCN or BiLSTM (SELFIES sequences) | Same, fine-tuned on GPCR |
| Prediction task | Binary binding | **3-class** (Active / Intermediate / Inactive) |
| Training strategy | Pre-training on BindingDB | **Transfer learning** (BindingDB → GPCR) |
| Multi-task | No | **Yes** (binding class + docking score regression) |
| Interpretability | Drug–protein attention | **Class-differential attention** (activation vs inhibition residues) |
| Topology analysis | No | **Residue co-attention network / overlap topology** |
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

The BindingDB splits used to pre-train the BiLSTM-BAN backbone are distributed with the base-model
repository, [BiLSTM-BAN-ProteinLigand](https://github.com/CHIHX12/BiLSTM-BAN-ProteinLigand):

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

### One-command reproduce (recommended)

All data is included in the repository. **Trained model parameters are deposited
separately at [doi:10.5281/zenodo.22841754](https://doi.org/10.5281/zenodo.22841754)** — see
[Model parameters](#model-parameters) below.

```bash
# Clone and set up environment
git clone https://github.com/CHIHX12/interpretable-attention-topology-drug-discovery.git
cd interpretable-attention-topology-drug-discovery
conda env create -f environment.yml
conda activate drugban

# Run the full analysis pipeline (~1 min on GPU)
bash reproduce.sh
```

This runs batch prediction on the provided fine-tuned model, extracts attention weights, performs class-differential analysis (active vs. inactive residues), generates consensus residue outputs including a PyMOL `.pml` script, and renders the publication network-overlap figures at **600 dpi**.

To re-run fine-tuning from scratch (~30 min on GPU):

```bash
bash reproduce.sh --retrain
```

### Model parameters

**Trained parameters are deposited openly at [doi:10.5281/zenodo.22841754](https://doi.org/10.5281/zenodo.22841754)**,
under CC BY-NC 4.0 with the additional terms in
[`MODEL-WEIGHTS-TERMS.md`](MODEL-WEIGHTS-TERMS.md). No request to the authors is
needed. That deposit holds the pre-trained backbone and all ten GHSR fine-tuned
checkpoints (seeds 42-51) with their per-seed epochs, test metrics and md5
checksums.

They live outside this repository for a licensing reason, not an access one: the
curated GHSR dataset inherits a ShareAlike obligation from ChEMBL and cannot
carry a NonCommercial term, so the two cannot sit under one licence in one
place. See [`LICENSE-DATA.md`](LICENSE-DATA.md).

| Expected path | Description | Val AUROC |
|---------------|-------------|-----------|
| `models/pretrained/DrugBAN_BiLSTM_BindingDB_epoch94.pth` | **BiLSTM-BAN base model** — pre-trained on BindingDB (binary binding, 50 epochs, best val); see [bioRxiv 2026](https://doi.org/10.64898/2026.05.10.724184) | — |
| `models/finetuned/DrugBAN_BiLSTM_GHSR_epoch36.pth` | **TEMA-ENM** — fine-tuned on GHSR, best epoch out of 50 total | **0.9621** |

Download the files from the deposit and place them at the paths above. Everything in this repository other than
the parameters is sufficient to retrain from scratch with `bash reproduce.sh --retrain`.

> **Training strategy**: the trainer runs for the full `MAX_EPOCH` (50) and saves the epoch with the best validation metric (val loss for multitask, AUROC for single-task). There is no early stopping — overfitting is prevented by selecting the best checkpoint rather than stopping training early.

### Step-by-step (manual)

**Step 1 — Fine-tune** (skip if using provided model):

```bash
python main.py \
    --cfg configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml \
    --data GPCR_resarch \
    --split random
```

**Step 2 — Batch prediction + attention extraction**:

```bash
python batch_predict_ghsr.py \
    --config configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml \
    --data_file datasets/GPCR_resarch/GHSR_training_data.csv \
    --model_path models/finetuned/DrugBAN_BiLSTM_GHSR_epoch36.pth \
    --output_dir datasets/GPCR_resarch/attention_results_reproduce
```

**Step 3 — Class-differential analysis** (active vs. inactive residues):

```bash
python analyze_class_attention_difference_en.py \
    --config configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml \
    --model_path models/finetuned/DrugBAN_BiLSTM_GHSR_epoch36.pth \
    --data_file datasets/GPCR_resarch/GHSR_training_data.csv \
    --output_dir result/class_diff_reproduce
```

**Step 4 — Consensus residue analysis**:

```bash
python consensus_analysis_ghsr.py \
    --attention_file datasets/GPCR_resarch/attention_results_reproduce/GHSR_attention_scores.npz \
    --output_dir datasets/GPCR_resarch/consensus_results_reproduce \
    --protein_length 523
```

**Step 5 — Publication network-overlap figures (600 dpi)**:

These scripts read the user-specified 25-pair tables
(`exact_five_{active,inactive}_25pairs.csv`, provided under
`Important_Analysis/{Active,Inactive}/`) and write into
`result/class_attention_analysis_pdb/`. All figures are rendered at **600 dpi**
with journal-sized fonts (PNG + vector PDF).

```bash
# Sharing matrix / overlap diagram / functional regions (Active & Inactive)
python analyze_network_overlap_active.py
python analyze_network_overlap_inactive.py

# 5×5 ΔImp heatmaps + constitutive pairing networks (Active & Inactive)
python create_exact_five_figures.py

# Inactive 3-way sharing diagram
python analyze_network_overlap_inactive_3way.py
```

| Script | Figures produced (600 dpi PNG + PDF) |
|--------|--------------------------------------|
| `analyze_network_overlap_active.py` / `_inactive.py` | `target_constitutive_sharing_matrix_*`, `network_overlap_diagram_*`, `functional_regions_3d_*` |
| `create_exact_five_figures.py` | `exact_five_{active,inactive}_heatmap`, `exact_five_{active,inactive}_network` |
| `analyze_network_overlap_inactive_3way.py` | `network_overlap_diagram_inactive_3way` |

> All plotting scripts set `savefig.dpi = 600` and enlarged journal fonts via
> `matplotlib.rcParams`. To change the export resolution, edit the
> `plt.rcParams.update({...})` block near the top of each script.

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
| `batch_predict_ghsr.py` | **Primary inference script** — runs all 1,539 GHSR pairs, extracts attention, saves `.npz` |
| `consensus_analysis_ghsr.py` | Consensus residues — finds positions with high attention frequency across all drugs (class-agnostic); generates PyMOL `.pml` with PDB numbering |
| `analyze_class_attention_difference_en.py` | Class-differential attention: active vs inactive (Welch's t-test per residue) |
| `analyze_class_attention_difference.py` | Same as above (Chinese-annotated version) |
| `analyze_class_attention_with_pdb_numbering.py` | Class-differential analysis with PDB residue numbering (dataset_pos + 2 = PDB number) |
| `analyze_constitutive_variable_residues.py` | Identify constitutive vs variable residues |
| `aggregate_attention_analysis.py` | Aggregate attention across multiple drugs per protein |
| `aggregate_attention_by_protein.py` | Per-protein attention aggregation |
| `train_save_all_epochs.py` | Train and save a checkpoint at **every** epoch (for checkpoint selection) |
| `scripts/extract_attention_weights.py` | Extract BAN attention for all drug-protein pairs |
| `scripts/plot_class_differential_attention.py` | Plot attention heatmaps |
| `scripts/statistical_tests.py` | Statistical significance testing |
| `scripts/evaluate_loro_results.py` | LORO cross-validation performance report |

### Attention Aggregation Definition

BAN produces raw attention logits of shape `[batch, heads, N_drug_atoms, L_protein]`.  
All scripts use the following two-step aggregation:

```
Step 1: mean over attention heads  →  [batch, N_drug_atoms, L_protein]
Step 2: mean over drug atoms       →  [batch, L_protein]
```

> **Why mean, not max?**  
> Max-pooling over drug atoms selects only the single most-attended atom per protein position. Across samples this collapses per-sample variance and eliminates class-differential signals — empirically, `max` reduces Glu124's unique attention values from ~1,500 to 2 and its std from 0.37 to ~0.  
> `mean` retains the full distribution and recovers the active-preferred signal at Glu124 (E124) and Ser125 (S125) consistently across all trained models.

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
├── reproduce.sh               # One-command reproducibility script
├── LICENSE.md                 # PolyForm Noncommercial 1.0.0 (source code)
├── LICENSE-DATA.md            # CC BY-NC 4.0 (data and figures)
├── MODEL-WEIGHTS-TERMS.md     # Model parameters — request-based, noncommercial
├── THIRD-PARTY-NOTICES.md     # DrugBAN MIT notice + dependency licenses
├── models/                    # NOT in the repository — see MODEL-WEIGHTS-TERMS.md
│   ├── pretrained/            #   place DrugBAN_BiLSTM_BindingDB_epoch94.pth here
│   └── finetuned/             #   place DrugBAN_BiLSTM_GHSR_epoch36.pth here
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
├── datasets/
│   ├── bindingdb/             # Pre-training data (large — see datasets/README.md)
│   └── GPCR_resarch/          # GPCR fine-tuning data (included in repo)
│       ├── GHSR_training_data.csv   # Full 1,539 drug-protein pairs
│       └── random/                  # Train / val / test splits (seed=42)
│
├── batch_predict_ghsr.py              # Primary inference + attention extraction
├── consensus_analysis_ghsr.py        # Consensus residue analysis + PyMOL script generation
├── train_save_all_epochs.py           # Save checkpoint every epoch (for selection)
├── analyze_class_attention_difference_en.py  # Class-differential residue analysis (EN)
├── analyze_class_attention_difference.py     # Class-differential residue analysis (ZH)
├── analyze_class_attention_with_pdb_numbering.py  # Same, with PDB numbering
├── analyze_constitutive_variable_residues.py
├── aggregate_attention_analysis.py
├── aggregate_attention_by_protein.py
│
├── analyze_network_overlap.py             # Network-overlap figures (base / active data)
├── analyze_network_overlap_active.py      # Active: sharing matrix, overlap, 3D regions (600 dpi)
├── analyze_network_overlap_inactive.py    # Inactive: sharing matrix, overlap, 3D regions (600 dpi)
├── analyze_network_overlap_inactive_3way.py  # Inactive 3-way sharing diagram (600 dpi)
├── create_exact_five_figures.py           # 5×5 ΔImp heatmaps + pairing networks (600 dpi)
├── Important_Analysis/                     # Curated 600-dpi figures + 25-pair input tables
│   ├── Active/    exact_five_active_25pairs.csv + figures
│   └── Inactive/  exact_five_inactive_25pairs.csv + figures
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

1. **Training**: The model learns which protein residues are attended when drugs bind with high vs. low affinity.
2. **Per-sample aggregation**: For each drug–protein pair, the raw BAN attention `[heads, N_atoms, L_protein]` is reduced to a per-residue vector `[L_protein]` by (i) averaging over heads, then (ii) averaging over drug atoms. See [Attention Aggregation Definition](#attention-aggregation-definition) for the rationale.
3. **Class-level comparison**: Residue attention vectors are grouped by activity class (active / inactive) and compared with Welch's t-test. Residues with significantly higher attention in active pairs are candidates for activation roles.
4. **Constitutive residues**: Residues with uniformly high attention across all classes form the core binding scaffold.

> **PDB numbering**: The protein sequence in the dataset starts at position 0. PDB residue number = dataset position + 2 (e.g., dataset index 122 → PDB Glu124).

This approach has recovered known binding site residues in GPCR structures with up to 33% precision using only sequence information.

---

## Citation

If you use this work, please cite **both** the BiLSTM-BAN base model and this repository:

**Base model (BiLSTM-BAN)** — the pre-trained backbone used for transfer learning:
```bibtex
@article{cheng2026bilstmban,
  title   = {BiLSTM-Powered Bilinear Attention for Protein--Ligand Prediction},
  author  = {Chih-Yang Cheng and Yi-An Chen and Feng-Yin Li and Suyong Re},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.05.10.724184},
  url     = {https://doi.org/10.64898/2026.05.10.724184}
}
```

**This Work (TEMA-ENM)**:
```bibtex
@article{cheng2026temaenm,
  title   = {TEMA-ENM: An Interpretable Attention-Topology Framework Decouples
             Affinity and Efficacy from Sequence Alone},
  author  = {Chih-Yang Cheng and Yi-Huan Wu and Feng-Yin Li},
  year    = {2026},
  url     = {https://github.com/CHIHX12/interpretable-attention-topology-drug-discovery}
}
```

---

## License

TEMA-ENM is **free for noncommercial use** — academic research, teaching, and
reproduction of the published results. Commercial use requires a separate license.

| Component | License |
|-----------|---------|
| Source code | [PolyForm Noncommercial 1.0.0](LICENSE.md) |
| Model parameters | [TEMA-ENM Model Parameters Terms of Use](MODEL-WEIGHTS-TERMS.md) — on request, noncommercial |
| GHSR dataset | [CC BY-SA 4.0](LICENSE-DATA.md) — ShareAlike, inherited from ChEMBL |
| Figures and analysis outputs | [CC BY-NC 4.0](LICENSE-DATA.md) |
| Upstream components | [Third-party notices](THIRD-PARTY-NOTICES.md) |

> The GHSR dataset is assembled from **ChEMBL** (CC BY-SA 3.0) and **BindingDB**
> (CC BY 4.0). ChEMBL's ShareAlike term means the dataset is **CC BY-SA 4.0** and
> cannot carry a NonCommercial restriction. This applies to the dataset only —
> not to the source code, and not to the model parameters.

Use by academic institutions, public research organizations, and government
research bodies is noncommercial under these terms, regardless of funding source.

**For commercial licensing**, contact Chih-Yang Cheng
([ORCID](https://orcid.org/0009-0002-2694-247X)), Department of Chemistry,
National Chung Hsing University.

### Third-party code

Portions of the training scaffold (`ban.py`, parts of `models.py`, `trainer.py`,
`configs.py`, `dataloader.py`) originate from [DrugBAN](https://github.com/peizhenbai/DrugBAN),
released under the MIT License (Copyright © 2022 Peizhen Bai). The MIT License
permits sublicensing; the original notice is retained in
[`THIRD-PARTY-NOTICES.md`](THIRD-PARTY-NOTICES.md) as required, and this does not
restrict anyone's independent right to obtain DrugBAN under its own MIT terms.

TEMA-ENM's scientific contribution — the BiLSTM encoders, transfer-learning
protocol, 3-class multi-task head, and class-differential attention-topology
analysis — is our own work, described in the publications under [Citation](#citation).

---

## References

1. Cheng, Chen, Li & Re (2026). BiLSTM-Powered Bilinear Attention for Protein–Ligand Prediction. *bioRxiv*. doi:10.64898/2026.05.10.724184.
2. Bai et al. (2023). Interpretable bilinear attention network with domain adaptation improves drug-target prediction. *Nature Machine Intelligence*. — upstream code base, see [Third-party code](#third-party-code).
3. Zdrazil et al. (2024). The ChEMBL Database in 2023. *Nucleic Acids Research*, 52(D1), D1180–D1192. — GHSR activity data source.
4. Liu et al. (2007). BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities. *Nucleic Acids Research*.
5. Huang et al. (2021). MolTrans: Molecular Interaction Transformer for drug-target interaction prediction. *Bioinformatics*.
6. Kim et al. (2018). Bilinear attention networks. *NeurIPS*.
