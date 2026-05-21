# Datasets

## Pre-training Data (BindingDB, BioSNAP, Human)

All pre-training data are from public sources. Download from the original DrugBAN repository:

```bash
# Download the original DrugBAN datasets
git clone https://github.com/peizhenbai/DrugBAN.git
cp -r DrugBAN/datasets/bindingdb ./bindingdb
cp -r DrugBAN/datasets/biosnap   ./biosnap
cp -r DrugBAN/datasets/human     ./human
```

### Expected directory structure after download:

```
datasets/
├── bindingdb/
│   ├── random/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── cluster/
│       ├── source_train.csv
│       ├── target_train.csv
│       └── target_test.csv
├── biosnap/
│   └── (same structure)
└── human/
    └── random/
        ├── train.csv
        ├── val.csv
        └── test.csv
```

### CSV Format

Each row represents a drug-protein pair:

```
SMILES,Protein,Y
CC(C)CC1=CC=CC=C1...,MGAASGRRGP...,1
```

- `SMILES`: Drug SMILES string
- `Protein`: Protein amino acid sequence (single-letter code)
- `Y`: Binary label (1 = binding, 0 = non-binding)

---

## GPCR Fine-tuning Data (GHSR)

The GPCR dataset used for transfer learning targets the **Growth Hormone Secretagogue Receptor (GHSR, UniProt Q92847)**.

### Data Format

```
SMILES,Protein,Y,cnnscore
CC(C)CC1=CC=CC=C1NC(=O)...,MGENSSPALLPETGEDKFPAMPL...,2,0.823
```

Additional columns:
- `Y`: Activity class
  - `0` = Inactive (IC50 > 10,000 nM)
  - `1` = Intermediate (100 nM < IC50 ≤ 10,000 nM)
  - `2` = Active (IC50 ≤ 100 nM)
- `cnnscore`: Docking CNNscore from AutoDock-GPU (used for multi-task regression)

### Preparing GPCR Splits

```bash
# Split into train/val/test with stratification
python split_ghsr_data.py \
    --input datasets/GPCR_resarch/GHSR_training_data.csv \
    --output datasets/GPCR_resarch/random \
    --stratify \
    --seed 42

# Create LORO (Leave-One-Receptor-Out) folds
python scripts/create_loro_splits.py \
    --input datasets/GPCR_resarch/GHSR_training_data.csv \
    --output datasets/GPCR_resarch \
    --n_folds 12
```

---

## References

1. Liu et al. (2007). BindingDB: a web-accessible database of experimentally determined protein-ligand binding affinities. *Nucleic Acids Research*, 35(suppl_1), D198-D201.
2. Huang et al. (2021). MolTrans: Molecular Interaction Transformer for drug-target interaction prediction. *Bioinformatics*, 37(6), 830-836.
3. Chen et al. (2020). TransformerCPI: improving compound-protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiments. *Bioinformatics*, 36(16), 4406-4414.
