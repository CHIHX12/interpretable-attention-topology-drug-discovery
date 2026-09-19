# Revision materials (v22)

Everything needed to reproduce the revised manuscript, in the order it is run.
Referee 1 could not locate the step that generates the pairing tables, or the
elastic-network scripts. Both are here, with their inputs and expected outputs.

All commands are run **from the repository root**, not from this directory.

```
revision/
├── *.py            data preparation, training, and the residue-level analysis
├── *.sh            the queues used to run the training grid
├── enm/            elastic-network and anisotropic-network analysis (Section 2.7, 3.6)
└── figures/        figure generation (main text Fig. 1-6)
```

## 1. Partitions

```bash
python revision/make_splits.py
```

Writes `datasets/GPCR_resarch/rev22_{compound,scaffold,nodual}_s{0..4}/` and the
audit report `rev22_split_summary.json`: class counts per subset, exact and
canonical SMILES overlap, test scaffolds seen in training, and the
nearest-neighbour ECFP4 Tanimoto distribution. This is the split with no
identical compound shared across partitions that Referee 1 asked for
(checklist 5c).

## 2. Training

```bash
bash revision/run_rev22_queue.sh      # descriptors supplied (the reported models)
bash revision/run_rev22_featoff.sh    # descriptors withheld (probing-trained control)
```

Both call `revision/train_rev22.py`, which takes the base configuration
`configs/DrugBAN_BiLSTM_GHSR_Reproduce.yaml` and overrides only the seed, the
partition and the initialisation. Ten fine-tuning seeds (42-51) on the deposited
partition, five partition replicates on each leakage-controlled family, each
trained both from the BindingDB checkpoint and from random initialisation.

Fine-tuning is deterministic on our hardware: re-running the deposited
configuration from the deposited pre-trained parameters reproduces the
fine-tuned parameter file bit for bit.

## 3. Reference models

```bash
python revision/ligand_baselines.py
```

Majority class, Tanimoto 1-nearest-neighbour, L2 logistic regression and a
500-tree random forest, all on ECFP4, on every partition. These are the
ligand-only baselines Referee 2 asked for, and they match or exceed the present
model on every leakage-controlled partition.

## 4. Attention read-out

```bash
python revision/extract_attention_masked.py --model <checkpoint> --out <npz>
python revision/differential_attention.py
```

The extraction saves five read-out variants per model. The analysed one is
`unmasked_max`: mean over heads, then maximum over the 290 drug-graph rows,
with the four physicochemical channels set to zero. The encoder pads to 1,200
positions, and the saved arrays cover the 523 real positions.

## 5. Tables and controls

| Command | Produces |
|---|---|
| `python revision/table1_rev22.py` | Table 1, and `result/rev22/table1_residues.csv` for all 302 residues |
| `python revision/table2_pairing.py` | Table 2, and the per-seed values behind it |
| `python revision/attention_probe.py` | the linear probe on the attention profile (SI 5.1) |
| `python revision/feature_dependence.py` | how much of Δ the four descriptors explain (SI 5.1) |
| `python revision/pocket_enrichment.py` | proximity to the ligand against two null models (SI 5.1) |
| `python revision/summarize_rev22.py` | `result/rev22/summary_table.csv`, the per-partition performance |
| `python revision/tema_rev22.py` | the pairing run on the pre-trained model |

`table2_pairing.py` is the step Referee 1 could not find. It applies the rule of
Section 2.6 — constitutive pool `|Δ| < 0.1` and `I > 1.5`, five partners per
target by the smallest importance gap, target excluded from its own pool — over
the ten fine-tuning seeds, and reproduces every value of Table 2.

## 6. Elastic network

See [enm/README.md](enm/README.md). Those scripts take experimental coordinates
only and no machine-learning input.

## 7. Figures

```bash
python revision/figures/make_schematics.py   # Fig. 1, Fig. 2
python revision/figures/make_figures.py      # Fig. 3, Fig. 4, Fig. 5
```

Fig. 6 is the original render, unchanged; `revision/figures/make_fig6_pymol.py`
and the `Fig6_*.pml` scripts build the alternative version that was not used.
They expect `6ko5.pdb` and `8jsr.pdb` beside them; copy those from
`revision/enm/` rather than keeping a second set in the repository.

## Outputs

The small tables the manuscript cites are tracked under `result/rev22/`. The
attention arrays (about 300 MB) and the model parameters are not; see the
repository README for how to obtain them.
